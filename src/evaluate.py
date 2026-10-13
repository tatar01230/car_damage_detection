"""Evaluate trained runs on the held-out test split and compare them.

For every run (runs/<ID>/weights/best.pt) this measures:
  - test mAP50, mAP50-95, precision, recall, per-class P/R/AP50/AP50-95, confusion matrix
  - inference latency at batch size 1 (median wall-clock ms/image incl. pre/postprocess)
    on the local GPU and on the CPU (pinned thread count, see --cpu-threads)
  - parameters, GFLOPs at the run's imgsz, best.pt size, training time (from summary.json)

Each model is evaluated at the imgsz it was trained with (read from the run's args.yaml).

Outputs (reports/evaluation/):
  test_metrics.csv, per_class.csv, accuracy_vs_speed.png, per_class_ap.png,
  confusion_matrix_<ID>.png

Usage:
    python -m src.evaluate --runs E1 E2 E3 E4 E5
"""

import argparse
import json
import platform
import shutil
import statistics
import time
from pathlib import Path

import cv2
import matplotlib
import pandas as pd
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BAR_COLOR, HIGHLIGHT = "#2a78d6", "#eb7a2e"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def cpu_name() -> str:
    if platform.system() == "Windows":
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def latency_ms(weights: Path, images: list, imgsz: int, device: str, cpu_threads: int, warmup: int = 20) -> float:
    """Median wall-clock latency per image at batch size 1 (preprocess + inference + NMS),
    images already decoded in memory. A fresh model instance per device: ultralytics keeps its
    predictor on the first device it was used with."""
    model = YOLO(str(weights))
    for img in images[:warmup]:
        model.predict(img, imgsz=imgsz, device=device, verbose=False)
    torch.set_num_threads(cpu_threads)  # ultralytics' select_device resets this to 8 on CPU
    times = []
    for img in images:
        start = time.perf_counter()
        model.predict(img, imgsz=imgsz, device=device, verbose=False)
        if device != "cpu":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def benchmark(runs: dict[str, tuple[Path, int]], images: list, n_cpu: int, cpu_threads: int, rounds: int) -> dict:
    """Latency per run and device. Runs are measured in interleaved rounds and the minimum of the
    per-round medians is kept: on this Windows desktop, background throttling made single
    measurements of the *same* model vary up to 4x, and the minimum filters that interference."""
    devices = (["0"] if torch.cuda.is_available() else []) + ["cpu"]
    results = {(r, d): [] for r in runs for d in devices}
    for i in range(rounds):
        for run_id, (weights, imgsz) in runs.items():
            for device in devices:
                imgs = images if device != "cpu" else images[:n_cpu]
                results[run_id, device].append(latency_ms(weights, imgs, imgsz, device, cpu_threads))
        print(f"latency round {i + 1}/{rounds}: " + ", ".join(f"{r}/{d}={v[-1]:.1f}" for (r, d), v in results.items()))
    return {k: min(v) for k, v in results.items()}


def evaluate_run(run_dir: Path, data: str, out: Path) -> tuple[dict, pd.DataFrame]:
    run_id = run_dir.name
    imgsz = yaml.safe_load((run_dir / "args.yaml").read_text())["imgsz"]
    summary = json.loads((run_dir / "summary.json").read_text())
    best = run_dir / "weights" / "best.pt"
    model = YOLO(str(best))

    m = model.val(
        data=data, split="test", imgsz=imgsz, batch=16, plots=True,
        project=str(run_dir.resolve()), name="test_eval", exist_ok=True, verbose=False,
    )
    shutil.copy(run_dir / "test_eval" / "confusion_matrix_normalized.png", out / f"confusion_matrix_{run_id}.png")

    names = model.names
    per_class = pd.DataFrame(
        {
            "run": run_id,
            "class": [names[c] for c in m.box.ap_class_index],
            "precision": m.box.p,
            "recall": m.box.r,
            "AP50": m.box.ap50,
            "AP50_95": m.box.ap,
        }
    )

    row = {
        "id": run_id,
        "model": summary["model"],
        "setting": summary["purpose"].split(" (")[0],
        "imgsz": imgsz,
        "test_mAP50": m.box.map50,
        "test_mAP50_95": m.box.map,
        "test_precision": m.box.mp,
        "test_recall": m.box.mr,
        "params_M": sum(p.numel() for p in model.model.parameters()) / 1e6,
        "GFLOPs": get_flops(model.model, imgsz),
        "best_pt_mb": best.stat().st_size / 2**20,
        "train_time_min": summary["train_time_min"],
        "train_gpu": summary["gpu"],
    }
    return row, per_class


def plot_accuracy_vs_speed(df: pd.DataFrame, path: Path, hw: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df["cpu_ms"], df["test_mAP50_95"], s=60, color=BAR_COLOR, zorder=3)
    for _, r in df.iterrows():
        label = f"{r['id']} {Path(r['model']).stem} @{r['imgsz']}" + (" freeze" if "frozen" in r["setting"] else "")
        ax.annotate(label, (r["cpu_ms"], r["test_mAP50_95"]), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel(f"CPU latency, batch 1 (ms/image, {hw})", fontsize=9, color=INK)
    ax.set_ylabel("test mAP50-95", fontsize=9, color=INK)
    ax.set_title("Accuracy vs speed (up and left is better; y-axis zoomed)", fontsize=10, color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_per_class(per_class: pd.DataFrame, run_id: str, path: Path) -> None:
    d = per_class[per_class["run"] == run_id].sort_values("AP50_95")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(d["class"], d["AP50_95"], color=BAR_COLOR, height=0.7)
    for y, v in enumerate(d["AP50_95"]):
        ax.text(v, y, f" {v:.2f}", va="center", fontsize=7, color=INK)
    ax.set_xlabel("test AP50-95", fontsize=9, color=INK)
    ax.set_title(f"Per-class AP50-95 on test ({run_id})", fontsize=10, color=INK, loc="left")
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["E1", "E2", "E3", "E4", "E5"])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--out", type=Path, default=Path("reports/evaluation"))
    parser.add_argument("--n-speed", type=int, default=200, help="test images used for GPU latency")
    parser.add_argument("--n-cpu", type=int, default=100, help="test images used for CPU latency")
    # Fixed thread count so CPU numbers are comparable across runs (6 = P-cores of the i5-13400F).
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=3, help="interleaved latency rounds (min of medians)")
    parser.add_argument("--per-class-run", default=None, help="run to plot per-class AP for (default: best mAP50-95)")
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.data).read_text())
    test_dir = Path(cfg["path"]) / cfg["test"]
    images = [cv2.imread(str(p)) for p in sorted(test_dir.iterdir(), key=lambda p: p.name)[: args.n_speed]]

    # Latency first, before any val() run leaves dataloader workers or plotting work behind
    runs = {
        r: (args.runs_dir / r / "weights" / "best.pt", yaml.safe_load((args.runs_dir / r / "args.yaml").read_text())["imgsz"])
        for r in args.runs
    }
    latency = benchmark(runs, images, args.n_cpu, args.cpu_threads, args.rounds)

    rows, per_class = [], []
    for run_id in args.runs:
        print(f"Evaluating {run_id} ...")
        row, pc = evaluate_run(args.runs_dir / run_id, args.data, args.out)
        row["gpu_ms"] = latency.get((run_id, "0"))
        row["cpu_ms"] = latency[run_id, "cpu"]
        rows.append(row)
        per_class.append(pc)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    cpu = f"{cpu_name()} ({args.cpu_threads} threads)"
    df = pd.DataFrame(rows).round(4)
    df["eval_gpu"], df["eval_cpu"] = gpu, cpu
    pcs = pd.concat(per_class).round(4)
    df.to_csv(args.out / "test_metrics.csv", index=False)
    pcs.to_csv(args.out / "per_class.csv", index=False)

    plot_accuracy_vs_speed(df, args.out / "accuracy_vs_speed.png", cpu)
    best_id = args.per_class_run or df.loc[df["test_mAP50_95"].idxmax(), "id"]
    plot_per_class(pcs, best_id, args.out / "per_class_ap.png")

    pd.set_option("display.width", 200)
    print(df.drop(columns=["eval_gpu", "eval_cpu", "train_gpu"]).to_string(index=False))
    print(f"\nLatency hardware: GPU={gpu}, CPU={cpu}")
    print(pcs.pivot(index="class", columns="run", values="AP50_95").to_string())


if __name__ == "__main__":
    main()
