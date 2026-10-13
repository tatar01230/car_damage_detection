"""Train one experiment (E1-E5) from its config file, with resume support.

Settings = configs/train_base.yaml, overridden by the experiment's `train:` block, overridden by
--set. Each run lives in <project>/<id>/ (use a Google Drive folder on Colab so a disconnect
loses at most one epoch):
  - weights/last.pt exists, no summary.json  -> resume the interrupted run
  - summary.json exists                       -> already finished, skip (use --force to retrain)

After training, best.pt is validated on the *val* split (the test split is reserved for the
final evaluation, Day 5) and one row per experiment is written to the results CSV.

Usage:
    python -m src.train --exp configs/experiments/E1.yaml --project runs
    python -m src.train --exp configs/experiments/E1.yaml --project runs/smoke --set epochs=1 fraction=0.05
"""

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import torch
import ultralytics
import yaml
from ultralytics import YOLO

BASE_CONFIG = Path("configs/train_base.yaml")


def load_config(exp_path: Path, overrides: list[str]) -> tuple[dict, dict]:
    exp = yaml.safe_load(exp_path.read_text())
    args = yaml.safe_load(BASE_CONFIG.read_text())
    args.update(exp.get("train") or {})
    for item in overrides:
        key, value = item.split("=", 1)
        args[key] = yaml.safe_load(value)  # "1" -> 1, "true" -> True, "0.05" -> 0.05
    return exp, args


def training_time_s(results_csv: Path) -> float:
    """Total training time from ultralytics' cumulative 'time' column. The timer restarts on
    resume, so sum the last value of every increasing segment."""
    times = pd.read_csv(results_csv)["time"].tolist()
    return sum(t for t, nxt in zip(times, times[1:] + [0.0]) if nxt < t)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def upsert_row(csv_path: Path, row: dict) -> None:
    """Write/replace this experiment's row so re-running never duplicates rows."""
    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    if not df.empty:
        df = df[df["id"] != row["id"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True).sort_values("id")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", type=Path, required=True, help="experiment config, e.g. configs/experiments/E1.yaml")
    parser.add_argument("--project", type=Path, default=Path("runs"), help="folder that holds one subfolder per run")
    parser.add_argument("--results-csv", type=Path, default=Path("reports/experiments.csv"))
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="extra overrides (smoke tests)")
    parser.add_argument("--force", action="store_true", help="delete an existing run and train from scratch")
    args = parser.parse_args()

    exp, train_args = load_config(args.exp, args.set)
    exp_id = exp["id"]
    # ultralytics nests *relative* project paths under its global runs_dir, so pass an absolute path
    project = args.project.resolve()
    run_dir = project / exp_id
    summary_path = run_dir / "summary.json"
    last = run_dir / "weights" / "last.pt"

    if args.force and run_dir.exists():
        shutil.rmtree(run_dir)

    if summary_path.exists():
        print(f"{exp_id} already finished ({summary_path}); use --force to retrain.")
        upsert_row(args.results_csv, json.loads(summary_path.read_text()))
        return

    if last.exists():
        print(f"Resuming {exp_id} from {last}")
        YOLO(str(last)).train(resume=True)  # restores all original args from the checkpoint
    else:
        print(f"Training {exp_id}: {exp['model']} with {train_args}")
        YOLO(exp["model"]).train(**train_args, project=str(project), name=exp_id, exist_ok=True)

    # Validate the best checkpoint on the val split
    best = run_dir / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(
        data=train_args["data"],
        split="val",
        imgsz=train_args["imgsz"],
        batch=train_args["batch"],
        project=str(run_dir),
        name="val_best",
        exist_ok=True,
        plots=False,
    )
    results = pd.read_csv(run_dir / "results.csv")
    results.columns = results.columns.str.strip()

    summary = {
        "id": exp_id,
        "model": exp["model"],
        "purpose": exp["purpose"],
        "overrides": json.dumps(exp.get("train") or {}),
        "epochs": int(results["epoch"].max()),
        "imgsz": train_args["imgsz"],
        "batch": train_args["batch"],
        "seed": train_args["seed"],
        "best_epoch": int(results.loc[results["metrics/mAP50-95(B)"].idxmax(), "epoch"]),
        "val_mAP50": round(float(metrics.box.map50), 4),
        "val_mAP50_95": round(float(metrics.box.map), 4),
        "val_precision": round(float(metrics.box.mp), 4),
        "val_recall": round(float(metrics.box.mr), 4),
        "train_time_min": round(training_time_s(run_dir / "results.csv") / 60, 1),
        "best_pt_mb": round(best.stat().st_size / 2**20, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "git_commit": git_commit(),
        "extra_overrides": " ".join(args.set),  # non-empty only for smoke tests
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    upsert_row(args.results_csv, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
