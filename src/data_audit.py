"""Audit a YOLO-format detection dataset and write a report.

Checks:
  - class distribution / imbalance per split
  - exact and near-duplicate images (perceptual hash), incl. leakage across splits
  - invalid labels: bad format, unknown class id, out-of-bounds, degenerate, duplicate boxes
  - empty / missing label files, orphan labels, corrupted images
  - image size / aspect ratio and box size distributions

Expected layout: <data>/data.yaml and <data>/<split>/{images,labels}/.

Usage:
    python -m src.data_audit --data data/raw/car-damage-v2 --out reports/data_audit/raw
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import imagehash
import matplotlib
import numpy as np
import yaml
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "val", "test")

TINY_AREA = 0.001  # box area / image area; about 20x20 px on a 640x640 image
MIN_SIDE_PX = 2  # boxes thinner than this (after clipping) carry no usable signal
DUP_BOX_IOU = 0.9  # two same-class boxes overlapping this much in one image = duplicate annotation
OOB_TOL = 1e-3  # tolerate float rounding at the image border

# Chart styling (single-hue bars; recessive axes)
BAR_COLOR = "#2a78d6"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


# ----------------------------------------------------------------------------- loading


def read_class_names(root: Path) -> list[str]:
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def parse_label_file(path: Path, width: int, height: int, num_classes: int) -> list[dict]:
    """Parse one YOLO label file. Every line becomes a box dict with a list of issues.

    Box coords are stored as normalized xyxy *clipped* to [0, 1]; 'out_of_bounds' records
    that clipping was needed.
    """
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        box = {"cls": None, "xyxy": None, "area": 0.0, "issues": []}
        boxes.append(box)
        try:
            values = [float(p) for p in parts]
        except ValueError:
            box["issues"].append("bad_format")
            continue
        if len(values) == 5:
            cls, cx, cy, w, h = values
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        elif len(values) >= 7 and len(values) % 2 == 1:  # polygon -> enclosing box
            cls, xs, ys = values[0], values[1::2], values[2::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            box["issues"].append("polygon")
        else:
            box["issues"].append("bad_format")
            continue

        if cls != int(cls) or not 0 <= cls < num_classes:
            box["issues"].append("unknown_class")
        box["cls"] = int(cls)
        if min(x1, y1) < -OOB_TOL or max(x2, y2) > 1 + OOB_TOL:
            box["issues"].append("out_of_bounds")
        x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
        box["xyxy"] = [x1, y1, x2, y2]
        box["area"] = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if (x2 - x1) * width < MIN_SIDE_PX or (y2 - y1) * height < MIN_SIDE_PX:
            box["issues"].append("degenerate")
        elif box["area"] < TINY_AREA:
            box["issues"].append("tiny")  # reported, not removed: small damage is real

    # Duplicate annotations: same class, near-identical box (keep the first one)
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            if a["xyxy"] and b["xyxy"] and a["cls"] == b["cls"] and "duplicate" not in b["issues"]:
                if box_iou(a["xyxy"], b["xyxy"]) >= DUP_BOX_IOU:
                    b["issues"].append("duplicate")
    return boxes


def load_dataset(root: Path, names: list[str]) -> tuple[list[dict], list[str]]:
    """Return one record per image, plus label files that have no matching image."""
    records, orphan_labels = [], []
    for split in SPLITS:
        image_dir, label_dir = root / split / "images", root / split / "labels"
        if not image_dir.is_dir():
            continue
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        image_stems = {p.stem for p in images}
        orphan_labels += [str(p) for p in label_dir.glob("*.txt") if p.stem not in image_stems]

        for image_path in images:
            rec = {
                "split": split,
                "image": str(image_path),
                "label": str(label_dir / f"{image_path.stem}.txt"),
                "width": None,
                "height": None,
                "phash": None,
                "issues": [],
                "boxes": [],
            }
            records.append(rec)
            try:
                with Image.open(image_path) as img:
                    img.load()  # force full decode so truncated files are caught
                    rec["width"], rec["height"] = img.size
                    rec["phash"] = str(imagehash.phash(img.convert("RGB")))
            except Exception:
                rec["issues"].append("corrupted")
                continue

            label_path = Path(rec["label"])
            if not label_path.exists():
                rec["issues"].append("missing_label")
                continue
            rec["boxes"] = parse_label_file(label_path, rec["width"], rec["height"], len(names))
            if not rec["boxes"]:
                rec["issues"].append("empty_label")
    return records, orphan_labels


# ----------------------------------------------------------------------------- duplicates


def find_duplicate_groups(records: list[dict], max_dist: int) -> tuple[list[list[int]], list[tuple]]:
    """Group images that are near-duplicates (pHash Hamming distance <= max_dist).
    Returns (groups of record indices, matched pairs).

    Note: Roboflow source file names are NOT used for grouping. In this dataset generic
    names like 'images-1-_jpg' were checked visually and belong to different photos.
    """
    idx = [i for i, r in enumerate(records) if r["phash"]]
    hashes = np.array([int(records[i]["phash"], 16) for i in idx], dtype=np.uint64)

    parent = list(range(len(records)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []  # (i, j, distance)
    for k in range(len(idx) - 1):
        dists = np.bitwise_count(hashes[k] ^ hashes[k + 1 :])
        for offset in np.nonzero(dists <= max_dist)[0]:
            i, j = idx[k], idx[k + 1 + offset]
            pairs.append((i, j, int(dists[offset])))

    for i, j, _ in pairs:
        parent[find(i)] = find(j)

    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1], pairs


# ----------------------------------------------------------------------------- summary


def summarize(records, orphan_labels, groups, pairs, names) -> dict:
    splits = sorted({r["split"] for r in records}, key=SPLITS.index)
    valid_boxes = [
        (r, b) for r in records for b in r["boxes"] if b["xyxy"] and "unknown_class" not in b["issues"]
    ]

    class_counts = {s: Counter() for s in splits}
    for r, b in valid_boxes:
        class_counts[r["split"]][names[b["cls"]]] += 1

    image_issues = Counter(i for r in records for i in r["issues"])
    box_issues = Counter(i for r in records for b in r["boxes"] for i in b["issues"])

    cross_split = [g for g in groups if len({records[i]["split"] for i in g}) > 1]
    label_conflicts = [
        g for g in groups if len({tuple(sorted(b["cls"] for b in records[i]["boxes"])) for i in g}) > 1
    ]

    sizes = Counter((r["width"], r["height"]) for r in records if r["width"])
    aspects = np.array([r["width"] / r["height"] for r in records if r["width"]])
    areas = np.array([b["area"] for _, b in valid_boxes])

    total = {name: sum(class_counts[s][name] for s in splits) for name in names}
    nonzero = [v for v in total.values() if v > 0]

    return {
        "images_per_split": {s: sum(r["split"] == s for r in records) for s in splits},
        "boxes_per_split": {s: sum(class_counts[s].values()) for s in splits},
        "class_counts": {s: {n: class_counts[s][n] for n in names} for s in splits},
        "class_counts_total": total,
        "imbalance_ratio_max_min": round(max(nonzero) / min(nonzero), 1) if nonzero else None,
        "image_issues": dict(image_issues),
        "box_issues": dict(box_issues),
        "orphan_labels": len(orphan_labels),
        "duplicates": {
            "near_duplicate_pairs": len(pairs),
            "groups": len(groups),
            "images_in_groups": sum(len(g) for g in groups),
            "redundant_images": sum(len(g) - 1 for g in groups),
            "cross_split_groups": len(cross_split),
            "groups_with_label_conflicts": len(label_conflicts),
        },
        "image_sizes_top5": [[f"{w}x{h}", n] for (w, h), n in sizes.most_common(5)],
        "unique_image_sizes": len(sizes),
        "aspect_ratio": {
            "min": round(float(aspects.min()), 2),
            "median": round(float(np.median(aspects)), 2),
            "max": round(float(aspects.max()), 2),
        },
        "box_area_ratio": {
            "p05": round(float(np.percentile(areas, 5)), 4),
            "median": round(float(np.median(areas)), 4),
            "p95": round(float(np.percentile(areas, 95)), 4),
            "tiny_share": round(float((areas < TINY_AREA).mean()), 4),
        },
        "boxes_per_image_mean": round(len(valid_boxes) / max(1, len(records)), 2),
    }


def write_markdown(summary: dict, path: Path, title: str) -> None:
    s = summary
    splits = list(s["images_per_split"])
    lines = [f"# Data audit: {title}", ""]
    lines += ["| | " + " | ".join(splits) + " | total |", "|---|" + "---|" * (len(splits) + 1)]
    for key, label in (("images_per_split", "images"), ("boxes_per_split", "boxes")):
        lines.append(f"| {label} | " + " | ".join(str(s[key][x]) for x in splits) + f" | {sum(s[key].values())} |")
    lines += ["", "## Class distribution (instances)", ""]
    lines += ["| class | " + " | ".join(splits) + " | total |", "|---|" + "---|" * (len(splits) + 1)]
    for name, tot in sorted(s["class_counts_total"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | " + " | ".join(str(s["class_counts"][x][name]) for x in splits) + f" | {tot} |")
    lines += [
        "",
        f"Imbalance ratio (largest / smallest class): **{s['imbalance_ratio_max_min']}**",
        "",
        "## Issues",
        "",
        f"- Image issues: {s['image_issues'] or 'none'}",
        f"- Box issues: {s['box_issues'] or 'none'}",
        f"- Orphan label files: {s['orphan_labels']}",
        "",
        "## Duplicates",
        "",
    ]
    lines += [f"- {k.replace('_', ' ')}: {v}" for k, v in s["duplicates"].items()]
    lines += [
        "",
        "## Images and boxes",
        "",
        f"- Unique image sizes: {s['unique_image_sizes']}; most common: {s['image_sizes_top5']}",
        f"- Aspect ratio (w/h): {s['aspect_ratio']}",
        f"- Box area / image area: {s['box_area_ratio']}",
        f"- Boxes per image (mean): {s['boxes_per_image_mean']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------- figures


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_class_distribution(summary: dict, path: Path) -> None:
    """Small multiples: one horizontal bar chart per split, same class order."""
    splits = list(summary["class_counts"])
    order = sorted(summary["class_counts_total"], key=summary["class_counts_total"].get)
    fig, axes = plt.subplots(1, len(splits), figsize=(4 * len(splits), 6), sharey=True)
    for ax, split in zip(np.atleast_1d(axes), splits):
        counts = [summary["class_counts"][split][c] for c in order]
        ax.barh(order, counts, color=BAR_COLOR, height=0.7)
        for y, v in enumerate(counts):
            ax.text(v, y, f" {v}", va="center", fontsize=7, color=INK)
        ax.set_title(f"{split} ({sum(counts)} boxes)", fontsize=10, color=INK, loc="left")
        _style(ax)
    fig.suptitle("Instances per class", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_box_and_image_sizes(records: list[dict], path: Path) -> None:
    areas = [b["area"] for r in records for b in r["boxes"] if b["area"] > 0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.hist(np.log10(areas), bins=40, color=BAR_COLOR, edgecolor="white", linewidth=0.5)
    ax1.axvline(np.log10(TINY_AREA), color=INK, linestyle="--", linewidth=1)
    ax1.text(np.log10(TINY_AREA), ax1.get_ylim()[1] * 0.95, " tiny threshold", fontsize=8, color=INK)
    ax1.set_xlabel("log10(box area / image area)", fontsize=9, color=INK)
    ax1.set_title("Box size distribution", fontsize=10, color=INK, loc="left")
    _style(ax1)

    wh = np.array([(r["width"], r["height"]) for r in records if r["width"]])
    ax2.scatter(wh[:, 0], wh[:, 1], s=8, color=BAR_COLOR, alpha=0.4, linewidths=0)
    ax2.set_xlabel("width (px)", fontsize=9, color=INK)
    ax2.set_ylabel("height (px)", fontsize=9, color=INK)
    ax2.set_title("Image sizes", fontsize=10, color=INK, loc="left")
    _style(ax2)
    ax2.grid(axis="y", color=GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def class_color(cls: int) -> tuple[int, int, int]:
    """Deterministic, well-spread BGR color per class id."""
    hsv = np.uint8([[[(cls * 37) % 180, 220, 230]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw_boxes(rec: dict, names: list[str], tile: int = 400):
    """Load an image, draw its labels, and letterbox it into a square tile."""
    img = cv2.imread(rec["image"])
    h, w = img.shape[:2]
    for b in rec["boxes"]:
        if not b["xyxy"]:
            continue
        x1, y1, x2, y2 = (int(v * s) for v, s in zip(b["xyxy"], (w, h, w, h)))
        bad = any(i in b["issues"] for i in ("unknown_class", "degenerate", "duplicate", "out_of_bounds"))
        color = (0, 0, 255) if bad else class_color(b["cls"])
        thickness = max(2, round(max(h, w) / 300))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        name = names[b["cls"]] if b["cls"] is not None and b["cls"] < len(names) else f"id={b['cls']}"
        scale = max(0.5, max(h, w) / 900)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = y1 if y1 - th - 6 >= 0 else y1 + th + 6  # put the tag inside the box at the top edge
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 4, ty), color, -1)
        cv2.putText(img, name, (x1 + 2, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    scale = tile / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    canvas = np.full((tile, tile, 3), 40, np.uint8)
    canvas[: img.shape[0], : img.shape[1]] = img
    return canvas


def save_montage(tiles: list, path: Path, cols: int = 4) -> None:
    rows = [tiles[i : i + cols] for i in range(0, len(tiles), cols)]
    rows[-1] = rows[-1] + [np.zeros_like(tiles[0])] * (cols - len(rows[-1]))
    cv2.imwrite(str(path), cv2.vconcat([cv2.hconcat(r) for r in rows]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def save_samples(records, names, out: Path, n: int, seed: int) -> None:
    rng = random.Random(seed)
    for split in sorted({r["split"] for r in records}):
        pool = [r for r in records if r["split"] == split and r["width"]]
        picked = rng.sample(pool, min(n, len(pool)))
        save_montage([draw_boxes(r, names) for r in picked], out / f"samples_{split}.jpg")


def save_duplicate_examples(records, pairs, names, out: Path, n: int = 8) -> None:
    """Side-by-side near-duplicate pairs (cross-split first), annotated with the pHash distance."""
    if not pairs:
        return
    ranked = sorted(pairs, key=lambda p: (records[p[0]]["split"] == records[p[1]]["split"], -p[2]))
    tiles = []
    for i, j, d in ranked[:n]:
        for k in (i, j):
            t = draw_boxes(records[k], names, tile=300)
            cv2.putText(t, f"{records[k]['split']} d={d}", (6, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(t)
    save_montage(tiles, out / "duplicate_pairs.jpg", cols=4)


# ----------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="dataset root containing data.yaml")
    parser.add_argument("--out", type=Path, required=True, help="report output folder")
    # 10 was chosen by visual inspection of random pairs: all pairs at d<=10 were the same photo
    # (re-crops / different watermarks); at d=12 distinct photos start to appear.
    parser.add_argument("--dup-dist", type=int, default=10, help="max pHash Hamming distance for near-duplicates")
    parser.add_argument("--samples", type=int, default=16, help="sample images to draw per split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    names = read_class_names(args.data)
    records, orphan_labels = load_dataset(args.data, names)
    groups, pairs = find_duplicate_groups(records, args.dup_dist)
    summary = summarize(records, orphan_labels, groups, pairs, names)

    (args.out / "audit.json").write_text(json.dumps(summary, indent=2))
    write_markdown(summary, args.out / "audit.md", str(args.data))
    plot_class_distribution(summary, args.out / "class_distribution.png")
    plot_box_and_image_sizes(records, args.out / "sizes.png")
    save_samples(records, names, args.out, args.samples, args.seed)
    save_duplicate_examples(records, pairs, names, args.out)

    print((args.out / "audit.md").read_text(encoding="utf-8"))
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
