"""Build a cleaned, re-split dataset version from the raw Roboflow export.

Cleaning rules (each one is justified in the README "Data audit" section):
  1. Drop corrupted images and images with an empty/missing label file
     (visual check: they show visible damage, i.e. missing annotations, not true negatives).
  2. Drop invalid boxes: bad format, unknown class id, degenerate (< 2 px side), duplicate
     annotations. Slightly out-of-bounds boxes are clipped to the image.
  3. De-duplicate: within each near-duplicate group (pHash distance <= --dup-dist) keep one
     image (most valid boxes, then highest resolution) and drop the rest. This also removes
     train/valid/test leakage of the original split.
  4. Drop classes left with fewer than --min-instances boxes and re-index class ids.
  5. Re-split train/val/test (default 70/15/15) with a fixed seed, greedily stratified on each
     image's rarest class so rare classes still appear in val and test.

Outputs:
  <out>/{train,val,test}/{images,labels}/, <out>/data.yaml
  <out>/manifest.csv  (kept images: new split <- raw path)
  <out>/removed.csv   (dropped images / boxes with the reason)
  configs/data.yaml   (training config pointing to <out>, committed to Git)

Usage:
    python -m src.clean_data --raw data/raw/car-damage-v2 --out data/clean/car-damage-v2
"""

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

from src.data_audit import find_duplicate_groups, load_dataset, read_class_names

DROP_BOX_ISSUES = {"bad_format", "unknown_class", "degenerate", "duplicate"}
DROP_IMAGE_ISSUES = {"corrupted", "missing_label", "empty_label"}


def valid_boxes(rec: dict) -> list[dict]:
    return [b for b in rec["boxes"] if not DROP_BOX_ISSUES.intersection(b["issues"])]


def stratified_split(records: list[dict], ratios: dict[str, float], seed: int) -> dict[str, list[dict]]:
    """Greedy stratification: handle images with the rarest classes first and put each one in
    the split that is furthest below its target share for that image's rarest class."""
    totals = Counter(b["cls"] for r in records for b in r["kept_boxes"])
    counts = {s: Counter() for s in ratios}
    n_images = {s: 0 for s in ratios}
    target_images = {s: ratios[s] * len(records) for s in ratios}

    items = records[:]
    random.Random(seed).shuffle(items)
    items.sort(key=lambda r: min(totals[b["cls"]] for b in r["kept_boxes"]))

    out = {s: [] for s in ratios}
    for rec in items:
        rarest = min((b["cls"] for b in rec["kept_boxes"]), key=lambda c: totals[c])

        def unfilled(s):
            class_gap = 1 - counts[s][rarest] / (ratios[s] * totals[rarest])
            image_gap = 1 - n_images[s] / target_images[s]
            return (class_gap, image_gap)

        split = max(ratios, key=unfilled)
        out[split].append(rec)
        n_images[split] += 1
        counts[split].update(b["cls"] for b in rec["kept_boxes"])
    return out


def to_yolo_line(cls: int, xyxy: list[float]) -> str:
    x1, y1, x2, y2 = xyxy
    return f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/car-damage-v2"))
    parser.add_argument("--out", type=Path, default=Path("data/clean/car-damage-v2"))
    parser.add_argument("--config-out", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--dup-dist", type=int, default=10)
    parser.add_argument("--min-instances", type=int, default=10)
    parser.add_argument("--split", type=float, nargs=3, default=[0.70, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    names = read_class_names(args.raw)
    records, _ = load_dataset(args.raw, names)
    removed = []  # (raw image path, what, reason)

    # 1 + 2: drop broken images, keep only valid boxes
    usable = []
    for rec in records:
        bad = DROP_IMAGE_ISSUES.intersection(rec["issues"])
        if bad:
            removed.append((rec["image"], "image", sorted(bad)[0]))
            continue
        for b in rec["boxes"]:
            dropped = DROP_BOX_ISSUES.intersection(b["issues"])
            if dropped:
                removed.append((rec["image"], f"box cls={b['cls']}", sorted(dropped)[0]))
        rec["kept_boxes"] = valid_boxes(rec)
        if rec["kept_boxes"]:
            usable.append(rec)
        else:
            removed.append((rec["image"], "image", "no_valid_boxes"))

    # 3: de-duplicate
    groups, _ = find_duplicate_groups(usable, args.dup_dist)
    drop = set()
    for group in groups:
        best = max(group, key=lambda i: (len(usable[i]["kept_boxes"]), usable[i]["width"] * usable[i]["height"]))
        for i in group:
            if i != best:
                drop.add(i)
                removed.append((usable[i]["image"], "image", f"near_duplicate_of {usable[best]['image']}"))
    usable = [r for i, r in enumerate(usable) if i not in drop]

    # 4: drop too-rare classes, re-index the rest
    counts = Counter(b["cls"] for r in usable for b in r["kept_boxes"])
    keep_classes = [c for c in range(len(names)) if counts[c] >= args.min_instances]
    remap = {old: new for new, old in enumerate(keep_classes)}
    new_names = [names[c] for c in keep_classes]
    for c in set(range(len(names))) - set(keep_classes):
        print(f"Dropping class '{names[c]}' ({counts[c]} instances < {args.min_instances})")
    final = []
    for rec in usable:
        for b in rec["kept_boxes"]:
            if b["cls"] not in remap:
                removed.append((rec["image"], f"box cls={b['cls']}", "rare_class"))
        rec["kept_boxes"] = [dict(b, cls=remap[b["cls"]]) for b in rec["kept_boxes"] if b["cls"] in remap]
        if rec["kept_boxes"]:
            final.append(rec)
        else:
            removed.append((rec["image"], "image", "no_boxes_after_class_drop"))

    # 5: re-split and write
    ratios = dict(zip(("train", "val", "test"), args.split))
    splits = stratified_split(final, ratios, args.seed)

    if args.out.exists():
        shutil.rmtree(args.out)
    manifest = []
    for split, recs in splits.items():
        (args.out / split / "images").mkdir(parents=True)
        (args.out / split / "labels").mkdir(parents=True)
        for rec in sorted(recs, key=lambda r: r["image"]):
            src = Path(rec["image"])
            shutil.copy2(src, args.out / split / "images" / src.name)
            lines = [to_yolo_line(b["cls"], b["xyxy"]) for b in rec["kept_boxes"]]
            (args.out / split / "labels" / f"{src.stem}.txt").write_text("\n".join(lines) + "\n")
            manifest.append((split, src.name, rec["image"], rec["split"]))

    with open(args.out / "manifest.csv", "w", newline="") as f:
        csv.writer(f).writerows([("split", "file", "raw_path", "raw_split"), *manifest])
    with open(args.out / "removed.csv", "w", newline="") as f:
        csv.writer(f).writerows([("raw_path", "what", "reason"), *removed])

    data_yaml = {
        "path": args.out.as_posix(),  # relative to the repo root (run commands from there)
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": dict(enumerate(new_names)),
    }
    for path in (args.out / "data.yaml", args.config_out):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    reasons = Counter(reason.split(" ")[0] for _, what, reason in removed if what == "image")
    box_reasons = Counter(reason for _, what, reason in removed if what != "image")
    print(f"Raw images: {len(records)} -> clean images: {len(final)}")
    print(f"Removed images by reason: {dict(reasons)}")
    print(f"Removed boxes by reason: {dict(box_reasons)}")
    print("Split sizes:", {s: len(r) for s, r in splits.items()})
    print(f"Wrote {args.out} and {args.config_out}")


if __name__ == "__main__":
    main()
