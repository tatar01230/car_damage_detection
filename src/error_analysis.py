"""Error analysis for one trained run: categorize false positives / false negatives.

1. Pick the confidence threshold that maximizes F1 on the *val* split (not test, to avoid
   tuning on the data we report on).
2. Predict on the test split at that threshold and match predictions to ground truth per image
   (greedy by confidence, same class, IoU >= 0.5).
3. Categorize every error:
     FP background    - IoU < 0.1 with every ground-truth box
     FP wrong_class   - overlaps (IoU >= 0.1) a box of another class, no same-class match
     FP localization  - same-class box with 0.1 <= IoU < 0.5
     FP duplicate     - same-class box with IoU >= 0.5 that was already matched
     FN missed        - no prediction with IoU >= 0.1
     FN wrong_class   - covered (IoU >= 0.5) by a prediction of another class
     FN localization  - only overlapped by predictions with IoU < 0.5
4. Write counts, per-class breakdown, class-confusion pairs and example montages
   (green = ground truth, red = prediction) for manual inspection.

Usage:
    python -m src.error_analysis --run E4
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

from src.data_audit import box_iou, save_montage

IOU_MATCH, IOU_TOUCH = 0.5, 0.1


def read_gt(label_path: Path) -> list[tuple[int, list[float]]]:
    boxes = []
    for line in label_path.read_text().splitlines():
        c, cx, cy, w, h = (float(v) for v in line.split())
        boxes.append((int(c), [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
    return boxes


def predict_split(model: YOLO, image_dir: Path, imgsz: int) -> list[dict]:
    """Predict at a very low threshold once; thresholds are applied afterwards."""
    out = []
    for img in sorted(image_dir.iterdir(), key=lambda p: p.name):
        r = model.predict(img, imgsz=imgsz, conf=0.001, verbose=False)[0]
        preds = [(int(c), b.tolist(), float(s)) for c, b, s in zip(r.boxes.cls, r.boxes.xyxyn, r.boxes.conf)]
        gt = read_gt(image_dir.parent / "labels" / f"{img.stem}.txt")
        out.append({"image": str(img), "gt": gt, "preds": preds})
    return out


def match_image(gt, preds, conf: float):
    """Return (pred_results, gt_results): lists of (category, other_class, iou) per box."""
    preds = sorted((p for p in preds if p[2] >= conf), key=lambda p: -p[2])
    matched = [False] * len(gt)
    pred_res = []
    for cls, box, score in preds:
        ious = [box_iou(box, g[1]) for g in gt]
        same = [(iou, j) for j, iou in enumerate(ious) if gt[j][0] == cls]
        free = [(iou, j) for iou, j in same if not matched[j]]
        best_free = max(free, default=(0.0, -1))
        if best_free[0] >= IOU_MATCH:
            matched[best_free[1]] = True
            pred_res.append(("TP", cls, box, score, None))
            continue
        best_same = max(same, default=(0.0, -1))
        best_any = max(ious, default=0.0)
        if best_same[0] >= IOU_MATCH:
            cat, other = "duplicate", None
        elif best_same[0] >= IOU_TOUCH:
            cat, other = "localization", None
        elif best_any >= IOU_TOUCH:
            cat, other = "wrong_class", gt[int(np.argmax(ious))][0]
        else:
            cat, other = "background", None
        pred_res.append((cat, cls, box, score, other))

    gt_res = []
    for j, (cls, box) in enumerate(gt):
        if matched[j]:
            gt_res.append(("TP", cls, box, None))
            continue
        overlaps = [(box_iou(box, p[1]), p[0]) for p in preds]
        other_class = [c for iou, c in overlaps if iou >= IOU_MATCH and c != cls]
        if other_class:
            gt_res.append(("wrong_class", cls, box, other_class[0]))
        elif any(iou >= IOU_TOUCH for iou, _ in overlaps):
            gt_res.append(("localization", cls, box, None))
        else:
            gt_res.append(("missed", cls, box, None))
    return pred_res, gt_res


def f1_at(items, conf: float) -> float:
    tp = fp = fn = 0
    for it in items:
        p, g = match_image(it["gt"], it["preds"], conf)
        tp += sum(r[0] == "TP" for r in p)
        fp += sum(r[0] != "TP" for r in p)
        fn += sum(r[0] != "TP" for r in g)
    return 2 * tp / max(1, 2 * tp + fp + fn)


def draw(item, pred_res, names, focus_box=None, tile=360):
    img = cv2.imread(item["image"])
    h, w = img.shape[:2]
    thick = max(2, round(max(h, w) / 250))
    scale = max(0.45, max(h, w) / 1000)

    def rect(box, color, text):
        x1, y1, x2, y2 = (int(v * s) for v, s in zip(box, (w, h, w, h)))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        cv2.putText(img, text, (x1 + 2, max(12, y1 + int(14 * scale) + 2)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1 + thick // 2, cv2.LINE_AA)

    for cls, box in item["gt"]:
        rect(box, (60, 200, 60), names[cls])
    for cat, cls, box, score, _ in pred_res:
        rect(box, (40, 40, 230), f"{names[cls]} {score:.2f}")
    if focus_box is not None:
        x1, y1, x2, y2 = (int(v * s) for v, s in zip(focus_box, (w, h, w, h)))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), thick * 2)
    s = tile / max(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)))
    canvas = np.full((tile, tile, 3), 40, np.uint8)
    canvas[: img.shape[0], : img.shape[1]] = img
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--out", type=Path, default=Path("reports/error_analysis"))
    parser.add_argument("--examples", type=int, default=16, help="images per example montage")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    run_dir = args.runs_dir / args.run
    imgsz = yaml.safe_load((run_dir / "args.yaml").read_text())["imgsz"]
    cfg = yaml.safe_load(Path(args.data).read_text())
    root = Path(cfg["path"])
    model = YOLO(str(run_dir / "weights" / "best.pt"))
    names = model.names

    val = predict_split(model, root / cfg["val"], imgsz)
    grid = [round(t, 2) for t in np.arange(0.05, 0.80, 0.05)]
    f1s = {t: f1_at(val, t) for t in grid}
    conf = max(f1s, key=f1s.get)
    print(f"Best val F1 {f1s[conf]:.3f} at conf={conf}")

    test = predict_split(model, root / cfg["test"], imgsz)
    pred_rows, gt_rows, per_image = [], [], []
    for it in test:
        p, g = match_image(it["gt"], it["preds"], conf)
        per_image.append((it, p, g))
        pred_rows += [{"image": it["image"], "category": c, "class": names[k], "conf": s, "gt_class": names.get(o)} for c, k, _, s, o in p]
        gt_rows += [{"image": it["image"], "category": c, "class": names[k], "pred_class": names.get(o)} for c, k, _, o in g]
    preds, gts = pd.DataFrame(pred_rows), pd.DataFrame(gt_rows)

    tp = int((preds["category"] == "TP").sum())
    summary = {
        "run": args.run,
        "conf_threshold": conf,
        "val_f1_at_threshold": round(f1s[conf], 3),
        "test_tp": tp,
        "test_precision": round(tp / len(preds), 3),
        "test_recall": round(tp / len(gts), 3),
        "test_f1": round(2 * tp / (len(preds) + len(gts)), 3),
        "false_positives": preds[preds["category"] != "TP"]["category"].value_counts().to_dict(),
        "false_negatives": gts[gts["category"] != "TP"]["category"].value_counts().to_dict(),
        "top_confusions_gt_to_pred": Counter(
            f"{r['class']} -> {r['pred_class']}" for _, r in gts[gts["category"] == "wrong_class"].iterrows()
        ).most_common(10),
    }
    per_class = (
        gts.assign(TP=gts["category"] == "TP", FN=gts["category"] != "TP")
        .groupby("class")[["TP", "FN"]].sum()
        .join(preds[preds["category"] != "TP"].groupby("class").size().rename("FP"), how="outer")
        .fillna(0).astype(int)
    )
    per_class["recall"] = (per_class["TP"] / (per_class["TP"] + per_class["FN"])).round(2)
    per_class["precision"] = (per_class["TP"] / (per_class["TP"] + per_class["FP"])).round(2)
    per_class = per_class.sort_values("recall")

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    per_class.to_csv(args.out / "per_class_errors.csv")
    preds.to_csv(args.out / "predictions.csv", index=False)
    gts.to_csv(args.out / "ground_truth.csv", index=False)

    # Example montages: highest-confidence FPs per category, and FNs per category
    for kind in ("background", "wrong_class", "localization"):
        cases = sorted(
            ((s, it, p, b) for it, p, _ in per_image for c, _, b, s, _ in p if c == kind), key=lambda x: -x[0]
        )[: args.examples]
        if cases:
            save_montage([draw(it, p, names, b) for _, it, p, b in cases], args.out / f"fp_{kind}.jpg")
    for kind in ("missed", "wrong_class", "localization"):
        cases = [(it, p, b) for it, p, g in per_image for c, _, b, _ in g if c == kind][: args.examples]
        if cases:
            save_montage([draw(it, p, names, b) for it, p, b in cases], args.out / f"fn_{kind}.jpg")

    print(json.dumps(summary, indent=2))
    print(per_class.to_string())


if __name__ == "__main__":
    main()
