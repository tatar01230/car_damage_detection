"""Image in -> annotated image + JSON claim report.

Weights: --weights path if given; otherwise weights/car_damage_yolo11n_416.pt if present;
otherwise downloaded from the Hugging Face model repo (HF_MODEL_REPO).

Usage:
    python -m src.predict car.jpg [more.jpg ...] --out outputs
    python -m src.predict car.jpg --weights runs/E4/weights/best.pt --conf 0.3
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.severity import damage_severity, overall_severity

LOCAL_WEIGHTS = Path("weights/car_damage_yolo11n_416.pt")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "tatar01230/car-damage-yolo11n")
HF_WEIGHTS_FILE = "car_damage_yolo11n_416.pt"
DEFAULT_CONF = 0.30  # maximizes F1 on the val split (see README, Error analysis)

SEVERITY_BGR = {"minor": (80, 175, 76), "moderate": (0, 152, 255), "severe": (54, 67, 244)}
NOTE = "Severity levels are rule-based heuristics (src/severity.py), not learned or validated against repair costs."


def load_model(weights: str | Path | None = None) -> YOLO:
    if weights:
        return YOLO(str(weights))
    if LOCAL_WEIGHTS.exists():
        return YOLO(str(LOCAL_WEIGHTS))
    if not HF_MODEL_REPO:
        raise FileNotFoundError(f"No weights: pass --weights, put them at {LOCAL_WEIGHTS}, or set HF_MODEL_REPO.")
    from huggingface_hub import hf_hub_download

    return YOLO(hf_hub_download(HF_MODEL_REPO, HF_WEIGHTS_FILE))


def analyze(model: YOLO, image_bgr: np.ndarray, conf: float = DEFAULT_CONF, image_name: str = "image") -> tuple[np.ndarray, dict]:
    """Detect damage, attach rule-based severity, return (annotated BGR image, report dict)."""
    h, w = image_bgr.shape[:2]
    result = model.predict(image_bgr, conf=conf, imgsz=model.overrides.get("imgsz", 640), verbose=False)[0]

    damages = []
    for cls, box, score in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        area_ratio = (x2 - x1) * (y2 - y1) / (w * h)
        damage_type = model.names[int(cls)]
        damages.append(
            {
                "type": damage_type,
                "confidence": round(score, 3),
                "bbox": [x1, y1, x2, y2],
                "area_ratio": round(area_ratio, 4),
                "severity": damage_severity(damage_type, area_ratio),
            }
        )
    damages.sort(key=lambda d: -d["confidence"])

    report = {
        "image": image_name,
        "damages": damages,
        "overall_severity": overall_severity([d["severity"] for d in damages]),
        "conf_threshold": conf,
        "note": NOTE,
    }
    return draw(image_bgr, damages), report


def draw(image_bgr: np.ndarray, damages: list[dict]) -> np.ndarray:
    img = image_bgr.copy()
    h, w = img.shape[:2]
    thick = max(2, round(max(h, w) / 300))
    scale = max(0.5, max(h, w) / 1200)
    for d in sorted(damages, key=lambda d: d["confidence"]):  # most confident drawn last, on top
        x1, y1, x2, y2 = d["bbox"]
        color = SEVERITY_BGR[d["severity"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        text = f"{d['type']} {d['confidence']:.2f} | {d['severity']}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = y1 if y1 - th - 8 >= 0 else y1 + th + 8
        cv2.rectangle(img, (x1, ty - th - 8), (x1 + tw + 6, ty), color, -1)
        cv2.putText(img, text, (x1 + 3, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    model = load_model(args.weights)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping {path}: not a readable image")
            continue
        annotated, report = analyze(model, image, args.conf, path.name)
        cv2.imwrite(str(args.out / f"{path.stem}_annotated.jpg"), annotated)
        (args.out / f"{path.stem}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
