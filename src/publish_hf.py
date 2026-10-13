"""Publish the final model to the Hugging Face Hub: weights + a model card with the test metrics.

The Streamlit demo and src/predict.py download the weights from this repo, so nothing large
lives in Git. Needs HF_TOKEN (write access) in the environment or the git-ignored .env file.

Usage:
    python -m src.publish_hf --user <hf-username>
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi

from src.predict import HF_WEIGHTS_FILE

GITHUB_URL = "https://github.com/tatar01230/car_damage_detection"


def model_card(metrics: pd.Series) -> str:
    return f"""---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
tags: [yolo, yolo11, object-detection, car-damage]
---

# Car damage detection: YOLO11n (imgsz 416)

YOLO11n fine-tuned from COCO weights to detect car damage per car part (16 classes, e.g.
`front-bumper-dent`, `Headlight-Damage`). It is the final model (experiment E4) of {GITHUB_URL}.

| Test split (427 images, 681 boxes) | Value |
|---|---|
| mAP50 | {metrics.test_mAP50:.3f} |
| mAP50-95 | {metrics.test_mAP50_95:.3f} |
| Precision / Recall | {metrics.test_precision:.3f} / {metrics.test_recall:.3f} |
| CPU latency, batch 1 | {metrics.cpu_ms:.1f} ms ({metrics.eval_cpu}) |

Training data: [Car Damage Detection by Root](https://universe.roboflow.com/root-6ntpq/car-damage-detection-u0q4r)
(Roboflow Universe, CC BY 4.0), de-duplicated and re-split. Known limitations (label noise, rare classes,
coarse part-level boxes) are documented in the GitHub README.

```python
from ultralytics import YOLO
model = YOLO("{HF_WEIGHTS_FILE}")
model.predict("car.jpg", imgsz=416, conf=0.30)
```

License: AGPL-3.0, inherited from Ultralytics YOLO.
"""


def publish_model(api: HfApi, repo: str, weights: Path) -> None:
    metrics = pd.read_csv("reports/evaluation/test_metrics.csv").set_index("id").loc["E4"]
    api.create_repo(repo, exist_ok=True)
    api.create_commit(
        repo,
        operations=[
            CommitOperationAdd(HF_WEIGHTS_FILE, str(weights)),
            CommitOperationAdd("README.md", model_card(metrics).encode()),
        ],
        commit_message="Upload final model (E4) and model card",
    )
    print(f"Model: https://huggingface.co/{repo}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", required=True, help="Hugging Face username or org")
    parser.add_argument("--model-name", default="car-damage-yolo11n")
    parser.add_argument("--weights", type=Path, default=Path("runs/E4/weights/best.pt"))
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set. Add a write token to .env (see .env.example).")
    api = HfApi(token=token)
    publish_model(api, f"{args.user}/{args.model_name}", args.weights)


if __name__ == "__main__":
    main()
