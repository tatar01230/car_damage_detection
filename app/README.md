---
title: Car Damage Detection
emoji: 🚗
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: agpl-3.0
short_description: YOLO11n car damage detection with a rule-based claim report
---

# Car damage detection → claim report

Upload a car photo to get damage boxes per car part, plus a JSON claim report with a rule-based
minor / moderate / severe label.

- Model: YOLO11n fine-tuned from COCO weights, imgsz 416. Weights are downloaded at startup from
  the model repo set in the `HF_MODEL_REPO` Space variable.
- Training data: [Car Damage Detection](https://universe.roboflow.com/root-6ntpq/car-damage-detection-u0q4r)
  by Root on Roboflow Universe (CC BY 4.0), cleaned and re-split. Example images come from its test split.
- Severity levels are hand-written heuristics, not learned.
- Code, evaluation and error analysis: https://github.com/tatar01230/car_damage_detection
