# Vehicle Damage Detection

Fine-tuning COCO-pretrained YOLO models (YOLOv8n / YOLO11n / YOLO11s) to detect car damage
**per car part** (e.g. front-bumper dent, headlight damage, windscreen damage), then turning the
detections into a simple insurance-claim style report.

> **Status:** work in progress (Day 2 of 7). Every number in this README comes from a run
> I actually performed. Sections marked `TODO` are not filled in yet.

## Problem & motivation
TODO (Day 7)

## Demo
TODO (Day 6): Hugging Face Spaces link + screenshot/GIF

## Dataset

**Source:** [Car Damage Detection by Root on Roboflow Universe](https://universe.roboflow.com/root-6ntpq/car-damage-detection-u0q4r),
version 2: 3,058 images, bounding-box annotations, no preprocessing or augmentation, CC BY 4.0.
17 classes that combine **car part + damage**:
`Bodypanel-Dent`, `bonnet-dent`, `boot-dent`, `doorouter-dent`, `fender-dent`, `front-bumper-dent`,
`pillar-dent`, `quaterpanel-dent`, `rear-bumper-dent`, `roof-dent`, `RunningBoard-Dent`,
`Front-Windscreen-Damage`, `Rear-windscreen-Damage`, `Headlight-Damage`, `Taillight-Damage`,
`Signlight-Damage`, `Sidemirror-Damage`.

The published split is 75/22/3 (only 79 test images), so the data is re-split after
de-duplication (see Data audit).

**Why this dataset.** Candidates compared on Day 1:

| Candidate | Images | Classes | Notes | Verdict |
|---|---|---|---|---|
| `root-6ntpq/car-damage-detection` v2 | 3,058 | 17 (part + damage) | Says *which part* is damaged, which is useful for claim cost estimation. Native boxes. No scratch/crack classes; some classes overlap in meaning (`Bodypanel-Dent` vs `doorouter-dent`) | **Chosen** |
| `dr-ulglr/carsdd` v3 (CarDD) | 4,000 | 6 (damage type) | Damage *type* only (dent, scratch, crack, ...). Original CarDD is licensed for non-commercial research only | Runner-up |
| `trialworkspace/car-damage-detection` | 182 | 3 | Too small for a reliable test set | Rejected |
| `ratchakrit/car-damage-merged` v2 | 8,971 | 6 | Merges likely overlapping CarDD copies (leakage risk); images stretched + histogram-equalized at export | Rejected |
| `yolov8tests/car-damage-detection-6` | 3,000 | 9, named `0`–`8` | Class meaning undocumented | Rejected |

**Trade-off accepted:** this dataset has no `scratch`, `crack` or `flat tire` classes, so the model
reports part-level dents and glass/lamp/mirror damage rather than fine-grained damage types.

**License note.** CC BY 4.0 as listed on Roboflow Universe; the original image source is not
documented. The dataset is downloaded by script and never redistributed in this repo.

### Data audit

`src/data_audit.py` checks class balance, duplicates/leakage (perceptual hash), invalid labels,
corrupted files, and image/box size distributions. `src/clean_data.py` applies the fixes and
writes a new dataset version. Full reports: [`reports/data_audit/raw`](reports/data_audit/raw/audit.md)
and [`reports/data_audit/clean`](reports/data_audit/clean/audit.md).

**What the audit found in the raw export (3,051 images, 4,727 boxes):**

| Finding | Count | Action |
|---|---|---|
| Near-duplicate images (pHash distance ≤ 10) | 151 groups, 169 redundant images | Keep one per group (most boxes, then highest resolution) |
| …of which span different splits (train/valid/test **leakage**) | 65 groups | Removed by de-duplication + re-split |
| …of which the copies are labeled differently | 44 groups | Same as above. This shows annotation inconsistency |
| Images with an empty label file | 70 (69 of the 78 test images) | Dropped. Visual check: they show clear damage, so labels are missing |
| Degenerate boxes (< 2 px side) | 3 | Dropped |
| Duplicate boxes (same class, IoU ≥ 0.9) | 1 | Dropped |
| `Bodypanel-Dent` class | 1 instance (a degenerate box) | Class removed → 16 classes |
| Corrupted images / unknown class ids / out-of-bounds boxes | 0 | — |

The original test split was effectively unusable: 78 images, but only 12 boxes.

**Before → after**

| | Raw | Clean |
|---|---|---|
| Images | 3,051 | 2,821 |
| Boxes | 4,727 | 4,510 |
| Classes | 17 | 16 |
| Split (train / val / test images) | 2,296 / 677 / 78 | 1,993 / 421 / 407 |
| Test boxes | 12 | 681 |
| Cross-split near-duplicate groups | 65 | 0 |
| Imbalance (largest / smallest class) | 809 | 34.7 |

**Key decisions**

- **Duplicate threshold = 10 (pHash Hamming distance, 64-bit hash).** Chosen by eyeballing random
  pairs at distances 8, 10, 12 and 14. Every sampled pair at ≤ 10 was the same photo (re-crops,
  different stock-photo watermarks). At 12, different photos start to appear.
- **Roboflow file names were not used for de-duplication.** Generic names such as `images-1-_jpg`
  repeat across *different* photos, which I checked visually.
- **Re-split 70/15/15, seed 42, stratified on each image's rarest class** so that rare classes
  (`pillar-dent`, `Signlight-Damage`) still appear in val/test. Running the cleaning script twice
  produces an identical `manifest.csv`.

**Things the audit surfaced that cleaning cannot fix** (they feed into Limitations):

- **Low resolution.** 1,644 / 2,821 images (58%) have a longest side under 320 px (e.g. 259×194
  thumbnails). 69% are under 640 px.
- **Rare classes.** `pillar-dent` (28 boxes) and `Signlight-Damage` (22) leave only 3–5 test
  instances, so their per-class AP will be noisy.
- **Label style.** Boxes are large (median box covers 12.6% of the image; no box is below 0.1%), so
  this is a coarse "damaged region per part" dataset, not a fine-grained scratch dataset. Some
  boxes are loose, and overlapping part classes are common (e.g. `Headlight-Damage` inside
  `front-bumper-dent`).
- **Stock-photo watermarks** (iStock, Shutterstock, Dreamstime, Alamy) appear on many images.
- pHash catches re-crops only up to a point. Heavily zoomed crops of the same scene can remain
  as separate images.

![Class distribution after cleaning](reports/data_audit/clean/class_distribution.png)

## Method
TODO (Day 3–4)

## Results
TODO (Day 5)

## Error analysis
TODO (Day 5)

## Severity rules
TODO (Day 6)

## Setup

Local environment: Python 3.12, NVIDIA GPU (for local inference/demo).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux/macOS: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
copy .env.example .env            # then put your ROBOFLOW_API_KEY in .env
python -m src.download_data       # -> data/raw/car-damage-v2/
python -m src.data_audit --data data/raw/car-damage-v2 --out reports/data_audit/raw
python -m src.clean_data          # -> data/clean/car-damage-v2/ + configs/data.yaml
python -m src.data_audit --data data/clean/car-damage-v2 --out reports/data_audit/clean
```

## Reproduce
TODO: train on Colab, evaluate, run demo

## Limitations
TODO

## What I learned
TODO

## Roadmap / future work
- Instance segmentation (`yolo11n-seg`) for precise damage shape and area
- Add a damage-*type* model trained on CarDD (dent / scratch / crack / ...) and combine it with
  this part-level model, e.g. "scratch on front bumper"
- OCR module (e.g. Thai license plates / claim documents) for an end-to-end claim pipeline
