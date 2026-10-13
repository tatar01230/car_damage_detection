"""Download the raw car damage dataset from Roboflow Universe.

The API key is read from the ROBOFLOW_API_KEY environment variable (or a git-ignored
.env file). The raw download is never modified afterwards; cleaning (Day 2) writes a
separate dataset version.

Usage:
    python -m src.download_data
    python -m src.download_data --out data/raw/car-damage-v2 --force
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

# Default dataset: part-level car damage detection on Roboflow Universe (17 classes, 3,058 images).
# https://universe.roboflow.com/root-6ntpq/car-damage-detection-u0q4r  (version 2: no preprocessing,
# no augmentation; its 75/22/3 split has only 79 test images, so we re-split on Day 2).
DEFAULT_WORKSPACE = "root-6ntpq"
DEFAULT_PROJECT = "car-damage-detection-u0q4r"
DEFAULT_VERSION = 2
# "yolov8" export = one .txt per image, one "class cx cy w h" line per box (normalized coords).
DEFAULT_FORMAT = "yolov8"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def summarize(root: Path) -> None:
    """Print image/label counts per split and check the total against Roboflow's README.

    The check exists because a manual zip extraction with Windows Explorer silently skipped
    7 images whose paths exceeded 260 characters.
    """
    total = 0
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = [p for p in (split_dir / "images").glob("*") if p.suffix.lower() in IMAGE_EXTS]
        labels = list((split_dir / "labels").glob("*.txt"))
        if images or labels:
            print(f"  {split_dir.name:<6} images={len(images):>5}  labels={len(labels):>5}")
            total += len(images)

    readme = root / "README.roboflow.txt"
    match = re.search(r"includes (\d+) images", readme.read_text()) if readme.exists() else None
    if match and int(match.group(1)) != total:
        sys.exit(f"ERROR: found {total} images but Roboflow lists {match.group(1)}. Re-download with --force.")
    print(f"  total  images={total:>5}" + (" (matches Roboflow README)" if match else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION)
    parser.add_argument("--format", default=DEFAULT_FORMAT)
    parser.add_argument("--out", type=Path, default=Path("data/raw/car-damage-v2"))
    parser.add_argument("--force", action="store_true", help="delete the folder and download again")
    args = parser.parse_args()

    out = args.out
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"{out} already exists, skipping download (use --force to re-download).")
        summarize(out)
        return

    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("ROBOFLOW_API_KEY is not set. Copy .env.example to .env and add your key.")

    from roboflow import Roboflow  # imported late so --help works without the package

    if out.exists():
        shutil.rmtree(out)  # --force: start clean so no stale files survive

    rf = Roboflow(api_key=api_key)
    version = rf.workspace(args.workspace).project(args.project).version(args.version)
    version.download(args.format, location=str(out), overwrite=True)

    print(f"Downloaded {args.workspace}/{args.project} v{args.version} ({args.format}) to {out}")
    summarize(out)


if __name__ == "__main__":
    main()
