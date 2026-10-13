"""Gradio demo: upload a car photo -> annotated damage boxes + JSON claim report.

Runs locally from the repo root (`python app/app.py`) and as a Hugging Face Space, where this
file sits next to a copy of src/ (see src/publish_hf.py).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE if (HERE / "src").exists() else HERE.parent))  # Space layout vs repo layout

import cv2  # noqa: E402
import gradio as gr  # noqa: E402

from src.predict import DEFAULT_CONF, analyze, load_model  # noqa: E402

MODEL = load_model()
EXAMPLES = sorted((HERE / "examples").glob("*.jpg"))

DESCRIPTION = """
# Car damage detection → claim report

Upload a photo of a damaged car. A fine-tuned **YOLO11n** (imgsz 416) detects damage per car part
(e.g. `front-bumper-dent`, `Headlight-Damage`). Each detection gets a **minor / moderate / severe**
label from simple, documented **rules**: damaged part, box size and number of damages. The rules are
heuristics, not learned.

Test-set accuracy is modest (mAP50 0.43); see the project README for the evaluation and error analysis.
Box colors: green = minor, orange = moderate, red = severe.
"""


def run(image_rgb, conf):
    if image_rgb is None:
        return None, {}
    annotated, report = analyze(MODEL, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), conf, "upload")
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), report


with gr.Blocks(title="Car Damage Detection") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="numpy", label="Car photo")
            conf_in = gr.Slider(0.05, 0.90, value=DEFAULT_CONF, step=0.05, label="Confidence threshold")
            button = gr.Button("Detect damage", variant="primary")
        with gr.Column():
            image_out = gr.Image(label="Detections")
            report_out = gr.JSON(label="Claim report")
    button.click(run, [image_in, conf_in], [image_out, report_out])
    conf_in.release(run, [image_in, conf_in], [image_out, report_out])
    if EXAMPLES:
        gr.Examples(
            [[str(p), DEFAULT_CONF] for p in EXAMPLES],
            [image_in, conf_in],
            label="Examples (test-set images the model gets right, CC BY 4.0, Roboflow Universe)",
        )

if __name__ == "__main__":
    demo.launch()
