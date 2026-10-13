"""Streamlit demo: upload a car photo -> annotated damage boxes + JSON claim report.

Local:  streamlit run app/streamlit_app.py
Cloud:  Streamlit Community Cloud, entrypoint app/streamlit_app.py (dependencies in app/requirements.txt,
        apt packages in the repo-root packages.txt). Weights are downloaded from the Hugging Face model repo.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # make `src` importable when Streamlit runs this file directly

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from src.predict import DEFAULT_CONF, analyze, load_model  # noqa: E402

EXAMPLES = sorted((ROOT / "app" / "examples").glob("*.jpg"))

st.set_page_config(page_title="Car Damage Detection", page_icon="🚗", layout="wide")


@st.cache_resource
def get_model():
    return load_model()  # loaded once per server process, shared by all users


st.title("Car damage detection → claim report")
st.markdown(
    "Upload a photo of a damaged car. A fine-tuned **YOLO11n** (imgsz 416) detects damage per car part "
    "(e.g. `front-bumper-dent`, `Headlight-Damage`). Each detection gets a **minor / moderate / severe** "
    "label from simple, documented **rules**: damaged part, box size and number of damages. The rules are "
    "heuristics, not learned. Test-set accuracy is modest (mAP50 0.43); see the "
    "[project README](https://github.com/tatar01230/car_damage_detection) for the evaluation and error analysis."
)

with st.sidebar:
    conf = st.slider("Confidence threshold", 0.05, 0.90, DEFAULT_CONF, 0.05,
                     help="0.30 maximizes F1 on the validation split. Lower = more detections, more false alarms.")
    st.markdown("Box colors: 🟩 minor · 🟧 moderate · 🟥 severe")
    example = st.selectbox(
        "…or try an example (test-set images the model gets right, CC BY 4.0)",
        ["—"] + [p.name for p in EXAMPLES],
    )

uploaded = st.file_uploader("Car photo", type=["jpg", "jpeg", "png", "webp"])
if uploaded is not None:
    image_bgr = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    name = uploaded.name
elif example != "—":
    image_bgr = cv2.imread(str(ROOT / "app" / "examples" / example))
    name = example
else:
    st.info("Upload a photo or pick an example in the sidebar.")
    st.stop()

if image_bgr is None:
    st.error("Could not read this file as an image.")
    st.stop()

with st.spinner("Detecting damage..."):
    annotated, report = analyze(get_model(), image_bgr, conf, name)

left, right = st.columns([3, 2])
with left:
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"{len(report['damages'])} damage(s) detected")
with right:
    st.metric("Overall severity", report["overall_severity"])
    st.json(report)
    st.download_button("Download report (JSON)", data=json.dumps(report, indent=2),
                       file_name=f"{Path(name).stem}_report.json", mime="application/json")
