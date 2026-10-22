from pathlib import Path
import base64
import os
import subprocess
import sys
import time
import requests
import streamlit as st
from PIL import Image
from io import BytesIO

BASE_DIR = Path(__file__).resolve().parent
API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")
API_SCRIPT = BASE_DIR / "api.py"

st.set_page_config(page_title="Gun Object Detection", page_icon="🎯", layout="wide")

def start_local_api():
    # Streamlit and FastAPI run together on the same machine. No separate API hosting required.
    if st.session_state.get("api_process") is None:
        log_path = BASE_DIR / "api.log"
        log_file = open(log_path, "a", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=str(BASE_DIR), stdout=log_file, stderr=subprocess.STDOUT
        )
        st.session_state.api_process = proc
        st.session_state.api_log_file = log_file
    for _ in range(20):
        try:
            r = requests.get(f"{API_URL}/health", timeout=1)
            if r.ok:
                return r.json()
        except requests.RequestException:
            time.sleep(0.5)
    return None

st.title("🎯 Gun Object Detection")
st.caption("YOLO object detection · Streamlit frontend + FastAPI backend in one local project")
with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    st.markdown("---")
    st.write("**Backend:** FastAPI")
    st.write("**Model:** YOLO11n fine-tuned on your dataset")

health = start_local_api()
if health is None:
    st.error("FastAPI backend did not start. Check api.log and install requirements.")
    st.stop()
if health.get("status") != "ok":
    st.warning("The backend is running, but the trained model is missing.")
    st.code("python train.py", language="bash")
    st.info("Train the model first. Once best.pt exists, restart this Streamlit app.")
    st.stop()

left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("Upload an image")
    uploaded = st.file_uploader("Choose JPG, JPEG, PNG, or WEBP", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        st.image(uploaded, caption="Original image", use_container_width=True)
        if st.button("Detect objects", type="primary", use_container_width=True):
            with st.spinner("Running YOLO detection..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/jpeg")},
                        params={"conf": confidence}, timeout=180
                    )
                    if not response.ok:
                        try:
                            detail = response.json().get("detail", response.text)
                        except Exception:
                            detail = response.text
                        st.error(f"Prediction failed: {detail}")
                    else:
                        st.session_state.last_result = response.json()
                except requests.RequestException as exc:
                    st.error(f"Could not reach FastAPI: {exc}")

with right:
    st.subheader("Detection results")
    result = st.session_state.get("last_result")
    if result:
        annotated = Image.open(BytesIO(base64.b64decode(result["annotated_image"])))
        st.image(annotated, caption="Annotated prediction", use_container_width=True)
        a, b = st.columns(2)
        a.metric("Objects detected", result["detections_count"])
        b.metric("Image size", f'{result["image_width"]} × {result["image_height"]}')
        if result["detections"]:
            st.dataframe([
                {"Class": d["class"], "Confidence": f'{d["confidence"]:.2f}%',
                 "Bounding box (x1,y1,x2,y2)": str(d["box"])}
                for d in result["detections"]
            ], use_container_width=True, hide_index=True)
        else:
            st.info("No objects found above the selected confidence threshold.")
    else:
        st.info("Upload an image and click **Detect objects** to see bounding boxes and confidence scores.")

st.markdown("---")
st.caption("Safety note: model predictions can be wrong. Do not use this demo as the sole basis for safety or law-enforcement decisions.")
