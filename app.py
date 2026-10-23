import base64
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

# This imports the same model-loading and inference functions used by FastAPI.
# No Uvicorn subprocess, localhost URL, or separate backend process is needed.
from api import MODEL_PATH, get_model, predict_image

st.set_page_config(page_title="Gun Object Detection", page_icon="🎯", layout="wide")


@st.cache_resource(show_spinner=False)
def load_yolo_model():
    # Streamlit caches the loaded model for the life of this app process.
    return get_model()


st.title("🎯 Gun Object Detection")
st.caption("YOLO object detection · Streamlit UI with in-process inference")

with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    st.markdown("---")
    st.write("**Inference:** Direct Python call (no separate FastAPI process)")
    st.write(f"**Model file:** `{MODEL_PATH.name}`")

if not Path(MODEL_PATH).is_file():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.info(
        "Place `best.pt` in the same folder as `app.py` and `api.py`, "
        "or configure the MODEL_PATH environment variable."
    )
    st.stop()

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Upload an image")
    uploaded = st.file_uploader(
        "Choose JPG, JPEG, PNG, or WEBP",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded is not None:
        try:
            original_image = Image.open(BytesIO(uploaded.getvalue())).convert("RGB")
        except Exception:
            st.error("This image could not be opened. Try a JPG, PNG, or WEBP file.")
            st.stop()

        st.image(original_image, caption="Original image", use_container_width=True)

        if st.button("Detect objects", type="primary", use_container_width=True):
            try:
                with st.spinner("Loading model and running detection..."):
                    load_yolo_model()
                    result = predict_image(
                        original_image,
                        conf=confidence,
                        filename=uploaded.name,
                    )
                st.session_state["last_result"] = result
            except Exception as exc:
                st.error(f"Detection failed: {exc}")

with right:
    st.subheader("Detection results")
    result = st.session_state.get("last_result")

    if result:
        try:
            annotated_bytes = base64.b64decode(result["annotated_image"])
            annotated_image = Image.open(BytesIO(annotated_bytes)).convert("RGB")
            st.image(
                annotated_image,
                caption="Annotated prediction",
                use_container_width=True,
            )

            metric_a, metric_b = st.columns(2)
            metric_a.metric("Objects detected", result["detections_count"])
            metric_b.metric(
                "Image size",
                f'{result["image_width"]} × {result["image_height"]}',
            )

            if result["detections"]:
                st.dataframe(
                    [
                        {
                            "Class": detection["class"],
                            "Confidence": f'{detection["confidence"]:.2f}%',
                            "Bounding box (x1, y1, x2, y2)": str(detection["box"]),
                        }
                        for detection in result["detections"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No objects found above the selected confidence threshold.")
        except Exception as exc:
            st.error(f"Could not display the result: {exc}")
    else:
        st.info("Upload an image and click **Detect objects** to see the results.")

st.markdown("---")
st.caption(
    "Safety note: model predictions can be wrong. Do not use this demo as the sole "
    "basis for safety or law-enforcement decisions."
)
