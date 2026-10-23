from pathlib import Path
from io import BytesIO
import base64
import os
import threading

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "best.pt")))

app = FastAPI(title="Gun Object Detection API", version="2.0.0")

_model = None
_model_lock = threading.Lock()
_prediction_lock = threading.Lock()


def get_model():
    """Load YOLO once, when first needed. Shared by the API and Streamlit."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}. "
                "Place best.pt beside api.py or set the MODEL_PATH environment variable."
            )
        _model = YOLO(str(MODEL_PATH))
    return _model


def predict_image(image: Image.Image, conf: float = 0.25, filename: str = "upload"):
    """Run detection on a PIL image; used directly by Streamlit and by FastAPI."""
    model = get_model()
    image = image.convert("RGB")
    conf = max(0.01, min(float(conf), 0.99))

    with _prediction_lock:
        result = model.predict(source=image, conf=conf, verbose=False)[0]

    detections = []
    if result.boxes is not None:
        names = result.names
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            detections.append({
                "class": str(names.get(class_id, class_id)),
                "confidence": round(float(box.conf[0].item()) * 100, 2),
                "box": [round(float(value), 2) for value in box.xyxy[0].tolist()],
            })

    # Ultralytics result.plot() returns a BGR NumPy array; OpenCV encodes it.
    plotted_bgr = result.plot()
    ok, encoded = cv2.imencode(".jpg", plotted_bgr)
    if not ok:
        raise RuntimeError("Could not encode the prediction image.")

    return {
        "filename": filename,
        "image_width": image.width,
        "image_height": image.height,
        "detections_count": len(detections),
        "detections": detections,
        "annotated_image": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


@app.get("/")
def root():
    return {"message": "Gun Object Detection API", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok" if _model is not None else "model_not_loaded",
        "model_path": str(MODEL_PATH),
        "hint": None if _model is not None else "Model loads on the first prediction request.",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.25):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image. Please upload JPG, PNG, or WEBP.",
        )

    try:
        return predict_image(image, conf=conf, filename=file.filename or "upload")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
