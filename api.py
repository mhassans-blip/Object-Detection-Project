from pathlib import Path
from io import BytesIO
import os
import threading
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "best.pt"))
app = FastAPI(title="Gun Object Detection API", version="1.0.0")
model = None
model_lock = threading.Lock()

@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        # Keep the API process alive so /health can explain the setup issue.
        print(f"MODEL NOT FOUND: {MODEL_PATH}. Run python train.py first.")
        return
    model = YOLO(str(MODEL_PATH))

@app.get("/")
def root():
    return {"message": "Gun Object Detection API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "model_not_loaded",
            "model_path": str(MODEL_PATH),
            "hint": None if model is not None else "Run python train.py to create best.pt."}

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.25):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Run python train.py first.")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image. Please upload JPG, PNG, or WEBP.")
    with model_lock:
        result = model.predict(source=image, conf=max(0.01, min(conf, 0.99)), verbose=False)[0]
    detections = []
    if result.boxes is not None:
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            detections.append({
                "class": str(names.get(cls_id, cls_id)),
                "confidence": round(float(box.conf[0].item()) * 100, 2),
                "box": [round(float(v), 2) for v in box.xyxy[0].tolist()]
            })
    plotted = result.plot()
    # Ultralytics returns BGR; convert to RGB before encoding.
    from PIL import Image as PILImage
    import cv2
    ok, encoded = cv2.imencode(".jpg", plotted)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode prediction image.")
    return {
        "filename": file.filename,
        "image_width": image.width,
        "image_height": image.height,
        "detections_count": len(detections),
        "detections": detections,
        "annotated_image": __import__("base64").b64encode(encoded.tobytes()).decode("ascii")
    }
