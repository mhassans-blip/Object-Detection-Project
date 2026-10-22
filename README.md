# Gun Object Detection: End-to-End YOLO + FastAPI + Streamlit

This project fine-tunes a YOLO11 nano detector on the supplied Roboflow dataset (`nc: 1`, class `gun`) and provides an image-upload UI.

**Architecture:** Streamlit (`app.py`) launches FastAPI (`api.py`) locally as a child process. You do not need to deploy the API separately for local use.

## 1. Environment

Python 3.10 or 3.11 is recommended for smoother ML package compatibility. Open a terminal in this project folder:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

A GPU is optional. Training on CPU can take a long time.

## 2. Train the model

The dataset archive `Gun.v1i.yolov11.zip` is included. Run:

```bash
python train.py
```

The script extracts the dataset, fine-tunes pretrained `yolo11n.pt` for up to 50 epochs, and copies the best checkpoint to `best.pt`. The first run may download pretrained weights, so internet access is required. Training duration and accuracy depend on your hardware and data.

## 3. Start the app

After training completes:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. Upload an image and click **Detect objects**. FastAPI starts automatically on `http://127.0.0.1:8000`; you can inspect its docs at `http://127.0.0.1:8000/docs`.

## Files

- `train.py`: dataset extraction and YOLO training
- `api.py`: FastAPI health and prediction endpoints
- `app.py`: Streamlit interface that starts the API and displays annotated predictions
- `best.pt`: generated after training
- `Gun.v1i.yolov11.zip`: supplied dataset

## Notes / troubleshooting

- If the UI says the model is missing, run `python train.py` and then restart Streamlit.
- If port 8000 is occupied, stop the other process or change the port in `app.py` and `API_URL` consistently.
- If package installation fails, use Python 3.10/3.11 in a clean virtual environment.
- Do not commit large trained weights or dataset archives to a public GitHub repository unless you intend to publish them.
- This is a computer-vision demo. Predictions are not guaranteed correct and should not be used as the sole basis for real-world safety decisions.
