# NeuroScan — Brain Tumor Classification (Flask API + Grad‑CAM + Gemini MRI Validation)

Production-ready Flask backend for brain tumor classification using a MobileNet TensorFlow model (.h5), with:
- Gemini Vision API gate that validates an upload is a human brain MRI before classification
- Grad‑CAM heatmaps for model interpretability
- Single-container Docker deployment (Railway/any container host)
- Optional HTML UI served by Flask (templates/NeuroScan.html)

Live app
- https://neuroscan.up.railway.app/

API base URL
- https://neuroscan.up.railway.app/

If you previously used the separate Next.js web app, this repo now focuses on the backend API. The Flask template provides a simple UI. You can still connect any frontend via the documented API.

## 🚀 Features

- TensorFlow/Keras MobileNet model loaded from `MODEL_PATH`
- Automatic model download at startup via `MODEL_URL` (handled by `entrypoint.sh`)
- Gemini Vision check to reject non‑MRI images before model inference
- Grad‑CAM heatmap generation for visual explanations
- Dockerfile + Gunicorn entrypoint for reproducible, production-friendly deploys
- CORS enabled for frontend integrations

## 🧱 Tech Stack

- Python 3.11, Flask, CORS
- TensorFlow/Keras, NumPy, OpenCV, Pillow
- Google Generative AI SDK (Gemini)
- Docker (optional), Railway-friendly

## 📁 Key Files

- `server1.py` — Flask app, routes, Gemini validation, Grad‑CAM
- `templates/NeuroScan.html` — Minimal UI served by Flask
- `requirements.txt` — Python dependencies
- `Dockerfile` — Container build
- `entrypoint.sh` — Downloads model to `MODEL_PATH` if missing, then starts Gunicorn

## ⚙️ Environment Variables

Set these in your deployment platform (e.g., Railway → Variables) or locally (export them in your shell).

- `GOOGLE_API_KEY` — Your Gemini API key (required for MRI validation)
- `GEMINI_MODEL` — Gemini model name. Default: `gemini-1.5-flash` (stable). If you have preview access, you can set `gemini-2.5-flash-preview-05-20`.
- `MODEL_URL` — HTTPS URL to the `.h5` file (used by entrypoint to auto-download on startup)
- `MODEL_PATH` — Filesystem path to the `.h5` at runtime (e.g., `/tmp/models/mobilenet_brain_tumor_classifier.h5`)
- `DATASET_PATH` — Path to dataset root for `/random` (mount a volume in production). Default: `./Dataset`
- `UPLOAD_FOLDER` — Where uploads are saved temporarily. Default: `Uploads`
- `PORT` — Server port. Default: `5050`
- Optional: `TF_CPP_MIN_LOG_LEVEL=2` — Reduce TensorFlow log verbosity

## 🧪 Run Locally (no Docker)

1) Create env and install deps
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Make the model available
- Option A: Point to an existing local model
```bash
export MODEL_PATH="/absolute/path/to/mobilenet_brain_tumor_classifier.h5"
```
- Option B: Download it manually
```bash
mkdir -p /tmp/models
curl -L "https://github.com/<owner>/<repo>/releases/download/<tag>/mobilenet_brain_tumor_classifier.h5" \
  -o /tmp/models/mobilenet_brain_tumor_classifier.h5
export MODEL_PATH="/tmp/models/mobilenet_brain_tumor_classifier.h5"
```

3) Configure Gemini (optional but recommended)
```bash
export GOOGLE_API_KEY="your-google-api-key"
# Optional if you lack preview access:
export GEMINI_MODEL="gemini-1.5-flash"
```

4) Start the app
```bash
export PORT=5050
python server1.py
# Visit http://localhost:5050
```

## 🐳 Run with Docker

Build and run the container. The entrypoint will download the model from `MODEL_URL` the first time, then start Gunicorn.

```bash
# Build
docker build -t neuroscan .

# Run (persist model across restarts by mounting a volume at /tmp)
docker run -p 8080:8080 \
  -e PORT=8080 \
  -e GOOGLE_API_KEY="your-google-api-key" \
  -e GEMINI_MODEL="gemini-1.5-flash" \
  -e MODEL_URL="https://github.com/<owner>/<repo>/releases/download/<tag>/mobilenet_brain_tumor_classifier.h5" \
  -e MODEL_PATH="/tmp/models/mobilenet_brain_tumor_classifier.h5" \
  -v neuroscan-data:/tmp \
  neuroscan
```

Notes:
- If deploying on Railway, mount a volume at `/tmp` and set `MODEL_PATH=/tmp/models/...` so the model persists between restarts.
- The container starts with Gunicorn per `entrypoint.sh`.

## 📡 API

Base URL (live): https://neuroscan.up.railway.app/

### POST `/predict`
- Body: multipart/form-data with `file` (JPEG/PNG/BMP)
- Behavior:
  - Validates MRI with Gemini (if configured). Non‑MRI → returns a synthetic “not_mri” class.
  - Otherwise runs the local TensorFlow model
- Response (example):
```json
{
  "class": "Glioma",
  "confidence": 0.93,
  "classes": [
    {"label": "Glioma", "percent": 93.0},
    {"label": "Meningioma", "percent": 5.2},
    {"label": "Pituitary", "percent": 1.7},
    {"label": "Notumor", "percent": 0.1},
    {"label": "Not mri", "percent": 0.0}
  ],
  "gemini": {
    "used": true,
    "raw": "YES_MRI"
  }
}
```

cURL example:
```bash
curl -F "file=@/path/to/non_mri.jpg" https://neuroscan.up.railway.app/predict
```

### POST `/heatmap`
- Body: multipart/form-data with `file`
- Validates MRI with Gemini, then returns a Grad‑CAM heatmap as base64 PNG.
- Response:
```json
{ "heatmap": "<base64-encoded-png>" }
```

cURL example:
```bash
curl -F "file=@/path/to/brain_mri.jpg" https://neuroscan.up.railway.app/heatmap
```

### GET `/random`
- Picks a random image from `DATASET_PATH/{Training|Testing}/{glioma|meningioma|notumor|pituitary}`
- Does NOT call Gemini (dataset is assumed valid)
- Response:
```json
{
  "class": "Meningioma",
  "confidence": 0.88,
  "classes": [ ... ],
  "image": "<base64-encoded-image>",
  "gemini": {"used": false, "raw": null}
}
```

### GET `/`
- Serves `templates/NeuroScan.html` simple UI
- Live: https://neuroscan.up.railway.app/

## 🔍 Logs and Troubleshooting

- “Gemini API configured successfully. model=…” → key loaded
- Per request you should see: “Gemini raw response text: 'YES_MRI'” or “'NO_MRI'”
- TensorFlow CUDA/TRT warnings are harmless on CPU-only hosts
- If you see “Model missing”:
  - Confirm `MODEL_PATH` equals the actual downloaded path
  - If using the entrypoint, ensure `MODEL_URL` is set and the container can reach it
  - On Railway, mount a volume at `/tmp` before downloading so the file persists

## 🔒 Security & Disclaimer

- This tool is for research and educational purposes only and must not be used for medical diagnosis.
- Always consult qualified medical professionals for medical advice and diagnosis.

## 📄 License

MIT License — see [LICENSE](LICENSE).

## 🙏 Acknowledgments

- Original NeuroScan project by Yash Naidu
- Google Gemini, TensorFlow, Flask communities

## 🧩 Frontend (Optional)

If you want a richer frontend, you can connect a Next.js app to this backend via the documented API:
- Set `NEXT_PUBLIC_API_URL` to your deployed backend URL
- POST uploads to `${NEXT_PUBLIC_API_URL}/predict` and `/heatmap`
- Consider CORS and file size limits
