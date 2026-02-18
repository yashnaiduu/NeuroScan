# NeuroScan — Deployment Guide

This guide covers the current production deployment and how to redeploy or run locally.

---

## Current Production Deployment

| Service | Platform | URL |
|:---|:---|:---|
| Frontend | Vercel | [neuroscan.vercel.app](https://neuroscan.vercel.app) |
| Backend API | Hugging Face Spaces | [yashnaiduu-neurosacn.hf.space](https://yashnaiduu-neurosacn.hf.space) |

---

## Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- Git
- Hugging Face account (for backend)
- Vercel account (for frontend)

---

## Environment Variables

| Variable | Required | Description |
|:---|:---:|:---|
| `PORT` | Optional | Backend port (default: `5050`, HF Spaces: `7860`) |
| `FLASK_ENV` | Optional | Set to `production` in deployment |
| `MODEL_URL` | Optional | Remote URL to download the `.h5` model file |

> No external API keys are required. MRI validation uses CLIP (runs locally).

---

## Backend — Hugging Face Spaces

The backend runs as a Docker container on Hugging Face Spaces.

### Redeploy

```bash
# Clone the HF Space
git clone https://yashnaiduu:YOUR_HF_TOKEN@huggingface.co/spaces/yashnaiduu/neurosacn hf_space
cd hf_space

# Copy updated files
cp ../server1.py ../requirements.txt ../Dockerfile ../entrypoint.sh .

# Commit and push (triggers rebuild)
git add . && git commit -m "Update deployment" && git push
```

### Key Files

| File | Purpose |
|:---|:---|
| `Dockerfile` | Container definition |
| `entrypoint.sh` | Startup script (downloads model, starts gunicorn) |
| `server1.py` | Flask application |
| `requirements.txt` | Python dependencies |

### Health Check

```bash
curl https://yashnaiduu-neurosacn.hf.space/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "clip_available": true,
  "uptime": 3600.5
}
```

---

## Frontend — Vercel

The frontend is a static HTML/CSS/JS site deployed on Vercel.

### Redeploy

Push to the `main` branch on GitHub — Vercel auto-deploys on every push.

```bash
git add client/
git commit -m "Update frontend"
git push
```

### Configuration

`vercel.json` handles routing — all requests are served from `client/index.html`.

---

## Local Development

```bash
# Clone the repository
git clone https://github.com/yashnaiduu/NeuroScan-Brain-Tumor-Classification.git
cd NeuroScan-Brain-Tumor-Classification

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend
python server1.py               # http://localhost:5050

# Run frontend (separate terminal)
cd client && python3 -m http.server 8000   # http://localhost:8000
```

---

## Local Docker

```bash
# Build
docker build -t neuroscan:latest .

# Run
docker run -d \
  --name neuroscan \
  -p 5050:5050 \
  -e FLASK_ENV=production \
  neuroscan:latest

# Check health
curl http://localhost:5050/health
```

---

## Troubleshooting

### Model Not Loading

**Symptom:** `/health` returns `model_loaded: false`

1. Check `MODEL_URL` is accessible and points to a valid `.h5` file
2. Check container logs for download errors
3. Ensure sufficient disk space (model is ~15MB)

### High Memory Usage

**Symptom:** Container crashes or restarts

1. Increase container memory to 2GB minimum
2. Reduce `WEB_CONCURRENCY` in `entrypoint.sh`
3. Monitor with `/stats` endpoint

### Slow Predictions

**Symptom:** Requests take longer than expected

1. CLIP model loads in the background on startup — first request may be slower
2. Increase `GUNICORN_TIMEOUT` if needed
3. CPU inference is expected; GPU instances will be significantly faster

---

## Support

**Yash Naidu** — [yashnnaidu@gmail.com](mailto:yashnnaidu@gmail.com)
