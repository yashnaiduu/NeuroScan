# NeuroScan — Brain Tumor Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Vercel](https://img.shields.io/badge/Vercel-Deployed-000000?style=for-the-badge&logo=vercel&logoColor=white)

**AI-powered brain tumor classification from MRI scans using MobileNetV2 + Grad-CAM**

[🌐 Live Demo](https://neuro-scan-brain-tumor-classification.vercel.app) · [🤗 Backend API](https://yashnaiduu-neurosacn.hf.space) · [📊 Dataset](https://huggingface.co/datasets/Sartajbhuvaji/Brain-Tumor-Classification-MRI)

</div>

---

## Overview

NeuroScan is a full-stack medical imaging web application that classifies brain MRI scans into four categories using a fine-tuned **MobileNetV2** deep learning model. It features real-time Grad-CAM heatmap visualization, CLIP-based MRI validation, and a clean, responsive frontend.

## Live Deployment

| Service | URL |
|---------|-----|
| **Frontend** | [neuro-scan-brain-tumor-classification.vercel.app](https://neuro-scan-brain-tumor-classification.vercel.app) |
| **Backend API** | [yashnaiduu-neurosacn.hf.space](https://yashnaiduu-neurosacn.hf.space) |

## Features

- 🧠 **4-Class Tumor Classification** — Glioma, Meningioma, Pituitary, No Tumor
- 🔥 **Grad-CAM Heatmaps** — Visual explanation of which brain regions influenced the prediction
- 🔍 **CLIP MRI Validation** — Rejects non-MRI images before classification using OpenAI CLIP
- 🎲 **Random Sample Testing** — Try the model with real MRI samples from the bundled dataset
- 📊 **Confidence Scores** — Per-class probability breakdown for every prediction
- 🌙 **Dark Mode UI** — Responsive, modern interface with smooth animations

## Architecture

```
┌─────────────────────┐         ┌──────────────────────────────┐
│   Frontend (Vercel) │  HTTP   │   Backend (Hugging Face)     │
│   Static HTML/JS    │ ──────► │   Flask API                  │
│                     │         │                              │
│  - Upload MRI       │         │  1. CLIP validates MRI       │
│  - View heatmap     │ ◄────── │  2. MobileNetV2 classifies   │
│  - See results      │  JSON   │  3. Grad-CAM generates map   │
└─────────────────────┘         └──────────────────────────────┘
```

### Model Architecture

| Layer | Details |
|-------|---------|
| Base Model | MobileNetV2 (pre-trained on ImageNet) |
| Input | 224×224 RGB |
| Feature Extraction | Depthwise separable convolutions |
| Pooling | Global Average Pooling |
| Regularization | Dropout (0.5) |
| Output | Dense (4 units) + Softmax |

- **Accuracy**: 96.8% on test set
- **Inference time**: <2s on CPU

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/stats` | GET | Model & system stats |
| `/predict` | POST | Classify uploaded MRI |
| `/heatmap` | POST | Generate Grad-CAM heatmap |
| `/random` | GET | Classify a random sample MRI |

## Tech Stack

**Backend**
- Python 3.9+, Flask, Flask-CORS
- TensorFlow / Keras (MobileNetV2)
- OpenCV (image processing)
- Transformers + PyTorch (CLIP validation)
- Deployed on **Hugging Face Spaces** (Docker)

**Frontend**
- HTML5, CSS3, Vanilla JavaScript
- Font Awesome, Phosphor Icons
- Deployed on **Vercel**

## Dataset

Uses the [Brain Tumor Classification (MRI)](https://huggingface.co/datasets/Sartajbhuvaji/Brain-Tumor-Classification-MRI) dataset with 4 classes:

| Class | Description |
|-------|-------------|
| Glioma | Primary brain tumors from glial cells |
| Meningioma | Tumors arising from the meninges |
| Pituitary | Tumors affecting the pituitary gland |
| No Tumor | Healthy brain scans |

## Local Development

### Prerequisites
- Python 3.9+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/yashnaiduu/NeuroScan-Brain-Tumor-Classification.git
cd NeuroScan-Brain-Tumor-Classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY (optional, for Gemini)

# Run backend
python server1.py
# Backend runs at http://localhost:5050

# Run frontend (in a separate terminal)
cd client && python3 -m http.server 8000
# Frontend at http://localhost:8000
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Optional | Gemini Vision API key for enhanced analysis |
| `PORT` | Optional | Backend port (default: 5050, HF Spaces: 7860) |

## Deployment

| Platform | Purpose | Config File |
|----------|---------|-------------|
| Hugging Face Spaces | Backend API | `Dockerfile`, `entrypoint.sh` |
| Vercel | Frontend | `vercel.json` |

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**Yash Naidu**  
[yash.22bce8038@vitapstudent.ac.in](mailto:yash.22bce8038@vitapstudent.ac.in)
