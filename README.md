<!--  ╔══════════════════════════════════════════════════╗  -->
<!--  ║               N E U R O S C A N                 ║  -->
<!--  ╚══════════════════════════════════════════════════╝  -->

<div align="center">

<br/>

<img src="https://readme-typing-svg.demolab.com?font=Inter&weight=800&size=52&duration=3500&pause=1000&color=FFFFFF&center=true&vCenter=true&width=700&lines=NeuroScan" alt="NeuroScan" />

<img src="https://readme-typing-svg.demolab.com?font=Inter&weight=400&size=18&duration=3500&pause=1000&color=888888&center=true&vCenter=true&width=700&lines=Brain+Tumor+Classification+from+MRI+Scans" alt="subtitle" />

<br/><br/>

<a href="https://neuroscan.vercel.app"><img src="https://img.shields.io/badge/LIVE%20DEMO-000000?style=for-the-badge&logo=vercel&logoColor=white" /></a>
&nbsp;
<a href="https://yashnaiduu-neurosacn.hf.space"><img src="https://img.shields.io/badge/BACKEND%20API-6d28d9?style=for-the-badge&logo=huggingface&logoColor=white" /></a>
&nbsp;
<a href="https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri"><img src="https://img.shields.io/badge/DATASET-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" /></a>

<br/><br/>

<img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" />
<img src="https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask&logoColor=white" />
<img src="https://img.shields.io/badge/OpenAI%20CLIP-412991?style=flat-square&logo=openai&logoColor=white" />

<br/><br/>

</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/yashnaiduu/NeuroScan-Brain-Tumor-Classification/main/preview.gif" width="640" alt="NeuroScan Preview" />
</div>

<br/>

---

<br/>

## &nbsp;What is NeuroScan?

NeuroScan is a full-stack medical imaging application that classifies brain MRI scans into four categories using a fine-tuned **MobileNetV2** model. Upload an MRI, and the app instantly tells you the tumor type, shows a **Grad-CAM heatmap** highlighting the affected region, and validates the image is actually an MRI using **OpenAI CLIP** — all in under 2 seconds.

<br/>

---

<br/>

## &nbsp;Features

<br/>

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>🧠&nbsp; 4-Class Classification</h3>
      <p>Identifies <strong>Glioma</strong>, <strong>Meningioma</strong>, <strong>Pituitary</strong> tumors, or a <strong>healthy</strong> scan with 96.8% accuracy.</p>
    </td>
    <td width="50%" valign="top">
      <h3>🔥&nbsp; Grad-CAM Heatmaps</h3>
      <p>Generates visual saliency maps that highlight exactly which brain regions influenced the model's prediction.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>�️&nbsp; CLIP MRI Validation</h3>
      <p>Uses OpenAI CLIP to reject non-MRI images before they reach the classifier — no garbage in, no garbage out.</p>
    </td>
    <td width="50%" valign="top">
      <h3>📊&nbsp; Confidence Scores</h3>
      <p>Returns a full probability breakdown across all four classes for every prediction.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>🎲&nbsp; Random Sample Testing</h3>
      <p>One-click testing using real MRI samples bundled with the application — no upload required.</p>
    </td>
    <td width="50%" valign="top">
      <h3>🌙&nbsp; Dark Mode UI</h3>
      <p>Sleek, responsive interface with smooth animations — designed for clarity and ease of use.</p>
    </td>
  </tr>
</table>

<br/>

---

<br/>

## &nbsp;Model Architecture


<div align="center">

<table>
  <tr>
    <td align="center"><b>📥 Input</b><br/><sub>224 × 224 RGB</sub></td>
    <td align="center">→</td>
    <td align="center" colspan="4">
      <table>
        <tr><td align="center" colspan="4"><sub><b>🧬 MobileNetV2 Feature Extractor</b></sub></td></tr>
        <tr>
          <td align="center"><sub>Conv2D<br/>32 filters</sub></td>
          <td align="center"><sub>→</sub></td>
          <td align="center"><sub>Expansion<br/>1×1 Conv</sub></td>
          <td align="center"><sub>→</sub></td>
          <td align="center"><sub>Depthwise<br/>3×3 Conv</sub></td>
          <td align="center"><sub>→</sub></td>
          <td align="center"><sub>Projection<br/>1×1 Conv</sub></td>
        </tr>
      </table>
    </td>
    <td align="center">→</td>
    <td align="center"><b>GAP</b><br/><sub>Global Avg Pool</sub></td>
    <td align="center">→</td>
    <td align="center"><b>Dropout</b><br/><sub>0.5</sub></td>
    <td align="center">→</td>
    <td align="center"><b>Dense</b><br/><sub>4 units</sub></td>
    <td align="center">→</td>
    <td align="center"><b>📤 Softmax</b><br/><sub>4 Classes</sub></td>
  </tr>
</table>

<br/>

| | |
|---|---|
| **Base Model** | MobileNetV2 — ImageNet pre-trained |
| **Input** | 224 × 224 RGB |
| **Backbone** | Depthwise separable convolutions |
| **Head** | GAP → Dropout(0.5) → Dense(4) → Softmax |
| **Accuracy** | 96.8% on held-out test set |
| **Inference** | < 2s on CPU |

</div>



---

<br/>

## &nbsp;API Reference

<div align="center">

| Endpoint | Method | Description |
|:---|:---:|:---|
| `/` | `GET` | API info & version |
| `/health` | `GET` | Health check |
| `/stats` | `GET` | Model & system stats |
| `/predict` | `POST` | Classify an uploaded MRI |
| `/heatmap` | `POST` | Generate Grad-CAM heatmap |
| `/random` | `GET` | Test with a random sample MRI |

</div>

<br/>

---

<br/>

## &nbsp;Tech Stack

<div align="center">

|  | Technologies |
|---|---|
| **Backend** | Python · Flask · TensorFlow/Keras · OpenCV · CLIP |
| **Frontend** | HTML · CSS · Vanilla JavaScript |
| **Hosting** | Hugging Face Spaces (backend) · Vercel (frontend) |

</div>

<br/>

---

<br/>

## &nbsp;Dataset

The [Brain Tumor Classification MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) dataset from Kaggle — 3,264 labeled MRI scans across four classes.

<div align="center">

| Class | Count | Description |
|:---|:---:|:---|
| Glioma | ~826 | Primary tumors from glial cells |
| Meningioma | ~822 | Tumors from the meninges |
| Pituitary | ~827 | Tumors of the pituitary gland |
| No Tumor | ~395 | Healthy brain scans |

</div>

<br/>

---

<br/>

## &nbsp;Run Locally

```bash
# 1. Clone
git clone https://github.com/yashnaiduu/NeuroScan-Brain-Tumor-Classification.git
cd NeuroScan-Brain-Tumor-Classification

# 2. Environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Start backend
python server1.py               # → http://localhost:5050

# 4. Start frontend (new terminal)
cd client && python3 -m http.server 8000   # → http://localhost:8000
```

> **Tip:** Set `PORT=7860` when deploying to Hugging Face Spaces.

<br/>

---

<br/>

## &nbsp;Deployment

| Platform | Role | Key Files |
|:---|:---|:---|
| Hugging Face Spaces | Backend API | `Dockerfile`, `entrypoint.sh` |
| Vercel | Frontend | `vercel.json` |

Detailed steps in [DEPLOYMENT.md](DEPLOYMENT.md).

<br/>

---

<br/>

## &nbsp;License

MIT — see [LICENSE](LICENSE).

<br/>

---

<br/>

<div align="center">

**Yash Naidu**

<a href="mailto:yashnnaidu@gmail.com"><img src="https://img.shields.io/badge/yashnnaidu%40gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white" /></a>

<br/><br/>

<sub>Built with TensorFlow &nbsp;·&nbsp; Deployed on Hugging Face & Vercel</sub>

<br/>

</div>
