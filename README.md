# 🧠 NeuroScan - Brain Tumor Classification

Create beautiful, interactive brain tumor classification with AI-powered analysis and Grad-CAM visualization! 

Drop a ✨ if you are here. It would mean a lot : )

## 🎬 Preview

<div align="center">

[![NeuroScan Brain Tumor Classification Demo](https://img.youtube.com/vi/LErblUjkCtU/maxresdefault.jpg)](https://youtu.be/LErblUjkCtU "NeuroScan Brain Tumor Classification Demo")

*Click the image above to watch the demo video*

</div>

## 🚀 Features

1. **Advanced AI Classification** - Accurately identifies 4 types of brain conditions
2. **Real-time Processing** - Fast inference for immediate results  
3. **Grad-CAM Visualization** - Generate heatmaps showing which regions influenced the prediction
4. **Gemini AI Validation** - Uses Google's Gemini Vision API to verify MRI authenticity
5. **Modern Web Interface** - Clean, responsive web interface with drag-and-drop functionality
6. **Small Model Size** - Optimized MobileNet model (~14MB compressed)
7. **High Accuracy** - ~95% accuracy on test dataset

## 📊 Classification Categories

| Category | Description | Medical Significance |
|----------|-------------|---------------------|
| **Glioma** | Primary brain tumors arising from glial cells | Most common primary brain tumor |
| **Meningioma** | Tumors from meninges (brain covering) | Usually benign, slow-growing |
| **No Tumor** | Normal brain tissue | Healthy brain scan |
| **Pituitary** | Tumors in pituitary gland | Can affect hormone production |

## 🛠️ Technology Stack

- **Backend**: Flask (Python web framework)
- **AI/ML**: TensorFlow, Keras, MobileNet
- **Computer Vision**: OpenCV, PIL
- **AI Validation**: Google Gemini Vision API
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Docker, Railway

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- Git
- Docker (for containerized deployment)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yashnaiduu/NeuroScan-Brain-Tumor-Classification.git
   cd NeuroScan-Brain-Tumor-Classification
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file (optional)
   echo "GOOGLE_API_KEY=your_gemini_api_key_here" > .env
   ```

5. **Run the application**
   ```bash
   python server1.py
   ```

6. **Access the application**
   Open your browser and navigate to `http://localhost:5050`

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t neuroscan .
   ```

2. **Run the container**
   ```bash
   docker run -p 5050:5050 neuroscan
   ```

## 🎮 Usage

### Web Interface

1. **Upload Image**
   - Click "Choose File" or drag-and-drop an MRI image
   - Supported formats: JPG, PNG, JPEG, BMP
   - Maximum file size: 16MB

2. **Get Results**
   - The system will automatically:
     - Validate if the image is a brain MRI
     - Classify the tumor type
     - Generate confidence scores
     - Provide detailed analysis

3. **View Heatmap**
   - Click "Generate Heatmap" to see which brain regions influenced the prediction
   - Red areas indicate high influence, blue areas indicate low influence

### API Endpoints

#### `POST /predict`
Classify a brain MRI image.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Image file

**Response:**
```json
{
  "prediction": "glioma",
  "confidence": 0.95,
  "probabilities": {
    "glioma": 0.95,
    "meningioma": 0.03,
    "notumor": 0.01,
    "pituitary": 0.01
  },
  "is_mri": true,
  "message": "Successfully classified as Glioma"
}
```

#### `GET /random`
Get a random sample prediction from the dataset.

#### `POST /heatmap`
Generate a Grad-CAM heatmap for the uploaded image.

## 🏥 Medical Disclaimer

⚠️ **Important Notice**

This application is designed for **research and educational purposes only**. 

- **Not for Clinical Use**: The system should not be used for actual medical diagnosis
- **Consult Professionals**: Always consult qualified medical professionals for diagnosis
- **Accuracy Limitations**: AI models may have limitations and biases
- **Data Privacy**: Ensure patient data privacy when using this system

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Application port | `5050` |
| `GOOGLE_API_KEY` | Gemini API key | Not set |
| `UPLOAD_FOLDER` | Upload directory | `Uploads` |
| `MODEL_PATH` | Model file path | `mobilenet_brain_tumor_classifier.h5` |

### Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 224x224 pixels
- **Output**: 4 classes (glioma, meningioma, notumor, pituitary)
- **Optimization**: Adam optimizer with categorical crossentropy loss

## 📈 Performance

- **Accuracy**: ~95% on test dataset
- **Inference Time**: < 2 seconds per image
- **Memory Usage**: ~500MB RAM
- **Model Size**: ~14MB (compressed)

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git add .
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Brain Tumor Classification Dataset
- **Model Architecture**: MobileNetV2 (Google)
- **AI Validation**: Google Gemini Vision API
- **Web Framework**: Flask (Pallets)
- **Computer Vision**: OpenCV

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yashnaiduu/NeuroScan-Brain-Tumor-Classification/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yashnaiduu/NeuroScan-Brain-Tumor-Classification/discussions)
- **Email**: yash.22bce8038@vitapstudent.ac.in

## Support

If you find this project helpful, consider buying me a coffee!

<p align="center">
  <a href="https://buymeacoffee.com/yahnaiduu" target="_blank">
    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;">
  </a>
</p>

---

<div align="center">

**Made with ❤️ for Medical AI Research**

[![GitHub stars](https://img.shields.io/github/stars/yashnaiduu/NeuroScan-Brain-Tumor-Classification?style=social)](https://github.com/yashnaiduu/NeuroScan-Brain-Tumor-Classification)
[![GitHub forks](https://img.shields.io/github/forks/yashnaiduu/NeuroScan-Brain-Tumor-Classification?style=social)](https://github.com/yashnaiduu/NeuroScan-Brain-Tumor-Classification)
[![GitHub issues](https://img.shields.io/github/issues/yashnaiduu/NeuroScan-Brain-Tumor-Classification)](https://github.com/yashnaiduu/NeuroScan-Brain-Tumor-Classification/issues)

</div>

## Thanks

If you read till here, thanks for taking interest in this. Hope I could be useful for you :) 
