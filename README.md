# 🧠 NeuroScan - Brain Tumor Classification

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **AI-Powered Brain Tumor Classification System with Grad-CAM Visualization**

NeuroScan is an advanced machine learning application that classifies brain MRI scans into four categories: **Glioma**, **Meningioma**, **No Tumor**, and **Pituitary Tumor**. The system combines deep learning with explainable AI techniques to provide both accurate predictions and visual heatmaps for medical professionals.

## 🎬 Preview

<div align="center">
  <video width="800" height="450" controls style="border-radius: 15px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);">
    <source src="/Users/yashnaidu/Desktop/preview.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

## 🚀 Features

### 🔬 **Advanced AI Classification**
- **Multi-class Classification**: Accurately identifies 4 types of brain conditions
- **Deep Learning Model**: Built with MobileNet architecture for optimal performance
- **Real-time Processing**: Fast inference for immediate results

### 🎯 **Explainable AI**
- **Grad-CAM Visualization**: Generate heatmaps showing which regions influenced the prediction
- **Interactive Heatmaps**: Visual explanations for medical professionals
- **Confidence Scores**: Detailed probability distributions for each class

### 🛡️ **Quality Assurance**
- **Gemini AI Validation**: Uses Google's Gemini Vision API to verify MRI authenticity
- **Image Preprocessing**: Automatic image enhancement and normalization
- **Error Handling**: Robust error management for various edge cases

### 🌐 **Web Interface**
- **Modern UI**: Clean, responsive web interface
- **File Upload**: Drag-and-drop or click-to-upload functionality
- **Real-time Results**: Instant classification with detailed reports
- **Random Sample Testing**: Test with pre-loaded dataset samples

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
   git clone https://github.com/yourusername/NeuroScan-Brain-Tumor-Classification.git
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

**Response:**
```json
{
  "prediction": "meningioma",
  "confidence": 0.87,
  "image_path": "path/to/sample.jpg",
  "probabilities": {...}
}
```

#### `POST /heatmap`
Generate a Grad-CAM heatmap for the uploaded image.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Image file

**Response:**
```json
{
  "heatmap_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "class_index": 0,
  "message": "Heatmap generated successfully"
}
```

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

## 🚀 Deployment

### Railway Deployment

1. **Connect to Railway**
   - Link your GitHub repository to Railway
   - Railway will automatically detect the Dockerfile

2. **Environment Variables**
   - Set `GOOGLE_API_KEY` in Railway dashboard
   - Configure `PORT` if needed

3. **Deploy**
   - Railway will automatically build and deploy your application

### Other Platforms

The application can be deployed on:
- **Heroku**: Use the provided Dockerfile
- **AWS**: Deploy using ECS or EC2
- **Google Cloud**: Use Cloud Run or Compute Engine
- **Azure**: Deploy to App Service or Container Instances

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

- **Issues**: [GitHub Issues](https://github.com/yourusername/NeuroScan-Brain-Tumor-Classification/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/NeuroScan-Brain-Tumor-Classification/discussions)
- **Email**: your.email@example.com

---

<div align="center">

**Made with ❤️ for Medical AI Research**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/NeuroScan-Brain-Tumor-Classification?style=social)](https://github.com/yourusername/NeuroScan-Brain-Tumor-Classification)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/NeuroScan-Brain-Tumor-Classification?style=social)](https://github.com/yourusername/NeuroScan-Brain-Tumor-Classification)
[![GitHub issues](https://img.shields.io/github/issues/yourusername/NeuroScan-Brain-Tumor-Classification)](https://github.com/yourusername/NeuroScan-Brain-Tumor-Classification/issues)

</div> 