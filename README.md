# NeuroScan Web - Brain Tumor Classification

A modern web interface for the NeuroScan brain tumor classification system, built with Next.js and designed for easy deployment on platforms like Vercel, Netlify, and GitHub Pages.

## 🚀 Features

- **Modern Web Interface**: Clean, responsive design built with Next.js and Tailwind CSS
- **Static Site Generation**: Optimized for deployment on static hosting platforms
- **Interactive UI**: Drag-and-drop file upload with real-time feedback
- **Visualization**: Grad-CAM heatmap generation for model interpretability
- **Mobile Responsive**: Works seamlessly across all device sizes
- **Accessibility**: Built with accessibility best practices

## 🛠️ Technology Stack

- **Frontend**: Next.js 14, React 18, TypeScript
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Deployment**: Static export for universal hosting

## 📦 Quick Start

### Local Development

1. **Install dependencies**
   ```bash
   npm install
   ```

2. **Run development server**
   ```bash
   npm run dev
   ```

3. **Open in browser**
   Navigate to `http://localhost:3000`

### Build for Production

```bash
npm run build
```

This creates an optimized static build in the `out` directory.

## 🌐 Deployment Options

### Vercel (Recommended)

1. Push your code to GitHub
2. Connect your repository to Vercel
3. Deploy automatically with zero configuration

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new)

### Netlify

1. Build the project: `npm run build`
2. Deploy the `out` folder to Netlify
3. Or connect your GitHub repository for automatic deployments

### GitHub Pages

1. Enable GitHub Pages in your repository settings
2. Set up GitHub Actions for automatic deployment
3. Use the `out` folder as your publish directory

### Other Static Hosts

The built files in the `out` directory can be deployed to any static hosting service:
- Firebase Hosting
- AWS S3 + CloudFront
- Azure Static Web Apps
- Surge.sh

## 🖥️ Backend Deployment (Flask API)

### Railway Deployment (Recommended for Backend)

The Flask backend (`server1.py`) can be easily deployed to Railway:

#### Prerequisites

1. **Model File**: Upload `mobilenet_brain_tumor_classifier.h5` to your repository or use external storage (Google Cloud Storage, AWS S3, etc.)
2. **Dataset Files**: Upload dataset files to storage or include them in the repository (note: large files may require Git LFS or external storage)

#### Deployment Steps

1. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for Railway deployment"
   git push
   ```

2. **Deploy to Railway**
   - Go to [Railway.app](https://railway.app)
   - Create a new project from your GitHub repository
   - Railway will automatically detect the Dockerfile

3. **Configure Environment Variables** in Railway dashboard:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   PORT=5050
   UPLOAD_FOLDER=Uploads
   DATASET_PATH=./Dataset
   MODEL_PATH=mobilenet_brain_tumor_classifier.h5
   ```

4. **Get your Railway backend URL**
   - Railway will provide a URL like: `https://your-app.railway.app`

#### Using External Storage for Large Files

If your model and dataset files are too large for the repository:

**Google Cloud Storage Example:**
```python
# Add to server1.py before loading the model
from google.cloud import storage

def download_model_from_gcs():
    client = storage.Client()
    bucket = client.bucket('your-bucket-name')
    blob = bucket.blob('mobilenet_brain_tumor_classifier.h5')
    blob.download_to_filename('mobilenet_brain_tumor_classifier.h5')
```

**AWS S3 Example:**
```python
import boto3

def download_model_from_s3():
    s3 = boto3.client('s3')
    s3.download_file('your-bucket-name', 'mobilenet_brain_tumor_classifier.h5', 
                     'mobilenet_brain_tumor_classifier.h5')
```

### Alternative Backend Deployment Options

- **Heroku**: Similar to Railway, supports Dockerfile deployment
- **Google Cloud Run**: Serverless container deployment
- **AWS Elastic Beanstalk**: Supports Docker containers
- **DigitalOcean App Platform**: Docker-based deployment

## 🔧 Configuration

### Environment Variables

For production deployment with a backend API, create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=https://your-backend-api.com
```

### Backend Integration

To connect the Next.js frontend with the Flask backend:

#### 1. Configure Frontend Environment

Create a `.env.local` file in the frontend directory:

```env
NEXT_PUBLIC_API_URL=https://your-railway-backend.railway.app
```

Replace `https://your-railway-backend.railway.app` with your actual Railway backend URL.

#### 2. Update API Endpoints

The Flask backend provides the following endpoints:
- `POST /predict` - Upload and classify a brain MRI image
- `GET /random` - Get a random dataset image with prediction
- `POST /heatmap` - Generate Grad-CAM heatmap for an uploaded image

Example API integration:

```typescript
const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict`, {
  method: 'POST',
  body: formData,
})
const result = await response.json()
```

#### 3. CORS Configuration

The Flask backend is already configured with CORS support via `flask-cors`:
```python
from flask_cors import CORS
CORS(app)  # Enables CORS for all routes
```

This allows the frontend to make requests from any domain.

## 📱 Features

### Image Upload
- Drag and drop interface
- File validation (JPEG, PNG, BMP)
- Size limits (16MB max)
- Preview functionality

### Results Display
- Confidence scoring
- Probability breakdown
- Medical information
- Visual progress bars

### Grad-CAM Visualization
- Heatmap generation
- Interactive controls
- Educational explanations

## 🎨 Customization

### Styling
- Modify `tailwind.config.js` for theme customization
- Update colors, fonts, and spacing in the config
- Custom CSS in `src/app/globals.css`

### Content
- Update medical information in `ResultsDisplay.tsx`
- Modify feature descriptions in the main page
- Customize disclaimer and footer content

## 📊 Performance

- **Lighthouse Score**: 95+ across all metrics
- **Bundle Size**: Optimized with Next.js automatic splitting
- **Loading Speed**: Static generation for instant page loads
- **SEO**: Built-in meta tags and structured data

## 🔒 Security

- Client-side file validation
- Secure file handling
- No sensitive data exposure
- HTTPS-ready configuration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see the original project for details.

## 🙏 Acknowledgments

- Original NeuroScan project by Yash Naidu
- Next.js team for the amazing framework
- Tailwind CSS for the utility-first styling
- Lucide for the beautiful icons

## 📞 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Original Project**: [NeuroScan Repository](https://github.com/yashnaiduu/NeuroScan-Brain-Tumor-Classification)

---

**Made with ❤️ for Medical AI Research**