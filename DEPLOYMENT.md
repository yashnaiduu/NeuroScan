# Railway Deployment Guide

This document provides step-by-step instructions for deploying the NeuroScan Brain Tumor Classification application to Railway.

## 📋 Overview

The application consists of:
- **Backend**: Flask API (`server1.py`) for ML model inference
- **Frontend**: Next.js web application for user interface

## 🚀 Backend Deployment (Railway)

### Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **GitHub Repository**: Your code should be pushed to GitHub
3. **Model File**: `mobilenet_brain_tumor_classifier.h5` (150-200MB)
4. **Dataset**: Brain MRI dataset files (optional for production)
5. **Gemini API Key**: Get one from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Step 1: Prepare Your Repository

Ensure these files are in your repository:
- ✅ `server1.py` (Flask application)
- ✅ `requirements.txt` (Python dependencies)
- ✅ `Dockerfile` (Container configuration)
- ✅ `mobilenet_brain_tumor_classifier.h5` (model file)

**Note**: If your model file is too large (>100MB), consider using Git LFS or external storage.

### Step 2: Deploy to Railway

1. **Go to Railway**: https://railway.app
2. **Create New Project**: Click "New Project"
3. **Deploy from GitHub**: 
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will automatically detect the Dockerfile

### Step 3: Configure Environment Variables

In the Railway project dashboard, add these environment variables:

| Variable | Value | Required |
|----------|-------|----------|
| `GOOGLE_API_KEY` | Your Gemini API key | Yes |
| `PORT` | 5050 (Railway auto-sets this) | No |
| `UPLOAD_FOLDER` | Uploads | No |
| `DATASET_PATH` | ./Dataset | No* |
| `MODEL_PATH` | mobilenet_brain_tumor_classifier.h5 | No |

*Only needed if you want the `/random` endpoint to work

### Step 4: Get Your Backend URL

After deployment completes:
1. Railway will provide a URL like: `https://your-app.railway.app`
2. Copy this URL - you'll need it for frontend configuration

### Step 5: Test Your Backend

Test the endpoints:
```bash
# Health check
curl https://your-app.railway.app/

# Upload and predict (with an image file)
curl -X POST https://your-app.railway.app/predict \
  -F "file=@/path/to/brain-mri.jpg"
```

## 🎨 Frontend Configuration

### Step 1: Create `.env.local`

In your Next.js frontend directory, create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=https://your-app.railway.app
```

Replace `https://your-app.railway.app` with your actual Railway backend URL.

### Step 2: Deploy Frontend

Deploy the frontend to Vercel, Netlify, or your preferred platform:

**Vercel (Recommended)**:
1. Push code to GitHub
2. Import project in Vercel dashboard
3. Add environment variable: `NEXT_PUBLIC_API_URL`
4. Deploy

**Netlify**:
1. Build: `npm run build`
2. Deploy the `out` folder
3. Add environment variable in Netlify dashboard

## 📦 Handling Large Files

### Option 1: Git LFS (Large File Storage)

For the model file:
```bash
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add mobilenet_brain_tumor_classifier.h5
git commit -m "Add model with Git LFS"
git push
```

### Option 2: External Storage

#### Using Google Cloud Storage

1. Upload model to GCS bucket
2. Add download code to `server1.py`:

```python
from google.cloud import storage

def download_model_from_gcs():
    if not os.path.exists('mobilenet_brain_tumor_classifier.h5'):
        client = storage.Client()
        bucket = client.bucket('your-bucket-name')
        blob = bucket.blob('mobilenet_brain_tumor_classifier.h5')
        blob.download_to_filename('mobilenet_brain_tumor_classifier.h5')
        logger.info("Model downloaded from GCS")

# Call before loading model
download_model_from_gcs()
```

3. Add `google-cloud-storage` to `requirements.txt`
4. Set `GOOGLE_APPLICATION_CREDENTIALS` in Railway

#### Using AWS S3

1. Upload model to S3 bucket
2. Add download code:

```python
import boto3

def download_model_from_s3():
    if not os.path.exists('mobilenet_brain_tumor_classifier.h5'):
        s3 = boto3.client('s3')
        s3.download_file('your-bucket', 
                        'mobilenet_brain_tumor_classifier.h5',
                        'mobilenet_brain_tumor_classifier.h5')
        logger.info("Model downloaded from S3")
```

3. Add `boto3` to `requirements.txt`
4. Set AWS credentials in Railway

### Option 3: Railway Volume

Railway supports persistent storage volumes:
1. Create a volume in Railway dashboard
2. Mount it to `/app/models`
3. Update `MODEL_PATH` to `/app/models/mobilenet_brain_tumor_classifier.h5`
4. Upload model to volume via Railway CLI

## 🔍 Troubleshooting

### Build Failures

**Issue**: Dockerfile build fails
- Check Railway build logs
- Verify all dependencies in `requirements.txt`
- Ensure sufficient memory (upgrade Railway plan if needed)

**Issue**: Model file not found
- Verify model file is in repository or accessible via storage
- Check `MODEL_PATH` environment variable

### Runtime Errors

**Issue**: Import errors
- Verify `requirements.txt` includes all dependencies:
  - `flask==2.3.3`
  - `flask-cors==4.0.0`
  - `gunicorn==20.1.0`
  - `tensorflow==2.15.0`
  - `google-generativeai==0.8.5`

**Issue**: CORS errors from frontend
- Verify `CORS(app)` is in `server1.py`
- Check frontend is using correct backend URL

**Issue**: Gemini API errors
- Verify `GOOGLE_API_KEY` is set in Railway
- Check API key is valid and has quota remaining

### Performance Issues

**Issue**: Slow predictions
- Increase Railway instance size
- Consider using GPU instances for TensorFlow
- Enable model caching

**Issue**: Timeout errors
- Increase gunicorn timeout (default: 120s)
- Optimize model inference code

## 📊 Monitoring

Railway provides:
- **Logs**: View application logs in real-time
- **Metrics**: CPU, memory, network usage
- **Deployments**: Track deployment history

Access these in the Railway project dashboard.

## 🔒 Security Best Practices

1. **Never commit API keys** to the repository
2. **Use environment variables** for all secrets
3. **Enable HTTPS** (Railway provides this automatically)
4. **Limit file upload sizes** (already set to 16MB)
5. **Validate all inputs** (already implemented)
6. **Keep dependencies updated** regularly

## 💰 Cost Considerations

Railway pricing (as of 2025):
- **Free Tier**: $5 credit/month
- **Developer Plan**: $20/month (more resources)
- **Team Plan**: Custom pricing

Model inference is compute-intensive. Monitor usage and upgrade as needed.

## 📚 Additional Resources

- [Railway Documentation](https://docs.railway.app)
- [Flask Deployment Best Practices](https://flask.palletsprojects.com/en/2.3.x/deploying/)
- [TensorFlow Model Optimization](https://www.tensorflow.org/model_optimization)
- [Gunicorn Configuration](https://docs.gunicorn.org/en/stable/configure.html)

## 🆘 Support

If you encounter issues:
1. Check Railway build/deployment logs
2. Review this deployment guide
3. Open an issue on GitHub
4. Consult Railway community/support

---

**Last Updated**: October 2025
**Maintained by**: Yash Naidu
