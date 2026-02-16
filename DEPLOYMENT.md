# NeuroScan Deployment Guide

## Overview

This guide covers deploying NeuroScan to production environments including Docker, cloud platforms, and configuration best practices.

## Prerequisites

- Docker installed (for containerized deployment)
- Model file (`mobilenet_brain_tumor_classifier.h5`) or URL to download it
- Google Gemini API key (optional but recommended)
- Cloud platform account (Railway, Render, AWS, GCP, etc.)

## Environment Configuration

### 1. Create Environment File

Copy the example environment file:

```bash
cp .env.example .env
```

### 2. Configure Required Variables

Edit `.env` and set the following:

```bash
# Required
GOOGLE_API_KEY=your_actual_gemini_api_key
MODEL_URL=https://your-storage/mobilenet_brain_tumor_classifier.h5

# Optional (defaults provided)
PORT=5050
FLASK_ENV=production
SECRET_KEY=generate_a_secure_random_key_here
```

### 3. Generate Secret Key

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

## Local Docker Deployment

### Build the Image

```bash
docker build -t neuroscan:latest .
```

### Run the Container

```bash
docker run -d \
  --name neuroscan \
  -p 5050:5050 \
  -e GOOGLE_API_KEY=your_api_key \
  -e MODEL_URL=https://your-storage/model.h5 \
  -e FLASK_ENV=production \
  neuroscan:latest
```

### Check Health

```bash
curl http://localhost:5050/health
```

## Cloud Platform Deployment

### Railway

1. **Create New Project**
   - Go to [Railway](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"

2. **Configure Environment Variables**
   ```
   GOOGLE_API_KEY=your_api_key
   MODEL_URL=https://your-storage/model.h5
   FLASK_ENV=production
   PORT=5050
   ```

3. **Deploy**
   - Railway will automatically detect the Dockerfile
   - Deployment starts automatically

### Render

1. **Create New Web Service**
   - Go to [Render](https://render.com)
   - Click "New" → "Web Service"
   - Connect your GitHub repository

2. **Configure Service**
   - **Environment**: Docker
   - **Region**: Choose closest to your users
   - **Instance Type**: Choose based on expected load

3. **Environment Variables**
   ```
   GOOGLE_API_KEY=your_api_key
   MODEL_URL=https://your-storage/model.h5
   FLASK_ENV=production
   ```

4. **Deploy**
   - Click "Create Web Service"

### AWS (Elastic Container Service)

1. **Push Image to ECR**
   ```bash
   aws ecr create-repository --repository-name neuroscan
   docker tag neuroscan:latest <account-id>.dkr.ecr.<region>.amazonaws.com/neuroscan:latest
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/neuroscan:latest
   ```

2. **Create ECS Task Definition**
   - Use the pushed image
   - Set environment variables
   - Configure 2GB memory, 1 vCPU minimum

3. **Create ECS Service**
   - Use Fargate launch type
   - Configure load balancer
   - Set desired count to 2+ for high availability

### Google Cloud Platform (Cloud Run)

1. **Build and Push to GCR**
   ```bash
   gcloud builds submit --tag gcr.io/<project-id>/neuroscan
   ```

2. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy neuroscan \
     --image gcr.io/<project-id>/neuroscan \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars GOOGLE_API_KEY=your_api_key,MODEL_URL=https://storage/model.h5
   ```

## Model File Hosting

### Option 1: Cloud Storage

**AWS S3:**
```bash
aws s3 cp mobilenet_brain_tumor_classifier.h5 s3://your-bucket/models/
aws s3 presign s3://your-bucket/models/mobilenet_brain_tumor_classifier.h5 --expires-in 31536000
```

**Google Cloud Storage:**
```bash
gsutil cp mobilenet_brain_tumor_classifier.h5 gs://your-bucket/models/
gsutil signurl -d 365d service-account-key.json gs://your-bucket/models/mobilenet_brain_tumor_classifier.h5
```

### Option 2: GitHub Releases

1. Create a new release in your repository
2. Upload the model file as a release asset
3. Use the direct download URL as MODEL_URL

### Option 3: Include in Docker Image

If model file is small enough, include it directly:

```dockerfile
# Add to Dockerfile before CMD
COPY mobilenet_brain_tumor_classifier.h5 /app/
```

## SSL/TLS Configuration

### Using Cloudflare (Recommended)

1. Add your domain to Cloudflare
2. Update DNS to point to your deployment
3. Enable "Full" SSL/TLS encryption mode
4. Cloudflare provides free SSL certificates

### Using Let's Encrypt with Nginx

```nginx
server {
    listen 80;
    server_name neuroscan.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name neuroscan.yourdomain.com;
    
    ssl_certificate /etc/letsencrypt/live/neuroscan.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/neuroscan.yourdomain.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:5050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring and Logging

### Health Checks

Configure your platform to check `/health` endpoint:

```bash
curl https://your-app.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gemini_available": true,
  "uptime": 3600.5
}
```

### Log Aggregation

**CloudWatch (AWS):**
- Automatically enabled for ECS/Fargate
- View logs in CloudWatch Logs console

**Google Cloud Logging:**
- Automatically enabled for Cloud Run
- View logs in Cloud Logging console

**Third-party (Datadog, New Relic):**
```bash
# Add to environment variables
DD_API_KEY=your_datadog_api_key
DD_SITE=datadoghq.com
```

## Performance Optimization

### Gunicorn Configuration

Adjust workers and threads based on your instance:

```bash
# For 2GB RAM, 1 vCPU
WEB_CONCURRENCY=2
WEB_THREADS=4

# For 4GB RAM, 2 vCPU
WEB_CONCURRENCY=4
WEB_THREADS=4
```

### Caching

The application includes built-in caching. Configure duration:

```bash
CACHE_DURATION=3600  # 1 hour in seconds
```

## Security Best Practices

1. **Never commit `.env` file** - Already in `.gitignore`
2. **Rotate API keys regularly** - Update in platform settings
3. **Use HTTPS only** - Configure SSL/TLS
4. **Enable rate limiting** - Already enabled in production
5. **Monitor logs** - Set up alerts for errors
6. **Keep dependencies updated** - Run `pip list --outdated` regularly

## Troubleshooting

### Model Not Loading

**Symptom:** `/health` returns `model_loaded: false`

**Solutions:**
1. Check MODEL_URL is accessible
2. Verify model file format (should be .h5)
3. Check container logs for download errors
4. Ensure sufficient disk space

### High Memory Usage

**Symptom:** Container crashes or restarts

**Solutions:**
1. Increase container memory to 2GB minimum
2. Reduce WEB_CONCURRENCY
3. Enable swap if available
4. Monitor with `/stats` endpoint

### Slow Predictions

**Symptom:** Requests timeout or take >10 seconds

**Solutions:**
1. Increase GUNICORN_TIMEOUT to 180
2. Use GPU-enabled instances if available
3. Reduce image size before upload
4. Check network latency to Gemini API

### Gemini API Errors

**Symptom:** Predictions work but validation fails

**Solutions:**
1. Verify GOOGLE_API_KEY is valid
2. Check API quota limits
3. Application will fallback gracefully
4. Monitor logs for specific error messages

## Backup and Disaster Recovery

### Database Backups

Currently, the application is stateless. If you add a database:

1. Enable automated backups on your cloud platform
2. Test restore procedures monthly
3. Store backups in different region

### Application Backups

1. **Code:** Version controlled in Git
2. **Model:** Store in multiple locations (S3, GCS, GitHub Releases)
3. **Configuration:** Document all environment variables

## Scaling

### Horizontal Scaling

**Railway/Render:**
- Increase number of instances in dashboard
- Configure load balancer

**AWS ECS:**
```bash
aws ecs update-service --cluster neuroscan --service neuroscan --desired-count 3
```

**GCP Cloud Run:**
- Automatically scales based on traffic
- Configure max instances in settings

### Vertical Scaling

Increase instance resources:
- Memory: 2GB → 4GB → 8GB
- CPU: 1 vCPU → 2 vCPU → 4 vCPU

## Cost Optimization

1. **Use spot instances** (AWS) or preemptible VMs (GCP) for non-critical workloads
2. **Enable auto-scaling** to scale down during low traffic
3. **Use CDN** for static assets
4. **Optimize model size** if possible
5. **Monitor usage** and set budget alerts

## Support

For deployment issues:
- Check application logs first
- Review this guide
- Open GitHub issue with logs and configuration (redact secrets)
- Email: yash.22bce8038@vitapstudent.ac.in
