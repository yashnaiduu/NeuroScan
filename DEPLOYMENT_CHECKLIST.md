# 🚀 Railway Deployment Checklist

Use this checklist to ensure a smooth deployment to Railway.

## 📝 Pre-Deployment

### Backend Preparation
- [ ] Verify `server1.py` has CORS enabled
- [ ] Confirm Gemini API key is NOT hardcoded in code
- [ ] Check `requirements.txt` includes all dependencies
- [ ] Verify `Dockerfile` uses gunicorn command
- [ ] Decide on model file storage strategy:
  - [ ] Option 1: Include in repository (if < 100MB)
  - [ ] Option 2: Use Git LFS for large files
  - [ ] Option 3: Upload to Google Cloud Storage
  - [ ] Option 4: Upload to AWS S3
  - [ ] Option 5: Use Railway volume

### Frontend Preparation
- [ ] Create `.env.local` file (copy from `.env.local.example`)
- [ ] Choose frontend hosting platform:
  - [ ] Vercel (recommended)
  - [ ] Netlify
  - [ ] Other

## 🔐 Environment Variables

### Railway Backend (Required)
- [ ] `GOOGLE_API_KEY` - Your Gemini API key from Google AI Studio

### Railway Backend (Optional)
- [ ] `PORT` - (Auto-set by Railway, leave as default)
- [ ] `UPLOAD_FOLDER` - Default: `Uploads`
- [ ] `DATASET_PATH` - Default: `./Dataset` (only if using /random endpoint)
- [ ] `MODEL_PATH` - Default: `mobilenet_brain_tumor_classifier.h5`

### Frontend (Required)
- [ ] `NEXT_PUBLIC_API_URL` - Your Railway backend URL

## 🎯 Railway Deployment Steps

1. **Create Railway Account**
   - [ ] Sign up at https://railway.app
   - [ ] Connect GitHub account

2. **Create New Project**
   - [ ] Click "New Project" in Railway
   - [ ] Select "Deploy from GitHub repo"
   - [ ] Choose your NeuroScan repository
   - [ ] Railway detects Dockerfile automatically

3. **Configure Environment Variables**
   - [ ] Open project settings in Railway
   - [ ] Add `GOOGLE_API_KEY` with your API key
   - [ ] Add any optional variables if needed

4. **Wait for Deployment**
   - [ ] Monitor build logs for errors
   - [ ] Wait for deployment to complete
   - [ ] Check for green "Active" status

5. **Get Backend URL**
   - [ ] Copy Railway-provided URL (e.g., `https://neuroscan.railway.app`)
   - [ ] Save this URL for frontend configuration

6. **Test Backend**
   - [ ] Visit backend URL in browser (should show homepage or API response)
   - [ ] Test predict endpoint with curl or Postman:
     ```bash
     curl -X POST https://your-backend.railway.app/predict \
       -F "file=@test-image.jpg"
     ```

## 🎨 Frontend Deployment Steps

### Option 1: Vercel (Recommended)
1. **Setup**
   - [ ] Go to https://vercel.com
   - [ ] Import your GitHub repository
   - [ ] Vercel auto-detects Next.js

2. **Configure**
   - [ ] Add environment variable: `NEXT_PUBLIC_API_URL` = Railway backend URL
   - [ ] Set framework preset to "Next.js"

3. **Deploy**
   - [ ] Click "Deploy"
   - [ ] Wait for deployment to complete
   - [ ] Get Vercel URL for your frontend

### Option 2: Netlify
1. **Setup**
   - [ ] Go to https://netlify.com
   - [ ] Import from GitHub

2. **Build Settings**
   - [ ] Build command: `npm run build`
   - [ ] Publish directory: `out`

3. **Environment Variables**
   - [ ] Add `NEXT_PUBLIC_API_URL` in site settings
   - [ ] Set to your Railway backend URL

4. **Deploy**
   - [ ] Trigger deployment
   - [ ] Wait for completion

## ✅ Post-Deployment Testing

### Backend Tests
- [ ] Homepage loads (`/`)
- [ ] Upload endpoint works (`/predict`)
- [ ] Random image endpoint works (`/random`) - if dataset included
- [ ] Heatmap endpoint works (`/heatmap`)
- [ ] No CORS errors in browser console
- [ ] Gemini API integration works (if key is set)

### Frontend Tests
- [ ] Website loads correctly
- [ ] Upload interface works
- [ ] Image upload to backend succeeds
- [ ] Predictions display correctly
- [ ] Heatmap visualization works
- [ ] No console errors
- [ ] Mobile responsive design works

### Integration Tests
- [ ] Frontend can communicate with backend
- [ ] CORS allows cross-origin requests
- [ ] API responses are correctly formatted
- [ ] Error handling works properly

## 🐛 Troubleshooting

If something doesn't work, check:

### Build Failures
- [ ] Check Railway build logs
- [ ] Verify all files are committed to Git
- [ ] Confirm Dockerfile syntax is correct
- [ ] Check requirements.txt has all dependencies

### Runtime Errors
- [ ] Check Railway runtime logs
- [ ] Verify environment variables are set
- [ ] Confirm model file is accessible
- [ ] Check Gemini API key is valid

### CORS Errors
- [ ] Verify `from flask_cors import CORS` in server1.py
- [ ] Confirm `CORS(app)` is called
- [ ] Check Flask-Cors is in requirements.txt

### Frontend Connection Issues
- [ ] Verify `NEXT_PUBLIC_API_URL` is set correctly
- [ ] Check Railway backend URL is accessible
- [ ] Confirm backend is running (not sleeping)

## 📊 Monitoring

After deployment, monitor:
- [ ] Railway logs for errors
- [ ] API response times
- [ ] Error rates
- [ ] Resource usage (CPU/memory)
- [ ] Gemini API quota usage

## 💡 Optimization Tips

- [ ] Enable Railway metrics
- [ ] Set up error tracking (Sentry, etc.)
- [ ] Monitor API performance
- [ ] Consider upgrading Railway plan if needed
- [ ] Optimize TensorFlow model if predictions are slow
- [ ] Implement caching for static predictions

## 📚 Documentation Reference

- [ ] Read `DEPLOYMENT.md` for detailed instructions
- [ ] Check `CHANGES_SUMMARY.md` for what was changed
- [ ] Review `README.md` for general information
- [ ] Reference `.env.example` for all environment variables

## 🎉 Success Criteria

Your deployment is successful when:
- ✅ Backend is accessible via Railway URL
- ✅ Frontend is accessible via Vercel/Netlify URL
- ✅ Users can upload images and get predictions
- ✅ No CORS errors in browser console
- ✅ Heatmaps generate correctly
- ✅ All tests pass

## 📞 Need Help?

If you're stuck:
1. Review error logs in Railway dashboard
2. Check troubleshooting section in `DEPLOYMENT.md`
3. Verify all checklist items above
4. Open an issue on GitHub with error details
5. Consult Railway documentation: https://docs.railway.app

---

**Ready to deploy?** Start from the top and check off each item! 🚀
