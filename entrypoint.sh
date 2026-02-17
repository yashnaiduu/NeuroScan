#!/bin/sh
set -e

echo "Starting NeuroScan application..."

# Configuration
MP="${MODEL_PATH:-/app/mobilenet_brain_tumor_classifier.h5}"
MODEL_DIR="$(dirname "$MP")"

# Check if model file exists
if [ ! -f "$MP" ]; then
    echo "Model file not found at: $MP"
    
    # Try to download model if URL is provided
    if [ -n "$MODEL_URL" ]; then
        echo "Downloading model from: $MODEL_URL"
        mkdir -p "$MODEL_DIR"
        
        # Download with retry logic
        MAX_RETRIES=3
        RETRY_COUNT=0
        
        while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
            if curl -fsSL --retry 3 --retry-delay 5 "$MODEL_URL" -o "$MP"; then
                echo "Model downloaded successfully"
                
                # Verify file is not empty
                if [ ! -s "$MP" ]; then
                    echo "Error: Downloaded model file is empty" >&2
                    rm -f "$MP"
                    exit 1
                fi
                
                break
            else
                RETRY_COUNT=$((RETRY_COUNT + 1))
                echo "Download attempt $RETRY_COUNT failed"
                
                if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
                    echo "Error: Failed to download model after $MAX_RETRIES attempts" >&2
                    exit 1
                fi
                
                echo "Retrying in 5 seconds..."
                sleep 5
            fi
        done
    else
        echo "Error: MODEL_URL not set and model file not found" >&2
        echo "Please set MODEL_URL environment variable or provide model file at: $MP" >&2
        exit 1
    fi
fi

echo "Model file ready at: $MP"

# Create necessary directories
mkdir -p Uploads cache

# Configuration
WEB_CONCURRENCY="${WEB_CONCURRENCY:-2}"
WEB_THREADS="${WEB_THREADS:-4}"
PORT="${PORT:-7860}"
TIMEOUT="${GUNICORN_TIMEOUT:-120}"

echo "Starting gunicorn with $WEB_CONCURRENCY workers and $WEB_THREADS threads per worker"
echo "Listening on port: $PORT"

# Start application with gunicorn
exec gunicorn server1:app \
    --workers "$WEB_CONCURRENCY" \
    --worker-class gthread \
    --threads "$WEB_THREADS" \
    --timeout "$TIMEOUT" \
    --bind "0.0.0.0:$PORT" \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --graceful-timeout 30 \
    --keep-alive 5
