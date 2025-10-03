# Alternative Dockerfile with minimal dependencies
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Update package list and install minimal dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p Uploads

# Expose port
EXPOSE 5050

# Run the application with gunicorn for production
# Railway will set the PORT environment variable
CMD gunicorn --bind 0.0.0.0:${PORT:-5050} --timeout 120 server1:app 