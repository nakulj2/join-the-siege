# Base image with Python 3.11
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_APPLICATION_CREDENTIALS=gcp-creds.json

# Create and set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gnupg \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies with retries and longer timeouts
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 --retries=10 -r requirements.txt

# Default command (you can override it in docker run)
CMD ["python", "src/app.py"]
