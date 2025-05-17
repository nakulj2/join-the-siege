# Base image with Python 3.11
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_APPLICATION_CREDENTIALS=gcp-creds.json

# Create and set working directory
WORKDIR /app

# Install system dependencies including Tesseract
RUN apt-get update && apt-get install -y \
    gnupg \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    git \
    curl \
    unzip \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Set PYTHONPATH so src/ is importable
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 --retries=10 -r requirements.txt

# Default command
CMD ["python", "src/app.py"]
