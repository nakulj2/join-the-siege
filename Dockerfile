# Base image with Python 3.12
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/src/classifiers/graphite-scout-419207-9810cb841b6d.json

# Create and set working directory
WORKDIR /app

# Install system packages needed by your project
RUN apt-get update && apt-get install -y \
    gnupg \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .
COPY src/classifiers/graphite-scout-419207-9810cb841b6d.json /app/src/classifiers/graphite-scout-419207-9810cb841b6d.json

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Default command (you can override it in docker run)
CMD ["python", "src/classifiers/bert_text.py"]
