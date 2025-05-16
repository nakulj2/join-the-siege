# Base image with Python 3.11
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/src/classifiers/graphite-scout-419207-9810cb841b6d.json

# Set working directory
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

# Install pip packages first for cache efficiency
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .
COPY src/classifiers/graphite-scout-419207-9810cb841b6d.json /app/src/classifiers/graphite-scout-419207-9810cb841b6d.json

# Authenticate with GCP and download model
# Install GCP CLI and fetch model
RUN apt-get update && apt-get install -y curl gnupg unzip && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-sdk && \
    gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS && \
    mkdir -p model/distilbert/text/checkpoint-28 && \
    gsutil -m cp -r gs://example_audio_bucket_nakulj2/distilbert-text/checkpoint-28/* model/distilbert/text/checkpoint-28/

# Set default run command
CMD ["python", "src/classifiers/bert_text.py"]
