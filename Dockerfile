# ✅ Use Google’s base image with gcloud + gsutil preinstalled
FROM gcr.io/google.com/cloudsdktool/cloud-sdk:slim

# Install Python 3.11 manually
RUN apt-get update && apt-get install -y python3.11 python3-pip ffmpeg libsndfile1 libgl1 git && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/src/classifiers/graphite-scout-419207-9810cb841b6d.json

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .
COPY src/classifiers/graphite-scout-419207-9810cb841b6d.json /app/src/classifiers/graphite-scout-419207-9810cb841b6d.json

# ✅ Authenticate and download model from GCS
RUN gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS && \
    mkdir -p model/distilbert/text/checkpoint-28 && \
    gsutil -m cp -r gs://example_audio_bucket_nakulj2/distilbert-text/checkpoint-28/* model/distilbert/text/checkpoint-28/

# Default command
CMD ["python", "src/classifiers/bert_text.py"]
