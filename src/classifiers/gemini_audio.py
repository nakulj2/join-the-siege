import os
import sys
from google.cloud import storage
from vertexai.generative_models import GenerativeModel, Part
from vertexai import init
import json

# Configuration
BUCKET_NAME = "example_audio_bucket_nakulj2"
GCP_PROJECT = "graphite-scout-419207"
GCP_REGION = "us-central1"
MODEL_NAME = "gemini-2.0-flash-lite-001"
CREDENTIALS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "graphite-scout-419207-9810cb841b6d.json")
)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH
LABELS = ["lecture", "ad", "song", "podcast"]

# Init Gemini
init(project=GCP_PROJECT, location=GCP_REGION)
model = GenerativeModel(MODEL_NAME)

def upload_to_gcs(bucket_name, source_file_path, destination_blob_name):
    if not os.path.exists(source_file_path):
        raise FileNotFoundError(f"❌ File not found: {source_file_path}")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    if not bucket.exists():
        raise ValueError(f"❌ Bucket {bucket_name} does not exist.")

    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path)
    print(f"✅ Uploaded: gs://{bucket_name}/{destination_blob_name}")
    return blob

import json

LABELS = ["lecture", "ad"]

def classify_audio(file_path: str, LABELS: list):
    file_name = os.path.basename(file_path)
    blob = upload_to_gcs(BUCKET_NAME, file_path, file_name)

    try:
        prompt = f"""
You are a classifier. You must choose only one label from the list: {LABELS}.
Analyze the given file and respond ONLY in this exact JSON format:

{{"label": "<label from list>"}}
"""
        response = model.generate_content([
            Part.from_uri(f"gs://{BUCKET_NAME}/{file_name}", mime_type="audio/mpeg"),
            prompt.strip()
        ])
        result_text = response.text.strip()
        print(f"🎧 Gemini Raw Response:\n{result_text}")

        # Try to parse the JSON
        try:
            result_json = json.loads(result_text)
            label = result_json.get("label")
        except json.JSONDecodeError:
            print("⚠️ Failed to parse JSON. Returning raw output.")
            label = "invalid_format"

    finally:
        blob.delete()
        print(f"🗑️ Deleted: gs://{BUCKET_NAME}/{file_name}")
    
    return label


def main(filepath):
    result = classify_audio(filepath, LABELS)
    print(f"\n🔍 Final Result:\n{result}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gemini_audio.py <path_to_audio_file>")
    else:
        main(sys.argv[1])
