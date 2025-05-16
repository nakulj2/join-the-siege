from google.cloud import storage
from vertexai.generative_models import GenerativeModel, Part
from vertexai import init
import os

# Set credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "graphite-scout-419207-9810cb841b6d.json")
)

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
    return blob  # Return blob object so we can delete it later

# === MAIN LOGIC ===

bucket_name = "example_audio_bucket_nakulj2"
source_file_path = "train_data/podcasts/podcast_4.mp3"
destination_blob_name = "podcast_4.mp3"

# Upload
blob = upload_to_gcs(bucket_name, source_file_path, destination_blob_name)

# Initialize Vertex AI
init(project="graphite-scout-419207", location="us-central1")

# Load the Gemini model
model = GenerativeModel("gemini-2.0-flash-lite-001")

# Generate response
response = model.generate_content([
    Part.from_uri(f"gs://{bucket_name}/{destination_blob_name}", mime_type="audio/mpeg"),
    "Classify this audio as either a lecture or an ad. Explain your reasoning."
])

# Output result
print(response.text)

# Delete the uploaded file
blob.delete()
print(f"🗑️ Deleted: gs://{bucket_name}/{destination_blob_name}")