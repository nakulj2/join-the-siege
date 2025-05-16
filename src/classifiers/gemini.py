from google.cloud import storage
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(
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

# Run it
upload_to_gcs(
    bucket_name="example_audio_bucket_nakulj2",
    source_file_path="test_data/lectures/lecture_1.mp3",
    destination_blob_name="lecture_1.mp3"
)

from vertexai.generative_models import GenerativeModel, Part
from vertexai import init

# Initialize Vertex AI (use your actual GCP project ID and region)
init(project="graphite-scout-419207", location="us-central1")  # or the region where Gemini 1.5 is enabled

# Load the Gemini model
model = GenerativeModel("gemini-2.0-flash-lite-001")

# Call Gemini with the GCS audio file
response = model.generate_content([
    Part.from_uri("gs://example_audio_bucket_nakulj2/lecture_1.mp3", mime_type="audio/mpeg"),
    "Classify this audio as either a lecture or an ad. Explain your reasoning."
])

# Print the result
print(response.text)