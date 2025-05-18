# Heron Coding Challenge - File Classifier

## Overview

At Heron, we’re using AI to automate document processing workflows in financial services and beyond. Each day, we handle over 100,000 documents that need to be quickly identified and categorised before we can kick off the automations.

This repository provides a basic endpoint for classifying files by their filenames. However, the current classifier has limitations when it comes to handling poorly named files, processing larger volumes, and adapting to new industries effectively.

**Your task**: improve this classifier by adding features and optimisations to handle (1) poorly named files, (2) scaling to new industries, and (3) processing larger volumes of documents.

This is a real-world challenge that allows you to demonstrate your approach to building innovative and scalable AI solutions. We’re excited to see what you come up with! Feel free to take it in any direction you like, but we suggest:


### Part 1: Enhancing the Classifier

- What are the limitations in the current classifier that's stopping it from scaling?

The current classifier depends solely on filenames to classify files. This approach fails when filenames are poorly or incorrectly named. It also lacks the flexibility to support expansion into new industries or handle documents with similar naming patterns but different semantic content.


- How might you extend the classifier with additional technologies, capabilities, or features?

To overcome these limitations, I implemented a two-stream pipeline:
Text-Based Stream: For PDF, PNG, JPEG, and DOC files, I extract textual content using pdfplumber and pytesseract (OCR fallback). This raw text is then classified using a DistilBERT classifier for improved accuracy and generalizability.


Multimodal Stream: For audio and video files (e.g., lectures, podcasts, songs, advertisements), I used Vertex AI’s multimodal API, allowing for semantic understanding across both audio and textual content. This approach ensures extensibility since industries like education, marketing, and entertainment rely on media-rich content which could be in diverse formats. 
This highly integrated setup acts as an extension to the current implementation. My enhancements allow the classifier to work beyond filenames, understanding semantic cues within documents and adapting across formats and industries (e.g., education, marketing, entertainment).



### Part 2: Productionising the Classifier 

- How can you ensure the classifier is robust and reliable in a production environment?

The classifier is fully containerized using Docker and exposed via a Flask API. Endpoints are tested through Postman and validated via unit and functional tests covering:
Text and audio classifiers
File upload and routing
Utility functions for OCR, transcription, and format handling
The solution was designed with modularity in mind—each step (text extraction, model inference, preprocessing) is testable independently. This improves fault isolation and ensures scalability.

- How can you deploy the classifier to make it accessible to other services and users?

The solution can be deployed using any cloud service or internal CI/CD pipeline, making the classifier publicly accessible and horizontally scalable. Authentication and GCP bucket integration have been added for cloud storage support. Additionally, the codebase includes CI-compatible structure (GitHub Actions-ready) and .gitignore hygiene for streamlined deployments.

We encourage you to be creative! Feel free to use any libraries, tools, services, models or frameworks of your choice

### Additional Enhancements

Scalability to New Industries: The classifier is agnostic to domain-specific keywords and can be fine-tuned on new datasets. 
A good use case for this tool is classifying a high volume of diverse documents from a marketing firm as advertisements, contracts, auditions, social media content etc. This tool is flexible in classifying different types of documents not implemented in the current version. Additionally this implementation can easily be refactored to extend this functionality to different text document types such as Excel and Word, and train the classifier to differentiate between content using semantic metadata structures.

Handling Larger Volumes: Efficient text models (DistilBERT) and streaming-based file reading ensure scalability. The modular design enables distributed processing.


Maintainability: The code is logically split into modules for classification, utilities, preprocessing, and API routing. Models are saved on GCP and loaded using joblib (for traditional classifiers).


## Instructions to Run the Project

Since much of the training and test data cannot be stored on GitHub, a zipped version of the code and data is included with this submission.

1. Clone the Repository

git clone <repository_url>
cd join-the-siege

2. Set Up the Environment

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

3. Prepare the Data

From the zipped folder, copy the train_data/ and test_data/ directories into the root of the project.

4. Train the Classifiers

Run the following scripts to train and save the model weights to the model/ directory:

python src/trainers/train_audio_baseline.py

python src/trainers/train_audio_bert.py

python src/trainers/train_audio_cnn.py

python src/trainers/train_text_baseline.py

python src/trainers/train_text_bert.py

5. Compare Model Performance

Text Models:

python src/compare_models_text.py

Audio Models:

Before running the audio comparison, update the GCP settings in src/classifiers/gemini_multimodal.py:

Place your GCP service account key in the src/classifiers/ folder.

Update the following line:

CREDENTIALS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "your-service-key.json")
)

Update the GCP configuration variables:

BUCKET_NAME = "your_bucket_name"

GCP_PROJECT = "your_project_id"

GCP_REGION = "your_region"

Then run:

python src/compare_models_audio.py

6. Run Tests

python -m pytest

7. Dockerize and Launch the API

Build Docker Image

docker build -t heron-classifier-test .

Run the Server

docker run --rm \
  -p 5000:8080 \
  -v "$(pwd)/model:/app/model" \
  -v "$(pwd)/test_data:/app/test_data" \
  -e GCP_CREDS_JSON_BASE64="$CREDS_BASE64" \
  heron-classifier-test

The server will be running at:

http://127.0.0.1:8080

8. Test the API

Using cURL:
curl -X POST http://localhost:5000/classify_text \
  -F "file=@test_data/drivers_license/drivers_license_1.png"
curl -X POST http://localhost:5000/classify_multimodal \
  -F "file=@test_data/ads/ad_1.mp4"

Using Postman:

POST http://localhost:5000/classify_text

POST http://localhost:5000/classify_multimodal

Upload a file using form-data.

9. Customizing Labels

To update audio multimodal labels, modify the LABELS list in src/app.py.
