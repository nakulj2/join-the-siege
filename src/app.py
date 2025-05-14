from flask import Flask, request, jsonify
from src.classifier_text import classify_text_file
from src.classifier_audio import classify_audio_file

app = Flask(__name__)

ALLOWED_TEXT = {'pdf', 'png', 'jpg', 'jpeg'}
ALLOWED_AUDIO = {'mp3'}

def extension(filename):
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

@app.route('/classify_file', methods=['POST'])
def classify_file_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if extension(file.filename) not in ALLOWED_TEXT:
        return jsonify({"error": "File not supported for text classification"}), 400

    label = classify_text_file(file)
    return jsonify({"file_class": label})

@app.route('/classify_audio', methods=['POST'])
def classify_audio_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if extension(file.filename) not in ALLOWED_AUDIO:
        return jsonify({"error": "File not supported for audio classification"}), 400

    label = classify_audio_file(file)
    return jsonify({"file_class": label})
