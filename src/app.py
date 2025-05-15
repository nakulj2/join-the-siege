from flask import Flask, request, jsonify
from src.classifiers.bert_text import classify as classify_text
from src.classifiers.bert_audio import classify as classify_audio

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

    try:
        file.filename = file.filename  # Ensure `filename` is set for extract_text
        from src.utils.extract_text import extract_text
        text = extract_text(file)
        label, probs = classify_text(text)
    except Exception as e:
        return jsonify({"error": f"Text extraction failed: {str(e)}"}), 500

    if label in ["model_not_loaded", "error"]:
        return jsonify({"error": label}), 500

    return jsonify({
        "file_class": label,
        "probabilities": probs
    })

@app.route('/classify_audio', methods=['POST'])
def classify_audio_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if extension(file.filename) not in ALLOWED_AUDIO:
        return jsonify({"error": "File not supported for audio classification"}), 400

    label, probs = classify_audio(file)
    if label in ["model_not_loaded", "error"]:
        return jsonify({"error": label}), 500

    return jsonify({
        "file_class": label,
        "probabilities": probs
    })

if __name__ == "__main__":
    app.run(debug=True)
