from werkzeug.datastructures import FileStorage
from src.utils.transcribe_audio import transcribe_audio
from src.utils.audio_features import extract_librosa_features
import joblib
import os

MODEL_PATH = "model/audio/logistic_regression.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def classify_audio_file(file: FileStorage):
    if not model:
        return "model_not_loaded"
    try:
        path = f"/tmp/{file.filename}"
        file.save(path)

        text = transcribe_audio(path)
        features = extract_librosa_features(path)
        input_text = f"{text} | features: {' '.join(map(str, features))}"
        return model.predict([input_text])[0]
    except Exception as e:
        print(f"[ERROR] Audio classification failed: {e}")
        return "error"
