# src/classifiers/baseline_audio.py

import os
import joblib
from werkzeug.datastructures import FileStorage
from src.utils.transcribe_audio import transcribe_audio
from src.utils.audio_features import extract_librosa_features

MODEL_PATH = "model/baseline/audio/naive_bayes.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"[ERROR] Failed to load audio model: {e}")
    model = None

def classify_audio_file(file: FileStorage):
    if not model:
        return "model_not_loaded"
    try:
        temp_path = f"/tmp/{file.filename}"
        file.save(temp_path)

        transcript = transcribe_audio(temp_path)
        features = extract_librosa_features(temp_path)
        combined = f"{transcript} | features: {' '.join(map(str, features))}"
        return model.predict([combined])[0]
    except Exception as e:
        print(f"[ERROR] Audio classification failed: {e}")
        return "error"
