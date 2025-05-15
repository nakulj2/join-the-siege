# src/classifiers/baseline_text.py

import os
import joblib
from werkzeug.datastructures import FileStorage
from src.utils.extract_text import extract_text

MODEL_PATH = "model/baseline/text/naive_bayes.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"[ERROR] Failed to load text model: {e}")
    model = None

def classify_text_file(file: FileStorage):
    if not model:
        return "model_not_loaded"
    try:
        text = extract_text(file)
        return model.predict([text])[0]
    except Exception as e:
        print(f"[ERROR] Text classification failed: {e}")
        return "error"
