from werkzeug.datastructures import FileStorage
from src.utils.extract_text import extract_text
import joblib
import os

MODEL_PATH = "model/text/logistic_regression.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def classify_text_file(file: FileStorage):
    if not model:
        return "model_not_loaded"
    try:
        text = extract_text(file)
        return model.predict([text])[0]
    except Exception as e:
        print(f"[ERROR] Text classification failed: {e}")
        return "error"