from werkzeug.datastructures import FileStorage
from src.utils.extract_text import extract_text
import joblib
import os

# Load model (only once at module level)
MODEL_PATH = "model/logistic_regression.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def classify_file(file: FileStorage):
    if not model:
        return "model_not_loaded"

    try:
        text = extract_text(file)
        prediction = model.predict([text])[0]
        return prediction
    except Exception as e:
        print(f"[ERROR] Failed to classify file: {e}")
        return "error"
