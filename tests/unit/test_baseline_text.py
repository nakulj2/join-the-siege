# 	Direct model load + predict
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from pathlib import Path
import joblib
from src.utils.extract_text import extract_text

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "model/baseline/text/logistic_regression.pkl"
BANK_FILE = BASE_DIR / "test_data/bank_statements/bank_statement_1.png"
LICENSE_FILE = BASE_DIR / "test_data/drivers_license/drivers_license_3.pdf"

model = joblib.load(MODEL_PATH)

def load_and_predict(file_path, model):
    with open(file_path, "rb") as f:
        f.filename = os.path.basename(file_path)
        text = extract_text(f)
        return model.predict([text])[0]

def test_valid_bank_statement_classification():
    assert os.path.exists(BANK_FILE), f"Missing file: {BANK_FILE}"
    prediction = load_and_predict(BANK_FILE, model)
    assert prediction == "bank_statements", f"Expected 'bank_statements', got '{prediction}'"

def test_valid_license_classification():
    assert os.path.exists(LICENSE_FILE), f"Missing file: {LICENSE_FILE}"
    prediction = load_and_predict(LICENSE_FILE, model)
    assert prediction == "drivers_license", f"Expected 'drivers_license', got '{prediction}'"

def test_batch_text_predictions():
    paths = [BANK_FILE, LICENSE_FILE]
    expected = ["bank_statements", "drivers_license"]

    predictions = [load_and_predict(p, model) for p in paths]
    assert predictions == expected, f"Expected {expected}, got {predictions}"