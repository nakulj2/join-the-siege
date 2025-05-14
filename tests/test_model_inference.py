import joblib
import os

def test_model_inference():
    model_path = "model/logistic_regression.pkl"
    assert os.path.exists(model_path), "Model not found"

    model = joblib.load(model_path)
    test_input = "Account Number: 12345678\nTransaction Date: 01/01/2023\nBalance: $10,000"
    pred = model.predict([test_input])[0]

    assert pred in ["invoices", "bank_statements", "drivers_license", "dummy"], f"Unexpected class: {pred}"
