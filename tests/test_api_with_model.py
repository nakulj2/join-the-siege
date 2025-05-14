import joblib
import os
from src.app import app
from src.utils.extract_text import extract_text

def test_file_classification_with_real_pdf():
    model_path = "model/text/logistic_regression.pkl"
    assert os.path.exists(model_path), "Trained model not found"
    model = joblib.load(model_path)

    file_path = "data/bank_statements/bank_statement_1.pdf"
    with open(file_path, "rb") as f:
        f.filename = os.path.basename(file_path)
        expected_text = extract_text(f)
        f.seek(0)  # Reset for upload
        expected_prediction = model.predict([expected_text])[0]

        client = app.test_client()
        response = client.post(
            "/classify_file",
            data={"file": (f, f.filename)},
            content_type="multipart/form-data"
        )

    assert response.status_code == 200
    result = response.get_json()
    assert result["file_class"] == expected_prediction, f"Expected: {expected_prediction}, Got: {result['file_class']}"