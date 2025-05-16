import sys
import os
import pytest

# Add project root to path for importing src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.app import app

@pytest.fixture
def client():
    return app.test_client()

def test_no_file_uploaded_to_text(client):
    response = client.post("/classify_file", data={})
    assert response.status_code == 400
    assert response.json["error"] == "No file part"

def test_no_file_uploaded_to_audio(client):
    response = client.post("/classify_audio", data={})
    assert response.status_code == 400
    assert response.json["error"] == "No file part"

def test_unsupported_file_type_text(client):
    data = {"file": (open(__file__, "rb"), "test.xyz")}
    response = client.post("/classify_file", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.json["error"] == "File not supported for text classification"

def test_unsupported_file_type_audio(client):
    data = {"file": (open(__file__, "rb"), "test.xyz")}
    response = client.post("/classify_audio", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.json["error"] == "File not supported for audio classification"
