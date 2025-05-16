# Tests /classify_audio endpoint

import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.app import app
import pytest
from pathlib import Path

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c

def test_audio_success(client):
    path = Path("test_data/songs/song_1.mp3")
    with path.open("rb") as f:
        response = client.post("/classify_audio", data={"file": path}, content_type="multipart/form-data")
        assert response.status_code == 200
        assert "file_class" in response.get_json()

def test_audio_wrong_file_type(client):
    path = Path("test_data/invoices/invoice_1.png")
    with path.open("rb") as f:
        response = client.post("/classify_audio", data={"file": (f, path.name)}, content_type="multipart/form-data")
        assert response.status_code == 400
