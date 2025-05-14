from src.app import app
import pytest
from pathlib import Path

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c

def test_audio_success(client):
    path = Path("data/songs/song_1.mp3")
    with path.open("rb") as f:
        response = client.post("/classify_audio", data={"file": (f, path.name)}, content_type="multipart/form-data")
        assert response.status_code == 200
        assert "file_class" in response.get_json()

def test_audio_wrong_file_type(client):
    path = Path("data/invoices/invoice_1.pdf")
    with path.open("rb") as f:
        response = client.post("/classify_audio", data={"file": (f, path.name)}, content_type="multipart/form-data")
        assert response.status_code == 400
