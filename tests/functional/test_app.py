import io
import os
import sys
import pytest
from pathlib import Path
from flask import Flask
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.app import app  # import your app object

BASE_DIR = Path(__file__).resolve().parents[2]
SONG_FILE = BASE_DIR / "test_data/songs/song_1.mp3"
PDF_FILE = BASE_DIR / "test_data/invoices/invoice_1.png"

@pytest.fixture
def client():
    app.config['TESTING'] = True
    return app.test_client()

def test_classify_audio(client):
    assert os.path.exists(SONG_FILE), f"Missing: {SONG_FILE}"
    with open(SONG_FILE, 'rb') as f:
        data = {'file': (io.BytesIO(f.read()), 'song_1.mp3')}
        response = client.post('/classify_audio', data=data, content_type='multipart/form-data')

    assert response.status_code == 200
    json_data = response.get_json()
    assert 'file_class' in json_data
    assert isinstance(json_data['probabilities'], list)

def test_classify_file(client):
    assert os.path.exists(PDF_FILE), f"Missing: {PDF_FILE}"
    with open(PDF_FILE, 'rb') as f:
        data = {'file': (io.BytesIO(f.read()), 'invoice_1.png')}
        response = client.post('/classify_file', data=data, content_type='multipart/form-data')

    assert response.status_code == 200
    json_data = response.get_json()
    assert 'file_class' in json_data
    assert isinstance(json_data['probabilities'], list)
