import sys
import os
import pytest
from io import BytesIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.app import app

client = app.test_client()

# Text tests
def test_classify_text_with_pdf():
    with open("test_data/invoices/invoice_2.pdf", "rb") as f:
        data = {"file": (BytesIO(f.read()), "invoice_2.pdf")}
        response = client.post("/classify_text", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    res = response.get_json()
    assert "file_class" in res
    assert "probabilities" in res
    assert isinstance(res["probabilities"], list)


def test_classify_text_with_png():
    with open("test_data/drivers_license/drivers_license_2.png", "rb") as f:
        data = {"file": (BytesIO(f.read()), "drivers_license_2.png")}
        response = client.post("/classify_text", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    res = response.get_json()
    assert "file_class" in res
    assert "probabilities" in res


def test_classify_text_with_unsupported_file_type():
    data = {"file": (BytesIO(b"some content"), "note.txt")}
    response = client.post("/classify_text", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["error"] == "File not supported for text classification"


def test_classify_text_with_no_file():
    response = client.post("/classify_text", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["error"] == "No file part"


def test_classify_text_with_empty_file():
    data = {"file": (BytesIO(b""), "empty.pdf")}
    response = client.post("/classify_text", data=data, content_type="multipart/form-data")
    assert response.status_code in [200, 500]  # text extractor may fail gracefully
    res = response.get_json()
    assert "error" in res or "file_class" in res

# Multimodal tests
def test_classify_multimodal_with_mp3():
    with open("test_data/podcasts/podcast_1.mp3", "rb") as f:
        data = {"file": (BytesIO(f.read()), "podcast_1.mp3")}
        response = client.post("/classify_multimodal", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    res = response.get_json()
    assert "file_class" in res
    assert isinstance(res["file_class"], str)


def test_classify_multimodal_with_mp4():
    with open("test_data/ads/ad_1.mp4", "rb") as f:
        data = {"file": (BytesIO(f.read()), "ad_1.mp4")}
        response = client.post("/classify_multimodal", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    res = response.get_json()
    assert "file_class" in res
    assert isinstance(res["file_class"], str)


def test_classify_multimodal_with_unsupported_type():
    data = {"file": (BytesIO(b"some audio"), "lecture.wav")}
    response = client.post("/classify_multimodal", data=data, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.get_json()["error"] == "File not supported for audio classification"


def test_classify_multimodal_with_no_file():
    response = client.post("/classify_multimodal", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["error"] == "No file part"


def test_classify_multimodal_with_empty_file():
    data = {"file": (BytesIO(b""), "empty.mp3")}
    response = client.post("/classify_multimodal", data=data, content_type="multipart/form-data")

    assert response.status_code in [200, 500]
    res = response.get_json()
    assert "error" in res or "file_class" in res

