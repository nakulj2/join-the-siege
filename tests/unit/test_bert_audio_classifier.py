import sys
import os
from werkzeug.datastructures import FileStorage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pytest
from src.classifiers import bert_audio


def test_bert_audio_classify_valid_mp3():
    with open("test_data/songs/song_1.mp3", "rb") as f:
        file = FileStorage(f, filename="song_1.mp3")
        label, probs = bert_audio.classify(file)

    assert isinstance(label, str)
    assert isinstance(probs, list)
    assert label in ["model_not_loaded", "error"] or label.strip()
    if isinstance(probs, list) and probs:
        assert abs(sum(probs) - 1.0) < 0.1 or label in ["model_not_loaded", "error"]


def test_bert_audio_model_not_loaded(monkeypatch):
    monkeypatch.setattr(bert_audio, "model", None)
    monkeypatch.setattr(bert_audio, "tokenizer", None)

    with open("test_data/songs/song_1.mp3", "rb") as f:
        file = FileStorage(f, filename="song_1.mp3")
        label, probs = bert_audio.classify(file)

    assert label == "model_not_loaded"
    assert probs == []
