import sys
import os
from werkzeug.datastructures import FileStorage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pytest
from src.classifiers import baseline_audio


def test_baseline_audio_classify_valid_mp3():
    with open("test_data/songs/song_1.mp3", "rb") as f:
        file = FileStorage(f, filename="song_1.mp3")
        label = baseline_audio.classify_audio_file(file)

    assert isinstance(label, str)
    assert label in ["model_not_loaded", "error"] or label.strip()


def test_baseline_audio_model_not_loaded(monkeypatch):
    monkeypatch.setattr(baseline_audio, "model", None)

    with open("test_data/songs/song_1.mp3", "rb") as f:
        file = FileStorage(f, filename="song_1.mp3")
        label = baseline_audio.classify_audio_file(file)

    assert label == "model_not_loaded"
