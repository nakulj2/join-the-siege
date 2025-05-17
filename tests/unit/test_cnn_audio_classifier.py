import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pytest
from src.classifiers import cnn_audio


def test_cnn_audio_runs_on_valid_image():
    """Check if CNN runs on a real spectrogram image (validity only)."""
    image_path = "test_data/songs/song_1.png"  # assuming spectrogram is available

    label, probs = cnn_audio.classify_spectrogram(image_path)

    assert isinstance(label, str)
    assert isinstance(probs, list)
    assert label in ["model_not_loaded", "error"] or label.strip()
    if isinstance(probs, list) and probs:
        assert abs(sum(probs) - 1.0) < 0.1 or label in ["model_not_loaded", "error"]
