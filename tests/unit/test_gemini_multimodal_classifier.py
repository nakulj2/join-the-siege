import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pytest
from src.classifiers.gemini_multimodal import classify_audio, LABELS


def test_gemini_rejects_unsupported_format():
    """Should raise ValueError for non-mp3/mp4 files."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        classify_audio("test_data/songs/song_1.wav", LABELS)


@pytest.mark.parametrize("file_path,expected", [
    ("test_data/songs/song_1.mp3", "song"),
    ("test_data/songs/song_2.mp3", "song"),
])
def test_gemini_classifies_valid_song_mp3(file_path, expected):
    label = classify_audio(file_path, LABELS)
    print(f"🎵 {file_path} → {label}")
    assert isinstance(label, str)
    assert label in LABELS, f"Invalid label {label}"
    assert label == expected, f"Expected {expected}, got {label}"


@pytest.mark.parametrize("file_path,expected", [
    ("test_data/podcasts/podcast_1.mp3", "podcast"),
    ("test_data/podcasts/podcast_2.mp3", "podcast"),
])
def test_gemini_classifies_valid_podcast_mp3(file_path, expected):
    label = classify_audio(file_path, LABELS)
    print(f"🎧 {file_path} → {label}")
    assert isinstance(label, str)
    assert label in LABELS, f"Invalid label {label}"
    assert label == expected, f"Expected {expected}, got {label}"


@pytest.mark.parametrize("file_path,expected", [
    ("test_data/ads/ad_1.mp4", "ad")
])
def test_gemini_classifies_valid_ad_mp4(file_path, expected):
    label = classify_audio(file_path, LABELS)
    print(f"📺 {file_path} → {label}")
    assert isinstance(label, str)
    assert label in LABELS, f"Invalid label {label}"
    assert label == expected, f"Expected {expected}, got {label}"
