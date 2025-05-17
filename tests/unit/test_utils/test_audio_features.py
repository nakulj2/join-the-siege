import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.utils.audio_features import extract_librosa_features

def test_extract_librosa_features_returns_tags():
    audio_path = "test_data/songs/song_1.mp3"
    tags = extract_librosa_features(audio_path)
    assert isinstance(tags, str)
    assert len(tags.split(",")) >= 3  # at least tempo, consistency, noise
