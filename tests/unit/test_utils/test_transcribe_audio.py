import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.utils.transcribe_audio import transcribe_audio

def test_transcribe_audio_returns_text():
    audio_path = "test_data/songs/song_1.mp3"
    text = transcribe_audio(audio_path)
    assert isinstance(text, str)
    assert len(text.strip()) > 0
