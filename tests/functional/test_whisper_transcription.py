from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.transcribe_audio import transcribe_audio
from src.utils.audio_features import extract_librosa_features

BASE_DIR = Path(__file__).resolve().parents[2]
PODCAST_FILE = BASE_DIR / "test_data/podcasts/podcast_1.mp3"

def test_whisper_transcription_output():
    assert os.path.exists(PODCAST_FILE), f"Missing: {PODCAST_FILE}"
    
    text = transcribe_audio(str(PODCAST_FILE))
    
    assert isinstance(text, str), "Transcription result is not a string"
    assert len(text.strip()) > 10, f"Transcription too short: {text[:50]}"

