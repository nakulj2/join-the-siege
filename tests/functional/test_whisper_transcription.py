import os
from pathlib import Path
from src.utils.transcribe_audio import transcribe_audio

BASE_DIR = Path(__file__).resolve().parents[2]
PODCAST_FILE = BASE_DIR / "data/podcasts/podcast_1.mp3"

def test_whisper_transcription_output():
    assert os.path.exists(PODCAST_FILE), f"Missing: {PODCAST_FILE}"
    
    text = transcribe_audio(str(PODCAST_FILE))
    
    assert isinstance(text, str), "Transcription result is not a string"
    assert len(text.strip()) > 10, f"Transcription too short: {text[:50]}"
