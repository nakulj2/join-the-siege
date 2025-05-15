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

def test_song_vs_podcast_audio_features():
    song_path = BASE_DIR / "test_data/songs/song_1.mp3"
    podcast_path = BASE_DIR / "test_data/podcasts/podcast_1.mp3"
    assert song_path.exists() and podcast_path.exists()

    song_features = extract_librosa_features(str(song_path))
    podcast_features = extract_librosa_features(str(podcast_path))

    song_tempo = song_features[0]
    podcast_tempo = podcast_features[0]

    # Songs should generally have higher tempo than speech-based content
    assert song_tempo > podcast_tempo, f"Expected song tempo > podcast tempo, got {song_tempo} vs {podcast_tempo}"

    # You can optionally also test that features are sufficiently "different"
    diff = abs(song_tempo - podcast_tempo)
    assert diff > 10, f"Tempo difference too small: {diff}"
