import joblib
import os
from src.utils.transcribe_audio import transcribe_audio
from src.utils.audio_features import extract_librosa_features
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
SONG_FILE = BASE_DIR / "data/songs/song_1.mp3"
PODCAST_FILE = BASE_DIR / "data/podcasts/podcast_1.mp3"
MODEL_PATH = "model/audio/logistic_regression.pkl"

def test_song_classification():
    assert os.path.exists(MODEL_PATH), "Trained model not found"
    model = joblib.load(MODEL_PATH)

    text = transcribe_audio(str(SONG_FILE))
    features = extract_librosa_features(SONG_FILE)
    input_text = f"{text} | features: {' '.join(map(str, features))}"

    prediction = model.predict([input_text])[0]
    assert prediction == "songs", f"Expected 'songs', got '{prediction}'"

def test_podcast_classification():
    assert os.path.exists(MODEL_PATH), "Trained model not found"
    model = joblib.load(MODEL_PATH)

    text = transcribe_audio(str(PODCAST_FILE))
    features = extract_librosa_features(PODCAST_FILE)
    input_text = f"{text} | features: {' '.join(map(str, features))}"

    prediction = model.predict([input_text])[0]
    assert prediction == "podcasts", f"Expected 'podcasts', got '{prediction}'"
