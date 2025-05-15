# Direct transcribe_audio, extract_librosa_features + model

import joblib
import os
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.transcribe_audio import transcribe_audio
from src.utils.audio_features import extract_librosa_features
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
SONG_FILE = BASE_DIR / "test_data/songs/song_1.mp3"
PODCAST_FILE = BASE_DIR / "test_data/podcasts/podcast_1.mp3"
MODEL_PATH = "model/baseline/audio/naive_bayes.pkl"

def load_and_predict(path, model):
    text = transcribe_audio(str(path))
    features = extract_librosa_features(str(path))
    input_text = f"{text} | features: {' '.join(map(str, features))}"
    return model.predict([input_text])[0]

def test_valid_song_classification():
    assert os.path.exists(SONG_FILE), f"Missing: {SONG_FILE}"
    model = joblib.load(MODEL_PATH)

    prediction = load_and_predict(SONG_FILE, model)
    assert prediction == "songs", f"Expected 'songs', got '{prediction}'"

def test_valid_podcast_classification():
    assert os.path.exists(PODCAST_FILE), f"Missing: {PODCAST_FILE}"
    model = joblib.load(MODEL_PATH)

    prediction = load_and_predict(PODCAST_FILE, model)
    assert prediction == "podcasts", f"Expected 'podcasts', got '{prediction}'"

def test_batch_audio_predictions():
    assert os.path.exists(SONG_FILE) and os.path.exists(PODCAST_FILE)
    model = joblib.load(MODEL_PATH)

    paths = [SONG_FILE, PODCAST_FILE]
    expected = ["songs", "podcasts"]

    predictions = []
    for path in paths:
        predictions.append(load_and_predict(path, model))

    assert predictions == expected, f"Expected {expected}, got {predictions}"