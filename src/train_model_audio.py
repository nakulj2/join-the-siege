# train_audio_model.py

import os
import joblib
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.utils.transcribe_audio import transcribe_audio
from src.utils.audio_features import extract_librosa_features
from collections import Counter
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

DATA_DIR = "data"
MODEL_DIR = "model/audio"

def load_audio_dataset():
    texts, labels = [], []
    allowed_ext = ".mp3"

    for label in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder_path):
            continue

        # Skip folders with no .mp3 files
        relevant_files = [f for f in os.listdir(folder_path) if f.lower().endswith(allowed_ext)]
        if not relevant_files:
            continue

        for fname in tqdm(relevant_files, desc=f"Loading {label}"):
            fpath = os.path.join(folder_path, fname)
            try:
                text = transcribe_audio(fpath)
                features = extract_librosa_features(fpath)
                combined = f"{text} | features: {' '.join(map(str, features))}"
                texts.append(combined)
                labels.append(label)
            except Exception as e:
                print(f"[WARN] Skipped {fpath}: {e}")
    return texts, labels

def train_and_eval(name, model_cls, X_train, X_test, y_train, y_test):
    print(f"🔁 Training {name}...")
    model = make_pipeline(TfidfVectorizer(), model_cls())
    model.fit(X_train, y_train)

    print(f"📊 {name} Report:")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{name.lower().replace(' ', '_')}.pkl")
    joblib.dump(model, path)
    print(f"✅ Saved {name} to {path}\n")
    return model

if __name__ == "__main__":
    X, y = load_audio_dataset()
    label_counts = Counter(y)

    if any(count <= 2 for count in label_counts.values()):
        print("⚠️ Not enough data per class — using all data for train/test.")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )

    print("Train labels:", Counter(y_train))
    print("Test labels:", Counter(y_test))

    train_and_eval("Logistic Regression", LogisticRegression, X_train, X_test, y_train, y_test)
    train_and_eval("Naive Bayes", MultinomialNB, X_train, X_test, y_train, y_test)
