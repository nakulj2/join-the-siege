import os
import joblib
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils.extract_text import extract_text
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from utils.transcribe_audio import transcribe_audio
from utils.audio_features import extract_librosa_features
from collections import Counter

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


DATA_DIR = "data"
MODEL_DIR = "model"

def load_dataset():
    texts, labels = [], []

    for label in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder_path):
            continue

        for fname in tqdm(os.listdir(folder_path), desc=f"Loading {label}"):
            fpath = os.path.join(folder_path, fname)
            try:
                if fname.endswith(".mp3"):
                    # New: handle audio files
                    text = transcribe_audio(fpath)
                    features = extract_librosa_features(fpath)
                    combined_text = f"{text} | features: {' '.join(map(str, features))}"
                    texts.append(combined_text)
                    labels.append(label)
                else:
                    # Existing: handle PDFs and images
                    with open(fpath, "rb") as f:
                        f.filename = fname
                        text = extract_text(f)
                        if text and text.strip():
                            texts.append(text)
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

    # Save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{name.lower().replace(' ', '_')}.pkl")
    joblib.dump(model, path)
    print(f"✅ Saved {name} to {path}\n")
    return model

if __name__ == "__main__":
    X, y = load_dataset()
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
    train_and_eval("Multinomial Naive Bayes", MultinomialNB, X_train, X_test, y_train, y_test)
