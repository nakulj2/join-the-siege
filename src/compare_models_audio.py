import os
import time
import joblib
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from utils.transcribe_audio import transcribe_audio
from utils.audio_features import extract_librosa_features

TEST_DIR = "test_data"
LABELS = sorted([
    name for name in os.listdir(TEST_DIR)
    if os.path.isdir(os.path.join(TEST_DIR, name)) and
       any(f.endswith(".mp3") for f in os.listdir(os.path.join(TEST_DIR, name)))
])

# ----------------------------
# Load Audio Test Data
# ----------------------------
def load_audio_test_dataset():
    texts, labels = [], []
    for label in LABELS:
        folder = os.path.join(TEST_DIR, label)
        for fname in tqdm(os.listdir(folder), desc=f"Loading {label}"):
            if not fname.endswith(".mp3"):
                continue
            path = os.path.join(folder, fname)
            try:
                text = transcribe_audio(path)
                features = extract_librosa_features(path)
                combined = f"{text} | features: {' '.join(map(str, features))}"
                texts.append(combined)
                labels.append(label)
            except Exception as e:
                print(f"[WARN] Skipped {path}: {e}")
    return texts, labels

# ----------------------------
# Confusion Matrix
# ----------------------------
def plot_confusion(y_true, y_pred, model_name, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=20)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.show()

# ----------------------------
# Confidence Visualizer
# ----------------------------
def plot_confidences(all_probs, labels, model_names, doc_ids):
    num_docs = len(doc_ids)
    num_classes = len(labels)
    width = 0.2
    x = range(num_classes)

    for i in range(num_docs):
        fig, ax = plt.subplots()
        for j, model_name in enumerate(model_names):
            probs = all_probs[j][i]
            ax.bar([p + width * j for p in x], probs, width=width, label=model_name)

        ax.set_xticks([p + width for p in x])
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylim([0, 1.1])
        ax.set_ylabel("Confidence")
        ax.set_title(f"Confidence for Audio Sample {i+1}")
        ax.legend()
        plt.tight_layout()
        plt.show()

# ----------------------------
# Main
# ----------------------------
def main():
    X_test, y_test = load_audio_test_dataset()

    # Load models
    logreg = joblib.load("model/baseline/audio/logistic_regression.pkl")
    nb = joblib.load("model/baseline/audio/naive_bayes.pkl")
    tokenizer = DistilBertTokenizerFast.from_pretrained("model/distilbert/audio")
    model = DistilBertForSequenceClassification.from_pretrained("model/distilbert/audio")

    model.eval()
    all_probs = []
    model_names = ["LogReg", "NaiveBayes", "DistilBERT"]

    # Logistic Regression
    start = time.time()
    log_probs = logreg.predict_proba(X_test)
    log_preds = logreg.predict(X_test)
    print(f"\n📊 Logistic Regression (Time: {round(time.time() - start, 2)}s):")
    print(classification_report(y_test, log_preds))
    plot_confusion(y_test, log_preds, "Logistic Regression", LABELS)
    all_probs.append(log_probs.tolist())

    # Naive Bayes
    start = time.time()
    nb_probs = nb.predict_proba(X_test)
    nb_preds = nb.predict(X_test)
    print(f"\n📊 Naive Bayes (Time: {round(time.time() - start, 2)}s):")
    print(classification_report(y_test, nb_preds))
    plot_confusion(y_test, nb_preds, "Naive Bayes", LABELS)
    all_probs.append(nb_probs.tolist())

    # DistilBERT
    bert_preds = []
    bert_probs = []

    start = time.time()
    with torch.no_grad():
        for text in X_test:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
            pred = torch.argmax(torch.tensor(probs)).item()
            bert_preds.append(LABELS[pred])
            bert_probs.append(probs)

    print(f"\n📊 DistilBERT (Time: {round(time.time() - start, 2)}s):")
    print(classification_report(y_test, bert_preds))
    plot_confusion(y_test, bert_preds, "DistilBERT", LABELS)
    all_probs.append(bert_probs)

    # Visualize Confidence Comparisons
    plot_confidences(all_probs, LABELS, model_names, list(range(len(X_test))))

if __name__ == "__main__":
    main()
