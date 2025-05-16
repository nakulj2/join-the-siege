import os
import time
import io
import torch
import joblib
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from utils.transcribe_audio import transcribe_audio
from utils.audio_features import extract_librosa_features
import librosa.display

TEST_DIR = "test_data"
LABELS = ["podcasts", "songs", "lectures"]

# ----------------------------
# Load Audio Test Data
# ----------------------------
def load_audio_test_dataset():
    texts, labels, paths = [], [], []
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
                paths.append(path)
            except Exception as e:
                print(f"[WARN] Skipped {path}: {e}")
    return texts, labels, paths

# ----------------------------
# In-Memory CNN Classification
# ----------------------------
def classify_with_cnn(audio_path, model, transform, label_list):
    try:
        y, sr = librosa.load(audio_path)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
        librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
        plt.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        img = Image.open(buf).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1).squeeze().tolist()
            pred_idx = torch.argmax(outputs, dim=1).item()

        return label_list[pred_idx], probs
    except Exception as e:
        print(f"[ERROR] CNN classification failed for {audio_path}: {e}")
        return "error", [0.0] * len(label_list)

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
    X_test, y_test, audio_paths = load_audio_test_dataset()

    # Load text+feature models
    logreg = joblib.load("model/baseline/audio/logistic_regression.pkl")
    nb = joblib.load("model/baseline/audio/naive_bayes.pkl")
    tokenizer = DistilBertTokenizerFast.from_pretrained("model/distilbert/audio")
    bert_model = DistilBertForSequenceClassification.from_pretrained("model/distilbert/audio")
    bert_model.eval()

    # Load CNN model
    cnn_model = models.resnet18(pretrained=False)
    cnn_model.fc = torch.nn.Linear(cnn_model.fc.in_features, len(LABELS))
    cnn_model.load_state_dict(torch.load("model/cnn/audio/resnet18.pth", map_location=torch.device("cpu")))
    cnn_model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    model_names = ["LogReg", "NaiveBayes", "DistilBERT", "CNN"]
    all_probs = []

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
    start = time.time()
    bert_preds, bert_probs = [], []
    with torch.no_grad():
        for text in X_test:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = bert_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
            pred = torch.argmax(torch.tensor(probs)).item()
            bert_preds.append(LABELS[pred])
            bert_probs.append(probs)

    print(f"\n📊 DistilBERT (Time: {round(time.time() - start, 2)}s):")
    print(classification_report(y_test, bert_preds))
    plot_confusion(y_test, bert_preds, "DistilBERT", LABELS)
    all_probs.append(bert_probs)

    # CNN Spectrogram Model
    cnn_preds, cnn_probs = [], []
    start = time.time()
    for path in tqdm(audio_paths, desc="Running CNN"):
        pred, probs = classify_with_cnn(path, cnn_model, transform, LABELS)
        cnn_preds.append(pred)
        cnn_probs.append(probs)

    print(f"\n📊 CNN Spectrogram (Time: {round(time.time() - start, 2)}s):")
    print(classification_report(y_test, cnn_preds))
    plot_confusion(y_test, cnn_preds, "CNN", LABELS)
    all_probs.append(cnn_probs)

    # Plot confidence bar chart
    plot_confidences(all_probs, LABELS, model_names, list(range(len(X_test))))

if __name__ == "__main__":
    main()
