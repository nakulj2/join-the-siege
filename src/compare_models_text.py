import os
import time
import joblib
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report
from tqdm import tqdm
from utils.extract_text import extract_text
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

TEST_DIR = "test_data"
LABELS = ['bank_statements', 'drivers_license', 'invoices']

# ----------------------------
# Load Test Data
# ----------------------------
def load_test_dataset():
    texts, labels = [], []
    for label in os.listdir(TEST_DIR):
        folder = os.path.join(TEST_DIR, label)
        if not os.path.isdir(folder):
            continue
        for fname in tqdm(os.listdir(folder), desc=f"Loading {label}"):
            path = os.path.join(folder, fname)
            try:
                with open(path, "rb") as f:
                    f.filename = fname
                    text = extract_text(f)
                if text.strip():
                    texts.append(text)
                    labels.append(label)
            except Exception as e:
                print(f"[WARN] Skipped {path}: {e}")
    return texts, labels

# ----------------------------
# Bar Chart Visualizer
# ----------------------------
def plot_confidences(all_probs, labels, model_names, doc_ids):
    num_models = len(model_names)
    num_docs = len(doc_ids)
    num_classes = len(labels)

    for i in range(num_docs):
        fig, ax = plt.subplots()
        width = 0.2
        x = range(num_classes)

        for j, model_name in enumerate(model_names):
            probs = all_probs[j][i]
            ax.bar(
                [p + width*j for p in x],
                probs,
                width=width,
                label=model_name
            )

        ax.set_xticks([p + width for p in x])
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylim([0, 1.1])
        ax.set_ylabel("Confidence")
        ax.set_title(f"Confidence for Test Sample {i+1}")
        ax.legend()
        plt.tight_layout()
        plt.show()

# ----------------------------
# Confusion matrix visualizer
# ----------------------------
def plot_confusion(y_true, y_pred, model_name, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=20)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.show()

# ----------------------------
# Main Comparison
# ----------------------------
def main():
    X_test, y_test = load_test_dataset()

    logreg = joblib.load("model/text/logistic_regression.pkl")
    nb = joblib.load("model/text/naive_bayes.pkl")

    distil_model_path = "model/distilbert/checkpoint-24"
    tokenizer = DistilBertTokenizerFast.from_pretrained(distil_model_path)
    model = DistilBertForSequenceClassification.from_pretrained(distil_model_path)

    all_probs = []
    model_names = ["LogReg", "NaiveBayes", "DistilBERT"]

    # Logistic Regression
    start = time.time()
    log_probs = logreg.predict_proba(X_test)
    log_preds = logreg.predict(X_test)
    print(f"\n📊 Logistic Regression Results (Time: {round(time.time() - start, 2)}s):")
    print(classification_report(y_test, log_preds))
    plot_confusion(y_test, log_preds, "Logistic Regression", LABELS)
    all_probs.append(log_probs.tolist())

    # Naive Bayes
    start = time.time()
    nb_probs = nb.predict_proba(X_test)
    nb_preds = nb.predict(X_test)
    print(f"\n📊 Naive Bayes Results (Time: {round(time.time() - start, 2)}s):")
    print(classification_report(y_test, nb_preds))
    plot_confusion(y_test, nb_preds, "Naive Bayes", LABELS)
    all_probs.append(nb_probs.tolist())

    # DistilBERT
    start = time.time()
    model.eval()
    bert_probs = []
    bert_preds = []

    with torch.no_grad():
        for text in X_test:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
            pred = torch.argmax(torch.tensor(probs)).item()
            bert_preds.append(LABELS[pred])
            bert_probs.append(probs)

    print(f"\n📊 DistilBERT Results (Time: {round(time.time() - start, 2)}s):")
    print(classification_report(y_test, bert_preds))
    plot_confusion(y_test, bert_preds, "DistilBERT", LABELS)
    all_probs.append(bert_probs)

    # Visualize
    plot_confidences(all_probs, LABELS, model_names, list(range(len(X_test))))

if __name__ == "__main__":
    main()
