# train_audio_bert.py

import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import torch
from tqdm import tqdm
from collections import Counter
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.transcribe_audio import transcribe_audio
from utils.audio_features import extract_librosa_features

DATA_DIR = "train_data"
MODEL_DIR = "model/distilbert/audio"

def load_audio_data():
    texts, labels, label_names = [], [], []
    label_map = {}
    idx_counter = 0

    for label in sorted(os.listdir(DATA_DIR)):
        folder = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder):
            continue

        mp3s = [f for f in os.listdir(folder) if f.lower().endswith(".mp3")]
        if not mp3s:
            continue

        if label not in label_map:
            label_map[label] = idx_counter
            label_names.append(label)
            idx_counter += 1

        for fname in tqdm(mp3s, desc=f"Loading {label}"):
            fpath = os.path.join(folder, fname)
            try:
                transcription = transcribe_audio(fpath)
                tempo_features = extract_librosa_features(fpath)
                combined_text = f"{transcription} | features: {' '.join(map(str, tempo_features))}"
                texts.append(combined_text)
                labels.append(label_map[label])
            except Exception as e:
                print(f"[WARN] Skipped {fpath}: {e}")

    return texts, labels, label_names

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def main():
    texts, labels, label_names = load_audio_data()
    print("Class counts:", Counter(labels))

    dataset = Dataset.from_dict({"text": texts, "label": labels})
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    tokenized = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(label_names)
    )

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=4,
        num_train_epochs=4,
        logging_steps=10,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=tokenized,
        tokenizer=tokenizer
    )

    trainer.train()

    preds = trainer.predict(tokenized)
    pred_labels = torch.argmax(torch.tensor(preds.predictions), axis=1).numpy()
    true_labels = tokenized["label"]

    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=label_names))

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"✅ Model and tokenizer saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
