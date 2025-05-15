# src/trainers/train_audio_bert.py

import os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from utils.transcribe_audio import transcribe_audio
from utils.audio_features import extract_librosa_features
from sklearn.metrics import classification_report

LABELS = ['songs', 'podcasts']
DATA_DIR = "train_data"

def load_audio_data():
    texts, labels = [], []
    for label in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(path):
            continue
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            try:
                text = transcribe_audio(fpath)
                features = extract_librosa_features(fpath)
                combined = f"{text} | features: {' '.join(map(str, features))}"
                texts.append(combined)
                labels.append(label)
            except Exception as e:
                print(f"[WARN] {fpath}: {e}")
    return texts, labels

def main():
    texts, labels = load_audio_data()
    dataset = Dataset.from_dict({"text": texts, "label": labels})

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding=True), batched=True)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(set(labels)))
    args = TrainingArguments("model/audio_bert", num_train_epochs=4, per_device_train_batch_size=4)

    trainer = Trainer(model=model, args=args, train_dataset=dataset, eval_dataset=dataset)
    trainer.train()

    preds = trainer.predict(dataset)
    pred_labels = torch.argmax(torch.tensor(preds.predictions), dim=1)
    print(classification_report(dataset["label"], pred_labels))

if __name__ == "__main__":
    main()
