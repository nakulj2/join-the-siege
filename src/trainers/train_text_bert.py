import os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import classification_report
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.extract_text import extract_text

LABELS = ['bank_statement', 'drivers_license', 'invoice']
DATA_DIR = "train_data"

def load_data():
    texts, labels = [], []
    for idx, label in enumerate(LABELS):
        folder = os.path.join(DATA_DIR, label + 's' if label != 'drivers_license' else label)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            try:
                with open(path, "rb") as f:
                    f.filename = path
                    text = extract_text(f)
                if text.strip():
                    texts.append(text)
                    labels.append(idx)
            except Exception as e:
                print(f"Failed to extract {path}: {e}")
    return texts, labels

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def main():
    texts, labels = load_data()
    dataset = Dataset.from_dict({"text": texts, "label": labels})
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    tokenized = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(LABELS))


    training_args = TrainingArguments(
        output_dir="./model/distilbert/text",
        per_device_train_batch_size=4,
        num_train_epochs=4,
        logging_steps=10
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=tokenized,
        tokenizer=tokenizer
    )

    trainer.train()

    # Evaluation
    preds = trainer.predict(tokenized)
    pred_labels = torch.argmax(torch.tensor(preds.predictions), axis=1).numpy()
    print("\nClassification Report:")
    print(classification_report(tokenized["label"], pred_labels, target_names=LABELS))

    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = main()
