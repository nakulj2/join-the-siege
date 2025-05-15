import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from utils.extract_text import extract_text  # assumes this returns full string for a file

LABELS = ['bank_statement', 'drivers_license', 'invoice']
DATA_DIR = "/Users/nakuljain/Desktop/HERON DATA/join-the-siege/data"

def load_data():
    texts, labels = [], []

    for idx, label in enumerate(LABELS):
        folder_path = os.path.join(DATA_DIR, label + 's' if label != 'drivers_license' else label)
        if not os.path.isdir(folder_path):
            print(f"❌ Missing folder: {folder_path}")
            continue

        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            print(f"📄 Processing: {fpath}")
            try:
                with open(fpath, "rb") as f:
                    f.filename = fpath
                    text = extract_text(f)
                if text.strip():
                    texts.append(text)
                    labels.append(idx)
                else:
                    print(f"⚠️ Empty text from: {fpath}")
            except Exception as e:
                print(f"❌ Error extracting from {fpath}: {e}")

    return texts, labels


def tokenize(batch, tokenizer):
    return tokenizer(batch['text'], padding=True, truncation=True)

def main():
    texts, labels = load_data()
    label_map = {i: label for i, label in enumerate(LABELS)}

    print(f"\nLoaded {len(texts)} documents.")
    from collections import Counter
    print("Label distribution:", Counter(labels))

    if len(texts) <= 3:
        print("⚠️ Not enough data to split. Training and evaluating on full dataset.")

        dataset = Dataset.from_dict({'text': texts, 'label': labels})
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        dataset = dataset.map(lambda x: tokenize(x, tokenizer), batched=True)

        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(LABELS))

        training_args = TrainingArguments(
            output_dir="./model/text",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=4,
            logging_steps=10
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            tokenizer=tokenizer
        )
    else:
        print("✅ Enough data, using 50/50 train-test split.")

        data = {'text': texts, 'label': labels}
        raw_dataset = Dataset.from_dict(data)
        train_test = raw_dataset.train_test_split(test_size=0.5, seed=42)

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        train_test = train_test.map(lambda x: tokenize(x, tokenizer), batched=True)

        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(LABELS))

        training_args = TrainingArguments(
            output_dir="./model/text",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=4,
            logging_steps=10
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_test["train"],
            eval_dataset=train_test["test"],
            tokenizer=tokenizer
        )

    trainer.train()

    # Inference on eval set
    eval_set = trainer.eval_dataset if hasattr(trainer, "eval_dataset") else dataset
    preds = trainer.predict(eval_set)
    pred_labels = torch.argmax(torch.tensor(preds.predictions), axis=1).numpy()
    true_labels = eval_set["label"]
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=LABELS))

    return model, tokenizer, label_map, texts, labels

def classify_text(text, model, tokenizer, label_map):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        return label_map[pred_id], probs.squeeze().tolist()


if __name__ == "__main__":
    model, tokenizer, label_map, texts, labels = main()

    # -----------------------
    # 🔍 Manual Test Examples
    # -----------------------
    test_texts = [
        "This is an Illinois Driver License with ID number and address.",
        "Account Summary from Wells Fargo dated March 2024.",
        "Invoice #2093 for services rendered to Acme Corp."
    ]

    print("\n📊 Manual Text Classification Results:")
    for txt in test_texts:
        label, probs = classify_text(txt, model, tokenizer, label_map)
        print(f"\n📝 Input: {txt}")
        print(f"🔍 Predicted: {label}")
        print(f"📈 Probabilities: {dict(zip(LABELS, [round(p, 3) for p in probs]))}")

    # ------------------------------
    # 📚 Training Set Memorization Check
    # ------------------------------
    print("\n📚 Training Set Prediction Check:")
    for txt, label_id in zip(texts, labels):
        label, probs = classify_text(txt, model, tokenizer, label_map)
        true_label = label_map[label_id]
        print(f"✅ True: {true_label} | 🔍 Predicted: {label} | 📈 Probs: {dict(zip(LABELS, [round(p, 3) for p in probs]))}")

    # --------------------------
    # 🧪 Real Document Evaluation
    # --------------------------
    test_image_path = "/Users/nakuljain/Desktop/HERON DATA/join-the-siege/data/drivers_license/drivers_license_4.png"
    with open(test_image_path, "rb") as f:
        f.filename = os.path.basename(test_image_path)  # Spoof .filename
        extracted_text = extract_text(f)

    label, probs = classify_text(extracted_text, model, tokenizer, label_map)
    print(f"\n🧾 Extracted Text from {test_image_path}:\n{extracted_text}")
    print(f"\n🔍 Predicted Label: {label}")
    print(f"📈 Probabilities: {dict(zip(LABELS, [round(p, 3) for p in probs]))}")

