# src/classifiers/bert_text.py

import os
import sys
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.utils.extract_text import extract_text

LABELS = ['bank_statement', 'drivers_license', 'invoice']
MODEL_PATH = "model/distilbert/text/checkpoint-28"

try:
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"[ERROR] Failed to load BERT model/tokenizer: {e}")
    tokenizer, model = None, None

def classify(text: str):
    if not tokenizer or not model:
        return "model_not_loaded", []
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
    return LABELS[pred], probs.squeeze().tolist()

def main(filepath):
    with open(filepath, "rb") as f:
        f.filename = filepath
        text = extract_text(f)
    print(f"\n📝 Extracted Text:\n{text[:300]}...\n")
    label, probs = classify(text)
    print(f"🔍 Predicted Label: {label}")
    print(f"📈 Probabilities: {dict(zip(LABELS, [round(p, 3) for p in probs]))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bert_text.py <path_to_file>")
    else:
        main(sys.argv[1])
