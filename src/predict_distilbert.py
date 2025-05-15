import sys
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from utils.extract_text import extract_text

LABELS = ['bank_statement', 'drivers_license', 'invoice']
MODEL_DIR = "model/distilbert/checkpoint-24"

# Load model and tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

def classify(text):
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
        print("Usage: python predict_distilbert.py <path_to_file>")
    else:
        main(sys.argv[1])
