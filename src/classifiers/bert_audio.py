# src/classifiers/bert_audio.py

import os
import torch
from werkzeug.datastructures import FileStorage
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.utils.transcribe_audio import transcribe_audio
from src.utils.audio_features import extract_librosa_features

MODEL_PATH = "model/distilbert/audio"

# Dynamically load labels (optional: you can hardcode if stable)
LABELS = sorted([
    name for name in os.listdir("train_data")
    if os.path.isdir(os.path.join("train_data", name)) and
       any(fname.endswith(".mp3") for fname in os.listdir(os.path.join("train_data", name)))
])

# Load tokenizer and model
try:
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"[ERROR] Failed to load audio BERT model/tokenizer: {e}")
    tokenizer, model = None, None

def classify(file: FileStorage):
    if not tokenizer or not model:
        return "model_not_loaded", []

    try:
        temp_path = f"/tmp/{file.filename}"
        file.save(temp_path)

        text = transcribe_audio(temp_path)
        features = extract_librosa_features(temp_path)
        combined = f"{text} | features: {' '.join(map(str, features))}"

        inputs = tokenizer(combined, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()

        return LABELS[pred], probs.squeeze().tolist()

    except Exception as e:
        print(f"[ERROR] BERT audio classification failed: {e}")
        return "error", []