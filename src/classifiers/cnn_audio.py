import os
import torch
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "model/cnn/audio/resnet18.pth"
LABELS_PATH = "model/cnn/audio/labels.txt"

# Load label list
try:
    with open(LABELS_PATH, "r") as f:
        LABELS = [line.strip() for line in f.readlines()]
except Exception as e:
    print(f"[ERROR] Failed to load label file: {e}")
    LABELS = []

# Load model
try:
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(LABELS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
except Exception as e:
    print(f"[ERROR] Failed to load CNN model: {e}")
    model = None

# Transform to match training config
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def classify_spectrogram(image_path: str):
    if not model or not LABELS:
        return "model_not_loaded", []

    try:
        img = Image.open(image_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1).squeeze().tolist()
            pred_idx = torch.argmax(outputs, dim=1).item()

        return LABELS[pred_idx], probs
    except Exception as e:
        print(f"[ERROR] CNN audio classification failed: {e}")
        return "error", []

# CLI usage for testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python cnn_audio.py <path_to_spectrogram.png>")
    else:
        label, probs = classify_spectrogram(sys.argv[1])
        print(f"🔍 Predicted: {label}")
        print(f"📈 Probabilities: {dict(zip(LABELS, [round(p, 3) for p in probs]))}")
