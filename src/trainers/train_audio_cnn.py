import os
import librosa
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import librosa.display
from pathlib import Path

DATA_DIR = "train_data"
SPEC_DIR = "spectrogram_data"
MODEL_DIR = "model/cnn/audio"
TARGET_LABELS = {"songs", "podcasts"}

# -----------------------------------
# Step 1: Convert MP3s to Spectrograms
# -----------------------------------
def generate_spectrograms():
    os.makedirs(SPEC_DIR, exist_ok=True)
    for label in os.listdir(DATA_DIR):
        if label not in TARGET_LABELS:
            continue  # ❌ Skip non-audio labels

        input_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(input_dir):
            continue  # ❌ Skip .DS_Store etc.

        output_dir = os.path.join(SPEC_DIR, label)
        os.makedirs(output_dir, exist_ok=True)

        for fname in os.listdir(input_dir):
            if not fname.endswith(".mp3"):
                continue
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname.replace(".mp3", ".png"))

            if os.path.exists(out_path):
                continue  # ✅ Already processed

            try:
                y, sr = librosa.load(in_path)
                S = librosa.feature.melspectrogram(y=y, sr=sr)
                S_dB = librosa.power_to_db(S, ref=np.max)

                plt.figure(figsize=(2.24, 2.24), dpi=100)
                librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
                plt.close()
            except Exception as e:
                print(f"[WARN] Skipping {in_path}: {e}")

# -----------------------------------
# Step 2: Train CNN on Spectrograms
# -----------------------------------
def train_cnn():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(SPEC_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(4):
        total_loss = 0
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"📉 Epoch {epoch+1} Loss: {total_loss:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "resnet18.pth"))
    with open(os.path.join(MODEL_DIR, "labels.txt"), "w") as f:
        for label in dataset.classes:
            f.write(label + "\n")
    print(f"✅ Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    generate_spectrograms()
    train_cnn()
