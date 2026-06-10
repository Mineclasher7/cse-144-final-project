import os
import random
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
from tqdm.auto import tqdm
from transformers import SiglipVisionModel, AutoImageProcessor
import matplotlib.pyplot as plt

# -----------------------------
# Seed + Device
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# -----------------------------
# Backbone (SigLIP-2 ViT)
# -----------------------------
MODEL_NAME = "google/siglip2-base-patch16-224"

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
backbone = SiglipVisionModel.from_pretrained(MODEL_NAME).to(device)
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False

embed_dim = backbone.config.hidden_size

# -----------------------------
# Dataset (DS_Store-safe)
# -----------------------------
class ImageDataset(Dataset):
    def __init__(self, root):
        self.files = []
        self.labels = []
        self.classes = []

        def sort_key(x):
            return int(x) if x.isdigit() else x

        for cls in sorted(os.listdir(root), key=sort_key):
            folder = os.path.join(root, cls)
            if not os.path.isdir(folder):
                continue  # skip .DS_Store, Thumbs.db, etc.

            self.classes.append(cls)
            class_idx = len(self.classes) - 1

            for f in sorted(os.listdir(folder)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.files.append(os.path.join(folder, f))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        label = self.labels[idx]
        return img, label

# -----------------------------
# Collate function (fixes PIL error)
# -----------------------------
def collate_pil(batch):
    imgs, labels = zip(*batch)
    return list(imgs), torch.tensor(labels, dtype=torch.long)

# -----------------------------
# Classifier head
# -----------------------------
class Classifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)

# -----------------------------
# Embedding helper
# -----------------------------
@torch.no_grad()
def get_embeddings(img_list):
    inputs = processor(images=img_list, return_tensors="pt").to(device)
    outputs = backbone(pixel_values=inputs["pixel_values"])
    return outputs.pooler_output  # (B, embed_dim)

# -----------------------------
# Plotting helpers
# -----------------------------
def save_history_csv(history, path="training_history.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                history["train_loss"][i],
                history["train_acc"][i],
                history["val_loss"][i],
                history["val_acc"][i],
            ])

def save_curves(history):
    epochs = range(1, len(history["train_acc"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], marker="o", label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], marker="o", label="Validation Accuracy")
    plt.title("SigLIP-2 Classifier Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_curve.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
    plt.title("SigLIP-2 Classifier Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=220)
    plt.close()

# -----------------------------
# Train
# -----------------------------
def main():
    train_dir = "/content/drive/MyDrive/train"
    EPOCHS = 35
    BATCH_SIZE = 32

    dataset = ImageDataset(train_dir)
    print(f"Loaded {len(dataset)} images, {len(dataset.classes)} classes")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_pil
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_pil
    )

    classifier = Classifier(embed_dim, len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4, weight_decay=0.05)

    best_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(EPOCHS):
        classifier.train()
        total, correct = 0, 0
        train_loss_sum = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            labels = labels.to(device)
            embeddings = get_embeddings(imgs)

            logits = classifier(embeddings)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += batch_size

        train_loss = train_loss_sum / total
        train_acc = correct / total

        classifier.eval()
        total, correct = 0, 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for imgs, labels in val_loader:
                labels = labels.to(device)
                embeddings = get_embeddings(imgs)
                logits = classifier(embeddings)
                loss = criterion(logits, labels)

                batch_size = labels.size(0)
                val_loss_sum += loss.item() * batch_size
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total += batch_size

        val_loss = val_loss_sum / total
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        save_history_csv(history)
        save_curves(history)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "classifier_state_dict": classifier.state_dict(),
                    "class_names": dataset.classes,
                    "model_name": MODEL_NAME,
                },
                "siglip2_classifier.pt",
            )
            print(f"  -> Saved best checkpoint (Val Acc: {best_acc:.4f})")

    print("Done. Best Val Acc:", best_acc)
    print("Saved accuracy_curve.png")
    print("Saved loss_curve.png")
    print("Saved training_history.csv")

if __name__ == "__main__":
    main()
