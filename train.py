import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
from tqdm.auto import tqdm
from transformers import SiglipVisionModel, AutoImageProcessor

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

        for cls in sorted(os.listdir(root)):
            folder = os.path.join(root, cls)
            if not os.path.isdir(folder):
                continue  # skip .DS_Store, Thumbs.db, etc.

            self.classes.append(cls)
            class_idx = len(self.classes) - 1

            for f in os.listdir(folder):
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
    return list(imgs), torch.tensor(labels)

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
# Train
# -----------------------------
def main():
    train_dir = "/content/drive/MyDrive/train"
    EPOCHS = 30
    BATCH_SIZE = 32

    dataset = ImageDataset(train_dir)
    print(f"Loaded {len(dataset)} images, {len(dataset.classes)} classes")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

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

    for epoch in range(EPOCHS):
        # ---- Train ----
        classifier.train()
        total, correct = 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            labels = labels.to(device)
            embeddings = get_embeddings(imgs)

            logits = classifier(embeddings)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # ---- Validation ----
        classifier.eval()
        total, correct = 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                labels = labels.to(device)
                embeddings = get_embeddings(imgs)
                logits = classifier(embeddings)
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

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

if __name__ == "__main__":
    main()
