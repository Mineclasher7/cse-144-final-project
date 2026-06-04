import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from transformers import SiglipVisionModel, AutoImageProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# -----------------------------
# Load SigLIP-2 ViT backbone
# -----------------------------
processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
backbone = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224").to(device)
backbone.eval()

# Freeze backbone
for p in backbone.parameters():
    p.requires_grad = False

# -----------------------------
# Dataset
# -----------------------------
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.files = []
        self.labels = []
        self.classes = sorted(os.listdir(root))

        for idx, cls in enumerate(self.classes):
            folder = os.path.join(root, cls)
            for f in os.listdir(folder):
                if f.lower().endswith(("jpg", "png", "jpeg")):
                    self.files.append(os.path.join(folder, f))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), self.labels[idx]

# -----------------------------
# Load data
# -----------------------------
train_dir = "/content/drive/MyDrive/train"
dataset = ImageDataset(train_dir)

train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=32, shuffle=False)

num_classes = len(dataset.classes)

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

classifier = Classifier(embed_dim=backbone.config.hidden_size, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4, weight_decay=0.05)

# -----------------------------
# Extract embeddings
# -----------------------------
def get_embeddings(batch):
    with torch.no_grad():
        outputs = backbone(pixel_values=batch)
        return outputs.pooler_output  # (B, hidden_dim)

# -----------------------------
# Train loop
# -----------------------------
best_acc = 0

for epoch in range(20):
    classifier.train()
    total, correct = 0, 0

    for px, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        px, y = px.to(device), y.to(device)

        with torch.no_grad():
            emb = get_embeddings(px)

        logits = classifier(emb)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    train_acc = correct / total

    # Validation
    classifier.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for px, y in val_loader:
            px, y = px.to(device), y.to(device)
            emb = get_embeddings(px)
            logits = classifier(emb)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "classifier": classifier.state_dict(),
            "class_names": dataset.classes
        }, "siglip_classifier.pt")
        print("Saved best checkpoint")

print("Done.")
