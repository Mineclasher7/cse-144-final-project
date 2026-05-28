import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm
from PIL import Image

# -----------------------------
# Device + Seed
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# -----------------------------
# Mixup
# -----------------------------
def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# -----------------------------
# Dataset + Augmentations
# -----------------------------
def get_dataloaders(train_dir, test_dir, batch_size=64, num_workers=2):

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(236),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_dataset = datasets.ImageFolder(root=train_dir)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    g = torch.Generator().manual_seed(42)
    train_idx, val_idx = random_split(range(len(full_dataset)), [train_size, val_size], generator=g)

    train_set = Subset(datasets.ImageFolder(train_dir, transform=train_tf), train_idx.indices)
    val_set   = Subset(datasets.ImageFolder(train_dir, transform=val_tf), val_idx.indices)

    class TestDataset(Dataset):
        def __init__(self, root, transform=None):
            self.root = root
            self.transform = transform
            self.files = sorted([f for f in os.listdir(root) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        def __len__(self):
            return len(self.files)
        def __getitem__(self, idx):
            img_path = os.path.join(self.root, self.files[idx])
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, self.files[idx]

    test_set = TestDataset(test_dir, transform=val_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, full_dataset.class_to_idx

# -----------------------------
# Model
# -----------------------------
def build_model(num_classes=100):
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    model = models.convnext_tiny(weights=weights)

    for p in model.parameters():
        p.requires_grad = True

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model.to(device)

# -----------------------------
# Training
# -----------------------------
def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            mixed_x, y_a, y_b, lam = mixup(x, y)
            y_pred = model(mixed_x)
            loss = lam * criterion(y_pred, y_a) + (1 - lam) * criterion(y_pred, y_b)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * x.size(0)

        # Real accuracy (not Mixup)
        with torch.no_grad():
            clean_preds = model(x).argmax(dim=1)
            correct += (clean_preds == y).sum().item()

        total += y.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for x, y in tqdm(loader, desc="Validation"):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)

        total_loss += loss.item()
        correct += (y_pred.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total

# -----------------------------
# Main
# -----------------------------
def main():
    train_dir = "/content/drive/MyDrive/train"
    test_dir  = "/content/drive/MyDrive/test"

    epochs = 30
    batch_size = 64

    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        train_dir, test_dir, batch_size=batch_size
    )

    model = build_model(num_classes=100)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 2e-5},
        {'params': model.classifier.parameters(), 'lr': 2e-4}
    ], weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[5e-5, 5e-4],
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )

    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0
    ckpt_path = "checkpoint.pt"

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx
            }, ckpt_path)

    print("Training complete. Best Val Acc:", best_acc)

if __name__ == "__main__":
    main()
