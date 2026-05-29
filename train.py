import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm
import copy

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
USE_CUDA = torch.cuda.is_available()
AUTOCAST_DEVICE = "cuda" if USE_CUDA else "cpu"

print(f"Using device: {device}")

# -----------------------------
# Mixup + CutMix
# -----------------------------
def mixup_cutmix(x, y, alpha=0.4):
    if random.random() < 0.5:
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(x.size(0)).to(x.device)
        mixed = lam * x + (1 - lam) * x[idx]
        return mixed, y, y[idx], lam
    else:
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(x.size(0)).to(x.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        mixed = x.clone()
        mixed[:, :, bbx1:bbx2, bby1:bby2] = x[idx, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1)*(bby2 - bby1) / (x.size(-1)*x.size(-2)))
        return mixed, y, y[idx], lam

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut = int(W * np.sqrt(1 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut//2, 0, W)
    y1 = np.clip(cy - cut//2, 0, H)
    x2 = np.clip(cx + cut//2, 0, W)
    y2 = np.clip(cy + cut//2, 0, H)
    return x1, y1, x2, y2

# -----------------------------
# Dataset + Augmentations
# -----------------------------
def get_loaders(train_dir, batch_size=32):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25)
    ])

    val_tf = transforms.Compose([
        transforms.Resize(236),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    full = datasets.ImageFolder(train_dir)
    class_to_idx = full.class_to_idx

    train_size = int(0.8 * len(full))
    val_size = len(full) - train_size

    train_set, val_set = random_split(full, [train_size, val_size])

    train_set.dataset.transform = train_tf
    val_set.dataset.transform = val_tf

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=USE_CUDA)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=USE_CUDA)

    return train_loader, val_loader, class_to_idx

# -----------------------------
# EMA
# -----------------------------
class EMA:
    def __init__(self, model, decay=0.995):
        self.shadow = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for s, p in zip(self.shadow.parameters(), model.parameters()):
                s.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def update_buffers(self, model):
        with torch.no_grad():
            for s, p in zip(self.shadow.buffers(), model.buffers()):
                s.data.copy_(p.data)

# -----------------------------
# Build Models
# -----------------------------
def build_convnext(num_classes):
    m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    in_f = m.classifier[2].in_features
    m.classifier[2] = nn.Linear(in_f, num_classes)
    return m.to(device)

def build_efficientnet(num_classes):
    m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, num_classes)
    return m.to(device)

def build_swin(num_classes):
    m = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    in_f = m.head.in_features
    m.head = nn.Linear(in_f, num_classes)
    return m.to(device)

# -----------------------------
# Train One Epoch
# -----------------------------
def train_epoch(model, loader, optimizer, scheduler, criterion, scaler, ema):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        mixed, y_a, y_b, lam = mixup_cutmix(x, y)

        with torch.amp.autocast(AUTOCAST_DEVICE, enabled=USE_CUDA):
            logits = model(mixed)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        ema.update(model)
        ema.update_buffers(model)

        total_loss += loss.item() * x.size(0)
        with torch.no_grad():
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total

# -----------------------------
# Validation
# -----------------------------
@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(AUTOCAST_DEVICE, enabled=USE_CUDA):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total

# -----------------------------
# Train Wrapper
# -----------------------------
def train_model(model, name, train_loader, val_loader, class_to_idx):
    print(f"\n===== Training {name} =====")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        steps_per_epoch=len(train_loader),
        epochs=30
    )
    scaler = torch.amp.GradScaler(AUTOCAST_DEVICE, enabled=USE_CUDA)
    ema = EMA(model)

    best_acc = 0
    ckpt_path = f"ckpt_{name}.pt"

    for epoch in range(30):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, ema)
        val_loss, val_acc = validate(ema.shadow, val_loader, criterion)

        print(f"Epoch {epoch+1:02d} | "
              f"Train Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": ema.shadow.state_dict(),
                "class_to_idx": class_to_idx
            }, ckpt_path)
            print(f"  -> Saved best checkpoint ({best_acc:.4f})")

# -----------------------------
# Main
# -----------------------------
def main():
    train_dir = "/content/drive/MyDrive/train"
    train_loader, val_loader, class_to_idx = get_loaders(train_dir)

    train_model(build_convnext(len(class_to_idx)), "convnext", train_loader, val_loader, class_to_idx)
    train_model(build_efficientnet(len(class_to_idx)), "efficientnet", train_loader, val_loader, class_to_idx)
    train_model(build_swin(len(class_to_idx)), "swin", train_loader, val_loader, class_to_idx)

if __name__ == "__main__":
    main()
