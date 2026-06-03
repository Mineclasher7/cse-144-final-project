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
# Mixup + CutMix (slightly softer)
# -----------------------------
def mixup_cutmix(x, y, alpha=0.2):
    if random.random() < 0.5:
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(x.size(0)).to(x.device)
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam
    else:
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(x.size(0)).to(x.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        mixed = x.clone()
        mixed[:, :, bbx1:bbx2, bby1:bby2] = x[idx, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2-bbx1)*(bby2-bby1) / (x.size(-1)*x.size(-2)))
        return mixed, y, y[idx], lam

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut = int(W * np.sqrt(1 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    return (np.clip(cx-cut//2,0,W), np.clip(cy-cut//2,0,H),
            np.clip(cx+cut//2,0,W), np.clip(cy+cut//2,0,H))

# -----------------------------
# Dataset — train/val split
# -----------------------------
def get_loaders(train_dir, batch_size=32, num_workers=2):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(236),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full = datasets.ImageFolder(train_dir)
    class_to_idx = full.class_to_idx
    print(f"Loaded {len(full)} images, {len(class_to_idx)} classes")

    train_size = int(0.8 * len(full))
    val_size   = len(full) - train_size
    train_set, val_set = random_split(full, [train_size, val_size])

    train_set.dataset.transform = train_tf
    val_set.dataset.transform   = val_tf

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=USE_CUDA)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=USE_CUDA)
    return train_loader, val_loader, class_to_idx

# -----------------------------
# EMA
# -----------------------------
class EMA:
    def __init__(self, model, decay=0.995):
        self.shadow = copy.deepcopy(model).eval()
        self.decay  = decay
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for s, p in zip(self.shadow.parameters(), model.parameters()):
                s.data.mul_(self.decay).add_(p.data, alpha=1-self.decay)

    def update_buffers(self, model):
        with torch.no_grad():
            for s, p in zip(self.shadow.buffers(), model.buffers()):
                s.data.copy_(p.data)

# -----------------------------
# Models — with Dropout head
# -----------------------------
def build_efficientnet(num_classes, dropout=0.3):
    m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_f, num_classes)
    )
    return m.to(device)

def build_swin(num_classes, dropout=0.3):
    m = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    in_f = m.head.in_features
    m.head = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_f, num_classes)
    )
    return m.to(device)

def set_stochastic_depth(model, drop_prob=0.1):
    for module in model.modules():
        if hasattr(module, 'drop_path') and hasattr(module.drop_path, 'p'):
            module.drop_path.p = drop_prob

# -----------------------------
# Gradual unfreeze schedule
# -----------------------------
def set_freeze(model, epoch):
    backbone_attr = 'features'   # both EfficientNetV2 and Swin use .features

    if epoch < 5:
        for p in getattr(model, backbone_attr).parameters():
            p.requires_grad = False
    elif epoch < 10:
        blocks = list(getattr(model, backbone_attr).children())
        for p in getattr(model, backbone_attr).parameters():
            p.requires_grad = False
        for p in blocks[-1].parameters():
            p.requires_grad = True
    else:
        for p in getattr(model, backbone_attr).parameters():
            p.requires_grad = True

# -----------------------------
# Train one epoch
# -----------------------------
def train_epoch(model, loader, optimizer, scheduler, criterion, scaler, ema, epoch):
    model.train()
    set_freeze(model, epoch)

    total_loss, correct, total = 0, 0, 0

    for x, y in tqdm(loader, desc=f"  Epoch {epoch+1}", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        mixed, y_a, y_b, lam = mixup_cutmix(x, y)

        with torch.amp.autocast(AUTOCAST_DEVICE, enabled=USE_CUDA):
            logits = model(mixed)
            loss   = lam * criterion(logits, y_a) + (1-lam) * criterion(logits, y_b)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        ema.update(model)
        ema.update_buffers(model)

        total_loss += loss.item() * x.size(0)
        with torch.no_grad():
            with torch.amp.autocast(AUTOCAST_DEVICE, enabled=USE_CUDA):
                correct += (model(x).argmax(1) == y).sum().item()
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
            loss   = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += y.size(0)

    return total_loss / total, correct / total

# -----------------------------
# Train wrapper
# -----------------------------
def train_model(name, model, train_loader, val_loader, epochs, ckpt_path, class_to_idx):
    set_stochastic_depth(model, drop_prob=0.1)
    ema = EMA(model, decay=0.995)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    backbone_attr = 'features'
    head_attr     = 'classifier' if hasattr(model, 'classifier') else 'head'
    optimizer = torch.optim.AdamW([
        {'params': getattr(model, backbone_attr).parameters(), 'lr': 1e-5},
        {'params': getattr(model, head_attr).parameters(),     'lr': 1e-4},
    ], weight_decay=0.1)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-5, 1e-4],
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100,
    )
    scaler = torch.amp.GradScaler(AUTOCAST_DEVICE, enabled=USE_CUDA)

    best_acc = 0.0

    print(f"\n{'='*50}\n  Training: {name}\n{'='*50}")
    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, ema, epoch
        )
        val_loss, val_acc = validate(ema.shadow, val_loader, criterion)

        print(f"  [{name}] Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": ema.shadow.state_dict(),
                "class_to_idx":     class_to_idx,
                "architecture":     name,
            }, ckpt_path)
            print(f"  -> Saved best checkpoint (Val Acc: {best_acc:.4f})")

# -----------------------------
# Main
# -----------------------------
def main():
    train_dir  = "/content/drive/MyDrive/train"
    EPOCHS     = 40
    BATCH_SIZE = 32

    train_loader, val_loader, class_to_idx = get_loaders(train_dir, batch_size=BATCH_SIZE)
    num_classes = len(class_to_idx)

    train_model("efficientnet", build_efficientnet(num_classes),
                train_loader, val_loader, EPOCHS,
                "ckpt_efficientnet.pt", class_to_idx)

    train_model("swin",         build_swin(num_classes),
                train_loader, val_loader, EPOCHS,
                "ckpt_swin.pt", class_to_idx)

    print("\nDone — run inference.py to generate submission.csv")

if __name__ == "__main__":
    main()
