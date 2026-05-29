import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm
import copy

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

USE_CUDA = torch.cuda.is_available()
AUTOCAST_DEVICE = 'cuda' if USE_CUDA else 'cpu'
print(f"Using device: {device} | autocast: {AUTOCAST_DEVICE}")

# -----------------------------
# Mixup + CutMix
# -----------------------------
def mixup_cutmix(x, y, alpha=0.4):
    if random.random() < 0.5:
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(x.size(0)).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        return mixed_x, y, y[index], lam
    else:
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(x.size(0)).to(x.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        mixed_x = x.clone()
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
        return mixed_x, y, y[index], lam

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# -----------------------------
# Dataset
# -----------------------------
def get_train_loader(train_dir, batch_size=32, num_workers=2):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandAugment(num_ops=3, magnitude=12),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.2)),
    ])

    dataset      = datasets.ImageFolder(root=train_dir, transform=train_tf)
    class_to_idx = dataset.class_to_idx
    print(f"Training on {len(dataset)} images across {len(class_to_idx)} classes")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=USE_CUDA)
    return loader, class_to_idx

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
            for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def update_buffers(self, model):
        with torch.no_grad():
            for ema_buf, buf in zip(self.shadow.buffers(), model.buffers()):
                ema_buf.data.copy_(buf.data)

# -----------------------------
# Model — Swin-T
# Shifted-window attention: best ViT-family model for small datasets
# 28M params, 81.3% ImageNet, fine-tunes well with local+global features
# -----------------------------
def build_model(num_classes, dropout=0.4):
    model   = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    in_feat = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_feat, num_classes)
    )
    return model.to(device)

def set_stochastic_depth(model, drop_prob=0.2):
    for module in model.modules():
        if hasattr(module, 'drop_path') and hasattr(module.drop_path, 'p'):
            module.drop_path.p = drop_prob

# -----------------------------
# Training
# -----------------------------
def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, ema, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0

    # Freeze backbone for first 2 epochs
    freeze = epoch < 2
    for param in model.features.parameters():
        param.requires_grad = not freeze

    for x, y in tqdm(loader, desc=f"  Epoch {epoch+1}", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        mixed_x, y_a, y_b, lam = mixup_cutmix(x, y)

        with torch.amp.autocast(AUTOCAST_DEVICE, enabled=USE_CUDA):
            y_pred = model(mixed_x)
            loss   = lam * criterion(y_pred, y_a) + (1 - lam) * criterion(y_pred, y_b)

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
# Main
# -----------------------------
def main():
    train_dir  = "/content/drive/MyDrive/train"
    EPOCHS     = 50
    BATCH_SIZE = 32

    loader, class_to_idx = get_train_loader(train_dir, batch_size=BATCH_SIZE)
    num_classes = len(class_to_idx)

    model = build_model(num_classes)
    set_stochastic_depth(model, drop_prob=0.2)
    ema   = EMA(model, decay=0.995)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = torch.optim.AdamW([
    {'params': model.features.parameters(), 'lr': 5e-5},   
    {'params': model.head.parameters(),     'lr': 5e-4},   
    {'params': model.norm.parameters(),     'lr': 5e-5},   
    ], weight_decay=0.2)


    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[5e-5, 5e-4, 5e-5],
        steps_per_epoch=len(loader),
        epochs=EPOCHS,
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100
    )

    scaler = torch.amp.GradScaler(AUTOCAST_DEVICE, enabled=USE_CUDA)

    print(f"\n{'='*50}\n  Training: Swin-T\n{'='*50}")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, loader, optimizer, scheduler, criterion, scaler, ema, epoch
        )
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    torch.save({
        "model_state_dict": ema.shadow.state_dict(),
        "class_to_idx":     class_to_idx,
        "architecture":     "swin_t",
    }, "ckpt_swint.pt")
    print("\nSaved -> ckpt_swint.pt")
    print("Run inference.py to generate submission.csv")

if __name__ == "__main__":
    main()