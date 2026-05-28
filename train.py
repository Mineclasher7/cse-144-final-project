import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm
from PIL import Image
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

# -----------------------------
# Mixup + CutMix
# -----------------------------
def mixup_cutmix(x, y, alpha=0.2):
    if random.random() < 0.5:
        # Mixup
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(x.size(0)).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        return mixed_x, y, y[index], lam
    else:
        # CutMix
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(x.size(0)).to(x.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        mixed_x = x.clone()
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
        return mixed_x, y, y[index], lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# -----------------------------
# Dataset + Augmentations
# -----------------------------
def get_dataloaders(train_dir, test_dir, batch_size=64, num_workers=2):

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
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
        transforms.Normalize(mean, std),
    ])

    full_dataset = datasets.ImageFolder(root=train_dir)
    class_to_idx = full_dataset.class_to_idx

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    g = torch.Generator().manual_seed(42)
    train_idx, val_idx = random_split(range(len(full_dataset)), [train_size, val_size], generator=g)


    train_set = Subset(datasets.ImageFolder(train_dir, transform=train_tf), train_idx.indices)
    train_set.dataset.class_to_idx = class_to_idx

    val_set = Subset(datasets.ImageFolder(train_dir, transform=val_tf), val_idx.indices)
    val_set.dataset.class_to_idx = class_to_idx

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

    return train_loader, val_loader, test_loader, class_to_idx

# -----------------------------
# Model + EMA
# -----------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = copy.deepcopy(model).eval()
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

def build_model(num_classes=100):
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    model = models.convnext_tiny(weights=weights)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model.to(device)

# -----------------------------
# Training
# -----------------------------
def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, ema, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0

    # Freeze first 5 epochs
    freeze = epoch < 1

    for name, param in model.named_parameters():
        if "features" in name:
            param.requires_grad = not freeze

    for x, y in tqdm(loader, desc=f"Training Epoch {epoch}"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        mixed_x, y_a, y_b, lam = mixup_cutmix(x, y)

        with torch.amp.autocast('cuda'):
            y_pred = model(mixed_x)
            loss = lam * criterion(y_pred, y_a) + (1 - lam) * criterion(y_pred, y_b)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        ema.update(model)

        total_loss += loss.item() * x.size(0)

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

    epochs = 80
    batch_size = 64

    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        train_dir, test_dir, batch_size=batch_size
    )

    model = build_model(num_classes=100)
    ema = EMA(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=0.1)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-4, 1e-3],
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )

    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0
    ckpt_path = "checkpoint.pt"

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, ema, epoch)
        val_loss, val_acc = evaluate(ema.shadow, val_loader, criterion)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": ema.shadow.state_dict(),
                "class_to_idx": class_to_idx
            }, ckpt_path)

    print("Training complete. Best Val Acc:", best_acc)

if __name__ == "__main__":
    main()
