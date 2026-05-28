import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Dataset + Augmentations
# -----------------------------
def get_dataloaders(train_dir, test_dir, batch_size=32, num_workers=2):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),  
        transforms.RandAugment(num_ops=3, magnitude=12),       
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),                  
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),       
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.2)),    
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_dataset = datasets.ImageFolder(root=train_dir)
    class_to_idx = full_dataset.class_to_idx
    num_classes = len(class_to_idx)
    print(f"Found {num_classes} classes, {len(full_dataset)} total images")

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
            self.files = sorted([f for f in os.listdir(root)
                                 if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        def __len__(self): return len(self.files)
        def __getitem__(self, idx):
            img_path = os.path.join(self.root, self.files[idx])
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, self.files[idx]

    test_set = TestDataset(test_dir, transform=val_tf)

    # OPT: Smaller batch size -> more gradient steps -> better generalization on tiny data
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=USE_CUDA)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=USE_CUDA)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=USE_CUDA)

    return train_loader, val_loader, test_loader, class_to_idx

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
            for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def update_buffers(self, model):
        with torch.no_grad():
            for ema_buf, buf in zip(self.shadow.buffers(), model.buffers()):
                ema_buf.data.copy_(buf.data)

# -----------------------------
# Model
# -----------------------------
def build_model(num_classes=100, dropout=0.4):
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    model = models.convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features

    # OPT: Dropout before final linear layer
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    return model.to(device)

# -----------------------------
# Stochastic Depth rate setter
# -----------------------------
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

    freeze = epoch < 2
    for name, param in model.named_parameters():
        if "features" in name:
            param.requires_grad = not freeze

    for x, y in tqdm(loader, desc=f"Train Epoch {epoch+1}"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        mixed_x, y_a, y_b, lam = mixup_cutmix(x, y)

        with torch.amp.autocast(AUTOCAST_DEVICE, enabled=USE_CUDA):
            y_pred = model(mixed_x)
            loss = lam * criterion(y_pred, y_a) + (1 - lam) * criterion(y_pred, y_b)

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
                clean_preds = model(x).argmax(dim=1)
            correct += (clean_preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for x, y in tqdm(loader, desc="Validation", leave=False):
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(AUTOCAST_DEVICE, enabled=USE_CUDA):
            y_pred = model(x)
            loss = criterion(y_pred, y)
        total_loss += loss.item() * x.size(0)
        correct += (y_pred.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate_tta(model, val_dataset, n_aug=6):
    model.eval()
    correct, total = 0, 0

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    tta_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    for idx in range(len(val_dataset)):
        try:
            real_idx = val_dataset.indices[idx]
            img_path, label = val_dataset.dataset.samples[real_idx]
            img_pil = Image.open(img_path).convert("RGB")

            logits_sum = None
            for _ in range(n_aug):
                t = tta_tf(img_pil).unsqueeze(0).to(device)
                with torch.amp.autocast(AUTOCAST_DEVICE, enabled=USE_CUDA):
                    logits = model(t)
                logits_sum = logits if logits_sum is None else logits_sum + logits

            pred = logits_sum.argmax(dim=1).item()
            correct += int(pred == label)
            total += 1
        except Exception:
            continue

    return correct / total if total > 0 else 0.0

# -----------------------------
# Main
# -----------------------------
def main():
    train_dir = "/content/drive/MyDrive/train"
    test_dir  = "/content/drive/MyDrive/test"

    epochs     = 120  
    batch_size = 32    

    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        train_dir, test_dir, batch_size=batch_size
    )
    num_classes = len(class_to_idx)

    model = build_model(num_classes=num_classes, dropout=0.4)

    set_stochastic_depth(model, drop_prob=0.2)

    ema = EMA(model, decay=0.995)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(),    'lr': 5e-5},  
        {'params': model.classifier.parameters(),  'lr': 5e-4},  
    ], weight_decay=0.2)   

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[5e-5, 5e-4],
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,       
        div_factor=10,        
        final_div_factor=100   #
    )

    scaler = torch.amp.GradScaler(AUTOCAST_DEVICE, enabled=USE_CUDA)

    best_acc     = 0
    best_tta_acc = 0
    ckpt_path         = "checkpoint.pt"
    ckpt_path_tta     = "checkpoint_tta_best.pt"

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, ema, epoch
        )
        val_loss, val_acc = evaluate(ema.shadow, val_loader, criterion)

        # OPT: TTA eval every 5 epochs (slow but more accurate signal)
        tta_str = ""
        if (epoch + 1) % 5 == 0:
            tta_acc = evaluate_tta(ema.shadow, val_loader.dataset, n_aug=6)
            tta_str = f" | TTA: {tta_acc:.4f}"
            if tta_acc > best_tta_acc:
                best_tta_acc = tta_acc
                torch.save({"model_state_dict": ema.shadow.state_dict(),
                            "class_to_idx": class_to_idx}, ckpt_path_tta)
                tta_str += " *"

        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}"
              + tta_str)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state_dict": ema.shadow.state_dict(),
                        "class_to_idx": class_to_idx}, ckpt_path)
            print(f"  -> Saved (val acc: {best_acc:.4f})")

    print(f"\nDone. Best Val Acc: {best_acc:.4f} | Best TTA Acc: {best_tta_acc:.4f}")

if __name__ == "__main__":
    main()