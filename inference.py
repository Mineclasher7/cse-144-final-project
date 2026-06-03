import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm.auto import tqdm
from torchvision import transforms, models
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
AUTOCAST_DEVICE = "cuda" if USE_CUDA else "cpu"
print(f"Inference on: {device}")

# -----------------------------
# Model Loaders — must match train.py architecture exactly
# -----------------------------
def load_efficientnet(path, num_classes=100):
    m = models.efficientnet_v2_s(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.0),
        nn.Linear(in_f, num_classes)
    )
    ckpt = torch.load(path, map_location=device)
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(device).eval(), ckpt["class_to_idx"]

def load_swin(path, num_classes=100):
    m = models.swin_t(weights=None)
    in_f = m.head.in_features
    m.head = nn.Sequential(
        nn.Dropout(p=0.0),
        nn.Linear(in_f, num_classes)
    )
    ckpt = torch.load(path, map_location=device)
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(device).eval(), ckpt["class_to_idx"]

# -----------------------------
# TTA — 11 views per image
# -----------------------------
@torch.no_grad()
def tta(model, img_pil, img_size=224, resize_size=256):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    norm = transforms.Normalize(mean, std)

    base = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(), norm,
    ])
    hflip = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(), norm,
    ])
    five_crop = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.FiveCrop(img_size),
        transforms.Lambda(lambda crops: torch.stack([
            norm(transforms.ToTensor()(c)) for c in crops
        ])),
    ])
    rand_crop = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm,
    ])

    views = [
        base(img_pil).unsqueeze(0),       # center crop
        hflip(img_pil).unsqueeze(0),      # h-flip
        five_crop(img_pil),               # 5 crops (4 corners + center)
        rand_crop(img_pil).unsqueeze(0),  # random crop 1
        rand_crop(img_pil).unsqueeze(0),  # random crop 2
        rand_crop(img_pil).unsqueeze(0),  # random crop 3
        rand_crop(img_pil).unsqueeze(0),  # random crop 4
    ]

    batch = torch.cat(views, dim=0).to(device)   # (11, C, H, W)
    with torch.amp.autocast(AUTOCAST_DEVICE, enabled=USE_CUDA):
        logits = model(batch)
    return F.softmax(logits, dim=1).mean(dim=0)  # (num_classes,)

# -----------------------------
# Main
# -----------------------------
def main():
    test_dir        = "/content/drive/MyDrive/test"
    submission_path = "/content/sample_submission.csv"

    print("Loading models...")
    eff,  class_to_idx = load_efficientnet("ckpt_efficientnet.pt")
    swin, _            = load_swin("ckpt_swin.pt")
    idx_to_class       = {v: k for k, v in class_to_idx.items()}

    files = sorted([f for f in os.listdir(test_dir)
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    results = []
    for fname in tqdm(files, desc="Ensemble TTA"):
        img = Image.open(os.path.join(test_dir, fname)).convert("RGB")

        p_eff  = tta(eff,  img, img_size=224, resize_size=256)
        p_swin = tta(swin, img, img_size=224, resize_size=256)

        # Equal weight ensemble — both models trained equally
        final = 0.5 * p_eff + 0.5 * p_swin
        pred  = final.argmax().item()
        results.append((fname, idx_to_class[pred]))

    df = pd.read_csv(submission_path)
    pred_map = {fname: label for fname, label in results}
    df["Label"] = df["ID"].map(pred_map)

    missing = df["Label"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} missing predictions")

    out_path = "/content/submission.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
