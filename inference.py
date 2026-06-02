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
# Load Models
# -----------------------------
def load_convnext(path, num_classes):
    m = models.convnext_tiny(weights=None)
    in_f = m.classifier[2].in_features
    m.classifier[2] = nn.Linear(in_f, num_classes)
    ckpt = torch.load(path, map_location=device)
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(device).eval(), ckpt["class_to_idx"]

def load_efficientnet(path, num_classes):
    m = models.efficientnet_v2_s(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, num_classes)
    ckpt = torch.load(path, map_location=device)
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(device).eval()

def load_swin(path, num_classes):
    m = models.swin_t(weights=None)
    in_f = m.head.in_features
    m.head = nn.Linear(in_f, num_classes)
    ckpt = torch.load(path, map_location=device)
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(device).eval()

# -----------------------------
# TTA
# -----------------------------
@torch.no_grad()
def tta(model, img):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    norm = transforms.Normalize(mean, std)

    base = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), norm
    ])

    flip = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(), norm
    ])

    crops = transforms.Compose([
        transforms.Resize(256),
        transforms.FiveCrop(224),
        transforms.Lambda(lambda c: torch.stack([norm(transforms.ToTensor()(x)) for x in c]))
    ])

    views = []
    views.append(base(img).unsqueeze(0))
    views.append(flip(img).unsqueeze(0))
    views.append(crops(img))

    batch = torch.cat(views, dim=0).to(device)

    with torch.amp.autocast(AUTOCAST_DEVICE, enabled=USE_CUDA):
        logits = model(batch)

    return F.softmax(logits, dim=1).mean(dim=0)

# -----------------------------
# Main
# -----------------------------
def main():
    test_dir = "/content/drive/MyDrive/test"

    # Load ConvNeXt FIRST and use its mapping
    conv, class_to_idx = load_convnext("ckpt_convnext.pt", 100)

    # Load the other two models (ignore their mappings)
    eff  = load_efficientnet("ckpt_efficientnet.pt", 100)
    swin = load_swin("ckpt_swin.pt", 100)

    # Build idx_to_class from ConvNeXt ONLY
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    results = []

    for fname in tqdm(files):
        img = Image.open(os.path.join(test_dir, fname)).convert("RGB")

        p1 = tta(conv, img)
        p2 = tta(eff, img)
        p3 = tta(swin, img)

        # Weighted ensemble
        final = 0.55*p1 + 0.25*p2 + 0.20*p3
        pred = final.argmax().item()

        results.append((fname, idx_to_class[pred]))

    df = pd.DataFrame(results, columns=["ID", "Label"])
    df.to_csv("submission.csv", index=False)
    print("Saved submission.csv")

if __name__ == "__main__":
    main()
