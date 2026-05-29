import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm.auto import tqdm
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# -----------------------------
# Device
# -----------------------------
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
USE_CUDA = torch.cuda.is_available()
AUTOCAST_DEVICE = 'cuda' if USE_CUDA else 'cpu'
print(f"Inference on: {device}")

# -----------------------------
# Test Dataset
# -----------------------------
class TestDataset(Dataset):
    def __init__(self, root):
        self.root  = root
        self.files = sorted([
            f for f in os.listdir(root)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        return self.files[idx]  
# -----------------------------
# Model Loader — matches train.py exactly
# -----------------------------
def load_model(ckpt_path, num_classes=100):
    model   = models.swin_t(weights=None)
    in_feat = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=0.0),          
        nn.Linear(in_feat, num_classes)
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt["class_to_idx"]

# -----------------------------
# TTA Inference — 11 views per image
# -----------------------------
@torch.no_grad()
def predict(model, test_root, num_classes=100, n_rand=4):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    norm = transforms.Normalize(mean, std)

    base_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), norm,
    ])
    hflip_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(), norm,
    ])
    five_crop_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.FiveCrop(224),
        transforms.Lambda(lambda crops: torch.stack([
            norm(transforms.ToTensor()(c)) for c in crops
        ])),
    ])
    rand_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm,
    ])

    files   = sorted([f for f in os.listdir(test_root)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    results = []

    for fname in tqdm(files, desc="TTA Inference"):
        img_pil = Image.open(os.path.join(test_root, fname)).convert("RGB")

        views = []
        views.append(base_tf(img_pil).unsqueeze(0))    
        views.append(hflip_tf(img_pil).unsqueeze(0))       
        views.append(five_crop_tf(img_pil))                 
        for _ in range(n_rand):
            views.append(rand_tf(img_pil).unsqueeze(0))     

        batch = torch.cat(views, dim=0).to(device)          

        with torch.amp.autocast(AUTOCAST_DEVICE, enabled=USE_CUDA):
            logits = model(batch)                            

        probs = F.softmax(logits, dim=1).mean(dim=0)         
        results.append((fname, probs.argmax().item()))

    return results

# -----------------------------
# Main
# -----------------------------
def main():
    test_dir        = "/content/drive/MyDrive/test"
    ckpt_path       = "ckpt_swint.pt"
    submission_path = "/content/sample_submission.csv"

    model, class_to_idx = load_model(ckpt_path, num_classes=100)
    idx_to_class        = {v: k for k, v in class_to_idx.items()}

    results  = predict(model, test_dir)
    pred_map = {fname: idx_to_class[pred] for fname, pred in results}

    df = pd.read_csv(submission_path)
    df["Label"] = df["ID"].map(pred_map)

    missing = df["Label"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} predictions missing")

    out_path = "/content/submission.csv"
    df.to_csv(out_path, index=False)
    print(f"Submission saved to: {out_path}")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
