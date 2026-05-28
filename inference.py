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
    def __init__(self, root, transform=None):
        self.root      = root
        self.transform = transform
        self.files     = sorted([
            f for f in os.listdir(root)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.files[idx])
        image    = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.files[idx]

# -----------------------------
# Model Loaders
# Each knows how to rebuild its own architecture
# -----------------------------
def load_convnext(ckpt_path, num_classes):
    model = models.convnext_tiny(weights=None)
    in_feat = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.0),          # dropout off at inference
        nn.Linear(in_feat, num_classes)
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt["class_to_idx"]

def load_efficientnet(ckpt_path, num_classes):
    model = models.efficientnet_v2_s(weights=None)
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.0),
        nn.Linear(in_feat, num_classes)
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt["class_to_idx"]

def load_swin(ckpt_path, num_classes):
    model = models.swin_t(weights=None)
    in_feat = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=0.0),
        nn.Linear(in_feat, num_classes)
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt["class_to_idx"]

# -----------------------------
# TTA transforms
# -----------------------------
def get_tta_transforms(img_size=224, resize_size=256):
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
    # 5 crops: 4 corners + center
    five_crop_tf = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.FiveCrop(img_size),
        transforms.Lambda(lambda crops: torch.stack([
            norm(transforms.ToTensor()(c)) for c in crops
        ])),
    ])
    # Random crop augmentations for extra TTA views
    rand_crop = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm,
    ])
    return base, hflip, five_crop_tf, rand_crop

# -----------------------------
# Single-model TTA inference
# Returns (N, num_classes) averaged softmax probabilities
# -----------------------------
@torch.no_grad()
def predict_single(model, test_root, img_size=224, resize_size=256, n_rand=4):
    base_tf, hflip_tf, five_crop_tf, rand_tf = get_tta_transforms(img_size, resize_size)

    # We iterate file by file to apply heterogeneous TTA
    files = sorted([f for f in os.listdir(test_root)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    all_probs  = []
    all_files  = []

    for fname in tqdm(files, desc=f"  TTA inference"):
        img_pil = Image.open(os.path.join(test_root, fname)).convert("RGB")

        views = []

        # View 1: center crop
        views.append(base_tf(img_pil).unsqueeze(0))

        # View 2: horizontal flip
        views.append(hflip_tf(img_pil).unsqueeze(0))

        # Views 3-7: five crops (corners + center)
        five = five_crop_tf(img_pil)   # shape (5, C, H, W)
        views.append(five)

        # Views 8+: random crops for extra diversity
        for _ in range(n_rand):
            views.append(rand_tf(img_pil).unsqueeze(0))

        batch = torch.cat(views, dim=0).to(device)  # (n_views, C, H, W)

        with torch.amp.autocast(AUTOCAST_DEVICE, enabled=USE_CUDA):
            logits = model(batch)                    # (n_views, num_classes)

        probs = F.softmax(logits, dim=1).mean(dim=0) # (num_classes,)
        all_probs.append(probs.cpu())
        all_files.append(fname)

    return torch.stack(all_probs), all_files  # (N, num_classes), [filenames]

# -----------------------------
# Ensemble inference
# Averages softmax probs across all models then argmax
# -----------------------------
def ensemble_predict(models_list, test_root, class_to_idx,
                     img_sizes, resize_sizes, n_rand=4):
    """
    models_list : list of (model, img_size, resize_size)
    Returns dict {filename: predicted_class_label}
    """
    all_model_probs = []
    file_order      = None

    for model, img_size, resize_size in models_list:
        probs, files = predict_single(model, test_root, img_size, resize_size, n_rand)
        all_model_probs.append(probs)
        if file_order is None:
            file_order = files

    # Average across models — shape (N, num_classes)
    ensemble_probs = torch.stack(all_model_probs).mean(dim=0)
    preds          = ensemble_probs.argmax(dim=1).tolist()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return {fname: idx_to_class[pred] for fname, pred in zip(file_order, preds)}

# -----------------------------
# Main
# -----------------------------
def main():
    test_dir         = "/content/drive/MyDrive/test"
    submission_path  = "/content/sample_submission.csv"

    # Paths to the FINAL checkpoints (trained on 100% of data)
    ckpt_convnext  = "ckpt_convnext.pt"
    ckpt_effnet    = "ckpt_effnetv2s.pt"
    ckpt_swin      = "ckpt_swint.pt"

    num_classes = 100

    print("Loading models...")
    convnext, class_to_idx = load_convnext(ckpt_convnext, num_classes)
    effnet,   _            = load_efficientnet(ckpt_effnet, num_classes)
    swin,     _            = load_swin(ckpt_swin, num_classes)

    # (model, img_size, resize_size)
    # EfficientNetV2-S works best at 384px but 300 is fine on Colab
    models_list = [
        (convnext, 224, 256),
        (effnet,   300, 320),
        (swin,     224, 256),
    ]

    print("\nRunning ensemble TTA inference...")
    print(f"  Models: ConvNeXt-Tiny | EfficientNetV2-S | Swin-T")
    print(f"  TTA views per model: 2 + 5 crops + 4 random = 11 views")

    pred_map = ensemble_predict(
        models_list, test_dir, class_to_idx,
        img_sizes=None, resize_sizes=None, n_rand=4
    )

    # Build submission
    df = pd.read_csv(submission_path)
    df["Label"] = df["ID"].map(pred_map)

    missing = df["Label"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} predictions missing — check file name matching")

    out_path = "/content/submission.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSubmission saved to: {out_path}")
    print(f"Preview:\n{df.head(10).to_string(index=False)}")

if __name__ == "__main__":
    main()
