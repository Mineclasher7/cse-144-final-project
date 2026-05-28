import os
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm.auto import tqdm
from torchvision import transforms, models
from PIL import Image

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# -----------------------------
# Dataset (FIXED)
# -----------------------------
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = sorted([
            f for f in os.listdir(root)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.files[idx]

# -----------------------------
# Model Loader (EMA‑compatible)
# -----------------------------
def load_model(num_classes, ckpt_path):
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(in_features, num_classes)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint["class_to_idx"]

# -----------------------------
# TTA Inference
# -----------------------------
@torch.no_grad()
def predict(model, dataset, batch_size=32):
    results = []

    base_tf = transforms.Compose([
        transforms.Resize(236),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    crop_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.FiveCrop(224)
    ])

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for img, filename in tqdm(loader, desc="Predicting"):
        img = img[0]  # remove batch dimension

        x1 = base_tf(img).unsqueeze(0).to(device)

        x2 = torch.flip(x1, dims=[3])

        crops = crop_tf(img)  
        crops = torch.stack([transforms.ToTensor()(c) for c in crops])
        crops = transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )(crops)
        crops = crops.to(device)

        # ----- Run model -----
        logits = []
        logits.append(model(x1))
        logits.append(model(x2))
        logits.append(model(crops))  

        # ----- Average softmax -----
        probs = torch.cat([
            F.softmax(logits[0], dim=1),
            F.softmax(logits[1], dim=1),
            F.softmax(logits[2], dim=1)
        ], dim=0)

        probs = probs.mean(dim=0)
        pred = probs.argmax().item()

        results.append((filename[0], pred))

    return results

# -----------------------------
# Main
# -----------------------------
def main():
    test_dir = "/content/drive/MyDrive/test"
    ckpt_path = "checkpoint.pt"
    submission_path = "/content/sample_submission.csv"

    base_dataset_tf = transforms.Compose([
        transforms.Resize(236),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    dataset = TestDataset(test_dir, transform=base_dataset_tf)

    model, class_to_idx = load_model(num_classes=100, ckpt_path=ckpt_path)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    results = predict(model, dataset)

    df = pd.read_csv(submission_path)
    pred_map = {fname: idx_to_class[pred] for fname, pred in results}

    df["Label"] = df["ID"].map(pred_map)
    df.to_csv(submission_path, index=False)

    print("Submission saved to:", submission_path)

if __name__ == "__main__":
    main()
