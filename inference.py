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
# Dataset
# -----------------------------
class TestDataset(torch.utils.data.Dataset):
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

# -----------------------------
# Model
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
# Inference
# -----------------------------
@torch.no_grad()
def predict(model, loader, num_augs=2):
    results = []

    for x, filenames in tqdm(loader, desc="Predicting"):
        x = x.to(device)

        probs = F.softmax(model(x), dim=1)

        if num_augs > 1:
            x_flip = torch.flip(x, dims=[3])
            probs += F.softmax(model(x_flip), dim=1)

        probs /= num_augs
        preds = probs.argmax(dim=1)

        results.extend(zip(filenames, preds.cpu().numpy()))

    return results

# -----------------------------
# Main
# -----------------------------
def main():
    test_dir = "/content/drive/MyDrive/test"
    ckpt_path = "checkpoint.pt"
    submission_path = "/content/sample_submission.csv"

    tf = transforms.Compose([
        transforms.Resize(236),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    test_set = TestDataset(test_dir, transform=tf)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    model, class_to_idx = load_model(num_classes=100, ckpt_path=ckpt_path)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    results = predict(model, test_loader)

    df = pd.read_csv(submission_path)
    pred_map = {fname: idx_to_class[pred] for fname, pred in results}

    df["Label"] = df["ID"].map(pred_map)
    df.to_csv(submission_path, index=False)

    print("Submission saved to:", submission_path)

if __name__ == "__main__":
    main()
