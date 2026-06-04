import os
import torch
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from transformers import SiglipVisionModel, AutoImageProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Inference on:", device)

# -----------------------------
# Load checkpoint
# -----------------------------
ckpt = torch.load("siglip2_classifier.pt", map_location=device)
class_names = ckpt["class_names"]
model_name = ckpt.get("model_name", "google/siglip2-base-patch16-224")

processor = AutoImageProcessor.from_pretrained(model_name)
backbone = SiglipVisionModel.from_pretrained(model_name).to(device)
backbone.eval()

embed_dim = backbone.config.hidden_size

class Classifier(torch.nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)

classifier = Classifier(embed_dim, len(class_names)).to(device)
classifier.load_state_dict(ckpt["classifier_state_dict"])
classifier.eval()

@torch.no_grad()
def get_embedding(img):
    inputs = processor(images=img, return_tensors="pt").to(device)
    outputs = backbone(pixel_values=inputs["pixel_values"])
    return outputs.pooler_output  # (1, embed_dim)

def main():
    test_dir = "/content/drive/MyDrive/test"
    files = sorted(
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    results = []
    for fname in tqdm(files, desc="Predicting"):
        img = Image.open(os.path.join(test_dir, fname)).convert("RGB")
        emb = get_embedding(img)
        logits = classifier(emb)
        pred = logits.argmax(1).item()
        label = class_names[pred]
        results.append((fname, label))

    df = pd.DataFrame(results, columns=["ID", "Label"])
    df.to_csv("submission.csv", index=False)
    print("Saved submission.csv")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    main()
