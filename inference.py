import os
import torch
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from transformers import SiglipVisionModel, AutoImageProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Inference on:", device)

# Load backbone
processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
backbone = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224").to(device)
backbone.eval()

# Load classifier
ckpt = torch.load("siglip_classifier.pt", map_location=device)
class_names = ckpt["class_names"]

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

classifier = Classifier(backbone.config.hidden_size, len(class_names)).to(device)
classifier.load_state_dict(ckpt["classifier"])
classifier.eval()

# Embedding helper
@torch.no_grad()
def get_emb(img):
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = backbone(**inputs)
    return out.pooler_output

# Predict
test_dir = "/content/drive/MyDrive/test"
files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(("jpg","png","jpeg"))])

results = []
for fname in tqdm(files, desc="Predicting"):
    img = Image.open(os.path.join(test_dir, fname)).convert("RGB")
    emb = get_emb(img)
    logits = classifier(emb)
    pred = logits.argmax(1).item()
    results.append((fname, class_names[pred]))

df = pd.DataFrame(results, columns=["ID", "Label"])
df.to_csv("submission.csv", index=False)
print("Saved submission.csv")
