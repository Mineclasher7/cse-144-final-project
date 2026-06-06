# How to Run Training and Inference

This repository provides two scripts:

- `train.py` — trains a classifier head on top of SigLIP‑2 ViT and saves `siglip2_classifier.pt`
- `inference.py` — loads the trained model and generates `submission.csv`

Follow the steps below to run each stage.

---

## Training


With the dataset uploaded into google drive as follows:

```text
MyDrive/
    train/                 # training images in ImageFolder format
        class_0/
        class_1/
        ...
    test/                  # unlabeled test images
        0.jpg
        1.jpg
        ...
```

To start training, run: 

```bash
python train.py
```
The script will:
- Load the dataset and split it into 80% train / 20% validation
- Extract embeddings using SigLIP‑2 ViT (google/siglip2-base-patch16-224)
- Train a lightweight MLP classifier head
- Track validation accuracy each epoch
- Save the best-performing model to:

```bash
siglip2_classifier.pt
```

Paths for the dataset are defined at the top of `train.py` and can be modified if needed.

---

## Inference

After `siglip2_classifier.pt` has been created, run: 

```bash
python inference.py
```

The script will:

- Load the trained model
- Apply Test-Time Augmentation (center crop, flip, 5-crop)
- Predict labels for all test images
- Write predictions into `submission.csv`

The output file will follow the required format:

```text
ID,Label
0.jpg,42
1.jpg,17
...
```

This file is ready for submission.
