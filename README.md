# How to Run Training and Inference

This repository provides two scripts:

- `train.py` — trains the ConvNeXt-Tiny model and saves `checkpoint.pt`
- `inference.py` — loads the trained model and generates `sample_submission.csv`

Follow the steps below to run each stage.

---

## Training

Make sure your directory structure looks like:

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
    sample_submission.csv  # provided template
```

To start training, run: 

```bash
python train.py
```

The script will:

- Load the training and validation sets
- Train ConvNeXt-Tiny with modern augmentations (RandAugment, Mixup, CutMix, RandomErasing)
- Use OneCycleLR and EMA for improved performance
- Save the best model to: 

```bash
checkpoint.pt
```

Paths for the dataset are defined at the top of `train.py` and can be modified if needed.

---

## Inference

After `checkpoint.pt` has been created, run: 

```bash
python inference.py
```

The script will:

- Load the trained model
- Apply Test-Time Augmentation (center crop, flip, 5-crop)
- Predict labels for all test images
- Write predictions into `sample_submission.csv`

The output file will follow the required format:

```text
ID,Label
0.jpg,42
1.jpg,17
...
```

This file is ready for submission.
