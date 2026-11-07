# Aerial Scene Classification: Custom CNN vs VGG16 Transfer Learning

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-UC%20Merced%20Land%20Use-green)
![Task](https://img.shields.io/badge/Task-21--class%20classification-blue)

## Overview

Multi-class classification of aerial images into **21 land-use classes** on the UC Merced
dataset (2,100 images). Benchmark between a CNN trained from scratch and **transfer learning**
from a VGG16 backbone pre-trained on ImageNet, in a small-data regime.

## Key Results

| Model | Trainable params | Test accuracy |
| :-- | --: | :-: |
| Custom CNN (from scratch) | 4.3M | 75.0% |
| **VGG16 transfer learning** (frozen backbone + dense head) | head only | **88.3%** |

Transfer learning improves test accuracy by **+13.3 points** over the from-scratch baseline.

## Model Architectures

**Custom CNN**
```
Input 256×256×3
 └─ 5 × [Conv2D 3×3 + ReLU ─► MaxPooling 2×2]     filters: 8 → 128
 └─ Flatten ─► Dense(512, ReLU) ─► Dense(21, softmax)
```

**Transfer learning**
```
Input 256×256×3
 └─ VGG16 (ImageNet weights, include_top=False, frozen)
 └─ Flatten ─► Dense(512, ReLU) ─► Dropout(0.5) ─► Dense(21, softmax)
```

## Training Setup

| | |
| :-- | :-- |
| Data split | 1,470 train · 315 validation · 315 test |
| Augmentation | rotation ±10°, width/height shift, zoom 0.3, horizontal and vertical flip |
| Optimizer | Adam, lr = 1e-4 |
| Loss | Categorical cross-entropy |
| Callbacks | Early stopping (patience 5), model checkpointing |

## Evaluation

- Test-set accuracy and per-class classification report
- Confusion matrix (21 × 21)
- One-vs-rest **ROC curves and AUC** for each class

## Tech Stack

Python · TensorFlow / Keras · scikit-learn · NumPy · Matplotlib · Seaborn

## Repository Structure

```
├── CNN_Classification+_Transfer_Learning.ipynb   # Pipeline, both models, training, evaluation
└── README.md
```

## Run

Open the notebook in Jupyter or Google Colab (GPU recommended) and run all cells.
