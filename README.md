# Aerial Image Classification: Custom CNN vs Transfer Learning

Classifying aerial images into **21 land-use classes** (airplane, harbor, freeway, forest...)
on the UC Merced dataset, only 2,100 images. The question: on a small dataset, is it better to
train a CNN from scratch or reuse a network pre-trained on ImageNet?

## Results (test set)

| Model | Parameters | Test accuracy |
| :-- | :-- | :-- |
| Custom CNN, trained from scratch | 4.3M | 75% |
| **VGG16 transfer learning** (frozen ImageNet base + new classifier) | 14.7M base | **88.3%** |

Transfer learning wins by **+13 points**: features learned on ImageNet transfer well to aerial
images, while the custom network does not have enough data to learn them from zero.

## What I did

- **Data pipeline** with augmentation (rotation, shift, zoom, flips) on 256×256 images
- **Custom CNN**: 5 conv blocks (filters 8 → 128) + dense classifier
- **Transfer learning**: VGG16 with frozen weights + new dense head
- **Evaluation**: test accuracy, confusion matrix, per-class ROC curves and AUC (one-vs-rest)
- Early stopping and checkpoints to avoid overfitting

## Stack

Python · TensorFlow / Keras · scikit-learn · Matplotlib · Seaborn

## Files

- [`CNN_Classification+_Transfer_Learning.ipynb`](CNN_Classification+_Transfer_Learning.ipynb):
  full pipeline, both models, evaluation
