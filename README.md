# 🌍 CNN Multi-Class Classification: UC Merced Land Use Dataset

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

This project implements a **Convolutional Neural Network (CNN)** to classify aerial imagery into 21 distinct land-use categories using the **UC Merced Land Use Dataset** [file:10]. The model is designed to recognize complex visual patterns from satellite data, distinguishing between classes such as "agricultural," "harbor," and "dense residential."

## 📌 Project Overview
Remote sensing scene classification is a key task in earth observation. This repository provides an end-to-end deep learning pipeline—from raw data extraction to model evaluation—demonstrating how to process and classify optical remote sensing images [file:10].

**Goal:** Accurately predict the land-use category of 256x256 RGB images across 21 classes.

## 📂 The Dataset
- **Source:** UC Merced Land Use Dataset
- **Classes:** 21 categories including:
  `agricultural`, `airplane`, `baseballdiamond`, `beach`, `buildings`, `chaparral`, `denseresidential`, `forest`, `freeway`, `golfcourse`, `harbor`, `intersection`, `mediumresidential`, `mobilehomepark`, `overpass`, `parkinglot`, `river`, `runway`, `sparseresidential`, `storagetanks`, `tenniscourt`.
- **Preprocessing:**
  - **Rescaling:** Pixel values normalized to [0, 1] range.
  - **Splitting:** Data divided into Training, Validation, and Test sets using `ImageDataGenerator` [file:10].

## 🧠 Model Architecture
The custom CNN architecture is built using the TensorFlow/Keras Functional API [file:10]:

1.  **Input:** `(256, 256, 3)` RGB images.
2.  **Feature Extraction:**
    -   5 Convolutional Blocks (`Conv2D` + `ReLU` + `MaxPooling2D`).
    -   Filter depths increase progressively to capture hierarchical features (edges → textures → objects).
3.  **Classification Head:**
    -   `Flatten` layer.
    -   **Dense Layer:** 512 units with `ReLU` activation.
    -   **Output Layer:** 21 neurons with `Softmax` activation for multi-class probability.
4.  **Optimization:**
    -   **Optimizer:** Adam (`lr=1e-4`).
    -   **Loss Function:** Categorical Crossentropy.

## 📊 Performance
-   **Training Accuracy:** Achieves >98% on the training set.
-   **Validation Accuracy:** ~60-65% (indicating potential for further regularization techniques like Dropout or Data Augmentation).
-   **Evaluation:** Includes Confusion Matrix visualization to identify misclassified categories (e.g., distinguishing between different densities of residential areas) [file:10].

## 🛠️ Tech Stack
-   **Deep Learning:** TensorFlow, Keras
-   **Data Processing:** NumPy, Pandas
-   **Visualization:** Matplotlib, Seaborn
-   **Environment:** Jupyter Notebook / Google Colab

## 🚀 Usage

### 1. Setup
Clone the repo and install dependencies:
```bash
git clone https://github.com/yourusername/uc-merced-cnn.git
pip install tensorflow numpy matplotlib pandas
