# 🌍 Aerial Land Use Classification (Custom CNN vs. Transfer Learning)

![Framework](https://img.shields.io/badge/Framework-TensorFlow_2.x-orange)
![Technique](https://img.shields.io/badge/Technique-Transfer_Learning-blue)
![Dataset](https://img.shields.io/badge/Dataset-UC_Merced-green)

This project tackles the multi-class classification problem using the **UC Merced Land Use Dataset**. It compares the performance of a custom-built Convolutional Neural Network (CNN) against a Transfer Learning approach using **VGG16** to classify satellite imagery into 21 distinct categories. [file:14]

## 🎯 Project Goals
*   **Data Pipeline:** Build a robust `ImageDataGenerator` pipeline with augmentation (rotations, zooms, flips) to handle 256x256 aerial images.
*   **Custom Architecture:** Design a VGG-like CNN from scratch with increasing filter depth (8 $\to$ 128) to understand feature extraction.
*   **Transfer Learning:** Implement VGG16 (pre-trained on ImageNet) using both **Feature Extraction** (frozen base) and **Fine-Tuning** strategies.

## 📂 Dataset
*   **Source:** UC Merced Land Use Dataset.
*   **Classes (21):** Agricultural, Airplane, Baseball Diamond, Beach, Buildings, Chaparral, Dense Residential, Forest, Freeway, Golf Course, Harbor, Intersection, Medium Residential, Mobile Home Park, Overpass, Parking Lot, River, Runway, Sparse Residential, Storage Tanks, Tennis Court.
*   **Preprocessing:** Images resized to `256x256`, normalized to `[0,1]`, and augmented. [file:14]

## 🧠 Architectures

### 1. Custom CNN
A sequential model built with 5 Convolutional blocks:
*   **Layers:** `Conv2D` (3x3, padding='same') $\to$ `ReLU` $\to$ `MaxPooling2D` (2x2).
*   **Depth:** Filters increase from 8 up to 128.
*   **Head:** `Flatten` $\to$ `Dense` (512) $\to$ `Dense` (21, Softmax).

### 2. Transfer Learning (VGG16)
*   **Base:** VGG16 (ImageNet weights, top layers excluded).
*   **Adaptation:** Added `GlobalAveragePooling2D` and a custom Dense classifier.
*   **Strategy:** Initially froze all base layers, then fine-tuned the top convolutional block to adapt to aerial features.

## 🛠️ Tech Stack
*   **Core:** Python 3, TensorFlow 2.x, Keras.
*   **Visualization:** Matplotlib (prediction grids), TensorBoard (loss/accuracy curves).
*   **Utils:** `ImageDataGenerator`, `ModelCheckpoint`, `EarlyStopping`.

## 📊 Results
The project demonstrates that Transfer Learning significantly outperforms the custom CNN on this small dataset (2100 images total), converging faster and achieving higher validation accuracy by leveraging pre-learned visual patterns.

## 👨‍💻 Author
**Alex Miguel Franco Martínez**
