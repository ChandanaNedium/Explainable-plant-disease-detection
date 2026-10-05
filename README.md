# Explainable Plant Disease Detection Using CNN

A deep learning-based tomato leaf disease classification system built using PyTorch and the PlantVillage dataset. The project combines CNN-based classification with ANN comparison, Grad-CAM explainability, confidence estimation, image validation, and a Streamlit web interface.

---

## 🌿 Project Overview

Plant disease detection from leaf images can help identify diseases at an early stage. However, a model that only provides a prediction does not explain why that prediction was made.

This project develops a CNN-based tomato leaf disease classifier and uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize the regions that influenced the model's prediction.

The project also compares CNN performance with a basic ANN and evaluates the model under both controlled dataset conditions and real-world uploaded-image conditions.

---

## 🎯 Objectives

- Build a CNN model for tomato leaf disease classification.
- Compare CNN performance with a basic ANN.
- Evaluate the model using accuracy, precision, recall, and F1-score.
- Analyze classification errors using a confusion matrix.
- Use Grad-CAM for visual model explainability.
- Calculate prediction confidence using softmax probabilities.
- Validate user-uploaded images before classification.
- Analyze real-world limitations such as domain shift and background bias.

---

## 🦠 Disease Classes

The model classifies four tomato leaf categories:

1. Bacterial Spot
2. Early Blight
3. Late Blight
4. Healthy

---

## 📊 Dataset

The project uses a four-class subset of the **PlantVillage dataset**.

### Dataset Distribution

| Split | Images |
|---|---:|
| Training | 4,638 |
| Validation | 994 |
| Testing | 995 |
| **Total** | **6,627** |

The data was divided using a stratified train/validation/test split to maintain class distribution.

> The complete PlantVillage dataset is not included in this repository.

---

## 🧠 Technologies Used

### Programming
- Python

### Deep Learning
- PyTorch
- Torchvision

### Computer Vision
- OpenCV
- PIL

### Data Analysis & Evaluation
- NumPy
- Matplotlib
- Scikit-learn

### Explainable AI
- Grad-CAM

### Deployment
- Streamlit

### Development Environment
- Google Colab
- NVIDIA T4 GPU

---

## 🏗️ CNN Architecture

The final CNN consists of:

- Conv2D: 3 → 32
- ReLU
- MaxPooling
- Conv2D: 32 → 64
- ReLU
- MaxPooling
- Conv2D: 64 → 128
- ReLU
- MaxPooling
- Conv2D: 128 → 256
- ReLU
- Adaptive Average Pooling
- Fully Connected Classification Layer

### Input

- RGB image
- Resized to **128 × 128**

---

## 🔄 Image Preprocessing

Each image undergoes:

1. Resizing to 128 × 128 pixels
2. Conversion to PyTorch tensor
3. Normalization using ImageNet mean and standard deviation

---

## ⚖️ ANN vs CNN Comparison

A basic Artificial Neural Network (ANN) was implemented as a baseline.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| ANN | ~85% | 0.82 |
| CNN | **93.67%** | **0.93** |

### Observation

The CNN performs better because convolutional layers preserve spatial relationships between nearby pixels and learn local image features such as textures, edges, and patterns.

The ANN flattens the image into a vector and therefore loses much of the spatial structure present in the original image.

---

## 🏆 Final CNN Performance

### Overall Results

- **Test Accuracy: 93.67%**
- **Macro F1: 0.93**

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Bacterial Spot | 0.93 | 0.98 | 0.95 |
| Early Blight | 0.91 | 0.82 | 0.86 |
| Late Blight | 0.93 | 0.90 | 0.91 |
| Healthy | 0.98 | 1.00 | 0.99 |

### Main Finding

The strongest performance was obtained for **Healthy** and **Bacterial Spot**.

The most challenging class was **Early Blight**, which showed confusion with **Late Blight**.

---

## 📌 Confusion Matrix

The confusion matrix was used to analyze class-wise errors on the held-out test set.

The major source of error was confusion between:

**Early Blight ↔ Late Blight**

![Confusion Matrix](screenshots/confusion_matrix.png)

---

## 🔍 Explainable AI — Grad-CAM

Grad-CAM was integrated to visualize the regions that contributed most strongly to the CNN's prediction.

The heatmap provides a visual explanation of the model's decision:

- 🔴 Red: stronger contribution
- 🟡 Yellow: moderate contribution
- 🔵 Blue: lower contribution

The GitHub example uses a held-out **PlantVillage test image** so that the visualization represents the model's intended dataset conditions.

![Grad-CAM](screenshots/gradcam.png)

> Grad-CAM shows regions influencing the model's prediction. It is not a precise disease-segmentation map and should not be interpreted as proof that the highlighted area is the exact disease lesion.

---

## 🎯 Confidence Estimation

The model uses softmax probabilities to obtain a confidence score for the predicted class.

Example:

```text
Prediction: Late Blight
Confidence: 98.02%
