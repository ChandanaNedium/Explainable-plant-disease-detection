# Explainable-plant-disease-detection
Explainable tomato leaf disease detection using CNN, Grad-CAM, ANN comparison, and image validation.

A deep learning-based tomato leaf disease classification system built using PyTorch and the PlantVillage dataset.

The project not only predicts the disease but also uses Grad-CAM to visualize the image regions that influenced the CNN's prediction. An ANN baseline is also implemented to compare traditional fully connected image classification with CNN-based spatial feature learning.

---

## Project Overview

Plant disease detection from leaf images can help identify diseases at an early stage. However, a classification model that only provides a prediction can behave like a black box.

This project addresses both classification and interpretability by combining:

- CNN-based image classification
- ANN vs CNN performance comparison
- Confidence estimation
- Grad-CAM explainability
- Confusion matrix and classification analysis
- Class imbalance analysis
- Image augmentation experiment
- User-uploaded image validation
- Streamlit web application

---

## Problem Statement

Traditional image classification systems may provide a disease prediction without showing what influenced the prediction.

The objective of this project is to develop a tomato leaf disease classification system using CNN and provide a visual explanation of the model's decision using Grad-CAM.

The project also evaluates the model on real-world uploaded images to identify limitations such as background bias and domain shift.

---

## Objectives

1. Build a CNN model for tomato leaf disease classification.
2. Compare CNN performance with a basic ANN.
3. Evaluate the model using accuracy, precision, recall and F1-score.
4. Visualize classification errors using a confusion matrix.
5. Use Grad-CAM to explain CNN predictions.
6. Estimate prediction confidence.
7. Validate uploaded images before classification.
8. Analyze model limitations on real-world images.

---

## Dataset

The project uses the PlantVillage dataset.

A subset containing four tomato classes was selected:

- Tomato___Bacterial_spot
- Tomato___Early_blight
- Tomato___Late_blight
- Tomato___healthy

### Dataset Distribution

Total images: 6,627

- Training: 4,638
- Validation: 994
- Testing: 995

The dataset was split using stratification to preserve class distribution.

> The complete dataset is not included in this repository.

---

## Technologies Used

### Programming Language
- Python

### Deep Learning
- PyTorch
- Torchvision

### Computer Vision
- OpenCV
- PIL

### Data Analysis and Evaluation
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

## Model Architecture

The CNN contains:

- Convolution layer: 3 → 32 channels
- ReLU activation
- Max Pooling
- Convolution layer: 32 → 64 channels
- ReLU activation
- Max Pooling
- Convolution layer: 64 → 128 channels
- ReLU activation
- Max Pooling
- Convolution layer: 128 → 256 channels
- ReLU activation
- Adaptive Average Pooling
- Fully connected classification layer

The model predicts four classes.

---

## Preprocessing

Each image is:

1. Resized to 128 × 128 pixels
2. Converted to a tensor
3. Normalized using ImageNet mean and standard deviation

---

## ANN vs CNN

A basic ANN was implemented as a baseline.

### Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| ANN | ~85% | 0.82 |
| CNN | 93.67% | 0.93 |

The CNN performed better because convolutional layers preserve and learn spatial patterns in images, whereas the ANN first flattens the image into a long vector.

---

## Final CNN Results

Test set:

- Accuracy: **93.67%**
- Macro F1: **0.93**

### Classification Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Bacterial Spot | 0.93 | 0.98 | 0.95 |
| Early Blight | 0.91 | 0.82 | 0.86 |
| Late Blight | 0.93 | 0.90 | 0.91 |
| Healthy | 0.98 | 1.00 | 0.99 |

The most difficult class was Early Blight, which showed confusion with Late Blight.

---

## Confusion Matrix

The confusion matrix was used to identify the classes where the model makes mistakes.

The largest confusion was observed between:

**Early Blight ↔ Late Blight**

This indicates that visually similar disease symptoms remain a major classification challenge.

![Confusion Matrix](screenshots/confusion_matrix.png)

---

## Explainable AI with Grad-CAM

Grad-CAM was used to visualize the regions that contributed most to the CNN prediction.

The heatmap uses colors to represent relative importance:

- Red: stronger contribution
- Yellow: moderate contribution
- Blue: lower contribution

This helps analyze whether the model is focusing on meaningful image regions.

![Grad-CAM](screenshots/gradcam.png)

> Grad-CAM indicates regions influencing the model's prediction. It should not be interpreted as a precise disease segmentation or medical diagnosis.

---

## Confidence Estimation

Softmax probabilities are used to calculate the model's prediction confidence.

Example:

**Prediction:** Late Blight  
**Confidence:** 98.02%

Confidence is used together with image validation rather than being treated as a guarantee of correctness.

---

## User-Uploaded Image Validation

Real-world images may contain:

- Hands
- Multiple leaves
- Complex backgrounds
- Different lighting
- Camera variations

A lightweight OpenCV-based image suitability check was added.

If an image appears unsuitable, the application displays:

> Image not suitable for reliable prediction.  
> Please upload a clear photo of ONE tomato leaf.

This prevents the system from automatically presenting a potentially misleading prediction for unsuitable images.

---

## Real-World Testing and Domain Shift

A real-world uploaded image was tested during development.

Although the CNN produced a high-confidence prediction, Grad-CAM sometimes highlighted hand/background regions.

This revealed a domain-shift problem:

**Training data:** controlled PlantVillage images

**Real-world image:** hand + multiple leaves + different background and lighting

This experiment demonstrates why high test accuracy on a controlled dataset does not necessarily guarantee the same performance on arbitrary real-world photographs.

---

## Data Augmentation Experiment

Image augmentation was experimentally evaluated using:

- Random horizontal flipping
- Random rotation
- Brightness changes
- Contrast changes

Under the current five-epoch training configuration:

- Augmented CNN Accuracy: **87.24%**
- Augmented CNN Macro F1: **0.84**

Since the original CNN performed better, the original model was selected as the final model.

This experiment was retained as part of the model analysis.

---

## Class Imbalance Experiment

The four classes had different numbers of images.

Class-weighted CrossEntropyLoss was tested to reduce the effect of class imbalance.

The experiment did not produce a significant improvement in overall performance, so the final model was selected based on the better observed test performance.

---

## Application Workflow

```text
User uploads image
        ↓
Image suitability check
        ↓
Is the image suitable?
     ↙       ↘
   No         Yes
   ↓           ↓
Ask user     CNN model
to upload       ↓
clear image   Disease prediction
                ↓
           Confidence score
                ↓
             Grad-CAM
                ↓
       Visual explanation
