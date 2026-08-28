# 🫁 Lung Disease Prediction Using Hybrid Deep Learning

## 📌 Project Overview

This project presents a deep learning-based system for classifying chest X-ray images into three categories:

- **Fibrosis**
- **Normal**
- **Pneumonia**

The system uses a hybrid deep learning architecture combining **MobileNetV2** and **EfficientNetB0** to extract complementary image features. The extracted features are combined and passed through fully connected layers to perform three-class classification.

A **Streamlit web application** is developed to allow users to upload a chest X-ray image and obtain the predicted lung condition.

---

## 🎯 Objectives

- Develop an automated chest X-ray classification system.
- Classify images into Fibrosis, Normal, and Pneumonia.
- Use transfer learning with MobileNetV2 and EfficientNetB0.
- Handle class imbalance using moderate oversampling.
- Provide an easy-to-use Streamlit interface for prediction.

---

## 🧠 Model Architecture

The final model uses a hybrid architecture:

```text
Input Chest X-ray
        ↓
Data Augmentation
        ↓
 ┌───────────────┐
 │               │
MobileNetV2   EfficientNetB0
 │               │
 ↓               ↓
Global Average Pooling
 │               │
 └───────┬───────┘
         ↓
Feature Concatenation
         ↓
Dense Layer (256)
         ↓
Batch Normalization
         ↓
Dropout
         ↓
Dense Layer (128)
         ↓
Dropout
         ↓
Output Layer (3 Classes)
