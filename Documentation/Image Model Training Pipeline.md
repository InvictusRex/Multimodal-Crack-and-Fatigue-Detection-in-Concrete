
# Concrete Crack Detection using ResNet50 (Image Modality)

## Overview

This module implements the visual inspection branch of a multimodal structural damage detection system. The goal is to detect visible cracks on concrete surfaces using a deep convolutional neural network trained on labeled concrete imagery.

The model uses transfer learning with ResNet50, a widely used deep residual network architecture originally trained on the ImageNet dataset.

This component forms the visual modality of the multimodal architecture and extracts high‑level visual features from concrete surfaces such as crack lines, fracture patterns, and surface discontinuities.

These extracted features are later fused with acoustic features in a multimodal model.

---

# System Pipeline

Concrete Image  
↓  
Image Preprocessing (Resize + Normalization)  
↓  
ResNet50 Backbone  
↓  
Deep Feature Representation (2048‑D)  
↓  
Fully Connected Layer  
↓  
Crack / No Crack Prediction

---

# Dataset

The image model uses the Concrete Crack Images Dataset, which is widely used in structural health monitoring research.

Dataset Properties

Number of Images: ~40,000  
Resolution: 227×227  
Classes: Crack / No Crack

Dataset Structure

dataset/  
    Positive/  
    Negative/

Positive images contain visible cracks, while negative images represent intact concrete surfaces.

---

# Image Preprocessing

Images are resized to 224×224 so they match the input size expected by ResNet50.

Images are normalized using ImageNet statistics:

mean = [0.485, 0.456, 0.406]  
std = [0.229, 0.224, 0.225]

This ensures compatibility with pretrained weights.

---

# Model Architecture

ResNet50 is a deep residual neural network containing 50 layers.

Residual Block Concept

Input → Conv → Conv → Add Skip Connection → Output

Mathematically:

H(x) = F(x) + x

This enables deep networks to train effectively without vanishing gradients.

---

# ResNet50 Architecture (Simplified)

Input Image (224×224×3)  
↓  
Conv Layer  
↓  
Max Pool  
↓  
Residual Blocks  
↓  
Global Average Pooling  
↓  
2048 Feature Vector  
↓  
Fully Connected Layer  
↓  
Binary Classification

---

# Transfer Learning

Instead of training from scratch, the model uses pretrained ImageNet weights.

Benefits:

• Faster convergence  
• Better feature extraction  
• Less data required

Only the final classification layer is modified.

Original layer: 2048 → 1000 classes  
Modified layer: 2048 → 2 classes

---

# Training Process

Loss Function: Cross Entropy

Loss = −Σ y log(p)

Optimizer: Adam  
Learning Rate: 1e‑4

Training Pipeline

Dataset → DataLoader → Batch → ResNet Forward Pass → Loss → Backpropagation → Optimizer Update

---

# Evaluation Metrics

Accuracy  
Precision  
Recall  
F1 Score

Typical Performance

Accuracy: 97‑99%  
Precision: ~0.98  
Recall: ~0.97  
F1 Score: ~0.97

---

# Role in Multimodal System

Concrete Image  
↓  
ResNet50  
↓  
2048‑D Visual Feature Vector

These features are fused with acoustic features extracted from the audio model.
