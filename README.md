# Multi-Modal Crack and Fatigue Detection

A machine learning pipeline for structural crack detection using multimodal sensing, combining acoustic emission signals and visual inspection images.

This project explores whether combining audio-based structural health monitoring with computer vision crack detection can improve reliability in identifying structural damage.

---

# Table of Contents

- [Introduction](#introduction)
- [Project Motivation](#project-motivation)
- [Datasets](#datasets)
- [Dataset Links](#dataset-links)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Current Challenges](#current-challenges)
- [Future Work](#future-work)
- [References](#references)

---

# Introduction

Structural cracks can lead to catastrophic failures in bridges, buildings, and mechanical components. Detecting cracks early is essential for maintaining infrastructure safety.

Traditional inspection methods rely heavily on manual visual inspection, which is time consuming and prone to human error. Additionally, visual inspection can only detect cracks that are visible on the surface.

However, cracks also generate acoustic emissions when they form or propagate inside materials. These signals can be captured using acoustic sensors and analyzed using signal processing and machine learning techniques.

This project combines two sensing modalities:

- Visual inspection using images
- Acoustic emission monitoring using sound signals

The goal is to develop a machine learning system capable of detecting structural cracks using both types of data.

---

# Project Motivation

Modern structural health monitoring systems often rely on multiple sensors. Each sensor type captures different information about the condition of the structure.

Images provide information about visible surface damage, while acoustic emission sensors capture internal events occurring inside materials.

By combining these two modalities, it is possible to build a more reliable crack detection system that can detect both surface cracks and internal structural damage.

This project aims to explore the potential of multimodal machine learning for structural health monitoring.

---

# Datasets

This project uses two datasets:

1. SDNET2018 Concrete Crack Image Dataset
2. Acoustic Crack Detection (ACD) Dataset

The image dataset provides visual information about cracks in concrete surfaces, while the acoustic dataset provides waveform signals generated during crack formation.

---

# Dataset Links

SDNET2018 Image Dataset

https://www.kaggle.com/datasets/aniruddhsharma/structural-defects-network-concrete-crack-images

Acoustic Crack Detection Dataset

https://github.com/jfernsler/Acoustic_Crack_Detection

---

# SDNET2018 Dataset

SDNET2018 contains more than 56,000 images of cracked and non-cracked concrete surfaces.

Images were collected from three types of structures:

- Bridge Decks
- Walls
- Pavements

Dataset structure:

SDNET2018  
├── D (Bridge Decks)  
│   ├── CD (Cracked)  
│   └── UD (Uncracked)  
├── W (Walls)  
│   ├── CW (Cracked)  
│   └── UW (Uncracked)  
└── P (Pavements)  
    ├── CP (Cracked)  
    └── UP (Uncracked)  

Each image represents a small patch of concrete surface and is labeled as cracked or uncracked.

---

# Acoustic Crack Detection Dataset

The Acoustic Crack Detection dataset contains acoustic emission signals recorded from metal specimens during fatigue testing experiments.

When cracks form or propagate, the material releases small bursts of elastic energy that propagate through the structure. These signals are captured by acoustic emission sensors.

Each event in the dataset contains the following fields:

channel  
crack  
hit  
testname  
time  
waveform  
waveform_noz  
wavelen_noz  

Where:

channel indicates the sensor used during recording.

crack is the binary label indicating crack presence.

waveform represents the raw acoustic signal.

waveform_noz represents the waveform after removing leading and trailing zeros.

---

# Data Preprocessing

Both datasets undergo preprocessing before training the machine learning models.

Image preprocessing includes:

- Reading images from dataset directories
- Assigning crack / non-crack labels
- Resizing images to 224x224
- Splitting the dataset into training, validation, and test sets

Acoustic preprocessing includes:

- Merging acoustic datasets
- Filtering signals from a specific sensor channel
- Removing zero padding from waveforms
- Normalizing waveform signals
- Padding or truncating signals to a fixed length of 4096 samples
- Performing stratified train-test split

The processed datasets are stored in the `data/processed` directory.

---

# Model Architecture

Two separate models are implemented in this project.

Image Model

The image pipeline uses a convolutional neural network (CNN) for crack classification. CNNs are well suited for visual crack detection because they can learn spatial patterns such as edges, textures, and fracture lines.

Typical architecture:

Input Image  
Conv Layer  
BatchNorm  
ReLU  
MaxPool  

Conv Layer  
BatchNorm  
ReLU  
MaxPool  

Conv Layer  
BatchNorm  
ReLU  
MaxPool  

Global Pooling  

Fully Connected Layer  

Output: Crack / No Crack

---

Audio Model

The audio pipeline uses a one-dimensional neural network that processes waveform signals directly.

Typical architecture:

Input Waveform (4096 samples)

Conv1D  
BatchNorm  
ReLU  
MaxPool  

Conv1D  
BatchNorm  
ReLU  
MaxPool 

Global Average Pooling  

Fully Connected Layer  

Output: Crack / No Crack

---

# Current Challenges

The acoustic dataset used in this project is relatively small compared to typical deep learning datasets.

Small datasets make it difficult for deep neural networks to generalize well, and models may suffer from instability or overfitting.

Improving dataset size and signal feature extraction will be important for improving model performance.

---

# Future Work

Possible future improvements include:

- Feature extraction from acoustic signals
- Spectrogram based audio models
- Multimodal fusion networks
- Larger acoustic datasets
- Transformer based architectures
- Real-time structural monitoring systems

The long-term goal is to build a robust multimodal crack detection system for structural health monitoring.

---

# References

Acoustic Crack Detection Dataset  
https://github.com/jfernsler/Acoustic_Crack_Detection

SDNET2018 Dataset  
https://www.kaggle.com/datasets/aniruddhsharma/structural-defects-network-concrete-crack-images

SDNET2018 Paper  
https://doi.org/10.1016/j.dib.2018.11.015
