
# Acoustic Anomaly Detection using Mel Spectrogram CNN

## Overview

This module implements the acoustic sensing branch of the multimodal anomaly detection system. Instead of visual inspection, this model analyzes machine sound signals to detect abnormal structural behavior.

Acoustic signals can reveal internal mechanical anomalies that may not be visible through visual inspection alone.

The dataset used is a subset of the MIMII (Malfunctioning Industrial Machine Investigation and Inspection) dataset.

---

# Audio Processing Pipeline

Raw Audio (.wav)  
↓  
Waveform Loading (librosa)  
↓  
Mel Spectrogram Generation  
↓  
Spectrogram Image  
↓  
CNN Feature Extraction  
↓  
Anomaly Classification

---

# Dataset

The system uses machine recordings from the MIMII dataset.

Machine Type Used: slider

Dataset Structure

slider/  
    id_00/  
        normal/  
        abnormal/

Multiple machine IDs are included to increase variation in acoustic patterns.

---

# Why Audio Helps

Visual inspection detects surface cracks.

Acoustic signals can detect:

• internal vibrations  
• mechanical resonance changes  
• structural anomalies

Combining visual and acoustic sensing increases detection reliability.

---

# Audio Representation

Raw audio is a time‑domain signal.

Amplitude vs Time waveform representation.

However neural networks learn better from frequency‑domain representations.

---

# Spectrogram Representation

A spectrogram converts audio into a frequency‑time image.

Frequency ↑  
Intensity pixels represent sound energy  
Time →

This allows CNNs to treat sound as an image.

---

# Mel Spectrogram

Instead of linear frequencies, the Mel scale approximates human auditory perception.

mel = 2595 log10(1 + f / 700)

Advantages:

• emphasizes perceptually important frequencies  
• compresses high‑frequency regions

---

# Spectrogram Generation Pipeline

Audio Signal  
↓  
Short‑Time Fourier Transform  
↓  
Power Spectrum  
↓  
Mel Filter Bank  
↓  
Mel Spectrogram

Output size is resized to 224×224.

---

# CNN Architecture

Input Spectrogram (1×224×224)  
↓  
Conv Layer (16 filters)  
↓  
Max Pool  
↓  
Conv Layer (32 filters)  
↓  
Max Pool  
↓  
Adaptive Pool  
↓  
Flatten  
↓  
Fully Connected Layer  
↓  
128‑D Feature Vector

---

# Training Pipeline

Audio Dataset  
↓  
Spectrogram Conversion  
↓  
Batch Loading  
↓  
CNN Forward Pass  
↓  
Loss Calculation  
↓  
Backpropagation

Loss Function: CrossEntropyLoss

---

# Evaluation Metrics

Accuracy  
Precision  
Recall  
F1 Score  
Confusion Matrix

Typical Performance

Accuracy: 85‑93%  
Precision: ~0.90  
Recall: ~0.85  
F1 Score: ~0.87

---

# Role in Multimodal System

Audio Signal  
↓  
CNN  
↓  
128‑D Feature Vector

These acoustic features are fused with visual features from the image model.

Visual Features (2048) + Audio Features (128) → Multimodal Fusion Network → Final Prediction
