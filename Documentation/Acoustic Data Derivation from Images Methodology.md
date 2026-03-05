
# Multimodal Concrete Crack Detection via Derived Acoustic Representation

## Overview

This document explains a practical technique for constructing a **paired multimodal dataset from a single visual dataset**. The objective is to simulate a **visual + acoustic multimodal learning pipeline** without requiring physical sensors or multiple synchronized datasets.

In many real-world structural health monitoring systems, engineers combine:

- Visual inspection (camera images)
- Acoustic emission sensors
- Ultrasonic testing
- Vibration measurements

However, publicly available datasets rarely contain **synchronized visual and acoustic data for the same concrete sample**. To address this limitation for experimental prototypes or demonstrations, we can derive a **signal representation directly from the image data itself**.

This approach produces a **paired multimodal dataset** where each sample contains:

- A **visual modality** (the original concrete image)
- A **derived acoustic-like modality** (generated signal or spectrogram)

Because both modalities originate from the **same sample**, the system satisfies the formal definition of **paired multimodal learning**.

---

# Core Idea

Concrete crack images contain meaningful **spatial patterns** such as:

- crack edges
- intensity discontinuities
- texture changes
- geometric crack structures

These spatial characteristics can be transformed into **signal representations**. The generated signal can then be processed similarly to an acoustic waveform.

The transformation pipeline looks like this:

```
Concrete Image
      ↓
Edge Detection / Intensity Profiling
      ↓
1D Signal Representation
      ↓
Spectrogram Generation
```

The resulting spectrogram behaves similarly to an **acoustic frequency representation** and can be processed using standard audio CNN architectures.

---

# Resulting Paired Multimodal Dataset

After transformation, each sample contains two modalities.

| Sample | Image | Acoustic Representation |
|------|------|------|
| Image_001 | crack image | generated spectrogram |
| Image_002 | no crack image | generated spectrogram |

Both modalities correspond to **the same structural sample**, which makes the dataset suitable for **paired multimodal learning**.

---

# Modalities in the System

## Visual Modality

Input:

- Concrete crack images from the dataset

Example dataset:

- Concrete Crack Images Dataset

The visual branch learns patterns such as:

- crack geometry
- surface discontinuities
- crack width and orientation
- texture anomalies

Model example:

- ResNet50
- EfficientNet
- MobileNetV2

Pipeline:

```
Image → CNN → Image Feature Vector
```

---

## Acoustic (Derived) Modality

Input:

- Spectrogram generated from image-derived signal

Although the signal is derived from visual patterns, the representation behaves similarly to acoustic or vibration signals.

Model example:

- Lightweight CNN
- Spectrogram CNN classifier

Pipeline:

```
Spectrogram → CNN → Audio Feature Vector
```

---

# Multimodal Architecture

Once both feature vectors are extracted, they are fused to make a final prediction.

Architecture:

```
Concrete Image
      │
      ▼
  CNN (ResNet)
      │
Image Features
      │
      ▼
   Fusion Layer
      ▲
      │
Generated Spectrogram
      │
   CNN Audio Model
      │
Audio Features
```

Fusion step:

```
Combined Features = Concatenate(Image Features, Audio Features)
Prediction = Classifier(Combined Features)
```

Because both inputs originate from the **same concrete image sample**, the system performs **paired multimodal learning**.

---

# Generating the Second Modality

There are several ways to derive a signal representation from an image.

## Method 1 — Intensity Signal

Convert image rows or columns into a waveform using average pixel intensity.

Example:

```
signal = mean_pixel_intensity_per_row
```

Steps:

1. Convert image to grayscale
2. Compute mean intensity for each row
3. Produce 1D signal vector
4. Generate spectrogram using STFT

This produces a waveform that reflects **intensity variation across the image**.

---

## Method 2 — Edge Response Signal

Edges often correspond to cracks and structural boundaries.

Pipeline:

```
edges = Canny(image)
signal = sum(edges along axis)
```

Steps:

1. Perform edge detection (Canny or Sobel)
2. Sum edge responses across rows or columns
3. Generate 1D signal
4. Convert to spectrogram

This method captures **crack geometry variations as signal fluctuations**.

---

## Method 3 — Frequency Spectrum Representation

Compute the frequency spectrum of the intensity signal.

Pipeline:

```
signal = mean_pixel_intensity_per_row
spectrum = FFT(signal)
```

Then convert the spectrum to a spectrogram or frequency representation.

This method simulates **vibration frequency behavior**.

---

# Spectrogram Generation

After generating the 1D signal, convert it into a spectrogram.

Typical pipeline:

```
signal
   ↓
Short-Time Fourier Transform (STFT)
   ↓
Magnitude Spectrogram
   ↓
Log Scaling / Mel Spectrogram
```

This produces a **2D representation** suitable for CNN input.

---

# Example Dataset Structure

After preprocessing, the dataset may look like this:

```
dataset/
│
├── images/
│   ├── crack_001.jpg
│   ├── crack_002.jpg
│
├── signals/
│   ├── crack_001_signal.npy
│   ├── crack_002_signal.npy
│
└── spectrograms/
    ├── crack_001_spec.png
    ├── crack_002_spec.png
```

Each sample now contains:

```
(image, signal/spectrogram)
```

This forms a **paired multimodal sample**.

---

# Inference Pipeline

During inference:

```
(image, signal) → multimodal model → crack prediction
```

Steps:

1. Load concrete image
2. Generate signal representation
3. Convert signal to spectrogram
4. Extract image features via CNN
5. Extract spectrogram features via CNN
6. Fuse features
7. Produce final prediction

---

# Advantages of This Approach

This technique has several advantages for prototyping and demonstrations.

### Single Dataset

Only one dataset is required.

### Paired Modalities

Both modalities correspond to the same sample.

### Simple Implementation

Signal generation requires minimal preprocessing.

### Clean Architecture

Allows demonstration of full multimodal architecture.

### No Sensor Hardware

No acoustic sensors or ultrasonic equipment required.

---

# Why This Works for Demonstrations

This approach allows researchers or students to demonstrate:

- multimodal learning architecture
- feature fusion strategies
- cross-modal feature extraction

without requiring:

- synchronized sensor setups
- multimodal data collection
- specialized hardware

It therefore serves as an effective **multimodal learning prototype**.

---

# Conclusion

By deriving an acoustic-like signal representation from spatial patterns in concrete crack images, it is possible to construct a **paired multimodal dataset using only a single visual dataset**.

The resulting system integrates:

- visual crack detection
- signal-based structural representation
- multimodal feature fusion

This technique provides a practical way to experiment with **multimodal structural inspection systems** when true multimodal datasets are unavailable.
