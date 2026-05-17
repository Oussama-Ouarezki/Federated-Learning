# Arabic Handwritten Character Recognition — Centralized & Federated Learning

> A comprehensive study on Arabic handwritten character recognition using CNNs under both centralized and federated learning settings across three benchmark datasets.

📄 **[Read the Full Report (PDF)](https://github.com/Oussama-Ouarezki/Federated-Learning/blob/main/main.pdf)**

---

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Centralized Learning](#centralized-learning)
- [Federated Learning](#federated-learning)
- [Results Summary](#results-summary)
- [Hardware](#hardware)
- [References](#references)

---

## Overview

Arabic handwritten character recognition is significantly more challenging than digit recognition due to:

- The cursive nature of Arabic script
- Multiple contextual letter forms (isolated, beginning, middle, end)
- Letters differing only by diacritical dots
- High variability in handwriting style, stroke thickness, scale, and orientation

This project investigates recognition under both **centralized** and **federated** learning settings, evaluating multiple CNN architectures and analyzing the impact of data heterogeneity.

---

## Datasets

Three benchmark datasets are used, each treated as an independent client in the federated learning setup:

| Dataset | Image Size | # Images | # Classes | Link |
|---|---|---|---|---|
| **AHCD** | 32 × 32 | 16,800 | 28 | [Kaggle](https://www.kaggle.com/datasets/mloey1/ahcd1) |
| **HMBD v1** | 300 × 300 | 54,115 | 28 letters × 4 positions | [GitHub](https://github.com/HossamBalaha/HMBD-v1) |
| **OIHACDB-28** | 128 × 128 | 5,600 | 28 | [MediaFire](https://www.mediafire.com/file/diikxls8qdi3ibb/OIHACDB.rar/file) |

### Preprocessing

All datasets were standardized to **32 × 32 grayscale images** before training:

- **OIHACDB-28:** Localized to the smallest bounding square, then resized — preserving character structure while reducing input dimensions. This reduced trainable parameters by **93%** (from 16.8M to 1.16M) and training time from ~47s to ~13.66s.
- **HMBD v1:** Color-inverted (white letters on black background), isolated-position-only samples kept, localized and resized to 32 × 32. Total images after preprocessing: **17,495**.
- **AHCD:** Already 32 × 32; file names unified and images merged into a single folder.

---

## Project Structure

```
├── datasets/
│   ├── AHCD/
│   ├── HMBD_v1/
│   └── OIHACDB/
├── preprocessing/
│   ├── one_folder.py          # Merge images into a single folder
│   ├── inverted_cropped.py    # Invert colors and localize characters
│   ├── resize.py              # Resize images to 32×32
│   └── adjustment.py         # Rename files to match labeling scheme
├── centralized/
│   ├── baseline_cnn.py
│   ├── improved_cnn.py
│   ├── resnet18.py
│   └── hybrid_attention_cnn.py
├── federated/
│   ├── fl_full_classes.py     # FL with all classes across clients
│   ├── fl_missing_classes.py  # FL with class-wise data fragmentation
│   └── fl_balanced.py         # FL with balanced missing classes
└── main.pdf
```

---

## Centralized Learning

### CNN Architecture (Baseline)

The baseline CNN processes 32 × 32 grayscale images through three feature extraction blocks:

```
Block 1:  Conv(1→32,  3×3) → ReLU → BN → MaxPool(2×2)  →  16×16×32
Block 2:  Conv(32→64, 3×3) → ReLU → BN → MaxPool(2×2)  →   8×8×64
Block 3:  Conv(64→128,3×3) → ReLU → BN → MaxPool(2×2)  →   4×4×128
Classifier: Flatten → Linear(2048→512) → ReLU → Dropout(0.5) → Linear(512→28)
```

**Parameters:** 1,157,085 | **Training Time:** ~170s | **Accuracy:** 96.50%

### Training Improvements on OIHACDB-28

| Technique | Accuracy | Training Time |
|---|---|---|
| Random weight initialization | 91.0% | ~47s |
| + Learning rate schedule (warm-up + cosine decay) | 93.0% | ~47s |
| + Data augmentation (rotation, scale, shear) | 97.68% | ~47s |
| + Resize to 32×32 (localize → resize) | **97.86%** | **~13.66s** |

### Model Comparison (Centralized, Full Dataset — 39,645 images)

| Model | Parameters | Training Time | Test Accuracy |
|---|---|---|---|
| Baseline CNN | 1,157,085 | ~2.8 min | 96.50% |
| Improved CNN (Double Conv + Adaptive Avg Pool) | 316,652 | ~7.6 min | 97.19% |
| ResNet-18 (adapted for 32×32) | 11,182,044 | ~21.4 min | 97.29% |
| Hybrid Attention CNN (CBAM + SE + Coord) | 1,244,560 | ~40 min | 95.75% |

> **Key finding:** The Baseline CNN offers the best trade-off between accuracy, complexity, and training efficiency. It was selected for all federated learning experiments.

---

## Federated Learning

Each dataset is treated as an **independent client**. Clients train local models on private data and share only model parameters with a central aggregation server (FedAvg), preserving data privacy.

```
Client 1 (AHCD)   ──┐
                     ├──► Aggregation Server ──► Global Model
Client 2 (HMBD)   ──┤
                     │
Client 3 (OIHACDB)──┘
```

### Experiment 1 — Full Classes (all 28 letters at each client)

- **Overall Accuracy:** 91.50%
- **Training Time:** ~711s (~11.9 min)
- **Letter performance:** 20/28 letters ≥ 90% accuracy

### Experiment 2 — Missing Classes (each letter exists at only one client)

Class distribution across clients:

| Client | Dataset | Letters |
|---|---|---|
| Client 1 | OIHACDB | AiinI, AlifI, BaaI, CaafI, DadI, DalI, DhaI, DhelI, FaaI, GhiinI |
| Client 2 | HMBD | HaI, HaaI, JiimI, KafI, KhaaI, LamI, MiimI, NounI, RaaI |
| Client 3 | AHCD | SadI, ShiinI, SiinI, TaaI, ThaI, ThaaI, WawI, YaaI, ZadI |

- **Overall Accuracy:** 84.98%
- **Training Time:** ~317s (~5.3 min)
- Classes absent from a client are less consistently recognized by the global model

### Experiment 3 — Balanced Missing Classes

OIHACDB (most affected client) was augmented with samples of missing letters from other datasets, and a fixed number of samples per class was enforced globally.

- **Effect:** Reduced gradient bias toward dominant classes; more stable and fair learning across all letters.

---

## Results Summary

| Training Strategy | Parameters | Training Time | Overall Accuracy |
|---|---|---|---|
| Centralized (Baseline CNN) | 1,157,085 | ~2.8 min | **96.50%** |
| Federated — Full Classes | 1,157,085 | ~11.9 min | 91.50% |
| Federated — Missing Classes | 1,157,085 | ~5.3 min | 84.98% |

**Key takeaways:**

- Centralized learning achieves the highest accuracy due to full data access.
- Federated learning with full class availability shows a moderate drop (~5%) with added communication overhead.
- Class-wise data fragmentation causes a significant further drop (~6.5%), highlighting the importance of balanced data distribution in non-IID federated settings.
- A more complex model does not guarantee better performance — simpler models with proper training are often preferable.
- Careful class balancing in federated learning is critical to prevent gradient bias and improve global model generalization.

---

## Hardware

All experiments were conducted on the following machine:

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX (see report for full specs) |
| CUDA | 12.4 (required for federated learning experiments) |
| Framework | PyTorch + Flower (FL) |

---

## References

1. A. El-Sawy, M. Loey, H. EL-Bakry — *Arabic Handwritten Characters Recognition using CNN*, WSEAS 2017.
2. H. Balaha — *HMBD-v1: Handwritten Arabic Multi-position Dataset*, GitHub 2020.
3. C. Boufenar et al. — *OIHACDB-28 Dataset*, 2020.
4. K. P. Murphy — *Probabilistic Machine Learning: An Introduction*, MIT Press 2022.
5. Y. LeCun et al. — *Gradient-based Learning Applied to Document Recognition*, IEEE 1998.
6. H. B. McMahan et al. — *Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)*, AISTATS 2017.
7. K. He et al. — *Deep Residual Learning for Image Recognition*, CVPR 2016.
