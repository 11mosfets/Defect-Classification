# Wafer Defect Classification

A machine learning pipeline for multi-class defect classification on the **WM-811K** wafer map dataset using a fine-tuned **ResNet-18** CNN.

---

## Overview

Semiconductor wafer manufacturing produces wafers that may contain one of eight distinct defect patterns. This project implements an end-to-end classification pipeline using transfer learning on ResNet-18 to identify defect types from wafer maps.

---

## Dataset

**WM-811K** — a publicly available wafer map dataset.

- **Source:** [Kaggle — WM-811K Wafer Map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map?resource=download)
- **Format:** `LSWMD.pkl` (Pickle file, ~2 GB)
- **Total wafer maps:** 811,457
- **Columns:** `waferMap`, `dieSize`, `lotName`, `waferIndex`, `trianTestLabel`, `failureType`

> ⚠️ The dataset file (`LSWMD.pkl`) is **not included** in this repository due to its size. See the [Replication Guide](REPLICATION.md) for download instructions.

### Original Dataset Train/Test Labels

| Label | Count |
|-------|-------|
| Unlabeled (NaN) | 638,507 |
| Test | 118,595 |
| Training | 54,355 |

### Die Size Statistics

| Stat | Value |
|------|-------|
| Mean | 1,841 |
| Std | 2,255 |
| Min | 3 |
| 25th percentile | 710 |
| Median | 953 |
| 75th percentile | 1,902 |
| Max | 48,099 |

### Defect Class Distribution

After filtering out unlabeled (`none`) entries, **25,519** wafer maps remain across 8 defect classes:

| Class | Count | Description |
|-------|-------|-------------|
| `Edge-Ring` | 9,680 | Defects concentrated at the wafer's outer ring |
| `Edge-Loc` | 5,189 | Localized edge defects |
| `Center` | 4,294 | Defects concentrated at the center |
| `Loc` | 3,593 | Localized defect cluster |
| `Scratch` | 1,193 | Linear scratch pattern |
| `Random` | 866 | Randomly distributed defects |
| `Donut` | 555 | Ring-shaped defect pattern |
| `Near-full` | 149 | Nearly full-wafer defect coverage |

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`data_exploration.ipynb`](data_exploration.ipynb) | Loads and visualizes the WM-811K dataset — class distributions, die size statistics, sample wafer maps per defect type, and original train/test label breakdown |
| [`cnn_pipeline.ipynb`](cnn_pipeline.ipynb) | Full training pipeline: data loading → stratified 90/10 split → ResNet-18 fine-tuning → evaluation with confusion matrix |

---

## Model & Training Details

| Parameter | Value |
|-----------|-------|
| Architecture | ResNet-18 (ImageNet pre-trained) |
| Input size | 64 × 64 RGB |
| Image encoding | Channel 0 = background die (value 1), Channel 1 = defect die (value 2) |
| Train / Test Split | 90% / 10% stratified per class |
| Train samples | 22,967 |
| Test samples | 2,552 |
| Optimizer | AdamW (lr = 1e-3) |
| Loss | CrossEntropyLoss |
| Batch size | 32 |
| Epochs | 20 |

### Per-Class Split

| Class | Total | Train (90%) | Test (10%) |
|-------|-------|-------------|------------|
| Center | 4,294 | 3,865 | 429 |
| Donut | 555 | 499 | 56 |
| Edge-Loc | 5,189 | 4,670 | 519 |
| Edge-Ring | 9,680 | 8,712 | 968 |
| Loc | 3,593 | 3,234 | 359 |
| Near-full | 149 | 134 | 15 |
| Random | 866 | 779 | 87 |
| Scratch | 1,193 | 1,074 | 119 |

---

## Results

### Training Progress

| Epoch | Train Loss | Train Accuracy |
|-------|-----------|----------------|
| 1 | 0.4250 | 84.82% |
| 5 | 0.1476 | 95.03% |
| 10 | 0.0607 | 98.02% |
| 15 | 0.0344 | 98.87% |
| 20 | 0.0241 | 99.27% |

### Test Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **92.79%** |

---

## Tech Stack

- Python 3.12
- PyTorch + TorchVision
- scikit-learn
- OpenCV
- pandas / NumPy
- Matplotlib / Seaborn
- Jupyter

---

## Quick Start

See the [Replication Guide](REPLICATION.md) for full setup and run instructions.

```bash
# Clone the repo
git clone https://github.com/11mosfets/Defect-Classification.git
cd Defect-Classification

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
