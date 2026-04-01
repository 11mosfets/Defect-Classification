# Wafer Defect Classification

A machine learning pipeline for multi-class defect classification on the **WM-811K** wafer map dataset using a fine-tuned **ResNet-18** CNN.

---

## Overview

Semiconductor wafer manufacturing produces wafers that may contain one of eight distinct defect patterns. This project implements an end-to-end classification pipeline using transfer learning on ResNet-18 to identify defect types from wafer maps.

### Defect Classes

| Class | Description |
|-------|-------------|
| `Edge-Ring` | Defects concentrated at the wafer's outer ring |
| `Edge-Loc` | Localized edge defects |
| `Center` | Defects concentrated at the center |
| `Loc` | Localized defect cluster |
| `Scratch` | Linear scratch pattern |
| `Random` | Randomly distributed defects |
| `Donut` | Ring-shaped defect pattern |
| `Near-full` | Nearly full-wafer defect coverage |

---

## Dataset

**WM-811K** — a publicly available wafer map dataset.

- **Total entries:** 811,457 wafer maps
- **Labeled defect samples:** 25,519 (across 8 classes)
- **Source:** [MIR Lab, National Cheng Kung University](http://mirlab.org/dataSet/public/)
- **Format:** `LSWMD.pkl` (Pickle file, ~2 GB)

> ⚠️ The dataset file (`LSWMD.pkl`) is **not included** in this repository due to its size. See the [Replication Guide](REPLICATION.md) for download instructions.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`data_exploration.ipynb`](data_exploration.ipynb) | Loads and visualizes the WM-811K dataset — class distributions, sample wafer maps, and basic statistics |
| [`cnn_pipeline.ipynb`](cnn_pipeline.ipynb) | Full training pipeline: data loading → stratified 95/5 split → ResNet-18 fine-tuning → evaluation with confusion matrix |

---

## Results

| Metric | Value |
|--------|-------|
| Train / Test Split | 95% / 5% (stratified per class) |
| Train Samples | 24,243 |
| Test Samples | 1,276 |
| Architecture | ResNet-18 (ImageNet pre-trained) |
| Epochs | 20 |
| Best Training Accuracy | ~99% |

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
git clone https://github.com/<your-username>/Defect_Classification.git
cd Defect_Classification

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
