# Replication Guide

Step-by-step instructions for reproducing the wafer defect classification results from scratch.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.12+ |
| pip | latest |
| macOS / Linux | (Windows untested) |
| RAM | ≥ 8 GB recommended |
| Disk Space | ≥ 3 GB (for dataset + venv) |

> 💡 Apple Silicon (M1/M2/M3) is supported — the pipeline auto-detects and uses the **MPS** backend for GPU acceleration.

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/Defect_Classification.git
cd Defect_Classification
```

---

## Step 2 — Download the Dataset

The dataset is **not included** in this repo due to its ~2 GB size.

1. Go to the [MIR Lab dataset page](http://mirlab.org/dataSet/public/) and download **LSWMD.pkl**, **or** use the direct Kaggle mirror:

   ```
   https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
   ```

2. Place the file in the project root so the path is:

   ```
   Defect_Classification/
   └── LSWMD.pkl      ← here
   ```

---

## Step 3 — Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
# .\venv\Scripts\activate     # Windows
```

---

## Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` installs:

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `torchvision` | Pre-trained ResNet-18 weights |
| `scikit-learn` | Stratified train/test split, metrics |
| `opencv-python` | Wafer map resizing |
| `pandas` / `numpy` | Data handling |
| `matplotlib` / `seaborn` | Visualization |
| `jupyter` / `ipykernel` | Notebook execution |
| `tqdm` | Progress bars |

---

## Step 5 — Register the Jupyter Kernel

```bash
python -m ipykernel install --user --name=venv --display-name "venv"
```

---

## Step 6 — Launch Jupyter

```bash
jupyter notebook
```

Open your browser to `http://localhost:8888` (should open automatically).

---

## Step 7 — Run the Notebooks (in order)

### 7a. Data Exploration

Open **`data_exploration.ipynb`** and run all cells (`Kernel → Restart & Run All`).

This notebook:
- Loads `LSWMD.pkl`
- Cleans nested array fields
- Shows class distribution (bar chart)
- Visualizes one sample wafer map per defect type

Expected output:
```
<class 'pandas.DataFrame'>
RangeIndex: 811457 entries, 0 to 811456
...
failureType
none         147431
Edge-Ring      9680
Edge-Loc       5189
...
```

---

### 7b. CNN Training Pipeline

Open **`cnn_pipeline.ipynb`** and run all cells (`Kernel → Restart & Run All`).

This notebook:
1. Loads and filters the dataset (removes `none` class)
2. Performs a **stratified 95% / 5% train/test split** (per class)
3. Builds a PyTorch `Dataset` that renders wafer maps as 64×64 RGB tensors
4. Fine-tunes **ResNet-18** (ImageNet weights) for 8-class classification
5. Trains for 20 epochs using **AdamW** + **CrossEntropyLoss**
6. Evaluates on the test set and plots a confusion matrix

Expected split output:
```
Total: 25519 | Train: 24243 (95%) | Test: 1276 (5%)

Per-class split (stratified):
Class             Total   Train (95%)  Test (5%)
-----------------------------------------------
Center             4294          4079        215
Donut               555           527         28
Edge-Loc           5189          4929        260
Edge-Ring          9680          9196        484
Loc                3593          3413        180
Near-full           149           141          8
Random              866           822         44
Scratch            1193          1133         60
```

Expected training output (approximate):
```
Epoch 1  | Loss: 0.89 | Acc: 72.8%
...
Epoch 20 | Loss: 0.11 | Acc: 97.1%
```

---

## Troubleshooting

### `LSWMD.pkl` loads with a VisibleDeprecationWarning
This is a known NumPy 2.x compatibility warning and does **not** affect results. It can be safely ignored.

### Out of memory during training
Reduce the `batch_size` in `cnn_pipeline.ipynb` (cell 3) from `32` to `16`.

### MPS / CUDA not detected
The pipeline falls back to CPU automatically. Training will be slower but functionally identical.

### Kernel not found in Jupyter
Re-run Step 5 (register the kernel) and refresh the browser.
