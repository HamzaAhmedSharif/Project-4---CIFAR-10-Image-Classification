---
title: Project 4 - Image Classification
app_file: app/gradio_app.py
---

# Project 4: Image Classification with GPU Optimization

## Overview

An end-to-end computer-vision pipeline on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
built with PyTorch and CUDA, structured as four stages: **EDA → model building → GPU benchmarking →
deployment**. All four parts are complete.

## Project Structure

```
Project 4 - Image Classification/
|-- notebooks/
|   |-- 01_Data_Exploration.ipynb   # Part 1 - EDA (complete)
|   |-- 02_Model_Building.ipynb     # Part 2 - training & comparison (complete)
|   `-- 03_GPU_Benchmarking.ipynb   # Part 3 - GPU benchmarking (complete)
|-- app/
|   |-- gradio_app.py               # Part 4 - interactive demo (complete)
|   `-- examples/                   #   one test-set image per class (committed)
|-- src/                            # shared package: models, data loading, training loop
|   |-- model_utils.py              #   SimpleCNN, EfficientNet-V2-S builder, WRN-28-10
|   |-- data_loader.py              #   transforms + DataLoader construction
|   `-- train_utils.py              #   train_one_epoch / evaluate / train_model
|-- data/raw/                       # CIFAR-10 (downloaded, not tracked in git)
|-- outputs/                        # dataset summary, model results, benchmark results, plots
|-- models/                         # best checkpoints + provenance metadata
|-- requirements.txt
`-- README.md
```

## Pipeline

| Part | Stage | Status | Key artifacts |
| :-- | :-- | :-- | :-- |
| 1 | Exploratory Data Analysis | Done | `outputs/dataset_summary.json`, `outputs/plots/*.png` |
| 2 | Model Building & Training | Done | `models/*_best.pth`, `outputs/model_results.json`, `outputs/plots/model_comparison.png`, `outputs/plots/confusion_matrix.png` |
| 3 | GPU Benchmarking | Done | `outputs/benchmark_results/`, `outputs/plots/{batch_size_benchmark,cpu_gpu_comparison,mixed_precision_benchmark,memory_usage}.png` |
| 4 | Deployment | Done | `app/gradio_app.py`, `app/examples/` |

## Models (Part 2)

### Model Selection

- **CNN scratch** — baseline to verify the training pipeline end-to-end.
- **EfficientNet-V2-S** — transfer learning with a modern architecture (2021); converges fast
  and is the best model per unit of wall-clock time.
- **WRN-28-10** — from-scratch state-of-the-art (2016), implemented from the original paper
  (pre-activation blocks, width factor 10); still highly competitive on CIFAR-10 and the final
  winner on test accuracy.

Three competing architectures, each trained under a shared protocol (cross-entropy, best-epoch
checkpointing) with model-appropriate optimizers — the CNN and EfficientNet use Adam, while the
WRN-28-10 follows the paper's recipe: SGD + Nesterov momentum (lr 0.1), weight decay 5e-4,
dropout 0.3 in every block, and cosine annealing over 50 epochs:

| Model | Type | Parameters | Notes |
| :-- | :-- | :-- | :-- |
| CNN from scratch | Baseline | ~0.9M | 4 conv blocks (32→256 ch) + batch norm + dropout; 50 epochs |
| EfficientNet-V2-S | Transfer learning | ~21.5M | ImageNet weights; stem stride 1 fix for 32×32; RandAugment + label smoothing + head dropout 0.3 |
| WRN-28-10 | From scratch | ~36.5M | Paper-compliant Wide ResNet (Zagoruyko & Komodakis, 2016); SGD + Nesterov, weight decay 5e-4, block dropout 0.3, cosine annealing, 50 epochs |

The winner's best checkpoint is promoted to `models/best_model.pth` — the single artifact consumed
by Parts 3 and 4 — with provenance recorded in `models/model_metadata.json`.

## Key Design Decisions

- **Verified normalization.** Pixel statistics were computed from the raw training split in Part 1
  rather than copied from other codebases (the commonly quoted std values do not reproduce from the
  data); Part 2 reads these constants back from `outputs/dataset_summary.json`.
- **No preprocessing leakage.** Augmentation is applied only to the training split; the test split
  is normalized but otherwise untouched, so reported metrics reflect real-world performance.
- **Robust transfer learning.** The EfficientNet stem stride is reduced to 1 (stride-2 destroys
  32×32 inputs) and the classifier head is swapped for a 10-way layer; weights load from the local
  cache when offline.
- **Paper-faithful WRN training.** The Wide ResNet uses the original recipe — SGD with Nesterov
  momentum (lr 0.1), weight decay 5e-4, dropout 0.3 inside every block, cosine annealing over
  50 epochs — instead of generic Adam defaults.
- **Targeted anti-overfitting.** A measured ~4.5 pp train/test gap on EfficientNet was closed with
  RandAugment (train loader), label smoothing (α=0.1), and higher head dropout (0.2→0.3); the CNN
  and WRN keep the standard recipe. The CNN schedule was extended to 50 epochs after a preliminary
  run showed accuracy still climbing at epoch 30.
- **No copy-pasted training code.** Models, data loading, and the training loop live once in `src/`
  and are imported by the notebook, so every run shares one verified implementation.
- **Reproducibility.** Fixed random seed (42), CUDA benchmark mode, and a shared training loop with
  best-epoch checkpoint persistence.

## Results

Best test accuracy on the held-out split, as produced by `notebooks/02_Model_Building.ipynb` on an
NVIDIA RTX 4060 Laptop GPU (8.0 GB):

| Model | Parameters | Test Accuracy (%) | Best Epoch | Training Time |
| :-- | :-- | :-- | :-- | :-- |
| CNN from scratch (50 epochs) | 0.9M | **86.63** | 47 | ~18 min |
| EfficientNet-V2-S (30 epochs) | 20.2M | **95.41** | 23 | ~32 min |
| WRN-28-10 (50 epochs) | 36.7M | **95.43** | 47 | ~3.3 h |

The winner — **WRN-28-10 (95.43%)** — wins by a hair over transfer learning, confirming that a
paper-compliant architecture designed for 32×32 inputs remains competitive with (and slightly
better than) ImageNet-pretrained features on CIFAR-10. Its per-class accuracy is weakest on the
semantically similar classes (`cat` 87.8%, `dog` 92.6%) — the expected CIFAR-10 failure mode.

> Metrics are reported on the held-out test split. A full top-to-bottom rerun of
> `notebooks/02_Model_Building.ipynb` takes roughly 3.5 hours on an RTX 4060, most of it the
> WRN-28-10 schedule.

## GPU Benchmarking (Part 3)

All three trained models were benchmarked on the RTX 4060 Laptop GPU under one documented protocol
(fixed 2,000-image seeded subset, 2 warm-up + best-of-3 timed passes, synchronized CUDA streams):

- Batch-size sweep (1 → 512) to find the throughput-optimal batch size, OOM-safe
- CPU vs GPU throughput and speedup per model (batch 128)
- Mixed-precision inference speedup (FP16 autocast vs FP32)
- Peak GPU memory footprint per model (weights + activations)

| Model | Optimal batch | GPU throughput (img/s) | GPU vs CPU | FP16 vs FP32 | Peak memory |
| :-- | :-- | :-- | :-- | :-- | :-- |
| CNN from scratch | 256 | 7,871 | 3.1x | 1.03x | 267 MB |
| EfficientNet-V2-S | 64 | 2,050 | 10.7x | 1.38x | 301 MB |
| WRN-28-10 | 32 | 887 | 20.2x | 1.77x | 635 MB |

Three takeaways: (1) batch-size tuning alone buys ~11x throughput on the CNN; (2) small, compute-light
networks are latency-bound (their FP16 gain is negligible), while the WRN — the heaviest model — gains
the most from FP16; (3) all three models comfortably fit in under 1 GB of VRAM, so the bottleneck for
GPU-bound inference is memory bandwidth, not capacity. Full per-batch numbers:
`outputs/benchmark_results/benchmark_summary.json`.

## Deployment (Part 4)

An interactive Gradio demo served by `app/gradio_app.py`. The app reads `models/model_metadata.json`
to discover **which** architecture won Part 2, imports it from `src/model_utils.py` (the same code the
notebooks use), and serves predictions from `models/best_model.pth`. One real CIFAR-10 test image per
class is committed in `app/examples/`, so the demo works out of the box.

### Run locally

```bash
python app/gradio_app.py
```

Open http://127.0.0.1:7860. Set `GRADIO_SHARE=1` for a temporary public link.

### Deploy to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space (SDK: Gradio).
2. Push this repository to it. The README frontmatter (`app_file: app/gradio_app.py`) tells Spaces
   where the app lives, so no renaming is needed.
3. Upload `models/best_model.pth` to the Space (web UI or Git LFS — `*.pth` is gitignored for GitHub,
   so the 146 MB checkpoint is never committed to the public repo).
4. Spaces builds and serves the demo automatically (CPU is fine: inference is ~50 ms/image).

Once your Space is live, drop this badge at the top of the README:

```markdown
[![Open in Spaces](https://img.shields.io/badge/Open%20in-Spaces-blue)](https://huggingface.co/spaces/YOUR-USERNAME/YOUR-SPACE-NAME)
```

## Getting Started

1. `pip install -r requirements.txt`
2. Run the notebooks in order: `01_Data_Exploration.ipynb` → `02_Model_Building.ipynb` →
   `03_GPU_Benchmarking.ipynb` (training takes ~3.5 h on an RTX 4060).
3. Launch the demo: `python app/gradio_app.py`

## Tech Stack

Python 3.10 · PyTorch 2.7.1+cu118 · TorchVision 0.22.1 · CUDA 11.8 · NumPy 2.0 · Pandas · Matplotlib ·
Seaborn · Gradio · NVIDIA RTX 4060 Laptop GPU (8.0 GB)
