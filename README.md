<div align="center">

# Image Classification with GPU Optimization

### End-to-End Deep Learning Pipeline on CIFAR-10

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

# Project 4: Image Classification with GPU Optimization

##  Overview

A production-ready computer vision pipeline built on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) using PyTorch and CUDA acceleration. This project demonstrates end-to-end ML engineering best practices through three comprehensive stages: **Exploratory Data Analysis → Model Development → GPU Benchmarking**.

###  Key Highlights

-  **95.85% Test Accuracy** — SOTA performance on CIFAR-10 (EfficientNet-V2-S)
-  **GPU-Optimized Training** — Full CUDA acceleration with mixed-precision support
-  **Production-Ready Architecture** — Modular, reusable, and well-documented codebase
-  **Comprehensive Benchmarking** — Detailed performance analysis across CPU/GPU, batch sizes, and precision modes
-  **Reproducible Research** — Fixed seeds, documented experiments, and complete artifacts

## Project Structure

```
Project 4 - CIFAR-10 Image Classification/
|-- notebooks/
|   |-- 01_Data_Exploration.ipynb   # Part 1 - EDA (complete)
|   |-- 02_Model_Building.ipynb     # Part 2 - training & comparison (complete)
|   `-- 03_GPU_Benchmarking.ipynb   # Part 3 - GPU benchmarking (complete)
|-- src/                            # shared package: models, data loading, training loop
|   |-- model_utils.py              #   SimpleCNN, EfficientNet-V2-S builder, WRN-28-10
|   |-- data_loader.py              #   transforms + DataLoader construction
|   `-- train_utils.py              #   train_one_epoch / evaluate / train_model
|-- data/raw/                       # CIFAR-10 (downloaded, not tracked in git)
|-- outputs/                        # dataset summary, model results, benchmark results, plots
|-- models/                         # best checkpoints + provenance metadata
|-- requirements.txt
|-- setup.py
`-- README.md
```

## Pipeline

| Part | Stage | Status | Key artifacts |
| :-- | :-- | :-- | :-- |
| 1 | Exploratory Data Analysis | Done | `outputs/dataset_summary.json`, `outputs/plots/*.png` |
| 2 | Model Building & Training | Done | `models/*_best.pth`, `outputs/model_results.json`, `outputs/plots/model_comparison.png`, `outputs/plots/confusion_matrix.png` |
| 3 | GPU Benchmarking | Done | `outputs/benchmark_results/`, `outputs/plots/{batch_size_benchmark,cpu_gpu_comparison,mixed_precision_benchmark,memory_usage}.png` |

## Models (Part 2)

### Model Selection

- **CNN scratch** — baseline to verify the training pipeline end-to-end.
- **EfficientNet-V2-S** — transfer learning with a modern architecture (2021); converges fast
  and achieves the highest test accuracy (95.85%).
- **WRN-28-10** — from-scratch state-of-the-art (2016), implemented from the original paper
  (pre-activation blocks, width factor 10); still highly competitive on CIFAR-10 at 95.54%.

Three competing architectures, each trained under a shared protocol (cross-entropy, best-epoch
checkpointing) with model-appropriate optimizers — the CNN and EfficientNet use Adam, while the
WRN-28-10 follows the paper's recipe: SGD + Nesterov momentum (lr 0.1), weight decay 5e-4,
dropout 0.3 in every block, and cosine annealing over 50 epochs:

| Model | Type | Parameters | Notes |
| :-- | :-- | :-- | :-- |
| CNN from scratch | Baseline | ~0.9M | 4 conv blocks (32→256 ch) + batch norm + dropout; 50 epochs |
| EfficientNet-V2-S | Transfer learning | ~20.2M | ImageNet weights; stem stride 1 fix for 32×32; RandAugment + label smoothing + head dropout 0.3 |
| WRN-28-10 | From scratch | ~36.7M | Paper-compliant Wide ResNet (Zagoruyko & Komodakis, 2016); SGD + Nesterov, weight decay 5e-4, block dropout 0.3, cosine annealing, 50 epochs |

The winner's best checkpoint is promoted to `models/best_model.pth` — the single artifact consumed
by Part 3 — with provenance recorded in `models/model_metadata.json`. The best model is
**EfficientNet-V2-S (95.85%)**, selected for its superior accuracy and faster inference speed.

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
| CNN from scratch (50 epochs) | 0.9M | **86.80** | 45 | ~17 min |
| EfficientNet-V2-S (30 epochs) | 20.2M | **95.85** | 28 | ~33 min |
| WRN-28-10 (50 epochs) | 36.7M | **95.54** | 49 | ~3.3 h |

The winner — **EfficientNet-V2-S (95.85%)** — outperforms the from-scratch WRN-28-10 while training
in a fraction of the time, confirming that transfer learning with proper adaptation (stem stride fix,
RandAugment, label smoothing) remains highly effective on CIFAR-10. Its per-class accuracy is
weakest on the semantically similar classes (`cat` 89.4%, `dog` 93.6%) — the expected CIFAR-10
failure mode.

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
| CNN from scratch | 512 | 8,219 | 2.9x | 0.95x | 267 MB |
| EfficientNet-V2-S | 64 | 2,082 | 9.4x | 1.26x | 301 MB |
| WRN-28-10 | 32 | 899 | 20.5x | 1.79x | 635 MB |

Three takeaways: (1) batch-size tuning alone buys ~10x throughput on the CNN; (2) small, compute-light
networks are latency-bound (their FP16 gain is negligible), while the WRN — the heaviest model — gains
the most from FP16; (3) all three models comfortably fit in under 1 GB of VRAM, so the bottleneck for
GPU-bound inference is memory bandwidth, not capacity. Full per-batch numbers:
`outputs/benchmark_results/benchmark_summary.json`.

## Reproducing the Results

### Prerequisites

```bash
git lfs install
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118
```

### Run the Notebooks in Order

```bash
jupyter notebook
```

1. `notebooks/01_Data_Exploration.ipynb` → generates `outputs/dataset_summary.json` and EDA plots
2. `notebooks/02_Model_Building.ipynb` → trains all 3 models, saves checkpoints to `models/`, produces `outputs/model_results.json`
3. `notebooks/03_GPU_Benchmarking.ipynb` → benchmarks all trained models, saves `outputs/benchmark_results/`

### Verify the Artifacts

```bash
# Check model checkpoints exist
ls -lh models/

# View results
cat outputs/model_results.json
cat outputs/benchmark_results/benchmark_summary.json
```

The winning checkpoint `models/best_model.pth` (EfficientNet-V2-S, 95.85%) is tracked via Git LFS and included in the repo.

##  Acknowledgments

- **Dataset**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) by Alex Krizhevsky
- **WRN Architecture**: [Wide Residual Networks](https://arxiv.org/abs/1605.07146) by Zagoruyko & Komodakis (2016)
- **EfficientNet**: [EfficientNetV2](https://arxiv.org/abs/2104.00298) by Tan & Le (2021)

##  Contact

For questions or feedback, please open an issue or reach out via [hamzaahmedsharif@example.com](mailto:hamzaahmedsharif@example.com).

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star!**

Made with ❤️ and PyTorch

</div>
