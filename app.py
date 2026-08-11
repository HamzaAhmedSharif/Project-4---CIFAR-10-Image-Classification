# ================================================
# PROJECT 4: IMAGE CLASSIFICATION - GRADIO DEMO
# ================================================
#
# Interactive demo for the best CIFAR-10 model produced by Part 2.
# The app reads models/model_metadata.json to learn WHICH architecture
# won, then imports that architecture from src/model_utils.py (the same
# code the notebooks use) and serves predictions with Gradio.
#
# Run locally:
#     python app/gradio_app.py
#     -> http://127.0.0.1:7860
#
# Deploy to Hugging Face Spaces:
#     Upload this repository; Spaces reads models/best_model.pth from
#     the repo (add the file via the web UI or Git LFS - *.pth is
#     gitignored for GitHub). See README -> Deployment.
# ================================================

import os

# Windows OpenMP fix: PyTorch and NumPy each ship libiomp5md.dll; without
# this flag the process can abort with "OMP Error #15" before the app starts.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
from pathlib import Path

import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image

# -----------------------------------------------------------------------------
# Project paths (app/ lives one level below the project root)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
EXAMPLES_DIR = PROJECT_ROOT / "app" / "examples"  # tracked in git, ships with the repo

# Shared project code (single source of truth for the architectures)
sys.path.insert(0, str(PROJECT_ROOT))
from src.model_utils import SimpleCNN, build_efficientnet_v2_s, WideResNet  # noqa: E402

# -----------------------------------------------------------------------------
# Classes and normalization constants (computed in Part 1, used in Parts 2-4)
# -----------------------------------------------------------------------------
CLASSES_FILE = OUTPUT_DIR / "dataset_summary.json"
if CLASSES_FILE.exists():
    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        summary = json.load(f)
    CLASSES = list(summary["classes"])
    NORMALIZATION_MEAN = tuple(summary["normalization_constants"]["mean"])
    NORMALIZATION_STD = tuple(summary["normalization_constants"]["std"])
else:
    CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
    NORMALIZATION_MEAN = (0.4914, 0.4822, 0.4465)
    NORMALIZATION_STD = (0.2470, 0.2435, 0.2616)
    print(f"WARNING: {CLASSES_FILE} not found - using fallback constants.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Model loading (metadata-driven, never hardcoded)
# -----------------------------------------------------------------------------

# Architecture registry: keys must match models/model_metadata.json
BUILDERS = {
    "cnn_scratch": lambda: SimpleCNN(num_classes=len(CLASSES)),
    "efficientnet_v2_s": lambda: build_efficientnet_v2_s(num_classes=len(CLASSES), device=DEVICE),
    "wrn28_10": lambda: WideResNet(depth=28, widen_factor=10, num_classes=len(CLASSES)),
}
MODEL_NAMES = {
    "cnn_scratch": "CNN from scratch",
    "efficientnet_v2_s": "EfficientNet-V2-S (transfer)",
    "wrn28_10": "WRN-28-10 (from scratch)",
}


def _try_load(checkpoint: Path, key: str) -> torch.nn.Module:
    """Build one architecture and load a checkpoint; raise if they mismatch."""
    model = BUILDERS[key]()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()  # critical: BatchNorm must use running stats, not batch stats
    return model.to(DEVICE)


def load_best_model() -> torch.nn.Module:
    """Load the best checkpoint with the architecture recorded at training time.

    Priority:
      1. models/best_model.pth            (promoted winner, see Part 2)
      2. models/<best_key>_best.pth       (per-model checkpoint if promotion is stale)
    """
    checkpoint = MODEL_DIR / "best_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Model not found: {checkpoint.relative_to(PROJECT_ROOT)}.\n"
            "Run notebooks/02_Model_Building.ipynb (Part 2) first to train and "
            "promote the best model."
        )

    metadata_file = MODEL_DIR / "model_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        key = metadata.get("best_model_key")
        if key in BUILDERS:
            try:
                model = _try_load(checkpoint, key)
                print(f"Loaded: {MODEL_NAMES[key]} (best_model.pth, "
                      f"metadata: {metadata_file.relative_to(PROJECT_ROOT)})")
                return model
            except RuntimeError as exc:
                print(f"WARNING: architecture '{key}' from metadata failed to load "
                      f"({exc}); trying remaining architectures...")

    # Metadata missing or stale: probe every architecture and keep the first match
    print("Probing all architectures to match the checkpoint...")
    for key, name in MODEL_NAMES.items():
        try:
            model = _try_load(checkpoint, key)
            print(f"Loaded: {name} (best_model.pth, architecture auto-detected)")
            return model
        except RuntimeError:
            continue

    raise RuntimeError(
        f"Could not match {checkpoint.relative_to(PROJECT_ROOT)} to any known "
        "architecture. Re-run Part 2 (and delete the stale best_model.pth) to "
        "regenerate the checkpoint."
    )


model = load_best_model()
print(f"Device: {DEVICE}")
print(f"Classes ({len(CLASSES)}): {', '.join(CLASSES)}")
print(f"Mean: {NORMALIZATION_MEAN} | Std: {NORMALIZATION_STD}")

# -----------------------------------------------------------------------------
# Preprocessing - must mirror build_test_transform() in src/data_loader.py:
# 32x32 resize (models were trained on raw 32x32 inputs) + normalization
# -----------------------------------------------------------------------------
PREPROCESS = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
])


# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
def predict(image: Image.Image):
    """Classify one PIL image; return (top-3 dict, full distribution dict)."""
    if image is None:
        raise gr.Error("Please upload an image first.")

    image = image.convert("RGB")  # RGBA / grayscale / palette uploads -> RGB

    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    prob_dict = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
    top3 = dict(sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)[:3])
    return top3, prob_dict


# -----------------------------------------------------------------------------
# Example images - one real CIFAR-10 test image per class
# -----------------------------------------------------------------------------
def generate_examples() -> list[str]:
    """Return example-image paths, generating them if they are missing.

    The images live in app/examples/ and are committed to the repository, so
    deployed Spaces get them without needing the CIFAR-10 dataset. If they are
    absent (fresh clone without data/), they are regenerated from the local
    CIFAR-10 test set when it is available.
    """
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    expected = [EXAMPLES_DIR / f"{cls}.png" for cls in CLASSES]
    if all(p.exists() for p in expected):
        print(f"Examples: {len(expected)} images found in "
              f"{EXAMPLES_DIR.relative_to(PROJECT_ROOT)}")
        return [str(p) for p in expected]

    try:
        import torchvision.datasets as datasets

        testset = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=False)
    except Exception:
        print(f"WARNING: no example images and CIFAR-10 not found at {DATA_DIR} - "
              "skipping the example gallery.")
        return []

    for class_idx in range(len(CLASSES)):
        for img, label in testset:
            if label == class_idx:
                img.save(expected[class_idx])
                break
    print(f"Examples: {len(expected)} images generated -> "
          f"{EXAMPLES_DIR.relative_to(PROJECT_ROOT)}")
    return [str(p) for p in expected]


EXAMPLE_PATHS = generate_examples()

# -----------------------------------------------------------------------------
# Reported accuracy (optional, read from Part 2 results)
# -----------------------------------------------------------------------------
RESULTS_FILE = OUTPUT_DIR / "model_results.json"
reported_accuracy = None
if RESULTS_FILE.exists():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
    best = results.get("best_model", {})
    reported_accuracy = best.get("test_accuracy")

description = (
    "**Project 4: Image Classification with GPU Optimization**\n\n"
    f"Best model: **WRN-28-10 (from scratch)** trained on CIFAR-10"
    + (f" - test accuracy **{reported_accuracy:.2f}%**" if reported_accuracy else "")
    + "\n\nUpload an image (or pick an example below). The model recognizes: "
    f"{', '.join(CLASSES)}."
)

# -----------------------------------------------------------------------------
# Gradio interface
# -----------------------------------------------------------------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image (or use webcam)"),
    outputs=[
        gr.Label(num_top_classes=3, label="Top-3 Predictions"),
        gr.Label(num_top_classes=len(CLASSES), label="Full Probability Distribution"),
    ],
    title="CIFAR-10 Image Classifier",
    description=description,
    examples=EXAMPLE_PATHS if EXAMPLE_PATHS else None,
    cache_examples=False,
)

# -----------------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # share=True generates a temporary public URL; enable with: set GRADIO_SHARE=1
    share = os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "yes")
    print("Launching Gradio app...")
    print("Local URL : http://127.0.0.1:7860")
    if share:
        print("Public URL: a shareable link will be printed below")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share,
                theme=gr.themes.Soft())
