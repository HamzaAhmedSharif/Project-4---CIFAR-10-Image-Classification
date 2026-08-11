# ================================================
# PROJECT 4: IMAGE CLASSIFICATION - STREAMLIT APP
# ================================================
# Deploy on Streamlit Cloud (FREE):
#   1. Upload this file + src/ + models/ + outputs/
#   2. Go to streamlit.io/cloud
#   3. Connect your GitHub and deploy!
# ================================================

import os
import json
import sys
from pathlib import Path

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EXAMPLES_DIR = PROJECT_ROOT / "app" / "examples"

# Add src to path so we can import model_utils
sys.path.insert(0, str(PROJECT_ROOT))
from src.model_utils import SimpleCNN, build_efficientnet_v2_s, WideResNet  # noqa: E402

# -----------------------------------------------------------------------------
# Load classes & normalization
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Load model (cached so it only loads once)
# -----------------------------------------------------------------------------
BUILDERS = {
    "cnn_scratch": lambda: SimpleCNN(num_classes=len(CLASSES)),
    "efficientnet_v2_s": lambda: build_efficientnet_v2_s(num_classes=len(CLASSES), device=DEVICE),
    "wrn28_10": lambda: WideResNet(depth=28, widen_factor=10, num_classes=len(CLASSES)),
}

@st.cache_resource
def load_model():
    checkpoint = MODEL_DIR / "best_model.pth"
    if not checkpoint.exists():
        st.error(f"Model not found at {checkpoint}. Please upload it.")
        return None

    metadata_file = MODEL_DIR / "model_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        key = metadata.get("best_model_key")
        if key in BUILDERS:
            try:
                model = BUILDERS[key]()
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
                model.eval()
                model.to(DEVICE)
                st.success(f"✅ Loaded: {key} on {DEVICE}")
                return model
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None
    st.error("model_metadata.json not found.")
    return None

# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------
PREPROCESS = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
])

# -----------------------------------------------------------------------------
# Prediction function
# -----------------------------------------------------------------------------
def predict(image):
    image = image.convert("RGB")
    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    model = load_model()
    if model is None:
        return None, None
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    prob_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_probs, prob_dict

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="CIFAR-10 Classifier", layout="centered")

st.title("🚀 CIFAR-10 Image Classifier")
st.markdown("**Best model: WRN-28-10 (from scratch) – 95.43% accuracy**")
st.markdown(f"Recognizes: {', '.join(CLASSES)}")

# Show example images
st.subheader("📸 Try an example image")
cols = st.columns(5)
example_paths = [EXAMPLES_DIR / f"{cls}.png" for cls in CLASSES]
selected_example = None

for i, path in enumerate(example_paths[:5]):  # Show first 5
    if path.exists():
        with cols[i]:
            if st.button(f"📷 {CLASSES[i]}"):
                selected_example = path

cols2 = st.columns(5)
for i, path in enumerate(example_paths[5:]):  # Show last 5
    if path.exists():
        with cols2[i]:
            if st.button(f"📷 {CLASSES[i+5]}"):
                selected_example = path

# File uploader
uploaded_file = st.file_uploader("Or upload your own image", type=["png", "jpg", "jpeg"])

# Determine which image to process
image_to_process = None
if uploaded_file:
    image_to_process = Image.open(uploaded_file)
elif selected_example:
    image_to_process = Image.open(selected_example)

# Display and predict
if image_to_process:
    st.image(image_to_process, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        sorted_probs, prob_dict = predict(image_to_process)

    if sorted_probs:
        # Top prediction
        top_class, top_conf = sorted_probs[0]
        st.success(f"### ✅ Prediction: **{top_class}**")
        st.metric("Confidence", f"{top_conf:.2%}")

        # Show top-3
        st.subheader("🔝 Top-3 Predictions")
        for cls, conf in sorted_probs[:3]:
            st.write(f"- **{cls}**: {conf:.2%}")

        # Show full distribution as a bar chart
        st.subheader("📊 Full Probability Distribution")
        chart_data = {cls: prob_dict[cls] for cls in CLASSES}
        st.bar_chart(chart_data)

# Footer
st.markdown("---")
st.caption("Built with PyTorch, Streamlit, and WRN-28-10")