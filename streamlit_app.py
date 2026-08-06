"""
Skin Lesion Classifier — Decision-Support Demo (educational).

Streamlit app: upload a dermatoscopic image (or pick an example); a ResNet18 model
(transfer-learned on HAM10000) returns its confidence across 7 lesion types.

⚕️  NOT a medical device. Not for diagnostic use. Educational proof-of-concept only.
"""
import base64
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from huggingface_hub import hf_hub_download
from PIL import Image
import streamlit as st
from st_clickable_images import clickable_images

# --- Classes: exact sorted order used during training (index i -> CLASSES[i]) ---
CLASSES = [
    "actinic_keratoses",
    "basal_cell_carcinoma",
    "benign_keratosis-like_lesions",
    "dermatofibroma",
    "melanocytic_Nevi",
    "melanoma",
    "vascular_lesions",
]
NUM_CLASSES = len(CLASSES)

# Trained weights live in a free HF *model* repo (model/dataset repos stay free)
HF_REPO = "sabrinahartung1010/skin-lesion-resnet18"
WEIGHTS_FILE = "resnet18_ham10000_classweights.pt"


@st.cache_resource
def load_model():
    """Download weights once, rebuild the architecture, load them in."""
    weights_path = hf_hub_download(repo_id=HF_REPO, filename=WEIGHTS_FILE)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(model, img):
    x = preprocess(img.convert("RGB")).unsqueeze(0)  # [1, 3, 224, 224]
    with torch.no_grad():
        probs = model(x).softmax(dim=1)[0]
    return {CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)}


# ---------------- UI ----------------
st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🔬")
st.title("Skin Lesion Classifier")
st.header("Decision-Support Demo")
st.markdown(
    "Upload a dermatoscopic image (or pick an example). The model — a **ResNet18** "
    "transfer-learned on **HAM10000** — returns its confidence across 7 lesion types."
)
st.warning(
    "⚕️ Educational proof-of-concept — **not a medical device and not for diagnostic use.**"
)

model = load_model()

example_files = sorted(str(p) for p in Path("examples").glob("*.jpg"))

# True labels for the example images — revealed only AFTER prediction, to show
# whether the model was right. Not shown in the UI before you click.
EXAMPLE_LABELS = {
    "example_1.jpg": "melanoma",
    "example_2.jpg": "melanocytic_Nevi",
    "example_3.jpg": "basal_cell_carcinoma",
    "example_4.jpg": "actinic_keratoses",
    "example_5.jpg": "benign_keratosis-like_lesions",
    "example_6.jpg": "dermatofibroma",
    "example_7.jpg": "vascular_lesions",
}

@st.cache_data
def example_data_uris(paths):
    """Encode example images as base64 data URIs for the clickable gallery."""
    uris = []
    for p in paths:
        with open(p, "rb") as f:
            uris.append("data:image/jpeg;base64," + base64.b64encode(f.read()).decode())
    return uris


# ----- Card 1: choose an image -----
with st.container(border=True):
    st.markdown("#### 1 · Choose an image")
    uploaded = st.file_uploader("Upload a dermatoscopic image", type=["jpg", "jpeg", "png"])
    st.markdown("**…or click an example:**")
    clicked = clickable_images(
        example_data_uris(example_files),
        div_style={"display": "flex", "flex-wrap": "wrap", "gap": "8px"},
        img_style={"height": "90px", "border-radius": "6px", "cursor": "pointer"},
    )

# An uploaded file always wins; otherwise use the clicked example (-1 = nothing clicked)
img, true_label = None, None
if uploaded is not None:
    img = Image.open(uploaded)
elif clicked > -1:
    path = example_files[clicked]
    img = Image.open(path)
    true_label = EXAMPLE_LABELS.get(Path(path).name)

# ----- Card 2: model prediction -----
with st.container(border=True):
    st.markdown("#### 2 · Model prediction")
    if img is not None:
        left, right = st.columns([1, 2])
        with left:
            st.image(img, caption="Input", width=200)
        with right:
            probs = predict(model, img)
            df = pd.DataFrame({"confidence": probs}).sort_values("confidence", ascending=False)
            top = df.index[0]
            st.markdown(f"**Top prediction:** {top} · {df.iloc[0, 0] * 100:.1f}%")
            if true_label is not None:
                if top == true_label:
                    st.success(f"✅ Correct — true label: **{true_label}**")
                else:
                    st.error(f"❌ Predicted **{top}**, true label: **{true_label}**")
            else:
                st.caption("ℹ️ Ground truth is unknown for uploaded images.")
        st.bar_chart(df)
    else:
        st.info("⬆️ Upload an image or click an example above to see a prediction.")

st.markdown(
    "---\n**How it was built:** ResNet18 transfer learning on HAM10000, handling severe "
    "class imbalance (~67% benign nevi), evaluated with a focus on melanoma recall. "
    "Full notebooks & honest analysis: "
    "[GitHub repo](https://github.com/sabrinahartung/ham10000-skin-lesion-classification)."
)
