"""
Skin Lesion Classifier — Decision-Support Demo (educational).

Streamlit app: upload a dermatoscopic image (or pick an example); a ResNet18 model
(transfer-learned on HAM10000) returns its confidence across 7 lesion types.

⚕️  NOT a medical device. Not for diagnostic use. Educational proof-of-concept only.
"""
import base64
from pathlib import Path

import matplotlib
import numpy as np
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


def gradcam(model, img, class_idx):
    """Grad-CAM heatmap for `class_idx`, as a raw (un-normalised) [7, 7] tensor.

    Weights the last conv block's feature maps by the gradient of that class's
    logit w.r.t. them: channels the score reacts to strongly count most.

    Deliberately *not* normalised here — the caller scales several classes
    against one shared maximum, so a weak explanation still looks weak.
    Per-map normalisation would stretch pure noise to full contrast.
    """
    x = preprocess(img.convert("RGB")).unsqueeze(0)
    layer = model.layer4[-1]  # last residual block -> 7x7 feature maps

    activations, gradients = {}, {}
    handles = [
        layer.register_forward_hook(lambda m, inp, out: activations.update(v=out)),
        layer.register_full_backward_hook(lambda m, gi, go: gradients.update(v=go[0])),
    ]
    try:
        logits = model(x)
        model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
    finally:
        for h in handles:
            h.remove()

    acts, grads = activations["v"][0], gradients["v"][0]  # both [C, 7, 7]
    weights = grads.mean(dim=(1, 2), keepdim=True)
    # ReLU keeps only evidence *for* the class; negative contributions are dropped.
    return (weights * acts).sum(dim=0).relu().detach()


def overlay_cam(img, cam, scale, alpha=0.5):
    """Blend the heatmap over the image (jet colormap, red = most influential).

    `scale` is the shared denominator across the classes shown side by side.
    """
    cam = (cam / scale).clamp(0, 1)
    img = img.convert("RGB")
    # Training resizes to a square 224x224, so scaling the CAM back to the
    # original size undoes exactly that squash — no misalignment.
    cam_img = Image.fromarray((cam.numpy() * 255).astype("uint8")).resize(
        img.size, Image.BICUBIC
    )
    heat = matplotlib.colormaps["jet"](np.asarray(cam_img) / 255.0)[..., :3]
    return Image.blend(img, Image.fromarray((heat * 255).astype("uint8")), alpha)


# A map whose maximum is (near) zero has no positive evidence at all: the ReLU
# above wiped it out. Below WEAK_FRACTION of the strongest map, what's left is
# mostly noise and shouldn't be read as an explanation.
FLAT_EPS = 1e-6
WEAK_FRACTION = 0.25
# Gradient magnitude only loosely tracks confidence — a class scoring 0.1% can
# still produce a mid-strength map, because the gradient says "what would raise
# this score", not "what the model believes". So gate on the probability too:
# below this the model has effectively ruled the class out, whatever its map
# looks like.
MIN_PROB = 0.05


# ---------------- UI ----------------
st.set_page_config(page_title="HAM10000 Skin Lesion Classifier", page_icon="🔬")
st.title("HAM10000 Skin Lesion Classifier")
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

# ----- Card 3: where the model looked -----
with st.container(border=True):
    st.markdown("#### 3 · Explainability: Where the model *looked* (Grad-CAM)")
    if img is not None:
        # Only the two classes actually in contention. For a class the model
        # rejects, its reason is the *absence* of features — which Grad-CAM's
        # ReLU discards by construction, leaving noise or an empty map.
        top2 = df.index[:2].tolist()
        cams = {c: gradcam(model, img, CLASSES.index(c)) for c in top2}
        # One shared scale for both maps, so their strengths stay comparable.
        scale = max(FLAT_EPS, *(float(c.max()) for c in cams.values()))

        cols = st.columns(3)
        with cols[0]:
            st.image(img, caption="Input", width="stretch")
        for col, cls in zip(cols[1:], top2):
            with col:
                cam, peak = cams[cls], float(cams[cls].max())
                if peak <= FLAT_EPS:
                    st.info(
                        f"**{cls}** · {probs[cls] * 100:.1f}%\n\n"
                        "No positive evidence anywhere in the image — the model rejects "
                        "this class because features are *missing*, which Grad-CAM cannot "
                        "show."
                    )
                else:
                    st.image(overlay_cam(img, cam, scale),
                             caption=f"{cls} · {probs[cls] * 100:.1f}%", width="stretch")
                    if probs[cls] < MIN_PROB:
                        st.caption(
                            f"⚠️ The model has ruled this class out ({probs[cls] * 100:.1f}%). "
                            "The map shows which regions *would* speak for it — not a reason "
                            "behind the actual prediction."
                        )
                    elif peak < WEAK_FRACTION * scale:
                        st.caption(
                            f"⚠️ Weak signal ({peak / scale * 100:.0f}% of the map beside "
                            "it) — closer to noise than to an explanation."
                        )
        st.caption(
            "Red marks the regions that pushed that class's score up, blue the ones that "
            "barely mattered; both maps share one colour scale, so a paler map really is "
            "weaker evidence. They are 7×7 pixels upscaled — read them as a rough area, "
            "not a lesion border. Attention on skin, ruler marks or vignetting is a hint "
            "the model latched onto an artefact rather than the lesion."
        )
    else:
        st.info("⬆️ Pick an image above to see the Grad-CAM heatmaps.")

st.markdown(
    "---\n**How it was built:** "
    "Full notebooks & honest analysis: "
    "[GitHub repo](https://github.com/sabrinahartung/ham10000-skin-lesion-classification)."
)
