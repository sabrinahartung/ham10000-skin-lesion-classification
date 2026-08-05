# Skin Lesion Classification — HAM10000

Transfer learning with a pretrained **ResNet18** to classify 7 types of dermatoscopic
skin lesions, with a focus on handling severe class imbalance so that rare, dangerous
classes (like melanoma) are still detected.

## Overview

| | |
|---|---|
| **Dataset** | [marmal88/skin_cancer](https://huggingface.co/datasets/marmal88/skin_cancer) — HAM10000, ~10k dermatoscopic images, 7 lesion classes |
| **Model** | ResNet18 pretrained on ImageNet, final layer adapted to 7 classes (transfer learning) |
| **Framework** | PyTorch (Apple Silicon / MPS) |
| **Core challenge** | ~67% of images are benign nevi (`nv`) — handled with class-weighted cross-entropy loss |

## Notebook

📓 **[`SkinLesions.ipynb`](SkinLesions.ipynb)** — full walkthrough: data loading, EDA,
preprocessing, training, and per-class evaluation.

*An improvement notebook focused on melanoma recall is planned and will be linked here.*

## Results (baseline)

- Validation accuracy ≈ **0.81**, comparable on the held-out test set → good generalization.
- The class-weighted loss works: all 7 classes are predicted (no collapse to the majority
  class), and the rare classes reach high recall.
- **Key limitation:** ~14% of true melanomas are classified as benign on the test set —
  the clinically most important error, and the focus of the next iteration.

> ⚕️ *Educational project — not a medical device and not intended for diagnostic use.*

## Setup

This project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run jupyter lab
```

## Roadmap

- [ ] Improve melanoma recall (resampling, longer fine-tuning, unfreezing the backbone)
- [ ] Compare architectures (ConvNeXt-Tiny, ViT-B/16)
- [ ] Interactive demo (Gradio / Hugging Face Space)
