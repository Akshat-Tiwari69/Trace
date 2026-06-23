"""Segmentation model: build, checkpoint I/O, and CPU inference (task A4).

We fine-tune a **SegFormer MiT-B0 encoder + U-Net decoder** via
``segmentation_models_pytorch`` (smp 0.3.4 ships the MiT/SegFormer encoders but
not a standalone Segformer decoder, and the U-Net decoder gives full-resolution
masks directly). ``encoder_weights="imagenet"`` fine-tunes pretrained weights —
never trains from scratch (``docs/Rules.md``). ~5.5M params; fits 8 GB.

``predict_mask`` is the pipeline's ``predict(tile) -> mask_array`` (``TRD.md``):
a trained model turns one RGB tile into a binary {0,1} road mask on CPU, which
P2 then skeletonises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import segmentation_models_pytorch as smp
import torch

# ImageNet stats — the MiT-b0 encoder is pretrained on ImageNet, so train-time
# augmentation and inference must normalise with the same constants.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model(
    encoder: str = "mit_b0",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> torch.nn.Module:
    """Build the SegFormer-MiT(U-Net) road segmenter.

    ``encoder_weights="imagenet"`` downloads pretrained encoder weights (the
    fine-tune starting point); pass ``None`` for a random encoder (tests/offline).
    """
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,  # raw logits — DiceBCELoss/metrics apply sigmoid
    )


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module, unwrapping DataParallel/DDP if present."""
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    meta: dict[str, Any] | None = None,
) -> None:
    """Save model weights + metadata (encoder, metrics, config) to ``path``.

    Unwraps DataParallel/DDP so the saved keys match a plain model on reload.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": _unwrap(model).state_dict(), "meta": meta or {}}, path)


def load_checkpoint(
    path: str | Path,
    encoder: str | None = None,
    map_location: str = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild the model (random encoder, no download) and load saved weights.

    The encoder defaults to whatever the checkpoint's ``meta['encoder']`` says
    (so a mit_b2 checkpoint rebuilds as mit_b2); pass ``encoder`` to override.
    """
    ckpt = torch.load(path, map_location=map_location)
    enc = encoder or ckpt.get("meta", {}).get("encoder", "mit_b0")
    model = build_model(encoder=enc, encoder_weights=None)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("meta", {})


def _to_chw_tensor(image: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8/float image → normalised (1,3,H,W) float tensor."""
    arr = image.astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


@torch.no_grad()
def predict_mask(
    model: torch.nn.Module,
    image: np.ndarray,
    device: str = "cpu",
    threshold: float = 0.5,
) -> np.ndarray:
    """Predict a binary {0,1} road mask for one RGB tile (``predict(tile)``).

    ``image`` is ``HxWx3`` (uint8 0–255 or float 0–1). Returns a ``uint8``
    ``HxW`` mask of 0/1, the §4 contract shape P2 consumes.
    """
    net = _unwrap(model).to(device)
    net.eval()
    x = _to_chw_tensor(image).to(device)
    logits = net(x)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return (prob >= threshold).astype(np.uint8)
