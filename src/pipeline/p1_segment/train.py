"""Training + evaluation loop for road segmentation (task A4).

Thin, importable helpers so the Colab/Kaggle notebook stays mostly
configuration. Uses AMP (mixed precision) — mandatory at 512², helpful at 256²
on 8 GB GPUs (``docs/Research.md``). On CPU, AMP is simply disabled.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import DataLoader

from src.pipeline.p1_segment.metrics import dice_score, iou_score


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str = "cpu",
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Run one training epoch; return the mean batch loss."""
    model.train()
    use_amp = scaler is not None and device != "cpu"
    total, n = 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda" if device != "cpu" else "cpu", enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, masks)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    threshold: float = 0.5,
    reduction: str = "global",
) -> dict[str, float]:
    """Evaluate IoU + Dice over a loader.

    ``reduction="global"`` (default) accumulates intersection/union over the
    **whole set** (paper-grade — unaffected by batch size or per-tile road
    coverage); ``"batch_mean"`` averages per-batch scores (legacy behaviour).
    """
    model.eval()
    if reduction == "batch_mean":
        iou_sum, dice_sum, n = 0.0, 0.0, 0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = (torch.sigmoid(model(images)) >= threshold).float()
            iou_sum += iou_score(preds, masks)
            dice_sum += dice_score(preds, masks)
            n += 1
        n = max(n, 1)
        return {"iou": iou_sum / n, "dice": dice_sum / n}

    # Accumulate on-device; convert once at the end (avoids a GPU sync per batch).
    inter = pred_sum = target_sum = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds = torch.sigmoid(model(images)) >= threshold
        targets = masks > 0.5
        inter = inter + (preds & targets).sum()
        pred_sum = pred_sum + preds.sum()
        target_sum = target_sum + targets.sum()
    inter, pred_sum, target_sum = float(inter), float(pred_sum), float(target_sum)
    union = pred_sum + target_sum - inter
    eps = 1e-7
    return {"iou": (inter + eps) / (union + eps),
            "dice": (2 * inter + eps) / (pred_sum + target_sum + eps)}
