"""Segmentation metrics for road masks (task A4).

IoU and Dice are the standard pixel-overlap scores; **Occlusion-Recall** is the
project-specific metric from ``docs/Evaluation.md`` — recall measured *only*
inside artificially occluded regions, i.e. how well the model "sees through"
trees/buildings/vehicles (the core brief, ``docs/Research.md``).

All functions take binary ``{0,1}`` tensors (or anything that compares equal to
1 for road). Use ``binarize`` to turn logits into a mask first.
"""

from __future__ import annotations

import torch


def binarize(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Sigmoid + threshold → a binary {0,1} float mask, same shape as input."""
    return (torch.sigmoid(logits) >= threshold).float()


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    """Intersection-over-Union of two binary masks (Jaccard)."""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return float((inter + eps) / (union + eps))


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    """Dice / F1 of two binary masks (2·|A∩B| / (|A|+|B|))."""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum()
    return float((2 * inter + eps) / (pred.sum() + target.sum() + eps))


def occlusion_recall(
    pred: torch.Tensor,
    target: torch.Tensor,
    occlusion_mask: torch.Tensor,
    eps: float = 1e-7,
) -> float:
    """Recall of road pixels that fall **inside the occluded region**.

    ``occlusion_mask`` is 1 where the input was deliberately occluded (e.g. the
    CoarseDropout boxes used in training/eval). Of the true road pixels hidden
    under occlusion, what fraction did the model still predict? High values mean
    the model recovers roads it cannot directly see — exactly the project goal.
    """
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    occ = (occlusion_mask > 0.5).float()

    hidden_road = target * occ          # true road pixels under occlusion
    recovered = pred * hidden_road      # of those, the ones we predicted
    return float((recovered.sum() + eps) / (hidden_road.sum() + eps))
