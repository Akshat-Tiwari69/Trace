"""Loss functions for road segmentation (task A4).

``DiceBCELoss`` = BCE-with-logits + soft Dice, optionally + **soft-clDice**.
BCE gives stable per-pixel gradients; Dice counters the heavy road/not-road
imbalance (roads are a few % of pixels); **soft-clDice** (Shit et al., CVPR
2021) is a *topology* loss computed on the soft skeleton — it rewards keeping
roads **connected**, which is what matters for routing/APLS downstream
(``docs/Research.md`` plans a Dice + soft-clDice + BCE combination).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# soft-clDice — differentiable skeleton-based connectivity loss
# --------------------------------------------------------------------------- #
def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Morphological erosion via min-pooling (min = -maxpool(-x))."""
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """Morphological dilation via max-pooling."""
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 3) -> torch.Tensor:
    """Differentiable soft skeleton of a probability map (values in [0, 1])."""
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(probs: torch.Tensor, target: torch.Tensor,
                iters: int = 3, smooth: float = 1.0) -> torch.Tensor:
    """soft-clDice loss: 1 - clDice between predicted probs and target mask.

    Compares each mask's skeleton against the other mask, so breaking a road
    (losing topology) is penalised even if few pixels change.
    """
    skel_pred = soft_skeletonize(probs, iters)
    skel_true = soft_skeletonize(target, iters)
    tprec = (skel_pred * target).sum() + smooth
    tprec = tprec / (skel_pred.sum() + smooth)        # skeleton-vs-mask precision
    tsens = (skel_true * probs).sum() + smooth
    tsens = tsens / (skel_true.sum() + smooth)        # skeleton-vs-mask sensitivity
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)


# --------------------------------------------------------------------------- #
# Combined loss
# --------------------------------------------------------------------------- #
class DiceBCELoss(nn.Module):
    """BCE-with-logits + soft Dice, optionally blended with soft-clDice.

    Args:
        bce_weight: weight on BCE within the BCE+Dice base (Dice gets the rest).
        cldice_weight: if > 0, final = (1-w)·(BCE+Dice) + w·soft-clDice. 0 keeps
            the original Dice+BCE behaviour exactly.
        cldice_iters: skeletonisation iterations (≈ max road half-width in px).
        eps: smoothing constant for the Dice denominator.
    """

    def __init__(self, bce_weight: float = 0.5, cldice_weight: float = 0.0,
                 cldice_iters: int = 3, eps: float = 1e-7) -> None:
        super().__init__()
        if not 0.0 <= bce_weight <= 1.0:
            raise ValueError("bce_weight must be in [0, 1]")
        if not 0.0 <= cldice_weight <= 1.0:
            raise ValueError("cldice_weight must be in [0, 1]")
        self.bce_weight = bce_weight
        self.cldice_weight = cldice_weight
        self.cldice_iters = cldice_iters
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = self.bce(logits, target)

        probs = torch.sigmoid(logits)
        inter = (probs * target).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (1.0 - (2 * inter + self.eps) / (denom + self.eps)).mean()

        base = self.bce_weight * bce + (1.0 - self.bce_weight) * dice
        if self.cldice_weight > 0.0:
            cl = soft_cldice(probs, target, self.cldice_iters)
            return (1.0 - self.cldice_weight) * base + self.cldice_weight * cl
        return base


# --------------------------------------------------------------------------- #
# Lovász-hinge — a direct, differentiable surrogate for IoU (Berman et al. 2018)
# --------------------------------------------------------------------------- #
def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovász extension of the Jaccard loss for sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary Lovász-hinge loss, averaged over the batch. ``logits`` are raw."""
    losses = []
    for i in range(logits.size(0)):
        log_i = logits[i].reshape(-1)
        lab_i = target[i].reshape(-1)
        if lab_i.numel() == 0:
            losses.append(log_i.sum() * 0.0)
            continue
        signs = 2.0 * lab_i.float() - 1.0
        errors = 1.0 - log_i * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        grad = _lovasz_grad(lab_i[perm])
        losses.append(torch.dot(F.relu(errors_sorted), grad))
    return torch.stack(losses).mean()


class ComboLoss(nn.Module):
    """Weighted BCE + soft-Dice + Lovász-hinge (+ optional soft-clDice).

    Lovász directly optimises IoU, Dice handles class imbalance, BCE stabilises
    early training, and clDice rewards connectivity. ``cldice_weight`` is a plain
    attribute so a training loop can ramp it on in the final epochs.
    """

    def __init__(self, bce_weight: float = 0.4, dice_weight: float = 0.4,
                 lovasz_weight: float = 0.2, cldice_weight: float = 0.0,
                 cldice_iters: int = 3, eps: float = 1e-7) -> None:
        super().__init__()
        for w in (bce_weight, dice_weight, lovasz_weight, cldice_weight):
            if w < 0.0:
                raise ValueError("loss weights must be non-negative")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight
        self.cldice_weight = cldice_weight
        self.cldice_iters = cldice_iters
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        total = logits.sum() * 0.0  # zero with the right device/dtype/graph
        if self.bce_weight:
            total = total + self.bce_weight * self.bce(logits, target)
        probs = torch.sigmoid(logits)
        if self.dice_weight:
            inter = (probs * target).sum(dim=(1, 2, 3))
            denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
            dice = (1.0 - (2 * inter + self.eps) / (denom + self.eps)).mean()
            total = total + self.dice_weight * dice
        if self.lovasz_weight:
            total = total + self.lovasz_weight * lovasz_hinge(logits, target)
        if self.cldice_weight > 0.0:
            total = total + self.cldice_weight * soft_cldice(probs, target, self.cldice_iters)
        return total
