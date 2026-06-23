"""Loss functions for road segmentation (task A4).

``DiceBCELoss`` = BCE-with-logits + soft Dice. BCE gives stable per-pixel
gradients; Dice counters the heavy road/not-road class imbalance (roads are a
few % of pixels — see the ~5.65% measured in A3). ``docs/Research.md`` plans a
Dice + soft-clDice + BCE combination; clDice (topology loss) can be layered on
later, but Dice+BCE is the solid, well-tested baseline to fine-tune with.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """Combined BCE-with-logits and soft Dice loss for binary masks.

    Args:
        bce_weight: weight on the BCE term; Dice gets ``1 - bce_weight``.
        eps: smoothing constant for the Dice denominator.
    """

    def __init__(self, bce_weight: float = 0.5, eps: float = 1e-7) -> None:
        super().__init__()
        if not 0.0 <= bce_weight <= 1.0:
            raise ValueError("bce_weight must be in [0, 1]")
        self.bce_weight = bce_weight
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = self.bce(logits, target)

        probs = torch.sigmoid(logits)
        inter = (probs * target).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1.0 - (2 * inter + self.eps) / (denom + self.eps)
        dice = dice.mean()

        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice
