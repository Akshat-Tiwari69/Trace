"""Unit tests for DiceBCELoss (task A4). CPU, no GPU."""

from __future__ import annotations

import pytest
import torch

from src.pipeline.p1_segment.losses import DiceBCELoss


def test_loss_is_scalar_and_differentiable():
    loss_fn = DiceBCELoss()
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    loss = loss_fn(logits, target)
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None


def test_confident_correct_beats_confident_wrong():
    loss_fn = DiceBCELoss()
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:6, 2:6] = 1
    # large positive logits where road is, negative elsewhere → near-perfect
    good = torch.where(target > 0.5, 8.0, -8.0)
    bad = -good
    assert loss_fn(good, target) < loss_fn(bad, target)


def test_bce_weight_validation():
    with pytest.raises(ValueError):
        DiceBCELoss(bce_weight=1.5)


def test_cldice_term_runs_and_differentiable():
    loss_fn = DiceBCELoss(bce_weight=0.5, cldice_weight=0.3)
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    target = (torch.rand(2, 1, 32, 32) > 0.5).float()
    loss = loss_fn(logits, target)
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None


def test_cldice_weight_validation():
    with pytest.raises(ValueError):
        DiceBCELoss(cldice_weight=1.5)


def test_soft_cldice_rewards_matching_topology():
    from src.pipeline.p1_segment.losses import soft_cldice
    target = torch.zeros(1, 1, 32, 32)
    target[..., 16, :] = 1.0                       # a horizontal road
    good = target.clone()                          # same road
    bad = torch.zeros(1, 1, 32, 32)
    bad[..., :, 16] = 1.0                          # perpendicular road (wrong topology)
    assert soft_cldice(good, target) < soft_cldice(bad, target)
