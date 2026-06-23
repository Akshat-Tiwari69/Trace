"""Unit tests for segmentation metrics (task A4). CPU, no GPU."""

from __future__ import annotations

import torch

from src.pipeline.p1_segment.metrics import (
    binarize,
    dice_score,
    iou_score,
    occlusion_recall,
)


def test_iou_perfect_and_disjoint():
    a = torch.zeros(1, 1, 8, 8)
    a[..., :4, :4] = 1
    assert iou_score(a, a.clone()) == 1.0
    b = torch.zeros(1, 1, 8, 8)
    b[..., 4:, 4:] = 1
    assert iou_score(a, b) < 0.01  # no overlap


def test_iou_half_overlap():
    pred = torch.zeros(1, 1, 4, 4)
    pred[..., :, :2] = 1          # left half
    target = torch.zeros(1, 1, 4, 4)
    target[..., :2, :] = 1        # top half
    # intersection = top-left 2x2 = 4; union = 8 + 8 - 4 = 12 → 1/3
    assert abs(iou_score(pred, target) - 1 / 3) < 1e-3


def test_dice_perfect():
    a = torch.zeros(1, 1, 8, 8)
    a[..., 2:6, 2:6] = 1
    assert abs(dice_score(a, a.clone()) - 1.0) < 1e-4


def test_binarize_threshold():
    logits = torch.tensor([[-2.0, 2.0]])  # sigmoid → ~0.12, ~0.88
    out = binarize(logits, threshold=0.5)
    assert out.tolist() == [[0.0, 1.0]]


def test_occlusion_recall_recovers_hidden_road():
    target = torch.zeros(1, 1, 4, 4)
    target[..., 0, :] = 1                 # a road along the top row (4 px)
    occ = torch.zeros(1, 1, 4, 4)
    occ[..., 0, :2] = 1                   # left half of that row is occluded
    pred = torch.zeros(1, 1, 4, 4)
    pred[..., 0, 0] = 1                   # model recovered 1 of the 2 hidden px
    assert abs(occlusion_recall(pred, target, occ) - 0.5) < 1e-3


def test_occlusion_recall_full_recovery():
    target = torch.ones(1, 1, 4, 4)
    occ = torch.ones(1, 1, 4, 4)
    pred = torch.ones(1, 1, 4, 4)
    assert abs(occlusion_recall(pred, target, occ) - 1.0) < 1e-4
