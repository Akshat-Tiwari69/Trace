"""Smoke test for the train/eval loop (task A4). CPU, tiny synthetic data."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pipeline.p1_segment.losses import DiceBCELoss
from src.pipeline.p1_segment.model import build_model
from src.pipeline.p1_segment.train import evaluate, train_one_epoch


def _tiny_loader(n: int = 2) -> DataLoader:
    images = torch.randn(n, 3, 64, 64)
    masks = (torch.rand(n, 1, 64, 64) > 0.5).float()
    return DataLoader(TensorDataset(images, masks), batch_size=2)


def test_train_one_epoch_returns_loss_and_steps():
    model = build_model(encoder_weights=None)
    loader = _tiny_loader()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, loader, opt, DiceBCELoss(), device="cpu")
    assert isinstance(loss, float) and loss > 0


def test_evaluate_reports_iou_and_dice():
    model = build_model(encoder_weights=None)
    scores = evaluate(model, _tiny_loader(), device="cpu")
    assert set(scores) == {"iou", "dice"}
    assert 0.0 <= scores["iou"] <= 1.0
    assert 0.0 <= scores["dice"] <= 1.0
