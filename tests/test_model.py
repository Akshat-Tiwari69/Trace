"""Unit tests for the segmentation model + inference (task A4). CPU, no GPU.

Built with ``encoder_weights=None`` so tests never hit the network.
"""

from __future__ import annotations

import numpy as np
import torch

from src.pipeline.p1_segment.model import (
    build_model,
    load_checkpoint,
    predict_mask,
    save_checkpoint,
)


def test_build_and_forward_full_resolution():
    model = build_model(encoder_weights=None)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 1, 64, 64)  # U-Net decoder → full-res logits


def test_predict_mask_is_binary_hw():
    model = build_model(encoder_weights=None)
    image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    mask = predict_mask(model, image)
    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})


def test_save_checkpoint_unwraps_dataparallel(tmp_path):
    # DataParallel can be constructed on CPU (no forward needed); save must
    # unwrap .module so the keys reload into a plain model.
    model = build_model(encoder_weights=None)
    dp = torch.nn.DataParallel(model)
    ckpt = tmp_path / "dp.pt"
    save_checkpoint(dp, ckpt, meta={"encoder": "mit_b0"})
    reloaded, _ = load_checkpoint(ckpt)
    assert reloaded.state_dict().keys() == model.state_dict().keys()


def test_checkpoint_roundtrip(tmp_path):
    model = build_model(encoder_weights=None)
    image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    before = predict_mask(model, image)

    ckpt = tmp_path / "seg.pt"
    save_checkpoint(model, ckpt, meta={"encoder": "mit_b0", "iou": 0.5})
    reloaded, meta = load_checkpoint(ckpt)

    after = predict_mask(reloaded, image)
    assert np.array_equal(before, after)        # weights preserved exactly
    assert meta["encoder"] == "mit_b0" and meta["iou"] == 0.5
