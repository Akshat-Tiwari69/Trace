"""Unit tests for the segmentation model + inference (task A4). CPU, no GPU.

Built with ``encoder_weights=None`` so tests never hit the network.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.pipeline.p1_segment.model import (
    build_model,
    load_checkpoint,
    predict_large,
    predict_mask,
    save_checkpoint,
)


def test_build_model_scse_unet_forwards():
    model = build_model(encoder_weights=None, decoder_attention_type="scse")
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 1, 64, 64)


def test_build_model_rejects_unknown_arch():
    with pytest.raises(ValueError):
        build_model(encoder_weights=None, arch="nope")


def test_build_model_rejects_attention_on_non_unet():
    with pytest.raises(ValueError):
        build_model(encoder_weights=None, arch="fpn", decoder_attention_type="scse")


def test_checkpoint_roundtrip_rebuilds_scse_arch(tmp_path):
    # The scse decoder adds parameters; load_checkpoint must read arch from meta
    # or load_state_dict would fail on mismatched keys.
    model = build_model(encoder_weights=None, decoder_attention_type="scse")
    image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    before = predict_mask(model, image)
    ckpt = tmp_path / "scse.pt"
    save_checkpoint(model, ckpt, meta={"encoder": "mit_b0", "arch": "unet",
                                       "decoder_attention_type": "scse"})
    reloaded, meta = load_checkpoint(ckpt)
    after = predict_mask(reloaded, image)
    assert np.array_equal(before, after)
    assert meta["decoder_attention_type"] == "scse"


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


def test_predict_large_stitches_to_full_size():
    model = build_model(encoder_weights=None)
    image = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)  # > one tile, ragged
    mask = predict_large(model, image, tile_size=256)
    assert mask.shape == (300, 400)         # stitched back to the original size
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})


def test_predict_large_matches_predict_mask_on_single_tile():
    model = build_model(encoder_weights=None)
    image = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    assert np.array_equal(predict_large(model, image, tile_size=256),
                          predict_mask(model, image))


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


def test_predict_mask_tta_runs_and_is_binary_hw():
    model = build_model(encoder_weights=None)
    image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    base = predict_mask(model, image)
    tta = predict_mask(model, image, tta=True)            # D4 averaging path
    assert tta.shape == base.shape == (64, 64)
    assert tta.dtype == np.uint8
    assert set(np.unique(tta)).issubset({0, 1})


def test_predict_large_tta_runs():
    model = build_model(encoder_weights=None)
    image = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
    mask = predict_large(model, image, tile_size=256, tta=True)
    assert mask.shape == (300, 400) and mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})


def test_predict_prob_returns_probabilities():
    import numpy as np
    from src.pipeline.p1_segment.model import build_model, predict_prob
    model = build_model(encoder_weights=None)
    image = np.random.default_rng(0).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    prob = predict_prob(model, image)
    assert prob.shape == (128, 128)
    assert prob.min() >= 0.0 and prob.max() <= 1.0


def test_predict_large_prob_blends_and_covers():
    import numpy as np
    from src.pipeline.p1_segment.model import build_model, predict_prob, predict_large_prob
    model = build_model(encoder_weights=None)
    # larger-than-tile image -> overlapping windows must cover every pixel, stay in [0,1]
    image = np.random.default_rng(1).integers(0, 255, (200, 260, 3), dtype=np.uint8)
    prob = predict_large_prob(model, image, tile_size=128, stride=96)
    assert prob.shape == (200, 260)
    assert prob.min() >= 0.0 and prob.max() <= 1.0
    assert not np.isnan(prob).any()
    # on a uniform image the blended prob ~ a single-tile prediction (no seam artifacts)
    uniform = np.full((200, 260, 3), 120, np.uint8)
    blended = predict_large_prob(model, uniform, tile_size=128, stride=96)
    single = predict_prob(model, uniform[:128, :128])
    assert abs(float(blended[64, 64]) - float(single[64, 64])) < 0.05
