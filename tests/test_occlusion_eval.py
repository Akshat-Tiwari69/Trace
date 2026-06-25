"""Tests for the heavy occlusion aug + occlusion-recall eval (task A8). CPU."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.pipeline.p1_segment.dataset import build_train_transform
from src.pipeline.p1_segment.model import build_model
from src.pipeline.p1_segment.occlusion_eval import evaluate_occlusion_recall, occlude


def test_heavy_transform_builds_and_applies():
    image = (np.random.rand(96, 96, 3) * 255).astype(np.uint8)
    mask = (np.random.rand(96, 96) > 0.5).astype(np.uint8)
    for occ in (False, True, "heavy"):
        t = build_train_transform(64, occlusion=occ)
        out = t(image=image, mask=mask)
        assert out["image"].shape == (3, 64, 64)        # CHW tensor after ToTensorV2
        assert out["mask"].shape == (64, 64)


def test_occlude_blanks_boxes_and_returns_mask():
    img = np.full((64, 64, 3), 200, np.uint8)
    occ_img, occ = occlude(img, np.random.default_rng(0))
    assert occ.sum() > 0                                 # some pixels occluded
    assert (occ_img[occ.astype(bool)] == 0).all()        # occluded region is black
    assert (occ_img[~occ.astype(bool)] == 200).all()     # rest untouched


def _write_pair(folder, stem, size=80):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(stem)) % 2**32)
    Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(folder / f"{stem}_sat.jpg")
    m = np.zeros((size, size), np.uint8)
    m[size // 2 - 2 : size // 2 + 2, :] = 255
    Image.fromarray(m, mode="L").save(folder / f"{stem}_mask.png")


def test_evaluate_occlusion_recall_runs(tmp_path):
    for i in range(2):
        _write_pair(tmp_path, f"p{i}")
    pairs = [(tmp_path / f"p{i}_sat.jpg", tmp_path / f"p{i}_mask.png") for i in range(2)]
    model = build_model(encoder_weights=None, decoder_attention_type="scse")
    res = evaluate_occlusion_recall(model, pairs, device="cpu", tile_size=64)
    assert 0.0 <= res["occlusion_recall"] <= 1.0
    assert 0.0 <= res["clean_iou"] <= 1.0
    assert res["n_pairs"] == 2
