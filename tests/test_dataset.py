"""Unit tests for RoadTileDataset + augmentation (task A4).

Needs albumentations + opencv; skipped cleanly if albumentations is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("albumentations")
pytest.importorskip("cv2")

import cv2  # noqa: E402

from src.pipeline.p1_segment.dataset import (  # noqa: E402
    RoadTileDataset,
    build_train_transform,
    build_val_transform,
    pair_deepglobe,
)


def _make_pair(root, name="1", size=80):
    img = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    mask = (np.random.rand(size, size) > 0.5).astype(np.uint8) * 255  # DeepGlobe 0/255
    cv2.imwrite(str(root / f"{name}_sat.jpg"), img)
    cv2.imwrite(str(root / f"{name}_mask.png"), mask)


def test_pair_deepglobe_finds_pairs(tmp_path):
    _make_pair(tmp_path, "1")
    _make_pair(tmp_path, "2")
    (tmp_path / "3_sat.jpg").write_bytes(b"")  # unpaired image → ignored
    pairs = pair_deepglobe(tmp_path)
    assert len(pairs) == 2


def test_train_item_shapes_and_binary_mask(tmp_path):
    _make_pair(tmp_path, "1")
    ds = RoadTileDataset(pair_deepglobe(tmp_path), build_train_transform(64, occlusion=True))
    image, mask = ds[0]
    assert image.shape == (3, 64, 64)
    assert mask.shape == (1, 64, 64)
    assert set(mask.unique().tolist()).issubset({0.0, 1.0})


def test_val_transform_is_deterministic_size(tmp_path):
    _make_pair(tmp_path, "1", size=120)
    ds = RoadTileDataset(pair_deepglobe(tmp_path), build_val_transform(64))
    image, mask = ds[0]
    assert image.shape == (3, 64, 64) and mask.shape == (1, 64, 64)


def test_empty_pairs_raises(tmp_path):
    with pytest.raises(ValueError):
        RoadTileDataset([], build_val_transform(64))
