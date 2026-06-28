"""Test the Massachusetts → DeepGlobe-format converter (A11). CPU, synthetic."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.pipeline.p1_segment.build_massachusetts_data import convert_massachusetts


def test_convert_produces_deepglobe_pairs(tmp_path):
    img_dir, lbl_dir, out = tmp_path / "train", tmp_path / "train_labels", tmp_path / "out"
    img_dir.mkdir(); lbl_dir.mkdir()
    # a 512x512 RGB "image" (.tiff) + a 0/255 road mask (.tif), Massachusetts-style
    Image.fromarray((np.random.rand(512, 512, 3) * 255).astype(np.uint8)).save(img_dir / "abc_15.tiff")
    m = np.zeros((512, 512), np.uint8)
    m[250:262, :] = 255                                      # a road band
    Image.fromarray(m, mode="L").save(lbl_dir / "abc_15.tif")

    n = convert_massachusetts(img_dir, lbl_dir, out, upsample=1.0, tile_size=512,
                              min_road_fraction=0.001)
    assert n >= 1
    sats = sorted(out.glob("*_sat.jpg"))
    masks = sorted(out.glob("*_mask.png"))
    assert len(sats) == len(masks) == n
    # mask is true DeepGlobe 0/255
    arr = np.asarray(Image.open(masks[0]).convert("L"))
    assert set(np.unique(arr).tolist()) <= {0, 255} and (arr == 255).any()


def test_empty_tiles_are_skipped(tmp_path):
    img_dir, lbl_dir, out = tmp_path / "train", tmp_path / "train_labels", tmp_path / "out"
    img_dir.mkdir(); lbl_dir.mkdir()
    Image.fromarray((np.random.rand(512, 512, 3) * 255).astype(np.uint8)).save(img_dir / "x_15.tiff")
    Image.fromarray(np.zeros((512, 512), np.uint8), mode="L").save(lbl_dir / "x_15.tif")  # no road
    n = convert_massachusetts(img_dir, lbl_dir, out, upsample=1.0, min_road_fraction=0.005)
    assert n == 0                                            # road-free tile dropped
