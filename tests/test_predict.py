"""Tests for A39 — P1 persisting its probability map for corridor-aware healing.

The blended (Hann-window) inference path already computes a per-pixel road
probability before thresholding it into the binary mask; bugs.md §4 asks that
it be persisted (not discarded) so P2's healing can tell an occluded real road
from open ground the model never thought looked like a road. These tests use
a tiny, untrained (``encoder_weights=None``) model — same pattern as
``test_model.py`` — so no checkpoint download and no GPU is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.pipeline.p1_segment.model import build_model, save_checkpoint
from src.pipeline.p1_segment.predict import run_inference


def _tiny_checkpoint(tmp_path: Path) -> Path:
    """A small random-weight checkpoint, just big enough to run inference."""
    model = build_model(encoder_weights=None)
    ckpt = tmp_path / "tiny.pt"
    save_checkpoint(model, ckpt, meta={"encoder": "mit_b0", "image_size": 64, "threshold": 0.5})
    return ckpt


def _geotiff(path: Path, size: int = 96) -> None:
    """A tiny georeferenced RGB tile (so ``write_manifest`` actually writes)."""
    import rasterio
    from rasterio.transform import from_origin

    transform = from_origin(72.8, 19.1, 1.0, 1.0)
    data = np.random.default_rng(0).integers(0, 255, (3, size, size), dtype=np.uint8)
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=3,
                       dtype=np.uint8, crs="EPSG:4326", transform=transform) as dst:
        dst.write(data)


def test_blended_inference_persists_prob_map_and_manifest_entry(tmp_path):
    ckpt = _tiny_checkpoint(tmp_path)
    image = tmp_path / "tile.tif"
    _geotiff(image, size=96)
    interim = tmp_path / "interim"

    out, _ = run_inference(image, ckpt, "probtest", interim, tile_size=64, device="cpu", blend=True)
    assert out.exists()

    prob_path = interim / "probtest" / "prob.png"
    assert prob_path.exists()

    from PIL import Image

    prob = np.asarray(Image.open(prob_path))
    assert prob.dtype == np.uint8
    assert prob.shape == (96, 96)
    # a real probability, not a degenerate all-0/all-255 placeholder
    assert prob.min() < prob.max()

    manifest = json.loads((interim / "probtest" / "manifest.json").read_text())
    assert manifest.get("prob_png") is True


def test_non_blend_inference_has_no_prob_map(tmp_path):
    """``blend=False`` (older non-overlapping tiling) never computes a full prob
    array, so it must persist neither the file nor the manifest flag."""
    ckpt = _tiny_checkpoint(tmp_path)
    image = tmp_path / "tile.tif"
    _geotiff(image, size=96)
    interim = tmp_path / "interim"

    run_inference(image, ckpt, "noblend", interim, tile_size=64, device="cpu", blend=False)

    assert not (interim / "noblend" / "prob.png").exists()
    manifest = json.loads((interim / "noblend" / "manifest.json").read_text())
    assert "prob_png" not in manifest


def test_build_graph_picks_up_persisted_prob_map(tmp_path, capsys):
    """The P1 prob.png this run persists is picked up by P2's build_graph — the
    corridor check switches on and the heal report carries the new stat."""
    ckpt = _tiny_checkpoint(tmp_path)
    image = tmp_path / "tile.tif"
    _geotiff(image, size=96)
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"

    run_inference(image, ckpt, "probgraph", interim, tile_size=64, device="cpu", blend=True)

    from src.pipeline.p2_graph.build_graph import build_graph
    from src.pipeline.p2_graph.config import GraphConfig

    cfg = GraphConfig(aoi="probgraph", interim_dir=interim, processed_dir=processed)
    assert cfg.prob_path.exists()

    capsys.readouterr()  # clear
    graph, report = build_graph(cfg)
    out = capsys.readouterr().out
    assert "corridor check ON" in out
    assert cfg.graphml_path.exists()
    assert "bridges_rejected_corridor" in graph.graph["heal"]
