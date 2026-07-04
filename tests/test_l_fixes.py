"""Regression tests for the A36 L-effort fixes (bugs.md).

Covers the pre-tiling / windowed-inference stage (§5H) and the upload-to-graph
analysis loop (§2B). Both run on synthetic inputs — no GPU, no checkpoint.
"""

from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# §5H — windowed reading + streamed inference for large rasters
# --------------------------------------------------------------------------- #
def _write_png(path, arr: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(arr).save(path)


def test_raster_dimensions_reads_header_only(tmp_path):
    from src.pipeline.p1_segment.raster_io import raster_dimensions

    img = np.zeros((300, 500, 3), np.uint8)
    p = tmp_path / "img.png"
    _write_png(p, img)
    assert raster_dimensions(p) == (300, 500)


def test_iter_windows_covers_image_with_overlap(tmp_path):
    from src.pipeline.p1_segment.raster_io import iter_windows

    img = np.zeros((1000, 1000, 3), np.uint8)
    p = tmp_path / "img.png"
    _write_png(p, img)

    covered = np.zeros((1000, 1000), bool)
    n = 0
    for window, r0, c0 in iter_windows(p, window_px=512, overlap_px=128):
        h, w = window.shape[:2]
        assert window.shape[2] == 3
        covered[r0:r0 + h, c0:c0 + w] = True
        n += 1
    assert n > 1                       # actually tiled, not one shot
    assert covered.all()               # every pixel is read by some window


class _BrightRoadModel(torch.nn.Module):
    """Position-independent stub: predicts 'road' where the input is bright.

    Reproducing the road purely from pixel brightness makes the windowed mosaic
    exactly reconstruct the whole-image result (no dependence on absolute tile
    position), so the streaming path can be verified deterministically.
    """

    def forward(self, x):  # x: normalised (B,3,H,W)
        return (x.mean(dim=1, keepdim=True)) * 20.0  # bright → large positive logit


def test_predict_large_raster_recovers_road_pattern(tmp_path):
    from src.pipeline.p1_segment.model import predict_large_raster

    # A white '+' road on black — larger than one 512 window so tiling matters.
    img = np.zeros((1100, 1100, 3), np.uint8)
    img[540:560, 50:1050] = 255      # horizontal road
    img[50:1050, 540:560] = 255      # vertical road
    p = tmp_path / "roads.png"
    _write_png(p, img)

    mask = predict_large_raster(_BrightRoadModel(), p, tile_size=256, threshold=0.5,
                                device="cpu", window_px=512, overlap_px=128)
    assert mask.shape == (1100, 1100)
    assert mask.dtype == np.uint8
    # Road captured along both arms; corners of the frame stay background.
    assert mask[549, 500] == 1 and mask[500, 549] == 1
    assert mask[50, 50] == 0 and mask[1049, 1049] == 0


# --------------------------------------------------------------------------- #
# §2B — upload → graph → resilience analysis (the "close the loop" feature)
# --------------------------------------------------------------------------- #
def test_analyze_mask_produces_graph_and_criticality():
    from src.app.upload_analysis import analyze_mask

    # A grid mask → a real connected road network.
    mask = np.zeros((256, 256), np.uint8)
    for r in (64, 128, 192):
        mask[r - 1:r + 2, 20:236] = 1
    for c in (64, 128, 192):
        mask[20:236, c - 1:c + 2] = 1

    result = analyze_mask(mask, resolution_m=0.5)
    assert result.graph.number_of_nodes() > 0
    assert result.graph.number_of_edges() > 0
    assert not result.criticality.empty
    assert {"node_id", "betweenness", "rank", "is_critical"} <= set(result.criticality.columns)
    assert 0.0 <= result.resilience_index <= 1.0
    # criticality is ranked 1..N and betweenness is normalised
    assert result.criticality["betweenness"].between(0, 1).all()


def test_analyze_mask_rejects_empty_mask():
    import pytest

    from src.app.upload_analysis import analyze_mask

    with pytest.raises(ValueError):
        analyze_mask(np.zeros((128, 128), np.uint8), resolution_m=0.5)
