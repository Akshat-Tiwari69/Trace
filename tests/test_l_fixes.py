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


# --------------------------------------------------------------------------- #
# A37 follow-up — upload-analysis concurrency guard + mask-size ceiling
# (bugs.md §5H "no concurrency/queueing guard" applied to the in-process
# CPU pipeline, not just the ARM-box-level deploy concern).
# --------------------------------------------------------------------------- #
def test_analyze_mask_rejects_oversized_mask(monkeypatch):
    import pytest

    from src.app import upload_analysis

    # Lower the ceiling instead of allocating a real 16MP+ array.
    monkeypatch.setattr(upload_analysis, "MAX_MASK_PIXELS", 100)
    with pytest.raises(ValueError, match="MP"):
        upload_analysis.analyze_mask(np.ones((20, 20), np.uint8), resolution_m=0.5)


def test_analyze_mask_busy_when_semaphore_saturated(monkeypatch):
    """A saturated semaphore returns AnalysisBusyError without running the pipeline."""
    import pytest

    from src.app import upload_analysis

    calls = []

    def _should_not_run(*args, **kwargs):
        calls.append(1)
        raise AssertionError("pipeline should not run while the semaphore is saturated")

    monkeypatch.setattr(
        "src.pipeline.p2_graph.skeleton_graph.mask_to_skeleton_with_distance",
        _should_not_run,
    )

    assert upload_analysis._analysis_semaphore.acquire(blocking=False)
    try:
        with pytest.raises(upload_analysis.AnalysisBusyError):
            upload_analysis.analyze_mask(np.ones((256, 256), np.uint8), resolution_m=0.5)
    finally:
        upload_analysis._analysis_semaphore.release()
    assert calls == []  # the heavy pipeline was never entered

    # The slot is free again afterwards — a subsequent call runs normally.
    mask = np.zeros((256, 256), np.uint8)
    for r in (64, 128, 192):
        mask[r - 1:r + 2, 20:236] = 1
    for c in (64, 128, 192):
        mask[20:236, c - 1:c + 2] = 1
    monkeypatch.undo()
    result = upload_analysis.analyze_mask(mask, resolution_m=0.5)
    assert result.n_nodes > 0


# --------------------------------------------------------------------------- #
# A37 follow-up — vectorized edge-style computation for the single GeoJson
# road layer (bugs.md §2C P2, replacing the per-edge folium.PolyLine loop).
# Pure pandas/numpy: importable and callable without a Streamlit runtime.
# --------------------------------------------------------------------------- #
def _tiny_edge_gdf():
    import geopandas as gpd
    from shapely.geometry import LineString

    return gpd.GeoDataFrame(
        {
            "u": [1, 3, 5, 7],
            "v": [2, 4, 6, 8],
            "is_bridged": [False, True, False, False],
            "is_bridge": [False, False, True, False],
            "geometry": [LineString([(0, 0), (1, 1)]) for _ in range(4)],
        }
    )


def _tiny_colour_scale():
    import branca.colormap as cm

    from src.app.app import TOKENS

    return cm.LinearColormap(
        colors=[TOKENS["ramp_0"], TOKENS["ramp_1"], TOKENS["ramp_2"], TOKENS["ramp_3"]],
        vmin=0.0,
        vmax=1.0,
    )


def test_compute_edge_styles_matches_original_precedence():
    from src.app.app import TOKENS, compute_edge_styles

    edges = _tiny_edge_gdf()
    scores = {1: 0.0, 2: 1.0, 3: 0.5, 4: 0.5, 5: 0.2, 6: 0.2, 7: 0.9, 8: 0.9}
    styled = compute_edge_styles(
        "test-fp-a", edges, scores, _tiny_colour_scale(),
        disabled_nodes=(7, 8), show_healed=True, show_spof=True,
    )
    by_pair = {(int(r.u), int(r.v)): r for r in styled.itertuples()}

    observed = by_pair[(1, 2)]
    assert observed.state == "observed link"
    assert observed.dash_array is None

    healed = by_pair[(3, 4)]
    assert healed.state == "healed link"
    assert healed.dash_array == "8 6"
    assert healed.weight == 4

    spof = by_pair[(5, 6)]
    assert spof.state == "critical bridge"
    assert spof.color == TOKENS["spof"]
    assert spof.weight == 5
    assert spof.dash_array is None

    disabled = by_pair[(7, 8)]
    assert disabled.state == "disabled link"
    assert disabled.color == TOKENS["disabled"]
    assert disabled.dash_array == "8 6"
    assert disabled.opacity == 0.45


def test_compute_edge_styles_show_healed_false_drops_bridged_edges():
    from src.app.app import compute_edge_styles

    edges = _tiny_edge_gdf()
    scores = {1: 0.0, 2: 1.0, 3: 0.5, 4: 0.5, 5: 0.2, 6: 0.2, 7: 0.9, 8: 0.9}
    styled = compute_edge_styles(
        "test-fp-b", edges, scores, _tiny_colour_scale(),
        disabled_nodes=(), show_healed=False, show_spof=True,
    )
    pairs = set(zip(styled["u"].astype(int), styled["v"].astype(int)))
    assert (3, 4) not in pairs  # the healed/bridged edge was suppressed
    assert (1, 2) in pairs
