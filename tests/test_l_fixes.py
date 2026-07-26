"""Regression tests for the A36 L-effort fixes.

Covers the A36 pre-tiling/windowed-inference stage and the A39 upload-to-graph
analysis loop. Both run on synthetic inputs — no GPU, no checkpoint.
"""

from __future__ import annotations

import numpy as np
import torch


def test_modal_client_sends_auth_header_outside_json(monkeypatch):
    """The shared key is authenticated before the server parses the body."""
    import json

    from src.app import modal_client

    captured = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit=-1):
            return b'{"mask_png_b64":"AA==","threshold":0.5}'

    def _urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setenv("MODAL_SEG_KEY", "test-secret")
    monkeypatch.setattr(modal_client, "MODAL_SEG_URL", "https://example.test/segment")
    monkeypatch.setattr(modal_client.urllib.request, "urlopen", _urlopen)
    body = json.dumps({"image_b64": "AA=="}).encode()

    modal_client.post_once(body)

    request = captured["request"]
    assert request.get_header("X-api-key") == "test-secret"
    assert json.loads(request.data) == {"image_b64": "AA=="}
    assert captured["timeout"] == 120


# --------------------------------------------------------------------------- #
# A36 — windowed reading + streamed inference for large rasters
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
# A39 upload → graph → resilience analysis
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


def test_representative_reroute_prioritizes_and_counts_disconnections():
    import networkx as nx

    from src.app.service import representative_reroute

    graph = nx.MultiGraph()
    for neighbour in (1, 2, 3):
        graph.add_edge(0, neighbour, length_m=1.0)
    graph.add_edge(1, 2, length_m=10.0)  # one finite alternative; node 3 is cut off

    route = representative_reroute(graph, 0)

    assert route is not None and route["rerouted_path"] is None
    assert route["disconnected_pairs"] == 2
    assert {route["origin"], route["destination"]} == {1, 3}


def test_analyze_mask_rejects_empty_mask():
    import pytest

    from src.app.upload_analysis import analyze_mask

    with pytest.raises(ValueError):
        analyze_mask(np.zeros((128, 128), np.uint8), resolution_m=0.5)


# --------------------------------------------------------------------------- #
# A37 follow-up — upload-analysis concurrency guard + mask-size ceiling
# applied to the in-process CPU pipeline, not just the host-level deploy concern.
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
