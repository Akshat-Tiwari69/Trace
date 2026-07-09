"""Regression tests for the A36 M-effort fixes (bugs.md).

Backend M-items: road-width recovery, false-bridge (edge-crossing) rejection,
auto k-sampling + bounded betweenness cache, and the provenance/orchestration
helpers.
"""

from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# §4 — road width from the distance transform
# --------------------------------------------------------------------------- #
def test_skeleton_graph_records_width_from_distance():
    from src.pipeline.p2_graph.skeleton_graph import (
        mask_to_skeleton_with_distance,
        skeleton_to_graph,
    )

    # A 5-px-wide horizontal road → half-width ~2.5 px → width_m ~5 at 1 m/px.
    mask = np.zeros((40, 40), np.uint8)
    mask[18:23, 4:36] = 1
    skel, dist = mask_to_skeleton_with_distance(mask)
    g = skeleton_to_graph(skel, transform=None, resolution_m=1.0, distance=dist)

    widths = [d["width_m"] for _, _, d in g.edges(data=True) if "width_m" in d]
    assert widths, "no width_m recorded"
    assert 3.0 <= max(widths) <= 7.0  # ~5 m road, sampled along the centreline


def test_skeleton_graph_omits_width_without_distance():
    from src.pipeline.p2_graph.skeleton_graph import mask_to_skeleton, skeleton_to_graph

    mask = np.zeros((20, 20), np.uint8)
    mask[10, 2:18] = 1
    g = skeleton_to_graph(mask_to_skeleton(mask), transform=None, resolution_m=1.0)
    assert all("width_m" not in d for _, _, d in g.edges(data=True))


# --------------------------------------------------------------------------- #
# §4 — false-bridge rejection (a bridge crossing a real road is dropped)
# --------------------------------------------------------------------------- #
def test_healing_rejects_bridge_that_crosses_a_road():
    import networkx as nx

    from src.pipeline.p2_graph.healing import heal_graph

    g = nx.Graph()
    # Two collinear stubs with a gap at x=10..30 along y=0 (want to bridge).
    g.add_node(0, x=0.0, y=0.0)
    g.add_node(1, x=10.0, y=0.0)
    g.add_edge(0, 1, length_m=10.0, geometry=[[0, 0], [10, 0]], is_bridged=False)
    g.add_node(2, x=30.0, y=0.0)
    g.add_node(3, x=40.0, y=0.0)
    g.add_edge(2, 3, length_m=10.0, geometry=[[30, 0], [40, 0]], is_bridged=False)
    # A perpendicular road slicing through the gap at x=20 (a real crossing road).
    g.add_node(4, x=20.0, y=-15.0)
    g.add_node(5, x=20.0, y=15.0)
    g.add_edge(4, 5, length_m=30.0, geometry=[[20, -15], [20, 15]], is_bridged=False)

    _, report = heal_graph(g, gap_max_m=40.0, angle_max_deg=60.0)
    # The 1↔2 bridge crosses the x=20 road → must be rejected, not added.
    assert report.bridges_rejected_crossing >= 1
    assert not g.has_edge(1, 2)


def test_healing_allows_clear_gap():
    import networkx as nx

    from src.pipeline.p2_graph.healing import heal_graph

    g = nx.Graph()
    g.add_node(0, x=0.0, y=0.0)
    g.add_node(1, x=10.0, y=0.0)
    g.add_edge(0, 1, length_m=10.0, geometry=[[0, 0], [10, 0]], is_bridged=False)
    g.add_node(2, x=25.0, y=0.0)
    g.add_node(3, x=35.0, y=0.0)
    g.add_edge(2, 3, length_m=10.0, geometry=[[25, 0], [35, 0]], is_bridged=False)

    _, report = heal_graph(g, gap_max_m=40.0, angle_max_deg=60.0)
    assert report.bridges_added >= 1
    assert report.bridges_rejected_crossing == 0


# --------------------------------------------------------------------------- #
# §4 — auto k-sampling + bounded cache
# --------------------------------------------------------------------------- #
def test_auto_k_thresholds():
    import networkx as nx

    from src.pipeline.p3_analysis.criticality import (
        AUTO_K_NODE_THRESHOLD,
        AUTO_K_SAMPLES,
        auto_k,
    )

    small = nx.path_graph(50)
    assert auto_k(small, None) is None            # exact below the threshold
    assert auto_k(small, 10) == 10                # explicit k always honoured
    big = nx.path_graph(AUTO_K_NODE_THRESHOLD + 5)
    assert auto_k(big, None) == AUTO_K_SAMPLES     # auto-sample above it


def test_betweenness_cache_is_bounded():
    import networkx as nx

    from src.pipeline.p3_analysis.criticality import BetweennessCache

    cache = BetweennessCache(maxsize=4)
    for i in range(10):  # 10 distinct graphs, cap 4
        g = nx.path_graph(3 + i)
        for u, v in g.edges():
            g.edges[u, v]["length_m"] = 1.0
        cache.get(g)
    assert len(cache) <= 4


# --------------------------------------------------------------------------- #
# §5A — provenance round-trip
# --------------------------------------------------------------------------- #
def test_provenance_build_and_roundtrip(tmp_path):
    from src.pipeline.p1_segment.provenance import (
        build_provenance,
        read_provenance,
        write_provenance,
    )

    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    rec = build_provenance(ckpt, {"encoder": "mit_b3", "arch": "unet"}, threshold=0.52)
    assert rec["checkpoint"] == "model.pt"
    assert rec["encoder"] == "mit_b3"
    assert rec["threshold"] == 0.52
    assert len(rec["model_sha256"]) == 64  # sha256 hex digest
    assert rec["created_utc"].endswith("+00:00")

    path = write_provenance(tmp_path / "sub" / "provenance.json", rec)
    assert read_provenance(path) == rec
    assert read_provenance(tmp_path / "missing.json") is None


# --------------------------------------------------------------------------- #
# §5B — PipelineConfig as the single source of truth
# --------------------------------------------------------------------------- #
def test_pipeline_config_from_json_and_graph_config(tmp_path):
    import json

    from src.pipeline.config import PipelineConfig

    cfg_file = tmp_path / "run.json"
    cfg_file.write_text(json.dumps({"aoi": "demo", "gap_max_m": 25.0, "curve_steps": 10}))
    cfg = PipelineConfig.from_file(cfg_file)
    assert cfg.aoi == "demo" and cfg.gap_max_m == 25.0 and cfg.curve_steps == 10

    gc = cfg.graph_config()
    assert gc.aoi == "demo" and gc.gap_max_m == 25.0  # P2 fields propagate

    # unknown keys must fail loudly, not be silently ignored
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"aoi": "demo", "not_a_field": 1}))
    with pytest.raises(ValueError, match="unknown config keys"):
        PipelineConfig.from_file(bad)


# --------------------------------------------------------------------------- #
# §5A — orchestrator idempotency + run.json summary
# --------------------------------------------------------------------------- #
def _synthetic_segment(image_path, checkpoint, aoi, interim_dir, tile_size=None, threshold=None,
                       device="cpu", tta=False, blend=True, postprocess=False,
                       min_component_size=50, pp_open_radius=0, pp_close_radius=0, fill_holes=0):
    """A deterministic grid-mask segmenter (no checkpoint) for orchestration tests."""
    from pathlib import Path

    from src.pipeline.p1_segment.osm_mask import save_binary_png

    mask = np.zeros((256, 256), np.uint8)
    for r in (64, 128, 192):
        mask[r - 1:r + 2, 20:236] = 1
    for c in (64, 128, 192):
        mask[20:236, c - 1:c + 2] = 1
    out = Path(interim_dir) / f"{aoi}_mask.png"
    save_binary_png(mask, out)
    return out, float(mask.mean())


def test_run_writes_summary_and_is_idempotent(tmp_path):
    import json

    from src.pipeline.run_pipeline import run

    kwargs = dict(interim_dir=tmp_path / "interim", processed_dir=tmp_path / "processed",
                  curve_steps=5, segment_fn=_synthetic_segment)
    first = run("img.jpg", "ckpt.pt", "idem", **kwargs)
    assert first["status"] == "ok" and first["nodes"] > 0
    assert all(s["ran"] for s in first["stages"])  # cold run: every stage ran

    run_json = tmp_path / "processed" / "idem_run.json"
    assert run_json.exists()
    saved = json.loads(run_json.read_text())
    assert saved["config"]["aoi"] == "idem" and "stages" in saved

    # Second run: all artifacts fresh → every stage skips.
    second = run("img.jpg", "ckpt.pt", "idem", **kwargs)
    assert all(not s["ran"] for s in second["stages"])

    # --from-stage p2 forces P2/P3 to rerun (P1 still skips as fresh).
    third = run("img.jpg", "ckpt.pt", "idem", from_stage="p2", **kwargs)
    ran = {s["stage"]: s["ran"] for s in third["stages"]}
    assert ran == {"p1": False, "p2": True, "p3": True}
