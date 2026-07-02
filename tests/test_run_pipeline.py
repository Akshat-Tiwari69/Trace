"""A5 walking-skeleton test: one tile flows P1→P2→P3→P4 (orchestration). CPU.

P1 (the real model) is covered by test_model; here we inject a deterministic
synthetic-mask segmenter so the P2→P3→P4 wiring is tested end-to-end without a
190 MB checkpoint, and assert the dashboard (P4) contract holds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.pipeline.p1_segment.osm_mask import save_binary_png
from src.pipeline.run_pipeline import DASHBOARD_CRITICALITY_COLUMNS, run


def _fake_segment(image_path, checkpoint, aoi, interim_dir, tile_size, threshold, device, tta=False,
                  postprocess=False, min_component_size=50, pp_close_radius=0, fill_holes=0):
    """Stand-in for P1: write a synthetic road grid mask at the contract path."""
    mask = np.zeros((256, 256), np.uint8)
    for r in (64, 128, 192):
        mask[r - 1 : r + 2, 20:236] = 1
    for c in (64, 128, 192):
        mask[20:236, c - 1 : c + 2] = 1
    out = Path(interim_dir) / f"{aoi}_mask.png"
    save_binary_png(mask, out)
    return out, float(mask.mean())


def test_walking_skeleton_flows_p1_to_p4(tmp_path):
    res = run(
        "unused.jpg", "unused.pt", "a5test",
        interim_dir=tmp_path / "interim", processed_dir=tmp_path / "processed",
        curve_steps=5, segment_fn=_fake_segment,
    )

    # P2 produced a real graph
    assert res["nodes"] > 0 and res["edges"] > 0

    # P3 artifacts exist
    assert (tmp_path / "processed" / "a5test_criticality.csv").exists()
    assert (tmp_path / "processed" / "a5test_resilience.csv").exists()
    assert (tmp_path / "processed" / "a5test_graph.geojson").exists()

    # P4 seam: the criticality CSV matches the dashboard's column contract
    assert res["p4"]["columns_match"] is True
    assert res["p4"]["geojson_exists"] is True

    # analysis summary came through
    assert "targeted_end_ri" in res["analysis"]


def test_dashboard_contract_columns_are_stable():
    # guard against a silent drift of the P4 contract
    assert DASHBOARD_CRITICALITY_COLUMNS == ["node_id", "betweenness", "rank", "is_critical", "x", "y"]
