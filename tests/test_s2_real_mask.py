"""Smoke test for the S2 driver — heal + analyse a (stand-in) predicted mask.

The real S2 input is a binary mask from the trained P1 model, which needs the
GPU checkpoint to produce. To test the P2/P3 consume-path **offline** (no torch,
no network), we synthesise a predicted-style mask: a pixel-space road grid with a
gap, exactly the shape ``predict.py`` writes (binary PNG, no alignment manifest).
This verifies S2 builds a graph, heals the gap, and writes the criticality CSV.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from src.pipeline.p1_segment.osm_mask import save_binary_png  # shared IO helper
from src.pipeline.p2_graph.config import GraphConfig
from src.pipeline.p2_graph.run_real_mask import run


def _synthetic_predicted_mask() -> np.ndarray:
    """A 3×3 road grid (3-px wide) with a gap punched in the top road."""
    mask = np.zeros((160, 160), dtype=np.uint8)
    for r in (40, 80, 120):
        mask[r - 1 : r + 2, 20:140] = 1   # horizontal roads
    for c in (40, 80, 120):
        mask[20:140, c - 1 : c + 2] = 1   # vertical roads
    mask[39:42, 78:92] = 0                 # ~14-px occlusion gap in the top road
    return mask


def test_s2_runs_on_pixelspace_predicted_mask(tmp_path):
    aoi = "stub_pred"
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    cfg = GraphConfig(aoi=aoi, interim_dir=interim, processed_dir=processed, resolution_m=1.0)

    # Write the stand-in predicted mask at the §4 contract path (no manifest).
    save_binary_png(_synthetic_predicted_mask(), cfg.mask_path)
    assert not cfg.manifest_path.exists()  # pixel-space: no georeferencing

    run(cfg, curve_steps=5)

    # P2 outputs exist and the graph is non-trivial
    assert cfg.graphml_path.exists()
    gj = json.loads(cfg.geojson_path.read_text())
    nodes = [f for f in gj["features"] if f["properties"]["feature_type"] == "node"]
    edges = [f for f in gj["features"] if f["properties"]["feature_type"] == "edge"]
    assert len(nodes) > 0 and len(edges) > 0

    # healing bridged the punched gap → at least one inferred edge
    assert any(e["properties"]["is_bridged"] for e in edges)

    # P3 output exists with the contracted columns and ranked rows
    rows = list(csv.DictReader((processed / f"{aoi}_criticality.csv").open()))
    assert len(rows) == len(nodes)
    assert set(rows[0]) >= {"node_id", "betweenness", "rank", "is_critical"}
    assert int(rows[0]["rank"]) == 1  # sorted, highest betweenness first
