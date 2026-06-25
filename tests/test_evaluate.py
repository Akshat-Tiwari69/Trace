"""Tests for the E1 graph-evaluation: GeoJSON round-trip + report generation."""

from __future__ import annotations

import json

import networkx as nx
import pytest

from src.pipeline.p2_graph.graph_io import (
    load_geojson_graph,
    save_geojson,
)
from src.pipeline.p3_analysis.evaluate import _healing_metrics, evaluate


def _bridged_graph() -> nx.Graph:
    """Two triangles joined by a single *bridged* edge (2—3)."""
    g = nx.Graph()
    for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        g.add_edge(a, b, length_m=1.0, is_bridged=False)
    g.add_edge(2, 3, length_m=1.0, is_bridged=True)  # the healed bridge
    for n in g.nodes:
        g.nodes[n]["x"], g.nodes[n]["y"] = float(n), 0.0
    return g


def test_geojson_roundtrip_preserves_graph(tmp_path):
    g = _bridged_graph()
    path = tmp_path / "rt_graph.geojson"
    save_geojson(g, path)
    back = load_geojson_graph(path)

    assert back.number_of_nodes() == g.number_of_nodes()
    assert back.number_of_edges() == g.number_of_edges()
    # edge attributes (incl. restored geometry) round-trip
    assert back.edges[2, 3]["is_bridged"] is True
    assert back.edges[0, 1]["length_m"] == 1.0
    assert back.edges[0, 1]["edge_betweenness"] == 0.0          # defaulted
    assert len(back.edges[0, 1]["geometry"]) == 2              # LineString restored
    # node coordinates preserved + defaulted properties reconstructed
    assert back.nodes[0]["x"] == pytest.approx(g.nodes[0]["x"])
    assert back.nodes[0]["y"] == pytest.approx(g.nodes[0]["y"])
    assert back.nodes[0]["type"] == "intersection"
    assert back.nodes[0]["betweenness"] == pytest.approx(0.0)
    assert back.nodes[0]["is_critical"] is False


def test_evaluate_handles_tiny_graph(tmp_path):
    """A 0–1 node graph must not crash the resilience summary (no ablation steps)."""
    g = nx.Graph()
    g.add_node(0, x=0.0, y=0.0)
    save_geojson(g, tmp_path / "one_graph.geojson")
    report = evaluate("one", sample_dir=tmp_path, curve_steps=10)
    assert report["resilience"]["ablation_steps"] == 0
    assert report["resilience"]["targeted_mean_ri"] == 0.0


def test_healing_metrics_reconstructs_connectivity():
    g = _bridged_graph()
    h = _healing_metrics(g)
    # removing the single bridge splits 6 nodes into 3 + 3
    assert h["bridges_added"] == 1
    assert h["components_before_heal"] == 2
    assert h["components_after_heal"] == 1
    assert h["largest_cc_before_heal"] == 3
    assert h["largest_cc_after_heal"] == 6
    assert h["connectivity_ratio_pct"] == 100.0  # 3 -> 6 nodes


def test_evaluate_writes_report_and_plot(tmp_path):
    save_geojson(_bridged_graph(), tmp_path / "tiny_graph.geojson")
    report = evaluate("tiny", sample_dir=tmp_path, curve_steps=3, top_n=3)

    assert (tmp_path / "tiny_graph_eval.json").exists()
    assert (tmp_path / "tiny_resilience_curve.png").exists()
    # report carries the headline sections
    assert report["graph"]["bridged_edges"] == 1
    assert report["resilience"]["targeted_degrades_faster"] is True
    on_disk = json.loads((tmp_path / "tiny_graph_eval.json").read_text())
    assert on_disk["healing"]["connectivity_ratio_pct"] == 100.0
