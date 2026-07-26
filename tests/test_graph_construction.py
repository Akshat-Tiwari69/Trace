"""Characterize the two mask-to-graph entry points before consolidation."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import numpy as np
from PIL import Image
import pytest


def _two_component_mask() -> np.ndarray:
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[63:66, 5:55] = 1
    mask[20:108, 24:27] = 1
    mask[63:66, 73:123] = 1
    mask[20:108, 102:105] = 1
    return mask


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _graph_payload(graph) -> dict:
    nodes = sorted(
        ([int(node), _plain(data)] for node, data in graph.nodes(data=True)),
        key=lambda item: item[0],
    )
    edges = sorted(
        (
            [int(u), int(v), int(key), _plain(data)]
            for u, v, key, data in graph.edges(keys=True, data=True)
        ),
        key=lambda item: item[:3],
    )
    return {"graph": _plain(graph.graph), "nodes": nodes, "edges": edges}


def _digest(value) -> str:
    encoded = json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_batch_mask_to_graph_characterization(tmp_path):
    from src.pipeline.p2_graph.build_graph import build_graph
    from src.pipeline.p2_graph.config import GraphConfig

    interim = tmp_path / "interim"
    interim.mkdir()
    Image.fromarray(_two_component_mask() * 255).save(interim / "characterization_mask.png")
    cfg = GraphConfig(
        aoi="characterization",
        interim_dir=interim,
        processed_dir=tmp_path / "processed",
        resolution_m=0.5,
        min_edge_len_m=0.5,
        consolidate=True,
    )

    graph, report = build_graph(cfg)

    assert asdict(report) == {
        "components_before": 2,
        "components_after": 1,
        "largest_cc_before": 5,
        "largest_cc_after": 10,
        "bridges_added": 1,
        "bridges_rejected_crossing": 0,
        "bridges_rejected_corridor": 0,
    }
    assert _digest(_graph_payload(graph)) == (
        "bac0d8fe91b57767b14a1c5bcdc81cc28d86a380211bbbba519ffcb54c4752e4"
    )


def test_upload_mask_to_graph_and_metrics_characterization():
    from src.app.upload_analysis import analyze_mask

    result = analyze_mask(_two_component_mask(), resolution_m=0.5, critical_fraction=0.1)
    payload = {
        "graph": _graph_payload(result.graph),
        "criticality": result.criticality.to_dict(orient="records"),
        "ri": result.resilience_index,
        "top": result.top_node,
        "n_nodes": result.n_nodes,
        "n_edges": result.n_edges,
        "summary": result.summary,
    }
    assert {
        key: result.summary[key]
        for key in ("efficiency_method", "efficiency_k", "efficiency_seed")
    } == {"efficiency_method": "exact", "efficiency_k": None, "efficiency_seed": 42}

    assert _digest(payload) == (
        "20303c7ac9988346cff5cec86ff967b6b448bf6076d09fa31e0d04e97db4b09d"
    )


def test_upload_resilience_keeps_the_baseline_node_universe():
    from src.app.upload_analysis import analyze_mask
    from src.pipeline.p3_analysis.resilience import resilience_index

    result = analyze_mask(_two_component_mask(), resolution_m=0.5)
    expected = resilience_index(result.graph, [result.top_node])["resilience_index"]

    assert result.resilience_index == pytest.approx(expected)


def test_shared_mask_to_graph_sequence_characterization():
    from src.pipeline.p2_graph.config import GraphConfig
    from src.pipeline.p2_graph.construction import construct_graph

    cfg = GraphConfig(
        aoi="characterization",
        resolution_m=0.5,
        min_edge_len_m=0.5,
        min_corridor_support=0.0,
        consolidate=True,
    )
    graph, heal, simplify, consolidate, polyline = construct_graph(_two_component_mask(), cfg)

    assert _digest(_graph_payload(graph)) == (
        "0e35046c81e709da659528d698dfdde2a83762cbb1f40d01441018342be7ad33"
    )
    assert [asdict(report) for report in (heal, simplify, consolidate, polyline)] == [
        {
            "components_before": 2,
            "components_after": 1,
            "largest_cc_before": 5,
            "largest_cc_after": 10,
            "bridges_added": 1,
            "bridges_rejected_crossing": 0,
            "bridges_rejected_corridor": 0,
        },
        {
            "nodes_before": 10,
            "nodes_after": 6,
            "edges_before": 9,
            "edges_after": 5,
            "components_before": 1,
            "components_after": 1,
            "stubs_pruned": 2,
            "nodes_collapsed": 2,
        },
        {
            "nodes_before": 6,
            "nodes_after": 6,
            "components_before": 1,
            "components_after": 1,
            "nodes_merged": 0,
        },
        {"vertices_before": 235, "vertices_after": 10, "edges": 5},
    ]
