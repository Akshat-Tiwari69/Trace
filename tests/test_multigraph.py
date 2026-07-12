"""Regression tests for the A37 MultiGraph migration (bugs.md §4).

The P2 graph moved from ``nx.Graph`` to ``nx.MultiGraph`` so parallel roads —
loops, dual carriageways — survive instead of being silently collapsed by
``add_edge`` overwrites. These tests pin the contract end-to-end: GraphML and
GeoJSON round-trips keep both keyed edges, the full skeleton->analysis
pipeline runs on a graph with a genuine parallel branch, and the dashboard's
``graph_from_features`` builds a routable MultiGraph instead of a Graph.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import pytest


# --------------------------------------------------------------------------- #
# GraphML round-trip keeps parallel edges (graph_io.save_graphml/load_graphml)
# --------------------------------------------------------------------------- #
def test_graphml_roundtrip_preserves_parallel_edges(tmp_path):
    from src.pipeline.p2_graph.graph_io import load_graphml, save_graphml

    g = nx.MultiGraph()
    g.add_node(0, x=0.0, y=0.0)
    g.add_node(1, x=1.0, y=0.0)
    g.add_edge(0, 1, length_m=10.0, geometry=[[0.0, 0.0], [1.0, 0.0]])
    g.add_edge(0, 1, length_m=14.0, geometry=[[0.0, 0.0], [0.5, 0.4], [1.0, 0.0]])

    path = tmp_path / "parallel.graphml"
    save_graphml(g, path)
    back = load_graphml(path)

    assert back.is_multigraph()
    assert back.number_of_edges() == 2
    lengths = sorted(d["length_m"] for _, _, d in back.edges(data=True))
    assert lengths == [10.0, 14.0]
    # geometry decoded back from its JSON string on both keyed edges
    geoms = [d["geometry"] for _, _, d in back.edges(data=True)]
    assert [0.0, 0.0] in [pt for geom in geoms for pt in geom]
    assert any(len(geom) == 3 for geom in geoms)  # the bent (longer) branch


def test_load_graphml_forces_multigraph_type_even_without_parallel_edges(tmp_path):
    """A file with no parallel edges must still load as a MultiGraph (type
    consistency prevents downstream code from silently branching differently
    per-file — see graph_io.load_graphml's force_multigraph=True)."""
    from src.pipeline.p2_graph.graph_io import load_graphml, save_graphml

    g = nx.MultiGraph()
    g.add_node(0, x=0.0, y=0.0)
    g.add_node(1, x=1.0, y=0.0)
    g.add_edge(0, 1, length_m=5.0, geometry=[[0.0, 0.0], [1.0, 0.0]])

    path = tmp_path / "single.graphml"
    save_graphml(g, path)
    back = load_graphml(path)
    assert back.is_multigraph()


# --------------------------------------------------------------------------- #
# GeoJSON round-trip keeps parallel edges (graph_io.graph_to_geojson/load_geojson_graph)
# --------------------------------------------------------------------------- #
def test_geojson_roundtrip_preserves_parallel_edges(tmp_path):
    from src.pipeline.p2_graph.graph_io import load_geojson_graph, save_geojson

    g = nx.MultiGraph()
    g.add_node(0, x=0.0, y=0.0)
    g.add_node(1, x=0.001, y=0.0)
    g.add_edge(0, 1, length_m=100.0, is_bridged=False)
    g.add_edge(0, 1, length_m=140.0, is_bridged=True)  # a healed second branch

    path = tmp_path / "parallel.geojson"
    save_geojson(g, path)
    back = load_geojson_graph(path)

    assert back.is_multigraph()
    assert back.number_of_edges() == 2
    edge_data = back[0][1]  # {key: attrs} for both keyed parallel edges
    assert len(edge_data) == 2
    lengths_and_bridged = sorted(
        (d["length_m"], d["is_bridged"]) for d in edge_data.values()
    )
    assert lengths_and_bridged == [(100.0, False), (140.0, True)]


def test_geojson_roundtrip_preserves_geometry_and_edge_keys(tmp_path):
    from src.pipeline.p2_graph.graph_io import load_geojson_graph, save_geojson

    g = nx.MultiGraph()
    g.add_node(0, x=0.0, y=0.0)
    g.add_node(1, x=0.001, y=0.0)
    curved = [[0.0, 0.0], [0.0005, 0.0004], [0.001, 0.0]]
    g.add_edge(0, 1, key=7, length_m=140.0, geometry=curved)
    path = tmp_path / "curved.geojson"
    save_geojson(g, path)

    back = load_geojson_graph(path)
    assert list(back[0][1]) == [7]
    assert back[0][1][7]["geometry"] == curved


# --------------------------------------------------------------------------- #
# End-to-end: a parallel branch survives skeleton -> simplify -> criticality
# -> resilience without breaking the pipeline (bugs.md §4 / A37).
# --------------------------------------------------------------------------- #
def _parallel_branch_skeleton() -> np.ndarray:
    """Two junctions joined by a straight chord + a longer diagonal detour
    (same fixture as test_s_fixes.py::test_skeleton_parallel_branches_preserved)."""
    skel = np.zeros((13, 13), np.uint8)
    skel[6, 0:13] = 1  # stub — A — chord — B — stub
    for r, c in [(5, 3), (4, 4), (3, 5), (2, 6), (3, 7), (4, 8), (5, 9)]:
        skel[r, c] = 1  # the diagonal detour
    return skel


def test_parallel_branch_survives_full_analysis_pipeline():
    from src.pipeline.p2_graph.simplify import collapse_degree2_nodes, prune_short_stubs
    from src.pipeline.p2_graph.skeleton_graph import skeleton_to_graph
    from src.pipeline.p3_analysis.criticality import annotate_criticality
    from src.pipeline.p3_analysis.resilience import global_efficiency, resilience_index

    graph = skeleton_to_graph(_parallel_branch_skeleton(), transform=None, resolution_m=1.0)
    assert graph.number_of_edges() == 4  # 2 stubs + both A-B branches

    collapse_degree2_nodes(graph)  # no-op here (no degree-2 nodes), but must not choke
    prune_short_stubs(graph, min_stub_len_m=1.0)  # stubs are 3 m, survive a 1 m floor

    bc = annotate_criticality(graph)
    assert graph.number_of_edges() >= 3  # both junction nodes + their parallel link survive

    top_node = max(bc, key=bc.get)
    result = resilience_index(graph, removed_nodes=[top_node])
    assert 0.0 < result["resilience_index"] <= 1.0

    # The parallel branch's redundancy is topological (it keeps a shorter AND a
    # longer route between the same two junctions), not a node-ablation benefit:
    # Dijkstra always uses the shorter of the two parallel edges, and removing
    # either endpoint node kills both edges regardless of how many parallel
    # edges existed. So resilience_index is (by design, not by bug) identical
    # whether the redundant longer branch is kept or dropped — the value it adds
    # is representational fidelity (dual-carriageway geometry, edge counts),
    # not a change to this particular node-ablation metric.
    single = graph.copy()
    # drop whichever parallel edge is longer between the two junction nodes
    junctions = [n for n, d in graph.degree() if d >= 3]
    assert len(junctions) == 2
    a, b = junctions
    parallel_keys = list(single[a][b].keys())
    assert len(parallel_keys) == 2
    longer_key = max(parallel_keys, key=lambda k: single[a][b][k]["length_m"])
    single.remove_edge(a, b, longer_key)

    assert global_efficiency(single) == pytest.approx(global_efficiency(graph))
    result_single = resilience_index(single, removed_nodes=[top_node])
    assert result_single["resilience_index"] == pytest.approx(result["resilience_index"])


# --------------------------------------------------------------------------- #
# app.py: graph_from_features builds a MultiGraph, path_length uses the shorter
# of a parallel pair (bugs.md §4 / A37).
# --------------------------------------------------------------------------- #
def _two_parallel_edge_features():
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    rows = [
        {"feature_type": "node", "node_id": 1, "u": None, "v": None,
         "length_m": None, "is_bridged": None, "geometry": Point(0.0, 0.0)},
        {"feature_type": "node", "node_id": 2, "u": None, "v": None,
         "length_m": None, "is_bridged": None, "geometry": Point(0.001, 0.0)},
        {"feature_type": "edge", "node_id": None, "u": 1, "v": 2,
         "length_m": 120.0, "is_bridged": False,
         "geometry": LineString([(0.0, 0.0), (0.001, 0.0)])},
        {"feature_type": "edge", "node_id": None, "u": 1, "v": 2,
         "length_m": 80.0, "is_bridged": False,
         "geometry": LineString([(0.0, 0.0), (0.0005, 0.0002), (0.001, 0.0)])},
    ]
    return gpd.GeoDataFrame(rows)


def test_graph_from_features_keeps_parallel_edges_and_uses_shorter_for_path_length():
    from src.app.app import graph_from_features, path_length

    features = _two_parallel_edge_features()
    graph = graph_from_features("test-fingerprint", features)

    assert graph.is_multigraph()
    assert graph.number_of_edges() == 2
    assert path_length(graph, [1, 2]) == 80.0  # the shorter of the two parallel edges
