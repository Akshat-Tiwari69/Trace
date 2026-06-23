"""Unit tests for Phase II healing — Union-Find + angle-aware MST bridging.

These are the load-bearing, deterministic graph functions (``docs/Rules.md`` →
Testing: "unit tests for MST/Union-Find healing"). They build small graphs by
hand (no skeletonisation, no network) so the healing logic is tested in isolation.
"""

from __future__ import annotations

import networkx as nx
import pytest

from src.pipeline.p2_graph.healing import UnionFind, heal_graph
from src.pipeline.p2_graph.skeleton_graph import (
    TYPE_BRIDGED,
    _annotate_degree_and_type,
    prune_degenerate_edges,
)


# --------------------------------------------------------------------------- #
# Union-Find
# --------------------------------------------------------------------------- #
def test_unionfind_basic_merge():
    uf = UnionFind([0, 1, 2, 3])
    assert uf.find(0) != uf.find(1)
    assert uf.union(0, 1) is True
    assert uf.find(0) == uf.find(1)
    assert uf.union(0, 1) is False  # already joined


def test_unionfind_transitive():
    uf = UnionFind(range(5))
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)
    assert uf.find(0) != uf.find(3)


# --------------------------------------------------------------------------- #
# Healing fixtures: a "road fragment" is two nodes joined by a straight edge.
# --------------------------------------------------------------------------- #
def _road(graph: nx.Graph, a: int, axy: tuple, b: int, bxy: tuple) -> None:
    """Add a straight road segment a—b with metric geometry + length."""
    graph.add_node(a, x=float(axy[0]), y=float(axy[1]))
    graph.add_node(b, x=float(bxy[0]), y=float(bxy[1]))
    length = ((bxy[0] - axy[0]) ** 2 + (bxy[1] - axy[1]) ** 2) ** 0.5
    graph.add_edge(a, b, length_m=length, geometry=[list(axy), list(bxy)], is_bridged=False)


def _collinear_gap(gap: float = 10.0) -> nx.Graph:
    """Two collinear fragments with a straight gap between nodes 1 and 2."""
    g = nx.Graph()
    _road(g, 0, (0.0, 0.0), 1, (10.0, 0.0))            # fragment A, endpoint 1 → +x
    _road(g, 2, (10.0 + gap, 0.0), 3, (20.0 + gap, 0.0))  # fragment B, endpoint 2 → -x
    _annotate_degree_and_type(g)
    return g


# --------------------------------------------------------------------------- #
# Heal: the happy path
# --------------------------------------------------------------------------- #
def test_heal_bridges_a_straight_gap():
    g = _collinear_gap(gap=10.0)
    assert nx.number_connected_components(g) == 2

    g, report = heal_graph(g, gap_max_m=40.0, angle_max_deg=60.0)

    assert nx.number_connected_components(g) == 1
    assert report.bridges_added == 1
    assert report.components_before == 2 and report.components_after == 1
    # the new edge is flagged as inferred, and weighted ~ the gap length
    assert g.edges[1, 2]["is_bridged"] is True
    assert g.edges[1, 2]["length_m"] == pytest.approx(10.0, abs=1e-6)
    # healed endpoints are retyped 'bridged' so the dashboard can mark them
    assert g.nodes[1]["type"] == TYPE_BRIDGED
    assert g.nodes[2]["type"] == TYPE_BRIDGED


def test_heal_raises_connectivity_ratio():
    g = _collinear_gap(gap=10.0)
    _, report = heal_graph(g, gap_max_m=40.0)
    # largest component grows from 2 nodes to 4 → +100%
    assert report.connectivity_ratio == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Heal: the guards
# --------------------------------------------------------------------------- #
def test_heal_respects_gap_max():
    g = _collinear_gap(gap=50.0)  # 50 m gap
    _, report = heal_graph(g, gap_max_m=40.0)  # budget only 40 m
    assert report.bridges_added == 0
    assert report.components_after == 2  # left disconnected, as it should be


def test_heal_gap_max_allows_when_raised():
    g = _collinear_gap(gap=50.0)
    _, report = heal_graph(g, gap_max_m=60.0)  # now within budget
    assert report.bridges_added == 1
    assert report.components_after == 1


def test_heal_rejects_sharp_turn():
    """A bridge that forces a ~90° turn is rejected at a tight angle budget."""
    g = nx.Graph()
    _road(g, 0, (0.0, 0.0), 1, (10.0, 0.0))     # endpoint 1 heads +x
    _road(g, 2, (10.0, 5.0), 3, (10.0, 40.0))   # endpoint 2 heads -y; bridge 1→2 is +y (90°)
    _annotate_degree_and_type(g)

    _, tight = heal_graph(g.copy(), gap_max_m=40.0, angle_max_deg=60.0)
    assert tight.bridges_added == 0  # 90° turn > 60° budget → not bridged

    _, loose = heal_graph(g.copy(), gap_max_m=40.0, angle_max_deg=100.0)
    assert loose.bridges_added == 1  # same gap allowed once the angle budget opens


def test_prune_removes_self_loops_and_short_edges():
    g = nx.Graph()
    _road(g, 0, (0.0, 0.0), 1, (10.0, 0.0))      # keep: 10 m
    g.add_node(2, x=20.0, y=0.0)
    g.add_edge(2, 2, length_m=0.0, geometry=[[20.0, 0.0], [20.0, 0.0]], is_bridged=False)  # self-loop
    _road(g, 3, (30.0, 0.0), 4, (30.3, 0.0))     # drop: 0.3 m sub-pixel edge
    _annotate_degree_and_type(g)

    removed = prune_degenerate_edges(g, min_edge_len_m=1.0)
    assert removed == 2                          # the self-loop + the 0.3 m edge
    assert g.has_edge(0, 1)                       # the real road survives
    assert 2 not in g.nodes                       # orphaned self-loop node dropped
    assert not any(u == v for u, v in g.edges())  # no self-loops remain


def test_heal_prefers_straight_over_kinked():
    """Given two reachable targets, the straighter continuation is chosen."""
    g = nx.Graph()
    _road(g, 0, (0.0, 0.0), 1, (10.0, 0.0))     # endpoint 1 heads +x
    _road(g, 2, (25.0, 0.0), 3, (35.0, 0.0))    # straight ahead target (endpoint 2)
    _road(g, 4, (12.0, 12.0), 5, (12.0, 30.0))  # off-axis target (endpoint 4)
    _annotate_degree_and_type(g)

    g, report = heal_graph(g, gap_max_m=40.0, angle_max_deg=80.0)
    # node 1 should bridge to the collinear fragment (node 2), not the kinked one
    assert g.has_edge(1, 2)
    assert g.edges[1, 2]["is_bridged"] is True
