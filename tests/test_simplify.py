"""Unit tests for S3 graph simplification — prune stubs + collapse degree-2."""

from __future__ import annotations

import networkx as nx
import pytest

from src.pipeline.p2_graph.simplify import (
    collapse_degree2_nodes,
    consolidate_graph,
    consolidate_nearby_nodes,
    prune_short_stubs,
    simplify_graph,
)
from src.pipeline.p2_graph.skeleton_graph import _annotate_degree_and_type


def _edge(g: nx.Graph, a: int, axy: tuple, b: int, bxy: tuple, is_bridged: bool = False) -> None:
    g.add_node(a, x=float(axy[0]), y=float(axy[1]))
    g.add_node(b, x=float(bxy[0]), y=float(bxy[1]))
    length = ((bxy[0] - axy[0]) ** 2 + (bxy[1] - axy[1]) ** 2) ** 0.5
    g.add_edge(a, b, length_m=length, geometry=[list(axy), list(bxy)], is_bridged=is_bridged)


# --------------------------------------------------------------------------- #
# collapse degree-2
# --------------------------------------------------------------------------- #
def test_collapse_merges_degree2_chain():
    g = nx.Graph()
    _edge(g, 0, (0, 0), 1, (10, 0))            # A—B
    _edge(g, 1, (10, 0), 2, (20, 0), is_bridged=True)  # B—C (a bridge)
    _annotate_degree_and_type(g)

    n = collapse_degree2_nodes(g)
    assert n == 1
    assert 1 not in g.nodes                      # B merged away
    assert g.has_edge(0, 2)
    assert g.edges[0, 2]["length_m"] == pytest.approx(20.0)   # lengths summed
    assert g.edges[0, 2]["geometry"] == [[0, 0], [10, 0], [20, 0]]  # stitched, no dup
    assert g.edges[0, 2]["is_bridged"] is True   # bridged flag carried through (OR)


def test_collapse_collapses_whole_chain():
    g = nx.Graph()
    for i in range(5):                            # 0—1—2—3—4 straight chain
        _edge(g, i, (i * 10, 0), i + 1, ((i + 1) * 10, 0))
    _annotate_degree_and_type(g)

    collapse_degree2_nodes(g)
    # only the two endpoints survive, joined by one edge of total length 50
    assert set(g.nodes) == {0, 5}
    assert g.edges[0, 5]["length_m"] == pytest.approx(50.0)


def test_collapse_skips_triangle():
    """Every triangle node is degree-2, but collapsing would duplicate an edge."""
    g = nx.Graph()
    _edge(g, 0, (0, 0), 1, (10, 0))
    _edge(g, 1, (10, 0), 2, (5, 8))
    _edge(g, 2, (5, 8), 0, (0, 0))
    _annotate_degree_and_type(g)

    assert collapse_degree2_nodes(g) == 0         # left intact
    assert g.number_of_nodes() == 3


# --------------------------------------------------------------------------- #
# prune short stubs
# --------------------------------------------------------------------------- #
def test_prune_removes_short_stub_keeps_long():
    g = nx.Graph()
    _edge(g, 0, (0, 0), 1, (50, 0))               # main road
    _edge(g, 1, (50, 0), 2, (53, 0))              # 3 m stub  -> prune
    _edge(g, 1, (50, 0), 3, (50, 40))             # 40 m dead-end -> keep
    _annotate_degree_and_type(g)

    removed = prune_short_stubs(g, min_stub_len_m=15.0)
    assert removed == 1
    assert 2 not in g.nodes
    assert 3 in g.nodes


def test_prune_is_iterative():
    """Trimming one stub can expose another short one behind it."""
    g = nx.Graph()
    _edge(g, 0, (0, 0), 1, (50, 0))               # anchor
    _edge(g, 1, (50, 0), 2, (54, 0))              # 4 m
    _edge(g, 2, (54, 0), 3, (58, 0))              # +4 m behind it
    _annotate_degree_and_type(g)

    removed = prune_short_stubs(g, min_stub_len_m=10.0)
    assert removed == 2                            # node 3 then node 2
    assert set(g.nodes) == {0, 1}


# --------------------------------------------------------------------------- #
# simplify_graph (combined) — never disconnects
# --------------------------------------------------------------------------- #
def test_simplify_drops_counts_preserves_components():
    g = nx.Graph()
    for i in range(6):                            # long chain 0..6
        _edge(g, i, (i * 10, 0), i + 1, ((i + 1) * 10, 0))
    _edge(g, 3, (30, 0), 99, (30, 3))             # a 3 m stub off the middle
    _annotate_degree_and_type(g)

    before_components = nx.number_connected_components(g)
    report = simplify_graph(g, min_stub_len_m=15.0)

    assert report.nodes_after < report.nodes_before
    assert report.edges_after < report.edges_before
    assert report.components_after == before_components   # connectivity preserved
    assert 99 not in g.nodes                              # stub gone
    assert nx.is_connected(g)                             # still one piece


# --------------------------------------------------------------------------- #
# S4 — near-duplicate node consolidation
# --------------------------------------------------------------------------- #
def test_consolidate_merges_near_duplicate_junction():
    g = nx.Graph()
    _edge(g, 0, (0, 0), 1, (1, 0))          # 1 m link: same junction, split in two
    _edge(g, 0, (0, 0), 2, (-20, 0))        # 20 m road out of node 0
    _edge(g, 1, (1, 0), 3, (21, 0))         # 20 m road out of node 1
    _annotate_degree_and_type(g)

    merged = consolidate_nearby_nodes(g, tol_m=5.0)
    assert merged == 1
    assert 1 not in g.nodes                  # 1 merged into 0
    assert set(g.neighbors(0)) == {2, 3}     # both external edges rewired to keeper
    assert nx.is_connected(g)


def test_consolidate_overpass_guard():
    """Nodes metres apart but with NO connecting edge (an overpass) must not merge."""
    g = nx.Graph()
    _edge(g, 0, (5.0, 5.0), 1, (5.5, 5.0))   # road A's two coincident nodes (0.5 m link)
    _edge(g, 2, (5.0, 5.3), 3, (5.5, 5.3))   # road B crossing over, 0.3 m away — no link
    _annotate_degree_and_type(g)

    merged = consolidate_nearby_nodes(g, tol_m=2.0)
    assert merged == 2                        # each road's own pair merges...
    assert 0 in g.nodes and 2 in g.nodes      # ...but the two roads do NOT merge
    assert not g.has_edge(0, 2)               # proximity alone never bridges grades


def test_consolidate_graph_preserves_components():
    g = nx.Graph()
    _edge(g, 0, (0, 0), 1, (1, 0))            # near-dup pair (cluster A)
    _edge(g, 1, (1, 0), 2, (30, 0))           # real road onward
    _edge(g, 5, (0, 99), 6, (1, 99))          # a separate component, also a near-dup pair
    _annotate_degree_and_type(g)

    before = nx.number_connected_components(g)
    report = consolidate_graph(g, tol_m=5.0)
    assert report.nodes_after < report.nodes_before
    assert report.nodes_merged >= 1
    assert report.components_after == before   # never splits (or merges) components
