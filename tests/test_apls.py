"""Unit tests for S7 APLS topology validation (offline, synthetic graphs)."""

from __future__ import annotations

import networkx as nx
import pytest

from src.pipeline.p3_analysis.apls import apls

SPACING_DEG = 0.001          # ~111 m at the equator
SEG_M = SPACING_DEG * 111_320.0


def _line_graph(n: int = 5) -> nx.Graph:
    """A straight chain 0—1—…—(n-1) with lon/lat coords + metric lengths."""
    g = nx.Graph()
    for i in range(n):
        g.add_node(i, x=73.82 + i * SPACING_DEG, y=15.49)
    for i in range(n - 1):
        g.add_edge(i, i + 1, length_m=SEG_M)
    return g


def test_apls_identical_is_one():
    g = _line_graph()
    result = apls(g, g.copy(), n_samples=50, tol_m=20.0)
    assert result["apls"] == pytest.approx(1.0, abs=1e-9)


def test_apls_drops_when_proposal_breaks_a_path():
    gt = _line_graph(5)
    prop = gt.copy()
    prop.remove_edge(2, 3)               # split the chain → some pairs unrouteable
    result = apls(gt, prop, n_samples=300, tol_m=20.0)
    assert result["apls"] < 1.0


def test_apls_zero_without_correspondence():
    gt = _line_graph(4)
    prop = _line_graph(4)
    for n in prop.nodes:                 # move the proposal ~100 km away
        prop.nodes[n]["x"] += 1.0
    result = apls(gt, prop, n_samples=50, tol_m=20.0)
    assert result["apls"] == 0.0         # nothing snaps within tolerance


def test_apls_detour_scores_between_zero_and_one():
    """A proposal that reroutes longer than truth scores a partial penalty."""
    gt = _line_graph(3)                  # 0—1—2, direct
    prop = _line_graph(3)
    # make the 1—2 leg much longer in the proposal (a detour), keep it routable
    prop.edges[1, 2]["length_m"] = SEG_M * 5
    result = apls(gt, prop, n_samples=200, tol_m=20.0)
    assert 0.0 < result["apls"] < 1.0
