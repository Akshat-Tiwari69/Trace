"""Unit tests for Phase III — global efficiency + resilience under ablation.

Protects the locked design decision (``docs/Tracker.md`` §8): the resilience
metric is **global efficiency**, which must stay *finite* when the graph
disconnects. Also checks the targeted-vs-random sanity property from
``docs/Evaluation.md``.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from src.pipeline.p3_analysis.criticality import compute_betweenness
from src.pipeline.p3_analysis.resilience import (
    global_efficiency,
    resilience_index,
)


def _path3() -> nx.Graph:
    """Path 0—1—2 with unit-length edges."""
    g = nx.Graph()
    g.add_edge(0, 1, length_m=1.0)
    g.add_edge(1, 2, length_m=1.0)
    return g


def _barbell() -> nx.Graph:
    """Two triangles {0,1,2} and {3,4,5} joined by a single bridge edge 2—3.

    Nodes 2 and 3 are the chokepoints; node 0 is peripheral.
    """
    g = nx.Graph()
    for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        g.add_edge(a, b, length_m=1.0)
    g.add_edge(2, 3, length_m=1.0)  # the lone bridge
    return g


# --------------------------------------------------------------------------- #
# global efficiency
# --------------------------------------------------------------------------- #
def test_global_efficiency_known_value():
    # ordered pairs: (0,1),(1,0),(1,2),(2,1) at d=1 → 4·1; (0,2),(2,0) at d=2 → 2·0.5
    # sum = 4 + 1 = 5; normalised by N(N-1)=6 → 5/6
    assert global_efficiency(_path3()) == pytest.approx(5.0 / 6.0)


def test_global_efficiency_tiny_graphs():
    assert global_efficiency(nx.Graph()) == 0.0
    g = nx.Graph()
    g.add_node(0)
    assert global_efficiency(g) == 0.0


def test_global_efficiency_finite_when_disconnected():
    """The whole reason we use this metric: no division by infinity."""
    g = nx.Graph()
    g.add_node(0)
    g.add_node(1)  # two nodes, no edge → unreachable pair
    eff = global_efficiency(g)
    assert eff == 0.0
    assert math.isfinite(eff)


# --------------------------------------------------------------------------- #
# resilience index
# --------------------------------------------------------------------------- #
def test_resilience_index_finite_after_split():
    """Removing the middle of a path splits it; RI must stay finite."""
    result = resilience_index(_path3(), removed_nodes=[1])
    assert math.isfinite(result["resilience_index"])
    assert result["resilience_index"] == 0.0  # 0 and 2 now have no path
    assert result["perturbed_efficiency"] == 0.0


def test_resilience_index_unchanged_input_graph():
    g = _path3()
    n_before = g.number_of_nodes()
    resilience_index(g, removed_nodes=[1])
    assert g.number_of_nodes() == n_before  # operated on a copy


def test_targeted_removal_hurts_more_than_peripheral():
    """Removing a high-betweenness chokepoint must drop resilience further than
    removing a peripheral node — the betweenness sanity check."""
    g = _barbell()
    bc = compute_betweenness(g)
    chokepoint = max(bc, key=bc.get)            # node 2 or 3 (the bridge)
    peripheral = min(bc, key=bc.get)            # a triangle corner

    ri_choke = resilience_index(g, [chokepoint])["resilience_index"]
    ri_periph = resilience_index(g, [peripheral])["resilience_index"]

    assert ri_choke < ri_periph  # the chokepoint matters more


def test_largest_cc_fraction_reported():
    g = _barbell()
    chokepoint = max(compute_betweenness(g), key=compute_betweenness(g).get)
    result = resilience_index(g, [chokepoint])
    # removing the bridge splits 6 nodes into 3 + 2 → largest CC = 3/5
    assert result["largest_cc_fraction"] == pytest.approx(3 / 5)
