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

from src.pipeline.p3_analysis.criticality import (
    annotate_criticality,
    annotate_cut_structure,
    compute_betweenness,
    rank_table,
)
from src.pipeline.p3_analysis.resilience import (
    ablation_curve,
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


def test_global_efficiency_k_sample():
    g = _barbell()
    exact = global_efficiency(g)
    # k >= N falls back to exact
    assert global_efficiency(g, k=g.number_of_nodes()) == pytest.approx(exact)
    # k < N gives a finite, positive estimate (won't be identical to exact)
    est = global_efficiency(g, k=3, seed=1)
    assert math.isfinite(est) and est > 0


def test_global_efficiency_rejects_nonpositive_k():
    # k <= 0 would make the sample empty and the normaliser 0 → guard it
    for bad in (0, -1):
        with pytest.raises(ValueError):
            global_efficiency(_barbell(), k=bad)


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
    bc = compute_betweenness(g)
    chokepoint = max(bc, key=bc.get)
    result = resilience_index(g, [chokepoint])
    # removing the bridge splits 6 nodes into 3 + 2 → largest CC = 3/5
    assert result["largest_cc_fraction"] == pytest.approx(3 / 5)


def test_targeted_ablation_requires_betweenness():
    """A 'targeted' curve with no betweenness must fail loudly, not silently."""
    with pytest.raises(ValueError):
        ablation_curve(_barbell(), order="targeted", betweenness=None)


# --------------------------------------------------------------------------- #
# critical_fraction semantics
# --------------------------------------------------------------------------- #
def test_critical_fraction_zero_flags_none():
    g = _barbell()
    annotate_criticality(g, critical_fraction=0.0)
    assert not any(d["is_critical"] for _, d in g.nodes(data=True))


def test_critical_fraction_positive_flags_at_least_one():
    g = _barbell()
    annotate_criticality(g, critical_fraction=0.10)  # 0.10·6 rounds to 1
    assert sum(d["is_critical"] for _, d in g.nodes(data=True)) >= 1


def test_critical_fraction_out_of_range_raises():
    with pytest.raises(ValueError):
        annotate_criticality(_barbell(), critical_fraction=1.5)


# --------------------------------------------------------------------------- #
# S8 — articulation points & bridge edges
# --------------------------------------------------------------------------- #
def test_cut_structure_finds_articulation_and_bridge():
    g = _barbell()                       # two triangles joined by the single edge 2—3
    counts = annotate_cut_structure(g)
    assert counts["n_articulation"] == 2 and counts["n_bridges"] == 1
    assert g.nodes[2]["is_articulation"] and g.nodes[3]["is_articulation"]
    assert not g.nodes[0]["is_articulation"]      # a triangle corner is not a cut node
    assert g.edges[2, 3]["is_bridge"] is True     # the lone bridge edge
    assert g.edges[0, 1]["is_bridge"] is False    # a cycle edge is not a bridge


def test_cut_structure_none_in_a_cycle():
    g = nx.Graph()
    for a, b in [(0, 1), (1, 2), (0, 2)]:        # a triangle has no cut node/edge
        g.add_edge(a, b, length_m=1.0)
    counts = annotate_cut_structure(g)
    assert counts["n_articulation"] == 0 and counts["n_bridges"] == 0


def test_cut_structure_handles_disconnected():
    g = _barbell()
    g.add_edge(10, 11, length_m=1.0)             # a separate 2-node component (its edge is a bridge)
    counts = annotate_cut_structure(g)
    assert counts["n_bridges"] == 2              # 2—3 plus the isolated 10—11


def test_rank_table_includes_is_articulation():
    g = _barbell()
    for n in g.nodes:                            # rank_table needs coords
        g.nodes[n]["x"], g.nodes[n]["y"] = float(n), 0.0
    bc = compute_betweenness(g)
    annotate_criticality(g)
    annotate_cut_structure(g)
    rows = rank_table(g, bc)
    assert "is_articulation" in rows[0]
    assert sum(r["is_articulation"] for r in rows) == 2
