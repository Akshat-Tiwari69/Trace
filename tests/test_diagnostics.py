"""Unit tests for A46 structural graph diagnostics (protocol step 5)."""

from __future__ import annotations

import networkx as nx

from src.pipeline.p3_analysis.diagnostics import (
    aggregate,
    compare_to_gt,
    format_summary,
    graph_diagnostics,
)


def _path_graph(n: int, length_m: float = 10.0) -> nx.Graph:
    """A connected chain 0—1—…—(n-1) with equal-length edges."""
    g = nx.Graph()
    for i in range(n - 1):
        g.add_edge(i, i + 1, length_m=length_m)
    return g


# --------------------------------------------------------------------------- #
# graph_diagnostics
# --------------------------------------------------------------------------- #
def test_diagnostics_on_a_connected_chain():
    d = graph_diagnostics(_path_graph(5))
    assert d["is_empty"] is False
    assert d["n_nodes"] == 5 and d["n_edges"] == 4
    assert d["n_components"] == 1
    assert d["largest_cc_node_fraction"] == 1.0        # all one piece
    assert d["n_isolated_nodes"] == 0
    assert d["total_length_m"] == 40.0
    assert d["median_edge_length_m"] == 10.0
    assert d["reachable_pair_fraction"] == 1.0


def test_diagnostics_flags_empty_graph():
    d = graph_diagnostics(nx.Graph())
    assert d["is_empty"] is True                        # must not average away silently
    assert d["n_components"] == 0 and d["reachable_pair_fraction"] == 0.0


def test_diagnostics_counts_fragmentation_and_isolates():
    g = _path_graph(3)          # 0—1—2
    g.add_edge(10, 11, length_m=10.0)   # a second component
    g.add_node(99)                       # an isolated dust node
    d = graph_diagnostics(g)

    assert d["n_components"] == 3                        # chain + pair + isolate
    assert d["n_isolated_nodes"] == 1
    assert d["largest_cc_node_fraction"] == round(3 / 6, 4)
    assert d["reachable_pair_fraction"] < 1.0            # not everything routes


# --------------------------------------------------------------------------- #
# compare_to_gt — the A18-style diagnosis
# --------------------------------------------------------------------------- #
def test_compare_detects_a_more_fragmented_prediction():
    gt = _path_graph(5)                  # one component, 40 m
    pred = _path_graph(5)
    pred.remove_edge(2, 3)               # same nodes/length-ish, but broken in two
    cmp = compare_to_gt(pred, gt)

    assert cmp["more_fragmented_than_gt"] is True
    assert cmp["component_delta"] == 1
    assert cmp["largest_cc_fraction_delta"] < 0          # lost the single big piece


def test_compare_separates_missing_roads_from_broken_roads():
    """length_fraction_of_gt distinguishes 'never found it' from 'found but broke it'."""
    gt = _path_graph(5)                                  # 40 m
    missing = _path_graph(3)                             # only 20 m found, still intact
    cmp = compare_to_gt(missing, gt)

    assert cmp["length_fraction_of_gt"] == 0.5           # half the road length
    assert cmp["more_fragmented_than_gt"] is False       # what it found is connected


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #
def test_aggregate_tallies_fleet_level_facts():
    gt = _path_graph(5)
    broken = _path_graph(5)
    broken.remove_edge(2, 3)
    comparisons = {
        "c1": compare_to_gt(broken, gt),        # fragmented
        "c2": compare_to_gt(gt, gt),            # perfect
        "c3": compare_to_gt(nx.Graph(), gt),    # empty prediction
    }
    summary = aggregate(comparisons)

    assert summary["n_chips"] == 3
    assert summary["more_fragmented_than_gt_chips"] == "1/3"
    assert summary["n_empty_pred"] == 1                  # the degenerate one is surfaced
    assert summary["mean_components_gt"] == 1.0
    assert "more fragmented than GT on 1/3 chips" in format_summary(summary)


def test_aggregate_handles_no_chips():
    assert aggregate({})["n_chips"] == 0
    assert format_summary({"n_chips": 0}) == "no chips diagnosed"
