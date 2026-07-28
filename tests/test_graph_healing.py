"""Unit tests for Phase II healing — Union-Find + angle-aware MST bridging.

These are the load-bearing, deterministic graph functions (``docs/Rules.md`` →
Testing: "unit tests for MST/Union-Find healing"). They build small graphs by
hand (no skeletonisation, no network) so the healing logic is tested in isolation.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from src.pipeline.p2_graph.healing import (
    UnionFind,
    _bridge_geometry,
    _polyline_length,
    heal_graph,
)
from src.pipeline.p2_graph.skeleton_graph import (
    TYPE_BRIDGED,
    _annotate_degree_and_type,
    build_metric_to_pixel,
    prune_degenerate_edges,
)
from src.pipeline.p2_graph.spike_osm import simulate_occlusion


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


def test_simulate_occlusion_patch_is_exact_size():
    """A patch zeroes an exactly patch_px×patch_px window (not 2·half)."""
    mask = np.ones((60, 60), dtype=np.uint8)
    out = simulate_occlusion(mask, n_patches=1, patch_px=11, seed=7)
    assert out is not mask  # returns a copy

    # The window is clamped inward at borders, so on a 60×60 image an 11-px patch
    # is always exactly 11×11 regardless of where its centre landed.
    removed = int((mask > 0).sum() - (out > 0).sum())
    assert removed == 11 * 11  # old 2·half slicing would zero 10×10


def test_simulate_occlusion_patch_exact_at_corner():
    """Even centred at the very corner, the clamped window stays patch_px²."""
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[0, 0] = 1  # the only road pixel → every patch centres on the corner
    out = simulate_occlusion(mask, n_patches=1, patch_px=7, seed=0)
    # the 7×7 window is shifted fully inside the image (rows/cols 0..6)
    assert (out[0:7, 0:7] == 0).all()
    assert mask.sum() - out.sum() == 1  # only the single road pixel was zeroed


def test_simulate_occlusion_zero_patches_returns_copy():
    mask = np.ones((5, 5), dtype=np.uint8)
    out = simulate_occlusion(mask, n_patches=0, patch_px=3)
    assert out is not mask and np.array_equal(out, mask)  # copy, unchanged


# --------------------------------------------------------------------------- #
# S6 — curved (smooth) bridge geometry
# --------------------------------------------------------------------------- #
def test_bridge_geometry_straight_when_collinear():
    """A collinear, facing-each-other gap stays a straight line."""
    geom = _bridge_geometry(
        np.array([0.0, 0.0]), np.array([10.0, 0.0]),
        np.array([1.0, 0.0]), np.array([-1.0, 0.0]),
    )
    assert max(abs(y) for _, y in geom) < 1e-6                  # no sideways bend
    assert _polyline_length(geom) == pytest.approx(10.0, abs=1e-6)


def test_bridge_geometry_curves_when_offset():
    """An offset gap is drawn as a smooth multi-point curve, longer than the chord."""
    p_u, p_v = np.array([0.0, 0.0]), np.array([10.0, 5.0])
    geom = _bridge_geometry(p_u, p_v, np.array([1.0, 0.0]), np.array([-1.0, 0.0]))
    assert len(geom) > 2                                        # a curve, not a segment
    assert geom[0] == [0.0, 0.0] and geom[-1] == [10.0, 5.0]    # endpoints exact
    chord = float(np.hypot(10.0, 5.0))
    assert _polyline_length(geom) > chord                       # bends, so longer


def test_bridge_geometry_falls_back_to_straight():
    geom = _bridge_geometry(np.array([0.0, 0.0]), np.array([3.0, 4.0]), None, None)
    assert geom == [[0.0, 0.0], [3.0, 4.0]]                     # undefined heading → segment


def test_bridge_geometry_is_magnitude_independent():
    """A non-unit heading yields the same curve as its normalised version."""
    p_u, p_v = np.array([0.0, 0.0]), np.array([10.0, 5.0])
    unit = _bridge_geometry(p_u, p_v, np.array([1.0, 0.0]), np.array([-1.0, 0.0]))
    scaled = _bridge_geometry(p_u, p_v, np.array([5.0, 0.0]), np.array([-9.0, 0.0]))
    assert np.allclose(np.array(unit), np.array(scaled))


def test_heal_draws_curved_bridge():
    """Healing an offset gap produces a multi-point curved bridge edge."""
    g = nx.Graph()
    _road(g, 0, (0.0, 0.0), 1, (10.0, 0.0))      # endpoint 1 heads +x
    _road(g, 2, (20.0, 6.0), 3, (30.0, 6.0))     # endpoint 2 heads -x, offset +6 in y
    _annotate_degree_and_type(g)

    g, report = heal_graph(g, gap_max_m=40.0, angle_max_deg=80.0)
    assert report.bridges_added == 1
    assert len(g.edges[1, 2]["geometry"]) > 2     # curved, not a straight 2-point line


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


def test_equal_score_healing_is_independent_of_insertion_order():
    """Two equally good bridges resolve by node id, not set/KD-tree order."""
    def parallel_fragments(reverse: bool) -> nx.Graph:
        graph = nx.Graph()
        roads = [
            (0, (0.0, 0.0), 1, (10.0, 0.0)),
            (2, (0.0, 10.0), 3, (10.0, 10.0)),
        ]
        for args in reversed(roads) if reverse else roads:
            _road(graph, *args)
        _annotate_degree_and_type(graph)
        return graph

    first, _ = heal_graph(parallel_fragments(False), gap_max_m=11.0, angle_max_deg=100.0)
    second, _ = heal_graph(parallel_fragments(True), gap_max_m=11.0, angle_max_deg=100.0)
    inferred = lambda graph: {
        tuple(sorted((u, v))) for u, v, data in graph.edges(data=True)
        if data.get("is_bridged")
    }
    assert inferred(first) == inferred(second) == {(0, 2)}


def test_bridge_touching_nonincident_road_is_rejected():
    graph = nx.Graph()
    _road(graph, 0, (0.0, 0.0), 1, (4.0, 0.0))
    _road(graph, 2, (6.0, 0.0), 3, (10.0, 0.0))
    _road(graph, 4, (5.0, 0.0), 5, (5.0, 4.0))
    _annotate_degree_and_type(graph)

    healed, report = heal_graph(graph, gap_max_m=3.0, angle_max_deg=30.0)
    assert report.bridges_added == 0
    assert report.bridges_rejected_crossing >= 1
    assert not healed.has_edge(1, 2)


def test_curved_bridge_crossing_nonincident_road_is_rejected():
    graph = nx.Graph()
    _road(graph, 0, (0.0, 0.0), 1, (-1.0, -1.0))
    _road(graph, 2, (10.0, 0.0), 3, (11.0, -1.0))
    _road(graph, 4, (5.0, 1.5), 5, (5.0, 2.0))
    _annotate_degree_and_type(graph)

    healed, report = heal_graph(graph, gap_max_m=10.1, angle_max_deg=46.0)
    assert not healed.has_edge(0, 2)
    assert report.bridges_rejected_crossing >= 1


def test_probability_support_samples_segment_interiors():
    from src.pipeline.p2_graph.healing import sample_prob_along_polyline

    probability = np.zeros((1, 11), dtype=np.float32)
    probability[0, (0, 10)] = 1.0
    support = sample_prob_along_polyline(
        [[0.0, 0.0], [10.0, 0.0]], probability,
        lambda x, y: (y, x),
    )
    assert support == pytest.approx(2 / 11)


# --------------------------------------------------------------------------- #
# A39 — probability-map corridor check
# --------------------------------------------------------------------------- #
# The collinear-gap fixture's bridge (nodes 1->2) is a straight line at y=0,
# x in [10, 20] (see _collinear_gap). ``metric_to_pixel`` here is the identity
# map (resolution_m=1.0, no transform), so a prob array's (row, col) lines up
# with the graph's (y, x) directly.
_M2P = build_metric_to_pixel(transform=None, resolution_m=1.0)


def test_heal_corridor_kept_when_prob_supports_it():
    """A high-probability corridor along the bridge keeps it (mask+prob agree)."""
    g = _collinear_gap(gap=10.0)
    prob = np.zeros((5, 30), dtype=np.float32)
    prob[0, 8:23] = 0.9  # covers the x in [10, 20] corridor with margin

    g, report = heal_graph(
        g, gap_max_m=40.0, angle_max_deg=60.0,
        prob=prob, metric_to_pixel=_M2P, min_corridor_support=0.3,
    )
    assert report.bridges_added == 1
    assert report.bridges_rejected_corridor == 0
    assert g.edges[1, 2]["is_bridged"] is True


def test_heal_corridor_rejects_unsupported_bridge():
    """A frontage-road-style bridge with no P1 probability support is rejected."""
    g = _collinear_gap(gap=10.0)
    prob = np.zeros((5, 30), dtype=np.float32)  # no road signal anywhere

    g, report = heal_graph(
        g, gap_max_m=40.0, angle_max_deg=60.0,
        prob=prob, metric_to_pixel=_M2P, min_corridor_support=0.3,
    )
    assert report.bridges_added == 0
    assert report.bridges_rejected_corridor == 1
    assert nx.number_connected_components(g) == 2  # left disconnected, correctly


def test_heal_corridor_disabled_when_prob_none_matches_baseline():
    """``prob=None`` must heal identically regardless of the corridor knobs.

    Mask-only input (an upload or an old artifact with no persisted prob.png)
    has to behave exactly as it did before this feature existed — even an
    impossible-to-satisfy ``min_corridor_support`` must not reject anything.
    """
    baseline = _collinear_gap(gap=10.0)
    _, baseline_report = heal_graph(baseline, gap_max_m=40.0, angle_max_deg=60.0)

    g = _collinear_gap(gap=10.0)
    _, report = heal_graph(
        g, gap_max_m=40.0, angle_max_deg=60.0,
        prob=None, metric_to_pixel=_M2P, min_corridor_support=0.9, corridor_samples=16,
    )
    assert report.bridges_added == baseline_report.bridges_added == 1
    assert report.bridges_rejected_corridor == 0
    assert g.edges[1, 2]["is_bridged"] is True


def test_heal_corridor_out_of_bounds_samples_count_as_zero():
    """Bridge points outside the prob raster count as 0, not an IndexError."""
    g = _collinear_gap(gap=10.0)
    # Raster only covers cols 0..10 — most of the x in [10, 20] corridor falls
    # outside it, so out-of-bounds-as-0 must drag the mean below the threshold.
    prob = np.ones((5, 11), dtype=np.float32)

    g, report = heal_graph(
        g, gap_max_m=40.0, angle_max_deg=60.0,
        prob=prob, metric_to_pixel=_M2P, min_corridor_support=0.3,
    )
    assert report.bridges_added == 0
    assert report.bridges_rejected_corridor == 1
