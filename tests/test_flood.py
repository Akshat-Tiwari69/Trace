"""Unit tests for S11 flood / elevation failure scenario."""

from __future__ import annotations

import networkx as nx

from src.pipeline.p3_analysis.flood import (
    flood_comparison,
    flood_order,
    nodes_below_elevation,
    nodes_in_polygon,
)


def _grid(n: int = 5) -> nx.Graph:
    """An n×n grid with integer (x, y) coords and unit-metre edges."""
    g = nx.Graph()
    for i in range(n):
        for j in range(n):
            g.add_node(i * n + j, x=float(j), y=float(i))
    for i in range(n):
        for j in range(n):
            if j + 1 < n:
                g.add_edge(i * n + j, i * n + j + 1, length_m=1.0)
            if i + 1 < n:
                g.add_edge(i * n + j, (i + 1) * n + j, length_m=1.0)
    return g


def test_nodes_in_polygon_selects_enclosed():
    g = _grid(4)
    poly = [[-0.5, -0.5], [1.5, -0.5], [1.5, 1.5], [-0.5, 1.5]]   # covers x,y ∈ [0,1]
    assert set(nodes_in_polygon(g, poly)) == {0, 1, 4, 5}          # the 2×2 corner


def test_nodes_below_elevation():
    g = _grid(2)
    for n in g.nodes:
        g.nodes[n]["elevation"] = float(n)
    assert set(nodes_below_elevation(g, 2.0)) == {0, 1}


def test_flood_order_lowest_elevation_first():
    g = _grid(2)
    for n in g.nodes:
        g.nodes[n]["elevation"] = float(3 - n)        # node 3 is lowest
    assert flood_order(g, [0, 1, 2, 3])[0] == 3


def test_flood_order_falls_back_to_centroid_distance():
    g = _grid(3)
    order = flood_order(g, [0, 4, 8])                  # no elevation → centroid-first
    assert order[0] == 4                               # the central node is closest to centroid


def test_flood_comparison_returns_three_curves():
    g = _grid(6)
    flooded = [14, 15, 20, 21]                          # a central 2×2 block
    result = flood_comparison(g, flooded)
    assert set(result["curves"]) == {"flood", "targeted", "random"}
    assert len(result["curves"]["flood"]) == len(flooded) + 1   # baseline + one per removal
    assert set(result["end_ri"]) == {"flood", "targeted", "random"}
    # damage ranking lists all three, most-damaging (lowest RI) first
    assert set(result["damage_ranking"]) == {"flood", "targeted", "random"}
    end = result["end_ri"]
    assert end[result["damage_ranking"][0]] <= end[result["damage_ranking"][-1]]
