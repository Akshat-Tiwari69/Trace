"""Unit tests for S10 demand-weighted (percolation) centrality."""

from __future__ import annotations

import networkx as nx
from scipy.stats import spearmanr

from src.pipeline.p3_analysis.criticality import compute_betweenness
from src.pipeline.p3_analysis.percolation import (
    compare_centralities,
    degree_demand,
    percolation_centrality,
    spatial_demand,
)


def _grid(n: int = 5) -> nx.Graph:
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


def test_uniform_demand_tracks_betweenness_ranking():
    """With equal demand everywhere, percolation ranks nodes much like betweenness
    (not identical — it has its own normalization — but strongly correlated)."""
    g = _grid(5)
    states = {n: 1.0 for n in g.nodes}
    pc = percolation_centrality(g, states)
    bc = compute_betweenness(g)
    nodes = list(g.nodes)
    rho = spearmanr([bc[n] for n in nodes], [pc[n] for n in nodes]).statistic
    assert rho > 0.9


def test_spatial_demand_concentrates_at_corner():
    g = _grid(5)
    states = spatial_demand(g, corner="sw")          # bump at (min x, min y) = node 0
    assert states[0] > states[24]                    # node 24 is the opposite (ne) corner
    assert 0.0 <= min(states.values()) and max(states.values()) <= 1.0


def test_degree_demand_normalised():
    g = _grid(4)
    d = degree_demand(g)
    assert max(d.values()) == 1.0                     # normalised to the busiest junction
    assert all(0.0 <= v <= 1.0 for v in d.values())


def test_compare_centralities_structure():
    g = _grid(5)
    result = compare_centralities(g, spatial_demand(g, "se"), top_n=5)
    assert set(result) >= {"spearman", "top_n_overlap", "top_betweenness", "top_percolation"}
    assert -1.0 <= result["spearman"] <= 1.0
    assert len(result["top_betweenness"]) == 5
