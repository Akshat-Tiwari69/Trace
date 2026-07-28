"""Unit tests for S10 demand-weighted (percolation) centrality."""

from __future__ import annotations

import networkx as nx
import pytest
import warnings
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


def test_spatial_demand_validates_empty_graph_corner_and_sigma():
    with pytest.raises(ValueError, match="at least one node"):
        spatial_demand(nx.Graph())
    with pytest.raises(ValueError, match="corner"):
        spatial_demand(_grid(2), corner="east")
    with pytest.raises(ValueError, match="sigma_frac"):
        spatial_demand(_grid(2), sigma_frac=0.0)


def test_percolation_validates_state_coverage_and_range():
    graph = _grid(2)
    with pytest.raises(ValueError, match="every graph node"):
        percolation_centrality(graph, {0: 1.0})
    states = {node: 1.0 for node in graph}
    states[0] = -0.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        percolation_centrality(graph, states)


@pytest.mark.parametrize("positive_nodes", [0, 1])
def test_percolation_rejects_fewer_than_two_positive_states(positive_nodes):
    graph = _grid(2)
    states = {node: float(index < positive_nodes) for index, node in enumerate(graph)}
    with pytest.raises(ValueError, match="at least two positive"):
        percolation_centrality(graph, states)


def test_compare_centralities_clamps_top_n_to_graph_size():
    graph = _grid(2)
    result = compare_centralities(graph, spatial_demand(graph), top_n=10)
    assert result["top_n"] == graph.number_of_nodes()
    assert len(result["top_betweenness"]) == graph.number_of_nodes()


def test_tiny_constant_graph_has_explicit_undefined_correlation_without_warning():
    graph = nx.path_graph(2)
    nx.set_edge_attributes(graph, 1.0, "length_m")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = compare_centralities(graph, {0: 1.0, 1: 1.0})

    assert result["spearman"] is None
    assert result["spearman_defined"] is False
    assert "constant" in result["spearman_reason"]
