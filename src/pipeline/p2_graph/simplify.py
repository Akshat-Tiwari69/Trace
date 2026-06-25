"""Graph simplification — prune stubs + collapse degree-2 chains (task S3).

A skeleton→sknw graph (even after healing) carries two kinds of redundancy that
bloat the network without adding routing information (``docs/Research.md`` →
Roadmap §B; the same idea as OSMnx's ``simplify_graph``):

* **Degree-2 interstitial nodes** — points that merely sit *along* a road, not at
  a junction. They can be merged away: ``A—B—C`` with ``B`` degree-2 becomes a
  single edge ``A—C`` whose geometry is the two polylines stitched together and
  whose ``length_m`` is their sum. This is **lossless** for routing — same path,
  fewer nodes.
* **Short dead-end stubs** — tiny degree-1 spurs thrown off by skeletonisation
  (not real cul-de-sacs). Trimmed iteratively below ``min_stub_len_m``.

Both operations **preserve connectivity**: collapsing a degree-2 node keeps its
two neighbours joined, and trimming a leaf never splits a component. We assert
that the connected-component count never increases. Pure CPU, NetworkX only.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from src.pipeline.p2_graph.healing import UnionFind
from src.pipeline.p2_graph.skeleton_graph import _annotate_degree_and_type

if TYPE_CHECKING:
    import networkx as nx


def _oriented_geometry(graph: "nx.Graph", u: int, v: int) -> list:
    """Return edge ``(u, v)``'s polyline oriented to run from ``u`` to ``v``."""
    geom = [list(p) for p in graph.edges[u, v].get("geometry", [])]
    if not geom:
        return [[graph.nodes[u]["x"], graph.nodes[u]["y"]],
                [graph.nodes[v]["x"], graph.nodes[v]["y"]]]
    u_xy = np.array([graph.nodes[u]["x"], graph.nodes[u]["y"]])
    # If the polyline's first point is nearer v than u, it's stored v→u: flip it.
    if np.hypot(*(np.array(geom[0]) - u_xy)) > np.hypot(*(np.array(geom[-1]) - u_xy)):
        geom = geom[::-1]
    return geom


def collapse_degree2_nodes(graph: "nx.Graph") -> int:
    """Merge every degree-2 pass-through node into a single edge. Returns count.

    Skips a node when collapsing it would create a self-loop (its two neighbours
    are the same node) or duplicate an existing edge (a parallel edge the simple
    graph can't hold) — those nodes are left in place rather than lose geometry.
    """
    collapsed = 0
    for node in list(graph.nodes):
        if node not in graph or graph.degree(node) != 2:
            continue
        a, c = list(graph.neighbors(node))
        if a == c or graph.has_edge(a, c):
            continue  # would self-loop or clash with an existing edge

        geom_a = _oriented_geometry(graph, a, node)
        geom_c = _oriented_geometry(graph, node, c)
        merged_geom = geom_a + geom_c[1:]  # stitch, dropping the duplicated middle
        merged_len = float(graph.edges[a, node].get("length_m", 0.0)
                           + graph.edges[node, c].get("length_m", 0.0))
        bridged = bool(graph.edges[a, node].get("is_bridged", False)
                       or graph.edges[node, c].get("is_bridged", False))

        graph.remove_node(node)  # drops both incident edges
        graph.add_edge(a, c, length_m=merged_len, geometry=merged_geom, is_bridged=bridged)
        collapsed += 1
    return collapsed


def prune_short_stubs(graph: "nx.Graph", min_stub_len_m: float, max_iter: int = 20) -> int:
    """Iteratively drop degree-1 spurs shorter than ``min_stub_len_m``. Returns count.

    Iterative because trimming one stub can expose another short one behind it.
    Leaves longer dead-ends (real cul-de-sacs) untouched.
    """
    removed = 0
    for _ in range(max_iter):
        stubs = [
            n for n in graph.nodes
            if graph.degree(n) == 1
            and float(next(iter(graph.edges(n, data=True)))[2].get("length_m", 0.0)) < min_stub_len_m
        ]
        if not stubs:
            break
        graph.remove_nodes_from(stubs)
        removed += len(stubs)
    return removed


@dataclasses.dataclass
class SimplifyReport:
    """Before/after sizes for a simplification pass."""

    nodes_before: int
    nodes_after: int
    edges_before: int
    edges_after: int
    components_before: int
    components_after: int
    stubs_pruned: int
    nodes_collapsed: int

    @property
    def node_reduction_pct(self) -> float:
        if self.nodes_before == 0:
            return 0.0
        return 100.0 * (self.nodes_before - self.nodes_after) / self.nodes_before


def simplify_graph(graph: "nx.Graph", min_stub_len_m: float = 15.0) -> SimplifyReport:
    """Prune short stubs then collapse degree-2 chains, in place. Returns a report.

    Guarantees the connected-component count does not *increase* (simplification
    never disconnects the network).
    """
    import networkx as nx

    n0, e0 = graph.number_of_nodes(), graph.number_of_edges()
    c0 = nx.number_connected_components(graph)

    pruned = prune_short_stubs(graph, min_stub_len_m)
    collapsed = collapse_degree2_nodes(graph)
    _annotate_degree_and_type(graph)

    c1 = nx.number_connected_components(graph)
    if c1 > c0:  # safety net — should never happen
        raise AssertionError(f"simplify split the graph: {c0} -> {c1} components")

    return SimplifyReport(
        nodes_before=n0, nodes_after=graph.number_of_nodes(),
        edges_before=e0, edges_after=graph.number_of_edges(),
        components_before=c0, components_after=c1,
        stubs_pruned=pruned, nodes_collapsed=collapsed,
    )


# --------------------------------------------------------------------------- #
# S4 — near-duplicate node consolidation
# --------------------------------------------------------------------------- #
def consolidate_nearby_nodes(graph: "nx.Graph", tol_m: float) -> int:
    """Merge clusters of near-coincident junction nodes into one. Returns count merged.

    A single physical intersection is often split by skeletonisation/healing into
    several nodes a few metres apart, joined by **sub-tolerance edges**. We union
    the endpoints of every edge shorter than ``tol_m`` into clusters and collapse
    each cluster to its centroid, rewiring outside edges to the kept node.

    **Overpass guard (Boeing 2025):** because we only merge along an *existing*
    short edge, two roads that merely *cross* at different grades — which share no
    edge between the levels — are never merged, even though their nodes may be
    metres apart in projection. Proximity alone is not enough; a road link is
    required.
    """
    uf = UnionFind(list(graph.nodes))
    for u, v, data in graph.edges(data=True):
        if u != v and float(data.get("length_m", np.inf)) < tol_m:
            uf.union(u, v)

    clusters: dict[int, list[int]] = {}
    for n in graph.nodes:
        clusters.setdefault(uf.find(n), []).append(n)

    merged = 0
    for members in clusters.values():
        if len(members) < 2:
            continue
        keeper = min(members)
        member_set = set(members)
        cx = float(np.mean([graph.nodes[n]["x"] for n in members]))
        cy = float(np.mean([graph.nodes[n]["y"] for n in members]))

        for n in members:
            if n == keeper:
                continue
            for m in list(graph.neighbors(n)):
                if m in member_set:
                    continue  # edge internal to the cluster → drop (would self-loop)
                data = dict(graph.edges[n, m])
                # keep the shorter of any parallel connection to the same outside node
                if (not graph.has_edge(keeper, m)
                        or data.get("length_m", np.inf)
                        < graph.edges[keeper, m].get("length_m", np.inf)):
                    graph.add_edge(keeper, m, **data)
            graph.remove_node(n)

        graph.nodes[keeper]["x"], graph.nodes[keeper]["y"] = cx, cy
        merged += len(members) - 1
    return merged


@dataclasses.dataclass
class ConsolidateReport:
    """Before/after sizes for a consolidation pass."""

    nodes_before: int
    nodes_after: int
    components_before: int
    components_after: int
    nodes_merged: int


def consolidate_graph(graph: "nx.Graph", tol_m: float = 10.0) -> ConsolidateReport:
    """Consolidate near-duplicate junctions, then tidy any new degree-2 nodes.

    Preserves connectivity (merging along edges only contracts, never splits).
    """
    import networkx as nx

    n0 = graph.number_of_nodes()
    c0 = nx.number_connected_components(graph)

    merged = consolidate_nearby_nodes(graph, tol_m)
    collapse_degree2_nodes(graph)  # consolidation can expose new pass-through nodes
    _annotate_degree_and_type(graph)

    c1 = nx.number_connected_components(graph)
    if c1 > c0:  # safety net
        raise AssertionError(f"consolidation split the graph: {c0} -> {c1} components")

    return ConsolidateReport(
        nodes_before=n0, nodes_after=graph.number_of_nodes(),
        components_before=c0, components_after=c1, nodes_merged=merged,
    )
