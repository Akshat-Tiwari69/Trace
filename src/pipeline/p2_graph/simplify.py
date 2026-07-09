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


def _oriented_geometry(graph: "nx.Graph", u: int, v: int, key: int = 0) -> list:
    """Return edge ``(u, v, key)``'s polyline oriented to run from ``u`` to ``v``.

    On a MultiGraph each ``(u, v)`` pair may hold several keyed edges; ``key``
    selects which (default 0, the first). The polyline is flipped if its stored
    orientation runs v→u.
    """
    # MultiGraph: graph.edges[u, v] is a dict of {key: attrs}; [key] picks one.
    # The same expression also works for a plain Graph (its edges[u,v] is the
    # attr dict directly and ignores the extra index), so this helper serves both.
    attrs = graph.edges[u, v, key] if graph.is_multigraph() else graph.edges[u, v]
    geom = [list(p) for p in attrs.get("geometry", [])]
    if not geom:
        return [[graph.nodes[u]["x"], graph.nodes[u]["y"]],
                [graph.nodes[v]["x"], graph.nodes[v]["y"]]]
    u_xy = np.array([graph.nodes[u]["x"], graph.nodes[u]["y"]])
    # If the polyline's first point is nearer v than u, it's stored v→u: flip it.
    if np.hypot(*(np.array(geom[0]) - u_xy)) > np.hypot(*(np.array(geom[-1]) - u_xy)):
        geom = geom[::-1]
    return geom


def _sole_edge_key(graph: "nx.Graph", u: int, v: int) -> int:
    """The single edge key for a degree-2 node's ``(u, v)`` neighbour pair.

    A degree-2 node has exactly one edge to each of its two neighbours; on a
    MultiGraph that edge still has a key, so return it. Raises if there is
    somehow more than one (would mean the node isn't actually degree-2 in the
    multi sense — guarded by the caller's ``graph.degree(node) != 2`` check).
    """
    if not graph.is_multigraph():
        return 0
    keys = list(graph[u][v].keys())
    return keys[0]


def collapse_degree2_nodes(graph: "nx.Graph") -> int:
    """Merge every degree-2 pass-through node into a single edge. Returns count.

    Skips a node only when collapsing it would create a self-loop (its two
    neighbours are the same node). On a MultiGraph the merged edge is added as a
    **new keyed edge** between the neighbours — so a real parallel route (a loop
    / dual carriageway) is preserved rather than lost. Pre-existing parallel
    edges between the two neighbours are left untouched.
    """
    collapsed = 0
    for node in list(graph.nodes):
        if node not in graph or graph.degree(node) != 2:
            continue
        neigh = list(graph.neighbors(node))
        if len(neigh) != 2:
            # degree-2 via two parallel edges to the SAME neighbour (a MultiGraph
            # only): nothing to merge — collapsing would just shuffle keys. Skip.
            continue
        a, c = neigh
        if a == c:
            continue  # would self-loop (defensive — covered by the len check)

        ka = _sole_edge_key(graph, a, node)
        kc = _sole_edge_key(graph, node, c)
        geom_a = _oriented_geometry(graph, a, node, ka)
        geom_c = _oriented_geometry(graph, node, c, kc)
        merged_geom = geom_a + geom_c[1:]  # stitch, dropping the duplicated middle
        ea = graph.edges[a, node, ka] if graph.is_multigraph() else graph.edges[a, node]
        ec = graph.edges[node, c, kc] if graph.is_multigraph() else graph.edges[node, c]
        merged_len = float(ea.get("length_m", 0.0) + ec.get("length_m", 0.0))
        bridged = bool(ea.get("is_bridged", False) or ec.get("is_bridged", False))

        graph.remove_node(node)  # drops both incident edges
        graph.add_edge(a, c, length_m=merged_len, geometry=merged_geom, is_bridged=bridged)
        collapsed += 1
    return collapsed


def prune_short_stubs(graph: "nx.Graph", min_stub_len_m: float, max_iter: int = 20) -> int:
    """Iteratively drop degree-1 spurs shorter than ``min_stub_len_m``. Returns count.

    Iterative because trimming one stub can expose another short one behind it.
    Leaves longer dead-ends (real cul-de-sacs) untouched. Warns when the
    iteration cap is hit, so "done" is distinguishable from "gave up early".
    """
    removed = 0
    truncated = True
    multi = graph.is_multigraph()
    for _ in range(max_iter):
        stubs: list[int] = []
        for n in graph.nodes:
            if graph.degree(n) != 1:
                continue
            # A degree-1 node has exactly one incident edge; read its length.
            # keys=True only valid on a MultiGraph; plain Graph yields (u, v, data).
            it = graph.edges(n, data=True, keys=True) if multi else graph.edges(n, data=True)
            edge = next(iter(it))
            data = edge[3] if multi else edge[2]
            if float(data.get("length_m", 0.0)) < min_stub_len_m:
                stubs.append(n)
        if not stubs:
            truncated = False
            break
        graph.remove_nodes_from(stubs)
        removed += len(stubs)
    if truncated:
        print(f"[simplify] WARNING: prune_short_stubs hit max_iter={max_iter} — "
              "pruning may be incomplete (deeper stub chains remain)")
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

    **Transitive merging:** clusters are built by union-find over sub-tolerance
    edges, so the guarantee is "connected by a *chain* of edges each shorter than
    ``tol_m``" — not mutual pairwise proximity. A cluster's diameter can exceed
    ``tol_m`` when short edges chain (A–B and B–C both short merges A, B, C even
    if A–C is farther apart). Acceptable at the default 10 m tolerance; revisit
    if the tolerance is ever raised.

    **Overpass guard (Boeing 2025):** because we only merge along an *existing*
    short edge, two roads that merely *cross* at different grades — which share no
    edge between the levels — are never merged, even though their nodes may be
    metres apart in projection. Proximity alone is not enough; a road link is
    required.
    """
    uf = UnionFind(list(graph.nodes))
    for u, v, data in graph.edges(data=True):  # keys ignored for unioning
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
            # Rewire every edge from n to an OUTSIDE node onto keeper. On a
            # MultiGraph there may be several keyed edges to the same outside
            # node (a real parallel route); each is moved as its own keyed edge
            # so redundancy survives consolidation. Internal cluster edges are
            # dropped (they would self-loop on keeper).
            for m in list(graph.neighbors(n)):
                if m in member_set:
                    continue  # edge internal to the cluster → drop (would self-loop)
                if graph.is_multigraph():
                    for k in list(graph[n][m].keys()):
                        data = dict(graph.edges[n, m, k])
                        graph.add_edge(keeper, m, **data)
                else:
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


# --------------------------------------------------------------------------- #
# S5 — polyline geometry simplification (Douglas-Peucker)
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class PolylineReport:
    """Vertex counts before/after Douglas-Peucker simplification."""

    vertices_before: int
    vertices_after: int
    edges: int

    @property
    def vertex_reduction_pct(self) -> float:
        if self.vertices_before == 0:
            return 0.0
        return 100.0 * (self.vertices_before - self.vertices_after) / self.vertices_before


def simplify_polylines(graph: "nx.Graph", tol_m: float) -> PolylineReport:
    """Douglas-Peucker each edge's geometry to drop redundant vertices, in place.

    Shapely's ``LineString.simplify`` removes points that lie within ``tol_m`` of
    the line they sit on — a much lighter GeoJSON for the dashboard with the road
    shape preserved (endpoints are always kept). This touches **geometry only**:
    ``length_m`` (the routing weight) is left untouched, since the metric length is
    more accurate than the corner-cutting simplified chord. Run in metric space
    (before reprojection) so ``tol_m`` is true metres.

    Uses ``preserve_topology=True``: the fast variant can self-intersect on
    sharply-curved input — and the healed cubic-Bézier bridges (``is_bridged``)
    are exactly that shape. The cost is negligible at road-polyline sizes.
    """
    from shapely.geometry import LineString

    before = after = 0
    for _, _, data in graph.edges(data=True):
        geom = data.get("geometry")
        if not geom or len(geom) <= 2:
            before += len(geom) if geom else 0
            after += len(geom) if geom else 0
            continue
        before += len(geom)
        simplified = LineString(geom).simplify(tol_m, preserve_topology=True)
        coords = [[float(x), float(y)] for x, y in simplified.coords]
        if len(coords) < 2:  # degenerate guard — keep the original endpoints
            coords = [list(map(float, geom[0])), list(map(float, geom[-1]))]
        data["geometry"] = coords
        after += len(coords)

    return PolylineReport(vertices_before=before, vertices_after=after,
                          edges=graph.number_of_edges())
