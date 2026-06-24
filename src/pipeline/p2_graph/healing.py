"""Graph healing — MST + Union-Find, scored by distance AND angular alignment.

A skeletonised mask is fragmented: trees, shadows and vehicles break roads, so
the raw graph has many disconnected components (``docs/PRD.md`` G2, FR6). Healing
reconnects them into a single routable network by bridging dangling **endpoints**
across components.

What makes this more than "connect nearest endpoints" (``docs/Research.md`` →
*Open Problems* #2) is the **angle-aware score**: a candidate bridge is cheap if
it is short *and* roughly continues the direction each road was already heading,
and expensive (or rejected) if it would force the road to turn unnaturally. We
then run **Kruskal's MST over a Union-Find** structure: sort candidate bridges by
score and add one only if it joins two still-separate components. The result is a
minimum-spanning *forest* of bridges — exactly the MST/Union-Find healing the
docs specify — with every added edge flagged ``is_bridged=True``.

Pure-Python + numpy/scipy only (CPU; ``docs/Tracker.md`` Shaivi lane). NetworkX
and scipy's KD-tree are imported lazily.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np

from src.pipeline.p2_graph.skeleton_graph import (
    TYPE_BRIDGED,
    _annotate_degree_and_type,
)

if TYPE_CHECKING:
    import networkx as nx


# --------------------------------------------------------------------------- #
# Union-Find (Disjoint Set Union)
# --------------------------------------------------------------------------- #
class UnionFind:
    """Classic disjoint-set with path compression + union by rank.

    Used both to track which graph component a node is in and to drive the
    Kruskal MST that selects healing bridges.
    """

    def __init__(self, items) -> None:
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        """Return the representative of ``x``'s set (with path compression)."""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # compress
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> bool:
        """Merge the sets of ``a`` and ``b``. Return False if already joined."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# --------------------------------------------------------------------------- #
# Candidate bridges
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Bridge:
    """A candidate healing edge between two endpoints in different components."""

    u: int
    v: int
    distance_m: float
    angle_deg: float   # worst turn the road makes to follow this bridge (0 = straight)
    score: float       # distance penalised by the turn; lower is better


def _endpoint_direction(graph: "nx.Graph", node: int) -> np.ndarray | None:
    """Unit vector of the road *heading outward* as it reaches a degree-1 node.

    Uses the last segment of the incident edge's metric polyline, oriented so it
    points away from the road interior (the direction a straight continuation
    would go). Returns ``None`` if the direction is undefined (zero-length).
    """
    neighbors = list(graph.neighbors(node))
    if not neighbors:
        return None
    edge = graph.edges[node, neighbors[0]]
    geom = np.asarray(edge["geometry"], dtype=float)
    node_xy = np.array([graph.nodes[node]["x"], graph.nodes[node]["y"]])
    # Orient the polyline so its last point is the endpoint node.
    if np.hypot(*(geom[0] - node_xy)) < np.hypot(*(geom[-1] - node_xy)):
        geom = geom[::-1]
    # Walk back from the endpoint to the first point that isn't coincident.
    for prev in range(len(geom) - 2, -1, -1):
        vec = geom[-1] - geom[prev]
        norm = float(np.hypot(*vec))
        if norm > 1e-9:
            return vec / norm
    return None


def _turn_angle_deg(direction: np.ndarray | None, bridge_vec: np.ndarray) -> float:
    """Angle (deg) between a road's outgoing heading and the bridge direction.

    0° = the bridge continues the road perfectly straight; 180° = it doubles
    back. If the heading is undefined, return 0 (don't penalise what we can't
    measure).
    """
    bnorm = float(np.hypot(*bridge_vec))
    if direction is None or bnorm < 1e-9:
        return 0.0
    cos = float(np.clip(np.dot(direction, bridge_vec / bnorm), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def find_candidate_bridges(
    graph: "nx.Graph",
    components: list[set[int]],
    gap_max_m: float,
    angle_max_deg: float,
    angle_penalty_factor: float,
) -> list[Bridge]:
    """Propose angle-aware bridges between endpoints of *different* components.

    For every endpoint (degree-1 node) we look up, via a KD-tree, the endpoints
    of other components within ``gap_max_m``. A candidate is kept only if the
    road turns ≤ ``angle_max_deg`` at *both* ends. The score is the gap distance
    scaled up by how sharply the road has to turn::

        score = distance_m · (1 + angle_penalty_factor · (worst_angle / 180))

    so a short, straight bridge wins over a short, kinked one of equal length.
    """
    from scipy.spatial import cKDTree

    comp_of: dict[int, int] = {}
    for idx, comp in enumerate(components):
        for n in comp:
            comp_of[n] = idx

    endpoints = [n for n in graph.nodes if graph.degree(n) == 1]
    if len(endpoints) < 2:
        return []

    coords = np.array([[graph.nodes[n]["x"], graph.nodes[n]["y"]] for n in endpoints])
    directions = {n: _endpoint_direction(graph, n) for n in endpoints}
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=gap_max_m)  # indices into ``endpoints``

    bridges: list[Bridge] = []
    seen: set[tuple[int, int]] = set()
    for i, j in pairs:
        u, v = endpoints[i], endpoints[j]
        if comp_of[u] == comp_of[v]:
            continue  # same component — bridging would add a shortcut, not heal
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen.add(key)

        bridge_vec = coords[j] - coords[i]
        distance_m = float(np.hypot(*bridge_vec))
        # Each endpoint should continue roughly along the bridge (opposite signs).
        angle_u = _turn_angle_deg(directions[u], bridge_vec)
        angle_v = _turn_angle_deg(directions[v], -bridge_vec)
        worst = max(angle_u, angle_v)
        if worst > angle_max_deg:
            continue
        score = distance_m * (1.0 + angle_penalty_factor * (worst / 180.0))
        bridges.append(Bridge(u, v, distance_m, worst, score))

    return bridges


# --------------------------------------------------------------------------- #
# Heal
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class HealReport:
    """Before/after summary of a healing pass (drives the Connectivity Ratio)."""

    components_before: int
    components_after: int
    largest_cc_before: int
    largest_cc_after: int
    bridges_added: int

    @property
    def connectivity_ratio(self) -> float:
        """% increase in the largest connected component (``docs/Evaluation.md``)."""
        if self.largest_cc_before == 0:
            return 0.0
        return 100.0 * (self.largest_cc_after - self.largest_cc_before) / self.largest_cc_before


def heal_graph(
    graph: "nx.Graph",
    gap_max_m: float = 40.0,
    angle_max_deg: float = 60.0,
    angle_penalty_factor: float = 2.0,
) -> tuple["nx.Graph", HealReport]:
    """Bridge fragmented components into one routable graph (in place + report).

    Runs Kruskal's MST over the candidate bridges using a Union-Find: bridges
    are added cheapest-first, each only if it joins two still-separate
    components. Added edges are straight metric segments flagged
    ``is_bridged=True``; their endpoints are retyped ``bridged``.
    """
    import networkx as nx

    comps_before = list(nx.connected_components(graph))
    largest_before = max((len(c) for c in comps_before), default=0)

    bridges = find_candidate_bridges(
        graph, comps_before, gap_max_m, angle_max_deg, angle_penalty_factor
    )
    bridges.sort(key=lambda b: b.score)

    uf = UnionFind(list(graph.nodes))
    for comp in comps_before:  # seed UF with the existing components
        members = iter(comp)
        first = next(members)
        for other in members:
            uf.union(first, other)

    added = 0
    for b in bridges:
        if uf.union(b.u, b.v):  # only if it actually merges two components
            graph.add_edge(
                b.u,
                b.v,
                length_m=max(b.distance_m, 1e-6),  # weight must be > 0 (Schema)
                geometry=[
                    [graph.nodes[b.u]["x"], graph.nodes[b.u]["y"]],
                    [graph.nodes[b.v]["x"], graph.nodes[b.v]["y"]],
                ],
                is_bridged=True,
            )
            graph.nodes[b.u]["type"] = TYPE_BRIDGED
            graph.nodes[b.v]["type"] = TYPE_BRIDGED
            added += 1

    _annotate_degree_and_type_preserving_bridged(graph)

    comps_after = list(nx.connected_components(graph))
    largest_after = max((len(c) for c in comps_after), default=0)
    report = HealReport(
        components_before=len(comps_before),
        components_after=len(comps_after),
        largest_cc_before=largest_before,
        largest_cc_after=largest_after,
        bridges_added=added,
    )
    return graph, report


def _annotate_degree_and_type_preserving_bridged(graph: "nx.Graph") -> None:
    """Recompute degree/type but keep the ``bridged`` label on healed endpoints."""
    bridged = {n for n, d in graph.nodes(data=True) if d.get("type") == TYPE_BRIDGED}
    _annotate_degree_and_type(graph)
    for n in bridged:
        graph.nodes[n]["type"] = TYPE_BRIDGED
