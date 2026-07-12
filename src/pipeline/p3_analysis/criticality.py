"""Criticality — betweenness centrality to find "Gatekeeper Nodes".

Betweenness counts how often a node sits on the shortest path between other
pairs: high betweenness ⇒ "everyone has to pass through here" ⇒ a chokepoint
(``docs/PRD.md`` G3, FR7). We weight paths by ``length_m`` so betweenness reflects
real travel distance, normalise to ``[0, 1]`` (``docs/Schema.md``), rank the nodes,
and flag the top fraction as critical.

For very large city graphs, exact betweenness is the heavy CPU step
(``docs/Research.md`` → Infrastructure); pass ``k`` to use NetworkX's k-sample
approximation (``docs/TRD.md`` performance, T-3 in ``docs/RiskRegister.md``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx


# Above this many nodes, exact all-pairs betweenness (O(V·E)) makes an
# interactive ablation click take tens of seconds; sample this many sources
# instead — benchmarked at Spearman ~0.98 vs exact on a 2,500-node grid (S9).
AUTO_K_NODE_THRESHOLD = 1000
AUTO_K_SAMPLES = 150


def auto_k(graph: "nx.Graph", k: int | None) -> int | None:
    """Resolve the betweenness sample size for the interactive path (bugs.md §4).

    Honours an explicit ``k`` if given; otherwise returns ``AUTO_K_SAMPLES`` once
    the graph exceeds ``AUTO_K_NODE_THRESHOLD`` (so city-scale graphs stay
    responsive) and ``None`` (exact) below it — small graphs like the committed
    demo are unaffected, so their numbers never drift.
    """
    if k is not None:
        return k
    if graph.number_of_nodes() > AUTO_K_NODE_THRESHOLD:
        return AUTO_K_SAMPLES
    return None


def compute_betweenness(
    graph: "nx.Graph",
    weight: str = "length_m",
    k: int | None = None,
    seed: int = 42,
) -> dict[int, float]:
    """Return ``{node_id: betweenness}`` normalised to ``[0, 1]``.

    ``k`` (if given and < node count) switches to k-sample approximate
    betweenness for speed on large graphs; ``k=None`` auto-selects sampling above
    ``AUTO_K_NODE_THRESHOLD`` nodes (see :func:`auto_k`). ``seed`` keeps it
    reproducible.
    """
    import networkx as nx

    n = graph.number_of_nodes()
    resolved = auto_k(graph, k)
    use_k = resolved if (resolved is not None and resolved < n) else None
    return nx.betweenness_centrality(
        graph, k=use_k, weight=weight, normalized=True, seed=seed
    )


def annotate_criticality(
    graph: "nx.Graph",
    weight: str = "length_m",
    k: int | None = None,
    critical_fraction: float = 0.10,
) -> dict[int, float]:
    """Compute betweenness and write ``betweenness, is_critical`` onto each node.

    The top ``critical_fraction`` of nodes by betweenness are flagged
    ``is_critical=True``. ``critical_fraction=0.0`` flags **none**; any positive
    fraction flags at least one (so "show me the critical nodes" never comes back
    empty on a small graph). Returns the betweenness dict for downstream ranking.
    """
    if not 0.0 <= critical_fraction <= 1.0:
        raise ValueError("critical_fraction must be in [0, 1]")

    bc = compute_betweenness(graph, weight=weight, k=k)
    ranked = sorted(bc, key=lambda n: bc[n], reverse=True)
    if critical_fraction == 0.0 or not ranked:
        n_critical = 0
    else:
        n_critical = max(1, int(round(len(ranked) * critical_fraction)))
    critical = set(ranked[:n_critical])

    for node_id, score in bc.items():
        graph.nodes[node_id]["betweenness"] = float(score)
        graph.nodes[node_id]["is_critical"] = node_id in critical
        graph.nodes[node_id]["is_disabled"] = False
    return bc


def annotate_cut_structure(graph: "nx.Graph") -> dict:
    """Flag **articulation points** (nodes) and **bridge edges** — a second,
    structural criticality layer alongside betweenness (``docs/Tracker.md`` S8).

    An *articulation point* is a node whose removal disconnects the network; a
    *bridge* is an edge whose removal does. These are the network's hard single
    points of failure — distinct from "high traffic share" (betweenness): a node
    can carry lots of traffic yet not be a cut node, and vice-versa. Sets
    ``is_articulation`` on every node and ``is_bridge`` on every edge (in place),
    and returns the counts.

    Note: ``is_bridge`` (graph-theoretic cut edge) is **not** ``is_bridged`` (an
    edge the healing step inferred) — different concepts, deliberately named
    distinctly. Works on disconnected graphs (NetworkX handles each component).

    On a MultiGraph, ``nx.bridges`` returns ``(u, v, key)`` triples — a parallel
    edge is only a bridge if its *individual* removal disconnects the graph
    (rare: both endpoints of a parallel pair are still connected via the sibling
    edge, so parallel edges are almost never bridges, which is correct).
    """
    import networkx as nx

    articulation = set(nx.articulation_points(graph))
    # nx.bridges yields (u, v) on both Graph and MultiGraph (the key is omitted
    # even for multigraphs in networkx 3.x — a parallel edge is never a bridge
    # because its sibling preserves connectivity). Compare by the unordered pair.
    raw_bridges = list(nx.bridges(graph))
    bridge_pairs = {frozenset((u, v)) for u, v in raw_bridges}

    for node_id in graph.nodes:
        graph.nodes[node_id]["is_articulation"] = node_id in articulation
    multi = graph.is_multigraph()
    if multi:
        for u, v, k in graph.edges(keys=True):
            graph.edges[u, v, k]["is_bridge"] = frozenset((u, v)) in bridge_pairs
    else:
        for u, v in graph.edges:
            graph.edges[u, v]["is_bridge"] = frozenset((u, v)) in bridge_pairs

    return {"n_articulation": len(articulation), "n_bridges": len(raw_bridges)}


def rank_table(graph: "nx.Graph", bc: dict[int, float]) -> list[dict]:
    """Build the ranked per-node criticality rows for ``{aoi}_criticality.csv``.

    Columns match the §4 contract: ``node_id, betweenness, rank, is_critical``
    (plus ``is_articulation`` from :func:`annotate_cut_structure`, and ``x, y`` so
    the dashboard can place the ranked list on the map).

    Note ``rank`` orders by betweenness — **traffic importance**, not failure
    importance: an articulation point with modest through-traffic can rank low
    while still being a hard single point of failure (see ``is_articulation``).

    Requires :func:`annotate_criticality` to have run first (fails loudly on
    un-annotated nodes rather than silently reporting ``is_critical=False``).
    """
    ranked = sorted(bc, key=lambda n: bc[n], reverse=True)
    rows = []
    for rank, node_id in enumerate(ranked, start=1):
        data = graph.nodes[node_id]
        if "betweenness" not in data:
            raise ValueError(
                f"node {node_id} lacks 'betweenness' — run annotate_criticality() "
                "before rank_table() (ordering bug in the caller)"
            )
        rows.append(
            {
                "node_id": int(node_id),
                "betweenness": round(float(bc[node_id]), 6),
                "rank": rank,
                "is_critical": bool(data.get("is_critical", False)),
                "is_articulation": bool(data.get("is_articulation", False)),
                "x": round(float(data["x"]), 7),
                "y": round(float(data["y"]), 7),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# S9 — caching + k-sample approximation for large graphs
# --------------------------------------------------------------------------- #
def _graph_fingerprint(graph: "nx.Graph", weight: str) -> int:
    """Cheap structural hash of the graph (node count + sorted weighted edges).

    On a MultiGraph the edge key is folded in so two graphs that differ only by
    a parallel branch hash differently (otherwise the cache would silently serve
    a simple-graph betweenness for a multi-edge graph).
    """
    multi = graph.is_multigraph()
    if multi:
        edges = tuple(sorted(
            (min(u, v), max(u, v), k, round(float(d.get(weight, 0.0)), 3))
            for u, v, k, d in graph.edges(data=True, keys=True)
        ))
    else:
        edges = tuple(sorted(
            (min(u, v), max(u, v), round(float(d.get(weight, 0.0)), 3))
            for u, v, d in graph.edges(data=True)
        ))
    return hash((graph.number_of_nodes(), edges))


class BetweennessCache:
    """Memoize betweenness by a structural fingerprint of the graph + params.

    The dashboard computes baseline betweenness **once** and reads it across many
    interactions (``docs/TRD.md`` performance: "precompute betweenness once"). A
    node-ablation click operates on a *perturbed* graph whose fingerprint differs,
    so it recomputes — but every read of the unchanged baseline is a free cache
    hit. Recompute only when the graph (or weight/k/seed) actually changes.

    Bounded (LRU, ``maxsize``) so a long session of distinct ablations can't grow
    the store without limit — the baseline plus a working set of recent
    perturbations is all that's ever hot (bugs.md §4).
    """

    def __init__(self, maxsize: int = 64) -> None:
        from collections import OrderedDict

        self._store: "OrderedDict" = OrderedDict()
        self._maxsize = maxsize

    def get(self, graph: "nx.Graph", weight: str = "length_m",
            k: int | None = None, seed: int = 42) -> dict[int, float]:
        key = (_graph_fingerprint(graph, weight), weight, k, seed)
        if key in self._store:
            self._store.move_to_end(key)  # mark most-recently-used
            return self._store[key]
        value = compute_betweenness(graph, weight=weight, k=k, seed=seed)
        self._store[key] = value
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)  # evict least-recently-used
        return value

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


def benchmark_betweenness(
    graph: "nx.Graph", k: int, weight: str = "length_m", seed: int = 42,
) -> dict:
    """Compare k-sample vs exact betweenness: speedup + Spearman rank correlation.

    The k-sample estimate (``docs/RiskRegister.md`` T-3) is only useful if it's
    both *faster* and *ranks nodes the same way* — what the dashboard's criticality
    ordering depends on. Returns timings, the speedup, and the Spearman rank
    correlation of the two betweenness vectors.
    """
    import time

    from scipy.stats import spearmanr

    t = time.perf_counter()
    # Bypass compute_betweenness's interactive auto-sampling: a benchmark labelled
    # "exact" must call NetworkX exact mode even above AUTO_K_NODE_THRESHOLD.
    import networkx as nx
    exact = nx.betweenness_centrality(
        graph, k=None, weight=weight, normalized=True, seed=seed)
    exact_s = time.perf_counter() - t

    t = time.perf_counter()
    approx = compute_betweenness(graph, weight=weight, k=k, seed=seed)
    ksample_s = time.perf_counter() - t

    nodes = list(graph.nodes)
    rho = float(spearmanr([exact[n] for n in nodes], [approx[n] for n in nodes]).statistic)
    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "k": k,
        "exact_s": round(exact_s, 3),
        "ksample_s": round(ksample_s, 3),
        "speedup": round(exact_s / ksample_s, 1) if ksample_s > 0 else float("inf"),
        "spearman": round(rho, 3),
    }
