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


def compute_betweenness(
    graph: "nx.Graph",
    weight: str = "length_m",
    k: int | None = None,
    seed: int = 42,
) -> dict[int, float]:
    """Return ``{node_id: betweenness}`` normalised to ``[0, 1]``.

    ``k`` (if given and < node count) switches to k-sample approximate
    betweenness for speed on large graphs; ``seed`` keeps it reproducible.
    """
    import networkx as nx

    n = graph.number_of_nodes()
    use_k = k if (k is not None and k < n) else None
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


def rank_table(graph: "nx.Graph", bc: dict[int, float]) -> list[dict]:
    """Build the ranked per-node criticality rows for ``{aoi}_criticality.csv``.

    Columns match the §4 contract: ``node_id, betweenness, rank, is_critical``
    (plus ``x, y`` so the dashboard can place the ranked list on the map).
    """
    ranked = sorted(bc, key=lambda n: bc[n], reverse=True)
    rows = []
    for rank, node_id in enumerate(ranked, start=1):
        data = graph.nodes[node_id]
        rows.append(
            {
                "node_id": int(node_id),
                "betweenness": round(float(bc[node_id]), 6),
                "rank": rank,
                "is_critical": bool(data.get("is_critical", False)),
                "x": round(float(data["x"]), 7),
                "y": round(float(data["y"]), 7),
            }
        )
    return rows
