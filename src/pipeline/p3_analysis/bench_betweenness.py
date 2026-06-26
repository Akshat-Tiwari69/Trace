"""S9 — benchmark k-sample vs exact betweenness on a large graph (CLI).

Betweenness is the heaviest CPU step for city-scale graphs (``docs/Research.md`` →
Infrastructure; ``RiskRegister.md`` T-3). This benchmarks the ``k``-sample
approximation against exact betweenness on a large synthetic road-like grid:
speedup and Spearman rank correlation (the ranking is what the dashboard's
criticality ordering needs). Re-runnable offline; writes a small JSON result.

Example::

    python -m src.pipeline.p3_analysis.bench_betweenness --side 50 --k 150
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline.p3_analysis.criticality import benchmark_betweenness


def _grid_graph(side: int):
    """A ``side``×``side`` grid road network with unit-metre edges (side² nodes)."""
    import networkx as nx

    graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
    for u, v in graph.edges:
        graph.edges[u, v]["length_m"] = 1.0
    return graph


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark k-sample vs exact betweenness.")
    p.add_argument("--side", type=int, default=50, help="grid side (nodes = side²; 50 → 2500)")
    p.add_argument("--k", type=int, default=150, help="k-sample source count")
    p.add_argument("--out", default="data/sample/betweenness_benchmark.json")
    args = p.parse_args()

    graph = _grid_graph(args.side)
    result = benchmark_betweenness(graph, k=args.k)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))

    print(
        f"\n=== betweenness benchmark ({result['n_nodes']} nodes, {result['n_edges']} edges) ===\n"
        f"exact {result['exact_s']}s vs k={args.k} {result['ksample_s']}s "
        f"-> {result['speedup']}x faster | Spearman rank corr {result['spearman']}\n"
        f"  (timings are machine-dependent; the speedup/correlation are the point)\n"
        f"  -> {args.out}"
    )


if __name__ == "__main__":
    main()
