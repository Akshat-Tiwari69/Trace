"""S10 — demand-weighted (percolation) centrality vs plain betweenness.

Plain betweenness treats every origin–destination pair as equally likely — pure
topology. But a junction's *real* importance also depends on **where people
actually travel from/to**. **Percolation centrality** (Piraveenan et al. 2013,
``nx.percolation_centrality``) generalises betweenness by weighting each shortest
path by the "percolation states" (demand/population) of its endpoints — so a
junction serving a populous district scores higher even if its raw topological
betweenness is only moderate.

Without a census/WorldPop layer we use a **synthetic demand**: a spatial Gaussian
concentrated at a chosen corner (a stand-in population centre), or a degree proxy.
The point is the *comparison* — how much demand-weighting reshuffles the
criticality ranking vs plain betweenness on the sample (``docs/Tracker.md`` S10).
Pure CPU, NetworkX + numpy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.pipeline.p2_graph.graph_io import load_geojson_graph
from src.pipeline.p3_analysis.criticality import compute_betweenness


def degree_demand(graph) -> dict[int, float]:
    """Demand proxy = normalised node degree (busier junctions ≈ more demand)."""
    deg = dict(graph.degree())
    mx = max(deg.values(), default=1) or 1
    return {n: deg[n] / mx for n in graph.nodes}


def spatial_demand(graph, corner: str = "se", sigma_frac: float = 0.33) -> dict[int, float]:
    """A Gaussian demand bump centred at a bbox ``corner`` — a synthetic population
    centre. ``sigma_frac`` is the falloff as a fraction of the AOI extent."""
    xs = [d["x"] for _, d in graph.nodes(data=True)]
    ys = [d["y"] for _, d in graph.nodes(data=True)]
    corners = {
        "se": (max(xs), min(ys)), "sw": (min(xs), min(ys)),
        "ne": (max(xs), max(ys)), "nw": (min(xs), max(ys)),
        "centre": ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2),
    }
    px, py = corners[corner]
    sigma = sigma_frac * max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    return {n: math.exp(-(((d["x"] - px) ** 2 + (d["y"] - py) ** 2) / (2 * sigma * sigma)))
            for n, d in graph.nodes(data=True)}


def percolation_centrality(graph, states: dict[int, float], weight: str = "length_m") -> dict[int, float]:
    """Demand-weighted betweenness via ``nx.percolation_centrality`` (states = demand)."""
    import networkx as nx

    return nx.percolation_centrality(graph, states=states, weight=weight)


def compare_centralities(graph, states: dict[int, float], weight: str = "length_m",
                         top_n: int = 10) -> dict:
    """Compare plain betweenness vs demand-weighted percolation centrality.

    Returns the Spearman rank correlation, the top-N overlap, and the two top-N
    lists — quantifying how much demand-weighting reshuffles the criticality order.
    """
    from scipy.stats import spearmanr

    bc = compute_betweenness(graph, weight=weight)
    pc = percolation_centrality(graph, states, weight=weight)
    nodes = list(graph.nodes)
    rho = float(spearmanr([bc[n] for n in nodes], [pc[n] for n in nodes]).statistic)

    top_bc = sorted(bc, key=bc.get, reverse=True)[:top_n]
    top_pc = sorted(pc, key=pc.get, reverse=True)[:top_n]
    overlap = len(set(top_bc) & set(top_pc)) / top_n if top_n else 1.0
    promoted = [int(n) for n in top_pc if n not in top_bc]  # rose into top-N under demand
    return {
        "spearman": round(rho, 4),
        "top_n": top_n,
        "top_n_overlap": round(overlap, 3),
        "top_betweenness": [int(n) for n in top_bc],
        "top_percolation": top_pc and [int(n) for n in top_pc],
        "promoted_by_demand": promoted,
    }


def run(aoi: str, sample_dir: Path = Path("data/sample"),
        corner: str = "se", top_n: int = 10) -> dict:
    """Compute demand-weighted vs plain criticality on the AOI sample; write a report."""
    graph = load_geojson_graph(sample_dir / f"{aoi}_graph.geojson")
    states = spatial_demand(graph, corner=corner)
    result = compare_centralities(graph, states, top_n=top_n)
    result.update({"aoi": aoi, "demand": f"spatial Gaussian @ {corner} corner"})

    out = sample_dir / f"{aoi}_percolation.json"
    out.write_text(json.dumps(result, indent=2))
    print(
        f"\n=== Demand-weighted criticality — {aoi} ===\n"
        f"demand: synthetic population @ {corner} corner\n"
        f"Spearman(betweenness, percolation) = {result['spearman']:.3f} | "
        f"top-{top_n} overlap {result['top_n_overlap']:.0%}\n"
        f"  top betweenness: {result['top_betweenness'][:5]}\n"
        f"  top percolation: {result['top_percolation'][:5]}\n"
        f"  promoted by demand (new in top-{top_n}): {result['promoted_by_demand']}\n"
        f"  -> {out}"
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Demand-weighted percolation vs betweenness.")
    p.add_argument("--aoi", default="panaji_demo")
    p.add_argument("--sample-dir", default="data/sample")
    p.add_argument("--corner", default="se", choices=["se", "sw", "ne", "nw", "centre"],
                   help="synthetic population centre")
    p.add_argument("--top-n", type=int, default=10)
    args = p.parse_args()
    run(args.aoi, Path(args.sample_dir), corner=args.corner, top_n=args.top_n)


if __name__ == "__main__":
    main()
