"""S11 — flood / elevation failure scenario + comparison.

The targeted/random ablation (``resilience.py``) asks "what if we lose the most
*central* junctions?" A flood asks a different, spatially-correlated question:
"what if we lose a whole low-lying *region* at once?" This disables the nodes
inside a flood polygon (or below an elevation threshold) and traces the
global-efficiency Resilience Index as the flood spreads, then compares that
degradation against **targeted** (betweenness-first) and **random** removal of
the same number of nodes (``docs/PRD.md`` future-enhancement #4; E4).

A flood removes a *connected cluster*, so it tends to fragment the network harder
than random loss — the three-way curve makes that visible. Polygon-based (needs
no DEM); elevation-based works when nodes carry an elevation attribute. Pure CPU,
shapely + NetworkX.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.pipeline.p2_graph.graph_io import load_geojson_graph
from src.pipeline.p3_analysis.criticality import compute_betweenness
from src.pipeline.p3_analysis.resilience import ablation_curve


def nodes_in_polygon(graph, polygon: list) -> list[int]:
    """Nodes whose ``(x, y)`` falls inside ``polygon`` (a list of ``[lon, lat]``)."""
    from shapely.geometry import Point, Polygon

    poly = Polygon(polygon)
    return [n for n, d in graph.nodes(data=True)
            if poly.covers(Point(d["x"], d["y"]))]


def nodes_below_elevation(graph, threshold: float, attr: str = "elevation") -> list[int]:
    """Nodes whose ``attr`` is below ``threshold`` (for DEM-driven flooding)."""
    return [n for n, d in graph.nodes(data=True)
            if attr in d and float(d[attr]) < threshold]


def flood_order(graph, flooded: list[int], attr: str = "elevation") -> list[int]:
    """Order flooded nodes by flood severity — lowest elevation first, else by
    proximity to the flood centroid (the deepest/most-central area drowns first)."""
    if flooded and all(attr in graph.nodes[n] for n in flooded):
        return sorted(flooded, key=lambda n: float(graph.nodes[n][attr]))
    if not flooded:
        return []
    pts = np.array([[graph.nodes[n]["x"], graph.nodes[n]["y"]] for n in flooded])
    centre = pts.mean(axis=0)
    return sorted(flooded, key=lambda n: (graph.nodes[n]["x"] - centre[0]) ** 2
                  + (graph.nodes[n]["y"] - centre[1]) ** 2)


def flood_comparison(graph, flooded: list[int], weight: str = "length_m",
                     seed: int = 42, bc: dict | None = None) -> dict:
    """Three-way RI curves over ``len(flooded)`` removals: flood vs targeted vs random.

    Returns the curves plus a summary (final RI of each, and the damage ranking).
    """
    order = flood_order(graph, flooded)
    steps = len(order)
    if bc is None:
        bc = compute_betweenness(graph, weight=weight)

    flood = ablation_curve(graph, sequence=order, steps=steps, weight=weight)
    targeted = ablation_curve(graph, "targeted", betweenness=bc, steps=steps, weight=weight)
    random_curve = ablation_curve(graph, "random", steps=steps, weight=weight, seed=seed)

    ends = {"flood": flood[-1].resilience_index,
            "targeted": targeted[-1].resilience_index,
            "random": random_curve[-1].resilience_index}
    ranked = sorted(ends, key=ends.get)  # most damaging (lowest RI) first
    return {
        "n_flooded": steps,
        "curves": {"flood": flood, "targeted": targeted, "random": random_curve},
        "end_ri": {k: round(v, 4) for k, v in ends.items()},
        "damage_ranking": ranked,  # most → least damaging
    }


def _box(graph, cx: float, cy: float, frac: float) -> list:
    """A rectangle of half-extent ``frac/2`` of the node bbox, centred at (cx, cy)."""
    xs = [d["x"] for _, d in graph.nodes(data=True)]
    ys = [d["y"] for _, d in graph.nodes(data=True)]
    hw, hh = (max(xs) - min(xs)) * frac / 2, (max(ys) - min(ys)) * frac / 2
    return [[cx - hw, cy - hh], [cx + hw, cy - hh], [cx + hw, cy + hh], [cx - hw, cy + hh]]


def _polygon_around_node(graph, node: int, frac: float = 0.4) -> list:
    """A flood box centred on ``node`` — models a flood inundating the area around
    a (typically low-lying, critical) junction, the worst-case a planner cares about."""
    return _box(graph, graph.nodes[node]["x"], graph.nodes[node]["y"], frac)


def _default_polygon(graph, frac: float = 0.45) -> list:
    """A central rectangle covering the middle ``frac`` of the node bbox (a redundant
    inland area — the survivable-flood control case)."""
    xs = [d["x"] for _, d in graph.nodes(data=True)]
    ys = [d["y"] for _, d in graph.nodes(data=True)]
    return _box(graph, (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, frac)


def _plot(result: dict, aoi: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {"flood": ("#2a9d8f", "Flood (spatial cluster)"),
              "targeted": ("#d1495b", "Targeted (betweenness)"),
              "random": ("#30638e", "Random")}
    for key, (colour, label) in styles.items():
        curve = result["curves"][key]
        ax.plot([p.n_removed for p in curve], [p.resilience_index for p in curve],
                marker="o", ms=3, lw=1.8, color=colour, label=label)
    ax.set_xlabel("Junctions removed")
    ax.set_ylabel("Resilience Index (global efficiency ratio)")
    ax.set_title(f"Flood vs targeted vs random failure — {aoi}")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(aoi: str, sample_dir: Path = Path("data/sample"),
        processed_dir: Path = Path("data/processed"), polygon: list | None = None,
        central: bool = False) -> dict:
    """Run the flood scenario for the AOI's sample graph; write the RI curve CSV.

    Default floods the area **around the top chokepoint** (the planning worst-case);
    ``central=True`` floods the redundant inland centre (survivable control case).
    """
    graph = load_geojson_graph(sample_dir / f"{aoi}_graph.geojson")
    bc = compute_betweenness(graph)
    if polygon is None:
        polygon = (_default_polygon(graph) if central
                   else _polygon_around_node(graph, max(bc, key=bc.get)))
    flooded = nodes_in_polygon(graph, polygon)
    if not flooded:
        raise SystemExit("flood polygon contains no nodes — widen it")

    result = flood_comparison(graph, flooded, bc=bc)

    processed_dir.mkdir(parents=True, exist_ok=True)
    csv_path = processed_dir / f"{aoi}_flood_resilience.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_removed", "flood_ri", "targeted_ri", "random_ri"])
        c = result["curves"]
        for i in range(len(c["flood"])):
            w.writerow([i, round(c["flood"][i].resilience_index, 6),
                        round(c["targeted"][i].resilience_index, 6),
                        round(c["random"][i].resilience_index, 6)])

    _plot(result, aoi, sample_dir / f"{aoi}_flood_curve.png")

    ends = result["end_ri"]
    region = "redundant inland centre" if central else "around the top chokepoint"
    print(
        f"\n=== Flood scenario — {aoi} ===\n"
        f"flooded {result['n_flooded']} junctions ({region})\n"
        f"end RI: flood {ends['flood']:.3f} | targeted {ends['targeted']:.3f} "
        f"| random {ends['random']:.3f}\n"
        f"damage ranking (worst first): {' > '.join(result['damage_ranking'])}\n"
        f"  -> {csv_path}\n"
        f"  -> {sample_dir / f'{aoi}_flood_curve.png'}"
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Flood/elevation failure scenario + comparison.")
    p.add_argument("--aoi", default="panaji_demo", help="AOI id")
    p.add_argument("--sample-dir", default="data/sample")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--central", action="store_true",
                   help="flood the redundant inland centre instead of the top chokepoint")
    args = p.parse_args()
    run(args.aoi, Path(args.sample_dir), Path(args.processed_dir), central=args.central)


if __name__ == "__main__":
    main()
