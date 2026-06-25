"""CLI: criticality + resilience analysis on a healed graph (Phase III / S1·S2).

Consumes the P2 contract ``data/processed/{aoi}_graph.graphml`` and writes the P3
contract ``data/processed/{aoi}_criticality.csv`` (per-node
``node_id, betweenness, rank, is_critical`` + ``x, y``) plus a resilience
degradation curve ``{aoi}_resilience.csv`` (``docs/Tracker.md`` §4).

It also re-saves the graph's GeoJSON with betweenness baked onto the nodes, so
the dashboard's criticality heatmap reads straight from one file.

Example::

    python -m src.pipeline.p3_analysis.analyze --aoi panaji_demo
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.pipeline.p2_graph.config import GraphConfig
from src.pipeline.p2_graph.graph_io import load_graphml, save_geojson
from src.pipeline.p3_analysis.criticality import (
    annotate_criticality,
    annotate_cut_structure,
    rank_table,
)
from src.pipeline.p3_analysis.resilience import ablation_curve


def _write_csv(rows: list[dict], path: Path) -> None:
    """Write a list of uniform dict rows to CSV (header from the first row)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    cfg: GraphConfig,
    critical_fraction: float = 0.10,
    k: int | None = None,
    curve_steps: int = 25,
    efficiency_k: int | None = None,
) -> dict:
    """Run criticality + resilience on the AOI's graph; write the P3 artifacts.

    ``k`` k-samples betweenness; ``efficiency_k`` k-samples global efficiency in
    the ablation curves. Both default to exact (None) — set them only to keep the
    CLI responsive on very large AOIs.
    """
    graph = load_graphml(cfg.graphml_path)

    bc = annotate_criticality(graph, k=k, critical_fraction=critical_fraction)
    cut = annotate_cut_structure(graph)  # articulation points + bridge edges (S8)
    rows = rank_table(graph, bc)

    criticality_path = cfg.processed_dir / f"{cfg.aoi}_criticality.csv"
    _write_csv(rows, criticality_path)

    # Resilience: targeted (high-betweenness first) vs. random — the sanity check
    # that betweenness finds genuinely critical nodes (docs/Evaluation.md).
    steps = min(curve_steps, max(0, graph.number_of_nodes() - 1))
    targeted = ablation_curve(graph, "targeted", betweenness=bc, steps=steps, k=efficiency_k)
    random_curve = ablation_curve(graph, "random", steps=steps, k=efficiency_k)
    curve_rows = [
        {
            "n_removed": t.n_removed,
            "targeted_efficiency": round(t.efficiency, 8),
            "targeted_resilience_index": round(t.resilience_index, 6),
            "targeted_largest_cc_fraction": round(t.largest_cc_fraction, 6),
            "random_efficiency": round(r.efficiency, 8),
            "random_resilience_index": round(r.resilience_index, 6),
            "random_largest_cc_fraction": round(r.largest_cc_fraction, 6),
        }
        for t, r in zip(targeted, random_curve)
    ]
    resilience_path = cfg.processed_dir / f"{cfg.aoi}_resilience.csv"
    _write_csv(curve_rows, resilience_path)

    # Refresh the GeoJSON so nodes now carry betweenness/is_critical for the map.
    save_geojson(graph, cfg.geojson_path)

    top = rows[0] if rows else {"node_id": None, "betweenness": 0.0}
    targeted_end = targeted[-1].resilience_index if targeted else 1.0
    random_end = random_curve[-1].resilience_index if random_curve else 1.0
    print(
        f"[{cfg.aoi}] criticality: {len(rows)} nodes ranked | "
        f"top node {top['node_id']} (betweenness {top['betweenness']:.4f})\n"
        f"  cut structure: {cut['n_articulation']} articulation points, {cut['n_bridges']} bridge edges\n"
        f"  resilience after {steps} removals - "
        f"targeted RI {targeted_end:.3f} vs random RI {random_end:.3f} "
        f"({'targeted degrades faster [ok]' if targeted_end <= random_end else 'check: random fell faster'})\n"
        f"  -> {criticality_path}\n"
        f"  -> {resilience_path}"
    )
    return {
        "criticality_path": criticality_path,
        "resilience_path": resilience_path,
        "targeted_end_ri": targeted_end,
        "random_end_ri": random_end,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Criticality + resilience on a healed graph.")
    p.add_argument("--aoi", required=True, help="AOI id (matches the graph filename)")
    p.add_argument("--critical-fraction", type=float, default=0.10, help="top fraction flagged critical")
    p.add_argument("--k", type=int, default=None, help="k-sample betweenness (large graphs)")
    p.add_argument("--curve-steps", type=int, default=25, help="ablation curve length")
    p.add_argument("--efficiency-k", type=int, default=None,
                   help="k-sample global efficiency in ablation (large graphs)")
    args = p.parse_args()

    cfg = GraphConfig(aoi=args.aoi)
    analyze(
        cfg,
        critical_fraction=args.critical_fraction,
        k=args.k,
        curve_steps=args.curve_steps,
        efficiency_k=args.efficiency_k,
    )


if __name__ == "__main__":
    main()
