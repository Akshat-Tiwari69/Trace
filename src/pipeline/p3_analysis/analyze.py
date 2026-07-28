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
from src.pipeline.p2_graph.graph_io import atomic_write, load_graphml, save_graph_pair
from src.pipeline.p3_analysis.criticality import (
    annotate_criticality,
    annotate_cut_structure,
    rank_table,
)
from src.pipeline.p3_analysis.resilience import (
    EFFICIENCY_SEED,
    RANDOM_REMOVAL_SEED,
    ablation_curve,
    efficiency_sampling_policy,
)


def _write_csv(rows: list[dict], path: Path) -> None:
    """Write a list of uniform dict rows to CSV atomically (header from row 0)."""

    def _write(tmp: Path) -> None:
        if not rows:
            tmp.write_text("")
            return
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    atomic_write(path, _write)


def analyze(
    cfg: GraphConfig,
    critical_fraction: float = 0.10,
    k: int | None = None,
    curve_steps: int = 25,
    efficiency_k: int | None = None,
    exact_efficiency: bool = False,
) -> dict:
    """Run criticality + resilience on the AOI's graph; write the P3 artifacts.

    ``k`` controls betweenness sampling. Global efficiency is exact through 256
    active nodes, then uses the disclosed deterministic source-sampling policy;
    ``efficiency_k`` overrides its sample size and ``exact_efficiency`` preserves
    an explicit offline-evaluation path.
    """
    graph = load_graphml(cfg.graphml_path)

    n = graph.number_of_nodes()
    if n < 2:
        raise ValueError(
            f"graph for {cfg.aoi} has {n} nodes — upstream P1/P2 failure, refusing to analyze"
        )

    bc = annotate_criticality(graph, k=k, critical_fraction=critical_fraction)
    cut = annotate_cut_structure(graph)  # articulation points + bridge edges (S8)
    rows = rank_table(graph, bc)

    criticality_path = cfg.processed_dir / f"{cfg.aoi}_criticality.csv"
    _write_csv(rows, criticality_path)

    # Resilience: targeted (high-betweenness first) vs. random — the sanity check
    # that betweenness finds genuinely critical nodes (docs/Evaluation.md). The
    # random curve is one seed-43 reference, not a distribution or "typical" case.
    steps = min(curve_steps, max(0, graph.number_of_nodes() - 1))
    policy = efficiency_sampling_policy(
        graph, k=efficiency_k, exact=exact_efficiency, seed=EFFICIENCY_SEED
    )
    targeted = ablation_curve(
        graph, "targeted", betweenness=bc, steps=steps,
        k=policy["k"], source_seed=policy["seed"],
    )
    random_curve = ablation_curve(
        graph, "random", steps=steps, k=policy["k"],
        seed=RANDOM_REMOVAL_SEED, source_seed=policy["seed"],
    )
    curve_rows = [
        {
            "n_removed": t.n_removed,
            "targeted_efficiency": round(t.efficiency, 8),
            "targeted_resilience_index": round(t.resilience_index, 6),
            "targeted_largest_cc_fraction": round(t.largest_cc_fraction, 6),
            "targeted_active_largest_cc_fraction": round(
                t.active_largest_cc_fraction, 6
            ),
            "random_efficiency": round(r.efficiency, 8),
            "random_resilience_index": round(r.resilience_index, 6),
            "random_largest_cc_fraction": round(r.largest_cc_fraction, 6),
            "random_active_largest_cc_fraction": round(
                r.active_largest_cc_fraction, 6
            ),
            "random_seed": RANDOM_REMOVAL_SEED,
            "efficiency_method": policy["method"],
            "efficiency_k": policy["k"],
            "efficiency_seed": policy["seed"],
        }
        for t, r in zip(targeted, random_curve)
    ]
    resilience_path = cfg.processed_dir / f"{cfg.aoi}_resilience.csv"
    _write_csv(curve_rows, resilience_path)

    # Refresh both canonical graph artifacts so P3 annotations are not present in
    # GeoJSON while silently absent from the lossless GraphML contract.
    save_graph_pair(graph, cfg.graphml_path, cfg.geojson_path)

    top = rows[0] if rows else {"node_id": None, "betweenness": 0.0}
    targeted_end = targeted[-1].resilience_index if targeted else 1.0
    random_end = random_curve[-1].resilience_index if random_curve else 1.0
    sampling_detail = policy["method"]
    if policy["k"]:
        sampling_detail += f" k={policy['k']} seed={policy['seed']}"
    print(
        f"[{cfg.aoi}] criticality: {len(rows)} nodes ranked | "
        f"top node {top['node_id']} (betweenness {top['betweenness']:.4f})\n"
        f"  cut structure: {cut['n_articulation']} articulation points, {cut['n_bridges']} bridge edges\n"
        f"  resilience after {steps} removals - "
        f"targeted RI {targeted_end:.3f} vs seeded random reference RI {random_end:.3f} "
        f"({'targeted degrades faster [ok]' if targeted_end <= random_end else 'check: random fell faster'})\n"
        f"  efficiency: {sampling_detail}\n"
        f"  -> {criticality_path}\n"
        f"  -> {resilience_path}"
    )
    return {
        "criticality_path": str(criticality_path),
        "resilience_path": str(resilience_path),
        "targeted_end_ri": targeted_end,
        "random_end_ri": random_end,
        "random_seed": RANDOM_REMOVAL_SEED,
        "efficiency_method": policy["method"],
        "efficiency_k": policy["k"],
        "efficiency_seed": policy["seed"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Criticality + resilience on a healed graph.")
    p.add_argument("--aoi", required=True, help="AOI id (matches the graph filename)")
    p.add_argument("--critical-fraction", type=float, default=0.10, help="top fraction flagged critical")
    p.add_argument("--k", type=int, default=None, help="k-sample betweenness (large graphs)")
    p.add_argument("--curve-steps", type=int, default=25, help="ablation curve length")
    p.add_argument("--efficiency-k", type=int, default=None,
                   help="k-sample global efficiency in ablation (large graphs)")
    p.add_argument("--exact-efficiency", action="store_true",
                   help="force exact global efficiency for offline evaluation")
    args = p.parse_args()

    cfg = GraphConfig(aoi=args.aoi)
    analyze(
        cfg,
        critical_fraction=args.critical_fraction,
        k=args.k,
        curve_steps=args.curve_steps,
        efficiency_k=args.efficiency_k,
        exact_efficiency=args.exact_efficiency,
    )


if __name__ == "__main__":
    main()
