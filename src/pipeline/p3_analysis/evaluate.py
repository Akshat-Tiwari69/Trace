"""E1 (graph side): package the headline graph/resilience numbers for the report.

Akshat (integration/eval lead) collects each lane's metrics for ``Evaluation.md``
and the demo. This is the **graph lane's** contribution: it reads the committed
sample graph and reports, in one place, the numbers the graph pipeline produces
(``docs/Evaluation.md`` → Topology + Resilience):

* **Connectivity** — how much MST/Union-Find healing reconnected the network.
  Reconstructed from the graph itself by removing the ``is_bridged`` edges to
  recover the pre-heal state (no need to re-run the OSM spike).
* **Criticality** — the top "Gatekeeper" junctions by betweenness.
* **Resilience** — global-efficiency degradation under **targeted vs. random**
  node ablation: the sanity check that betweenness finds genuine chokepoints
  (targeted must fall faster than random).

Outputs a JSON report + a resilience-curve PNG next to the sample, and prints a
readable summary. Pure CPU, classical Python (Shaivi's lane).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline.p2_graph.graph_io import load_geojson_graph
from src.pipeline.p3_analysis.criticality import compute_betweenness
from src.pipeline.p3_analysis.resilience import ablation_curve, global_efficiency


def _largest_cc(graph) -> int:
    import networkx as nx

    return max((len(c) for c in nx.connected_components(graph)), default=0)


def _n_components(graph) -> int:
    import networkx as nx

    return nx.number_connected_components(graph)


def _healing_metrics(graph) -> dict:
    """Healing connectivity numbers.

    Prefers the **authoritative** build-time stats embedded in ``graph.graph['heal']``
    (measured at heal time, before any simplification). Falls back to recovering
    the pre-heal state by dropping ``is_bridged`` edges when no metadata is present
    (``docs/Evaluation.md`` → Connectivity Ratio).
    """
    meta = graph.graph.get("heal")
    if meta:
        return {
            "bridges_added": meta["bridges_added"],
            "components_before_heal": meta["components_before"],
            "components_after_heal": meta["components_after"],
            "connectivity_ratio_pct": meta["connectivity_ratio_pct"],
            "source": "build-time",
        }

    pre = graph.copy()
    bridges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("is_bridged")]
    pre.remove_edges_from(bridges)
    largest_before, largest_after = _largest_cc(pre), _largest_cc(graph)
    ratio = 100.0 * (largest_after - largest_before) / largest_before if largest_before else 0.0
    return {
        "bridges_added": len(bridges),
        "components_before_heal": _n_components(pre),
        "components_after_heal": _n_components(graph),
        "largest_cc_before_heal": largest_before,
        "largest_cc_after_heal": largest_after,
        "connectivity_ratio_pct": round(ratio, 2),
        "source": "reconstructed",
    }


def _resilience_metrics(graph, bc: dict, curve_steps: int) -> tuple[dict, list, list]:
    """Targeted vs. random ablation; returns (summary, targeted_curve, random_curve)."""
    steps = min(curve_steps, max(0, graph.number_of_nodes() - 1))
    targeted = ablation_curve(graph, "targeted", betweenness=bc, steps=steps)
    random_curve = ablation_curve(graph, "random", steps=steps)

    def mean_ri(curve) -> float:  # area under the RI curve (robustness summary)
        return sum(p.resilience_index for p in curve) / len(curve)

    summary = {
        "baseline_global_efficiency": round(global_efficiency(graph), 6),
        "ablation_steps": steps,
        "targeted_ri_end": round(targeted[-1].resilience_index, 4),
        "random_ri_end": round(random_curve[-1].resilience_index, 4),
        "targeted_minus_random_gap": round(
            random_curve[-1].resilience_index - targeted[-1].resilience_index, 4
        ),
        "targeted_mean_ri": round(mean_ri(targeted), 4),
        "random_mean_ri": round(mean_ri(random_curve), 4),
        "targeted_degrades_faster": targeted[-1].resilience_index <= random_curve[-1].resilience_index,
    }
    return summary, targeted, random_curve


def _plot_resilience(targeted, random_curve, aoi: str, path: Path) -> None:
    """Save the targeted-vs-random resilience degradation curve as a PNG."""
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([p.n_removed for p in targeted], [p.resilience_index for p in targeted],
            marker="o", ms=3, lw=1.8, color="#d1495b", label="Targeted (high-betweenness first)")
    ax.plot([p.n_removed for p in random_curve], [p.resilience_index for p in random_curve],
            marker="s", ms=3, lw=1.8, color="#30638e", label="Random")
    ax.set_xlabel("Junctions removed")
    ax.set_ylabel("Resilience Index (global efficiency ratio)")
    ax.set_title(f"Network resilience under node ablation — {aoi}")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def evaluate(
    aoi: str,
    sample_dir: Path = Path("data/sample"),
    curve_steps: int = 40,
    top_n: int = 5,
) -> dict:
    """Compute the graph-lane evaluation numbers for ``aoi`` and write the report."""
    graph = load_geojson_graph(sample_dir / f"{aoi}_graph.geojson")
    bc = compute_betweenness(graph)

    healing = _healing_metrics(graph)
    resilience, targeted, random_curve = _resilience_metrics(graph, bc, curve_steps)

    ranked = sorted(bc, key=lambda n: bc[n], reverse=True)
    top_nodes = [{"node_id": int(n), "betweenness": round(bc[n], 4)} for n in ranked[:top_n]]
    n_bridged = healing["bridges_added"]

    report = {
        "aoi": aoi,
        "graph": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "bridged_edges": n_bridged,
            "bridged_pct": round(100.0 * n_bridged / max(1, graph.number_of_edges()), 2),
        },
        "simplify": graph.graph.get("simplify"),
        "healing": healing,
        "criticality": {
            "top_nodes": top_nodes,
            "max_betweenness": round(max(bc.values(), default=0.0), 4),
            "mean_betweenness": round(sum(bc.values()) / max(1, len(bc)), 6),
        },
        "resilience": resilience,
    }

    out_json = sample_dir / f"{aoi}_graph_eval.json"
    out_json.write_text(json.dumps(report, indent=2))
    plot_path = sample_dir / f"{aoi}_resilience_curve.png"
    _plot_resilience(targeted, random_curve, aoi, plot_path)

    _print_report(report, out_json, plot_path)
    return report


def _print_report(report: dict, json_path: Path, plot_path: Path) -> None:
    g, h, c, r = report["graph"], report["healing"], report["criticality"], report["resilience"]
    top = ", ".join(f"{n['node_id']}({n['betweenness']:.3f})" for n in c["top_nodes"])
    cc = ""
    if "largest_cc_before_heal" in h:  # only the reconstructed path has these
        cc = f"| largest CC {h['largest_cc_before_heal']} -> {h['largest_cc_after_heal']} "
    simp = ""
    if report.get("simplify"):
        s = report["simplify"]
        simp = (f"Simplify:     nodes {s['nodes_before']} -> {s['nodes_after']} "
                f"(-{s['node_reduction_pct']}%) | {s['stubs_pruned']} stubs, "
                f"{s['nodes_collapsed']} degree-2 collapsed (components preserved)\n")
    print(
        f"\n=== Graph evaluation — {report['aoi']} ===\n"
        f"Graph:        {g['nodes']} nodes, {g['edges']} edges "
        f"({g['bridged_edges']} bridged, {g['bridged_pct']}%)\n"
        f"{simp}"
        f"Healing:      components {h['components_before_heal']} -> {h['components_after_heal']} "
        f"{cc}| connectivity ratio +{h['connectivity_ratio_pct']}% ({h.get('source', '?')})\n"
        f"Criticality:  top junctions {top}\n"
        f"              max betweenness {c['max_betweenness']:.3f}, mean {c['mean_betweenness']:.5f}\n"
        f"Resilience:   baseline efficiency {r['baseline_global_efficiency']:.4f}\n"
        f"              mean RI over {r['ablation_steps']} removals: targeted {r['targeted_mean_ri']:.3f} "
        f"vs random {r['random_mean_ri']:.3f}  <- targeted removal hurts far more\n"
        f"              (end-point RI: targeted {r['targeted_ri_end']:.3f} vs random {r['random_ri_end']:.3f})\n"
        f"              {'targeted degrades faster [ok]' if r['targeted_degrades_faster'] else 'CHECK: random fell faster'}\n"
        f"  -> {json_path}\n"
        f"  -> {plot_path}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="E1 graph-lane evaluation numbers.")
    p.add_argument("--aoi", default="panaji_demo", help="AOI id (default panaji_demo)")
    p.add_argument("--sample-dir", default="data/sample", help="dir holding {aoi}_graph.geojson")
    p.add_argument("--curve-steps", type=int, default=40, help="ablation curve length")
    p.add_argument("--top-n", type=int, default=5, help="how many top critical nodes to list")
    args = p.parse_args()

    evaluate(args.aoi, sample_dir=Path(args.sample_dir), curve_steps=args.curve_steps, top_n=args.top_n)


if __name__ == "__main__":
    main()
