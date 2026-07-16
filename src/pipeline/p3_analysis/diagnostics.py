"""A46 — structural graph diagnostics (promotion-protocol step 5).

APLS is one number. When a candidate's routing score moves, it alone cannot say
*why*: did the graph shatter into components, sprout isolated dust nodes, or
simply miss half the road length? The A46 promotion protocol therefore requires
**component / isolated-node / edge-length / reachability diagnostics reported
alongside APLS** (``docs/Evaluation.md`` → "Promotion protocol for A46", step 5).

That diagnosis has been done before, but by hand: the A18 verdict records "LoRA
averages 11.43 components vs 3.46 for GT, largest-component node fraction 0.376 vs
0.719, 4.26 isolated nodes, six empty graphs and roughly 48% of GT edge length …
more fragmented than GT on 114/127 chips", concluding that **missing
nodes/edges/connectivity — not background false positives — is the immediate
target**. This module makes exactly that diagnosis a reusable, tested function so
every A46 candidate is characterised the same way instead of re-derived ad hoc.

Report-only: nothing here gates or promotes. Pure CPU, NetworkX.
"""

from __future__ import annotations

import numpy as np


def graph_diagnostics(graph, weight: str = "length_m") -> dict:
    """Structural health of a single graph.

    Returns the fields protocol step 5 asks for: size, fragmentation
    (components + largest-component share), isolated nodes, edge-length
    distribution and reachability. ``is_empty`` flags a degenerate prediction —
    the A18 run had six of them, and they must not silently average away.
    """
    import networkx as nx

    from src.pipeline.p3_analysis.apls import _reachable_pair_fraction

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    if n_nodes == 0:
        return {
            "is_empty": True, "n_nodes": 0, "n_edges": 0, "n_components": 0,
            "largest_cc_node_fraction": 0.0, "n_isolated_nodes": 0,
            "isolated_node_fraction": 0.0, "total_length_m": 0.0,
            "median_edge_length_m": 0.0, "p95_edge_length_m": 0.0,
            "reachable_pair_fraction": 0.0,
        }

    components = list(nx.connected_components(graph))
    largest = max((len(c) for c in components), default=0)
    isolated = sum(1 for n in graph.nodes if graph.degree(n) == 0)
    lengths = np.array(
        [float(d.get(weight, 0.0)) for *_, d in graph.edges(data=True)], dtype=float
    ) if n_edges else np.zeros(0)

    return {
        "is_empty": False,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_components": len(components),
        "largest_cc_node_fraction": round(largest / n_nodes, 4),
        "n_isolated_nodes": isolated,
        "isolated_node_fraction": round(isolated / n_nodes, 4),
        "total_length_m": round(float(lengths.sum()), 2),
        "median_edge_length_m": round(float(np.median(lengths)), 2) if n_edges else 0.0,
        "p95_edge_length_m": round(float(np.percentile(lengths, 95)), 2) if n_edges else 0.0,
        "reachable_pair_fraction": round(_reachable_pair_fraction(graph), 4),
    }


def compare_to_gt(pred, gt, weight: str = "length_m") -> dict:
    """Diagnose a predicted graph against its ground truth.

    The A18-style fragmentation diagnosis, as a function: how much more broken is
    the prediction than the GT it is trying to reproduce, and how much of the road
    length did it find at all. ``length_fraction_of_gt`` separates "found the roads
    but broke them" from "never found the roads" — different fixes.
    """
    pred_d = graph_diagnostics(pred, weight)
    gt_d = graph_diagnostics(gt, weight)
    gt_length = gt_d["total_length_m"]
    return {
        "pred": pred_d,
        "gt": gt_d,
        "component_delta": pred_d["n_components"] - gt_d["n_components"],
        "more_fragmented_than_gt": pred_d["n_components"] > gt_d["n_components"],
        "length_fraction_of_gt": (round(pred_d["total_length_m"] / gt_length, 4)
                                  if gt_length > 0 else 0.0),
        "largest_cc_fraction_delta": round(
            pred_d["largest_cc_node_fraction"] - gt_d["largest_cc_node_fraction"], 4
        ),
    }


def aggregate(comparisons: dict[str, dict]) -> dict:
    """Fleet-level summary over ``{chip_id: compare_to_gt(...)}``.

    Reproduces the shape of the A18 verdict line — mean components (pred vs GT),
    mean largest-component share, mean isolated nodes, empty-graph count, mean
    length fraction, and the "more fragmented on N/M chips" tally — so a candidate
    can be characterised in one line without hand analysis.
    """
    if not comparisons:
        return {"n_chips": 0}

    rows = list(comparisons.values())

    def mean(path: str, key: str) -> float:
        return round(float(np.mean([r[path][key] for r in rows])), 4)

    n_more_fragmented = sum(1 for r in rows if r["more_fragmented_than_gt"])
    return {
        "n_chips": len(rows),
        "mean_components_pred": mean("pred", "n_components"),
        "mean_components_gt": mean("gt", "n_components"),
        "mean_largest_cc_fraction_pred": mean("pred", "largest_cc_node_fraction"),
        "mean_largest_cc_fraction_gt": mean("gt", "largest_cc_node_fraction"),
        "mean_isolated_nodes_pred": mean("pred", "n_isolated_nodes"),
        "mean_reachable_pair_fraction_pred": mean("pred", "reachable_pair_fraction"),
        "n_empty_pred": sum(1 for r in rows if r["pred"]["is_empty"]),
        "mean_length_fraction_of_gt": round(
            float(np.mean([r["length_fraction_of_gt"] for r in rows])), 4
        ),
        "more_fragmented_than_gt_chips": f"{n_more_fragmented}/{len(rows)}",
    }


def format_summary(summary: dict) -> str:
    """One-line-per-fact rendering of :func:`aggregate`, mirroring the A18 verdict."""
    if not summary.get("n_chips"):
        return "no chips diagnosed"
    return "\n".join([
        f"structural diagnosis over {summary['n_chips']} chips:",
        f"  components:      pred {summary['mean_components_pred']} vs GT "
        f"{summary['mean_components_gt']}",
        f"  largest-CC share: pred {summary['mean_largest_cc_fraction_pred']} vs GT "
        f"{summary['mean_largest_cc_fraction_gt']}",
        f"  isolated nodes:   {summary['mean_isolated_nodes_pred']} mean | "
        f"empty graphs: {summary['n_empty_pred']}",
        f"  road length found: {summary['mean_length_fraction_of_gt']:.0%} of GT",
        f"  reachable pairs:  {summary['mean_reachable_pair_fraction_pred']}",
        f"  more fragmented than GT on {summary['more_fragmented_than_gt_chips']} chips",
    ])
