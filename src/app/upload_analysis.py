"""Close the upload loop: a road mask → routable graph → resilience analysis.

The dashboard's upload flow previously dead-ended at a mask preview (bugs.md
§2B): the extracted mask never became a graph, never got a resilience score.
This module runs the existing **CPU** P2→P3 pipeline in-process on an uploaded
mask and returns the same analysis the demo AOI ships with — turning the app
from "viewer of one canned city" into "analyse your own imagery".

Kept free of Streamlit imports so it is unit-testable without a UI runtime; the
app layer handles rendering. Coordinates stay in **image space** (pixels scaled
by ``resolution_m``): an uploaded neighbourhood capture is not georeferenced, so
the honest presentation is an overlay on the image itself, not a fake world-map
placement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading

import networkx as nx
import numpy as np
import pandas as pd

# One heavy CPU analysis (skeletonize -> heal -> simplify -> betweenness) at a
# time: this runs in-process on a 4-core ARM box (bugs.md §5H), and a single
# pass already uses multiple cores itself, so N concurrent uploads would
# starve every other session rather than just queue behind one job. Mirrors
# app.py's `_modal_semaphore` GPU-concurrency guard (bugs.md §5C) but tighter
# (1 vs 2) since this is the CPU every other Streamlit session also depends
# on. `acquire(blocking=False)` so a saturated box surfaces an explicit
# "busy, retry" message instead of silently queuing behind the scenes.
# A full filesystem job queue (bugs.md §5H, M-L) stays deferred; this
# semaphore is the interim guard.
_ANALYSIS_MAX_INFLIGHT = 1
_analysis_semaphore = threading.BoundedSemaphore(_ANALYSIS_MAX_INFLIGHT)

# Reject masks larger than this before the CPU pipeline runs — same ceiling as
# app.py's `Image.MAX_IMAGE_PIXELS` decompression-bomb guard (4096x4096), kept
# in sync so an upload that passed the earlier PIL check doesn't blow up the
# skeletonize/betweenness pass instead.
MAX_MASK_PIXELS = 4096 * 4096


class AnalysisBusyError(RuntimeError):
    """Raised when another upload analysis is already running on this box."""


@dataclass
class AnalysisResult:
    """Everything the dashboard needs to present an uploaded network's analysis."""

    graph: nx.MultiGraph  # skeleton_to_graph builds a MultiGraph (parallel branches, A37)
    criticality: pd.DataFrame            # node_id, betweenness, rank, is_critical, is_articulation, x, y
    resilience_index: float              # RI after losing the #1 chokepoint (worst single failure)
    top_node: int | None                 # the #1 critical junction (None for a trivial graph)
    n_nodes: int
    n_edges: int
    resolution_m: float
    summary: dict = field(default_factory=dict)


def analyze_mask(
    mask01: np.ndarray,
    resolution_m: float = 0.5,
    critical_fraction: float = 0.10,
) -> AnalysisResult:
    """Run mask → skeleton → heal → simplify → criticality → resilience, in memory.

    ``resolution_m`` scales pixel geometry to metres (uploads have no georeference,
    so ``length_m`` is only as accurate as this estimate — surfaced to the user).
    Raises ``ValueError`` when the mask yields a degenerate graph (<2 junctions),
    i.e. the segmentation found essentially no roads — the same fail-loud contract
    the batch pipeline uses, so a blank result is never silently presented; also
    raised when the mask is over the pixel-count ceiling (checked before any
    work runs). Raises ``AnalysisBusyError`` when another analysis is already
    running on this box (bugs.md §5H concurrency guard) — checked after the
    size guard so an oversized mask is rejected without taking the slot.
    """
    mask01 = np.asarray(mask01)
    if mask01.size > MAX_MASK_PIXELS:
        raise ValueError(
            f"That image is {mask01.size / 1e6:.1f} MP — larger than the "
            f"{MAX_MASK_PIXELS / 1e6:.0f} MP limit for in-app analysis. Crop or "
            "downscale it and try again."
        )

    if not _analysis_semaphore.acquire(blocking=False):
        raise AnalysisBusyError(
            "Another analysis is running on this box right now — please retry "
            "in a moment."
        )
    try:
        from src.pipeline.p2_graph.healing import heal_graph
        from src.pipeline.p2_graph.simplify import (
            consolidate_graph,
            simplify_graph,
            simplify_polylines,
        )
        from src.pipeline.p2_graph.skeleton_graph import (
            mask_to_skeleton_with_distance,
            prune_degenerate_edges,
            skeleton_to_graph,
        )
        from src.pipeline.p3_analysis.criticality import (
            annotate_criticality,
            annotate_cut_structure,
            rank_table,
        )
        from src.pipeline.p3_analysis.resilience import global_efficiency

        binary = (mask01 > 0).astype(np.uint8)
        skeleton, distance = mask_to_skeleton_with_distance(binary)
        graph = skeleton_to_graph(skeleton, transform=None, resolution_m=resolution_m, distance=distance)
        prune_degenerate_edges(graph, min_edge_len_m=resolution_m)  # drop sub-pixel/self-loop edges

        graph, _heal = heal_graph(graph)
        simplify_graph(graph)
        consolidate_graph(graph)
        simplify_polylines(graph, tol_m=1.5)

        if graph.number_of_nodes() < 2 or graph.number_of_edges() < 1:
            raise ValueError(
                "the extracted mask has essentially no road network — try a clearer "
                "or better-resolution capture (~0.5 m/pixel neighbourhood zoom)."
            )

        bc = annotate_criticality(graph, critical_fraction=critical_fraction)
        annotate_cut_structure(graph)
        rows = rank_table(graph, bc)
        criticality = pd.DataFrame(rows)

        # Headline number: how much routing efficiency the worst single junction
        # loss costs — the product's core "resilience" statement for the
        # uploaded network.
        base_eff = global_efficiency(graph)
        top_node = int(criticality.iloc[0]["node_id"]) if not criticality.empty else None
        if top_node is not None and base_eff > 0:
            perturbed = graph.copy()
            perturbed.remove_node(top_node)
            ri = global_efficiency(perturbed) / base_eff
        else:
            ri = 1.0
        ri = float(max(0.0, min(1.0, ri)))

        return AnalysisResult(
            graph=graph,
            criticality=criticality,
            resilience_index=ri,
            top_node=top_node,
            n_nodes=graph.number_of_nodes(),
            n_edges=graph.number_of_edges(),
            resolution_m=resolution_m,
            summary={
                "critical_junctions": int(criticality["is_critical"].sum()) if not criticality.empty else 0,
                "articulation_points": int(criticality["is_articulation"].sum())
                if "is_articulation" in criticality else 0,
                "worst_junction": top_node,
            },
        )
    finally:
        _analysis_semaphore.release()


def render_graph_overlay(image_rgb: np.ndarray, result: AnalysisResult):
    """Draw the extracted network over the uploaded image (image-space, honest).

    Returns a matplotlib ``Figure`` — roads as thin links, junctions coloured by
    criticality, the #1 chokepoint ringed — so the user sees *their* network, not
    a world-map guess. Imported lazily so headless analysis needs no matplotlib.
    """
    from matplotlib.figure import Figure

    inv = 1.0 / max(result.resolution_m, 1e-9)  # metres → pixels for plotting
    fig = Figure(figsize=(7, 7), dpi=110)
    ax = fig.add_subplot(111)
    ax.imshow(image_rgb)
    ax.set_axis_off()

    for _u, _v, data in result.graph.edges(data=True):
        geom = data.get("geometry")
        if not geom:
            continue
        xs = [p[0] * inv for p in geom]
        ys = [p[1] * inv for p in geom]
        ax.plot(xs, ys, color="#0EA5E9", linewidth=1.0, alpha=0.7)

    crit = result.criticality
    if not crit.empty:
        ax.scatter(crit["x"] * inv, crit["y"] * inv, s=8, c="#94A3B8", alpha=0.6, zorder=3)
        hot = crit[crit["is_critical"].astype(bool)]
        ax.scatter(hot["x"] * inv, hot["y"] * inv, s=26, c="#F59E0B", zorder=4,
                   label="critical junction")
        if result.top_node is not None:
            top = crit[crit["node_id"] == result.top_node]
            ax.scatter(top["x"] * inv, top["y"] * inv, s=90, facecolors="none",
                       edgecolors="#EF4444", linewidths=2, zorder=5, label="#1 chokepoint")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    fig.tight_layout(pad=0.2)
    return fig
