"""Close the upload loop: a road mask → routable graph → resilience analysis.

The dashboard's upload flow previously dead-ended at a mask preview. A39 closes
that loop by turning the extracted mask into a graph and resilience score.
This module runs the existing **CPU** P2→P3 pipeline in-process on an uploaded
mask and returns the same analysis contract the demo AOI ships with.

Kept free of web-framework imports so it is unit-testable without a UI runtime.
Coordinates stay in **image space** (pixels scaled
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
# time: this runs in-process on a 4-core ARM box (A36), and a single
# pass already uses multiple cores itself, so N concurrent uploads would
# starve the public application rather than queue behind one job. This is
# tighter than the two-request GPU guard because all CPU analyses share one
# host. `acquire(blocking=False)` so a saturated box surfaces an explicit
# "busy, retry" message instead of silently queuing behind the scenes.
# The public upload path uses the filesystem queue. Keep this semaphore for
# direct callers and tests so two analyses cannot saturate one worker.
_ANALYSIS_MAX_INFLIGHT = 1
_analysis_semaphore = threading.BoundedSemaphore(_ANALYSIS_MAX_INFLIGHT)

# Reject masks larger than this before the CPU pipeline runs. It matches the
# API's 4096×4096 decompression ceiling so an upload that passed the earlier
# PIL check cannot blow up the
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
    running on this box (A36 concurrency guard) — checked after the
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
        from src.pipeline.p2_graph.config import GraphConfig
        from src.pipeline.p2_graph.construction import construct_graph
        from src.pipeline.p3_analysis.criticality import (
            annotate_criticality,
            annotate_cut_structure,
            rank_table,
        )
        from src.pipeline.p3_analysis.resilience import (
            efficiency_sampling_policy,
            resilience_index,
        )

        binary = (mask01 > 0).astype(np.uint8)
        graph_cfg = GraphConfig(
            "upload", resolution_m=resolution_m, min_edge_len_m=resolution_m,
            min_corridor_support=0.0,
        )
        graph, _heal, _, _, _ = construct_graph(binary, graph_cfg)

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
        efficiency_policy = efficiency_sampling_policy(graph)
        top_node = int(criticality.iloc[0]["node_id"]) if not criticality.empty else None
        if top_node is not None:
            metrics = resilience_index(
                graph, [top_node], k=efficiency_policy["k"],
                seed=efficiency_policy["seed"],
            )
            ri = metrics["resilience_index"]
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
                "efficiency_method": efficiency_policy["method"],
                "efficiency_k": efficiency_policy["k"],
                "efficiency_seed": efficiency_policy["seed"],
            },
        )
    finally:
        _analysis_semaphore.release()
