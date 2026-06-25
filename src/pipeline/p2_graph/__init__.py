"""P2 · Graph build + healing lane (Shaivi).

Mask → skeletonise → sknw graph → MST/Union-Find healing → a single routable,
weighted graph (``docs/PRD.md`` G2). CPU-only, classical Python.
"""

from src.pipeline.p2_graph.config import GraphConfig
from src.pipeline.p2_graph.healing import HealReport, UnionFind, heal_graph
from src.pipeline.p2_graph.simplify import (
    ConsolidateReport,
    SimplifyReport,
    collapse_degree2_nodes,
    consolidate_graph,
    consolidate_nearby_nodes,
    prune_short_stubs,
    simplify_graph,
)
from src.pipeline.p2_graph.skeleton_graph import (
    mask_to_skeleton,
    reproject_graph_to_wgs84,
    skeleton_to_graph,
)

__all__ = [
    "GraphConfig",
    "HealReport",
    "UnionFind",
    "heal_graph",
    "SimplifyReport",
    "ConsolidateReport",
    "collapse_degree2_nodes",
    "consolidate_graph",
    "consolidate_nearby_nodes",
    "prune_short_stubs",
    "simplify_graph",
    "mask_to_skeleton",
    "reproject_graph_to_wgs84",
    "skeleton_to_graph",
]
