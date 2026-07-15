"""Shared in-memory mask-to-graph construction sequence."""

from __future__ import annotations

from src.pipeline.p2_graph import healing, simplify, skeleton_graph
from src.pipeline.p2_graph.config import GraphConfig


def construct_graph(mask, cfg: GraphConfig, *, transform=None, prob=None, metric_to_pixel=None):
    """Build, heal, and simplify a graph from one binary road mask."""
    skeleton, distance = skeleton_graph.mask_to_skeleton_with_distance(mask)
    graph = skeleton_graph.skeleton_to_graph(
        skeleton, transform=transform, resolution_m=cfg.resolution_m, distance=distance)
    skeleton_graph.prune_degenerate_edges(graph, cfg.min_edge_len_m)

    graph, heal_report = healing.heal_graph(
        graph,
        gap_max_m=cfg.gap_max_m, angle_max_deg=cfg.angle_max_deg,
        angle_penalty_factor=cfg.angle_penalty_factor,
        prob=prob, metric_to_pixel=metric_to_pixel,
        min_corridor_support=cfg.min_corridor_support, corridor_samples=cfg.corridor_samples,
    )
    simplify_report = simplify.simplify_graph(graph, cfg.min_stub_len_m) if cfg.simplify else None
    consolidate_report = simplify.consolidate_graph(
        graph, cfg.consolidate_tol_m) if cfg.consolidate else None
    polyline_report = simplify.simplify_polylines(
        graph, cfg.geom_tol_m) if cfg.simplify_geom else None
    return graph, heal_report, simplify_report, consolidate_report, polyline_report
