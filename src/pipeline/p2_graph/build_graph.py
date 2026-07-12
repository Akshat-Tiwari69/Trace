"""CLI: build a healed routable graph from a road mask (Phase II / task S1·S2).

Consumes the P1 contract input ``data/interim/{aoi}_mask.png`` (binary {0,1}) and
its optional alignment manifest, runs skeletonise → sknw → MST/Union-Find
healing, and writes the P2 contract outputs ``data/processed/{aoi}_graph.graphml``
and ``.geojson`` (``docs/Tracker.md`` §4).

The *same* command serves both S1 (mask produced from OSM by the spike) and S2
(mask produced by the real segmentation model) — only the upstream mask differs.

Example::

    python -m src.pipeline.p2_graph.build_graph --aoi panaji_demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.pipeline.p2_graph.config import GraphConfig
from src.pipeline.p2_graph.graph_io import save_geojson, save_graphml
from src.pipeline.p2_graph.healing import HealReport, heal_graph, sample_prob_along_polyline
from src.pipeline.p2_graph.simplify import (
    consolidate_graph,
    simplify_graph,
    simplify_polylines,
)
from src.pipeline.p2_graph.skeleton_graph import (
    build_metric_to_pixel,
    ensure_metric_transform,
    mask_to_skeleton_with_distance,
    prune_degenerate_edges,
    reproject_graph_to_wgs84,
    skeleton_to_graph,
)


def _load_mask(path: Path) -> np.ndarray:
    """Load a binary {0,1} road mask PNG as a 2-D uint8 array."""
    from PIL import Image

    if not path.exists():
        raise SystemExit(
            f"mask not found: {path}\n"
            "  Run the OSM spike (S1) or wait for the P1 model mask (S2)."
        )
    arr = np.asarray(Image.open(path).convert("L"))
    return (arr > 0).astype(np.uint8)


def _load_prob(path: Path) -> np.ndarray | None:
    """Load P1's probability map (bugs.md §4) as a float [0,1] array, or ``None``.

    Present only when P1 ran the blended inference path; a mask-only input
    (upload, OSM spike, old artifact) has no such file — healing then falls back
    to distance/angle/crossing only, exactly as before this feature existed.
    """
    if not path.exists():
        return None
    from PIL import Image

    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _load_alignment(manifest_path: Path):
    """Return ``(transform, crs)`` from the mask manifest, or ``(None, None)``.

    Without a manifest the graph is built in pixel space (still valid, just not
    georeferenced) — so the pipeline degrades gracefully for a bare mask.
    """
    if not manifest_path.exists():
        return None, None
    from affine import Affine

    meta = json.loads(manifest_path.read_text())
    transform = Affine(*meta["transform"]) if "transform" in meta else None
    crs = meta.get("crs")
    return transform, crs


def build_graph(cfg: GraphConfig) -> tuple[object, HealReport]:
    """Run mask → skeleton → graph → heal → reproject → save. Returns (graph, report)."""
    mask = _load_mask(cfg.mask_path)
    transform, crs = _load_alignment(cfg.manifest_path)
    if transform is None:
        print(f"[{cfg.aoi}] WARNING: no alignment manifest at {cfg.manifest_path} — "
              f"graph is built in PIXEL space (resolution_m={cfg.resolution_m}); "
              "length_m and resilience numbers are not true metres unless "
              "--resolution-m matches the imagery's real GSD")
    transform, crs = ensure_metric_transform(transform, crs, mask.shape[1], mask.shape[0])

    skeleton, distance = mask_to_skeleton_with_distance(mask)
    graph = skeleton_to_graph(skeleton, transform=transform, resolution_m=cfg.resolution_m,
                              distance=distance)
    prune_degenerate_edges(graph, cfg.min_edge_len_m)  # drop sub-pixel/self-loop edges

    # Probability-map corridor check (bugs.md §4): loaded only when P1's blended
    # path persisted one; mask-only input (upload/OSM spike/old artifact) heals
    # exactly as before. Loudly log which mode is active — silent behaviour
    # change here would be a nasty surprise for a resilience number.
    prob = _load_prob(cfg.prob_path)
    corridor_on = prob is not None and cfg.min_corridor_support > 0
    if corridor_on:
        print(f"[{cfg.aoi}] healing: probability-map corridor check ON "
              f"(min_corridor_support={cfg.min_corridor_support}, prob map {cfg.prob_path})")
    else:
        why = "no prob.png found" if prob is None else "min_corridor_support=0"
        print(f"[{cfg.aoi}] healing: probability-map corridor check OFF ({why}) "
              "— distance/angle/crossing only")
    metric_to_pixel = build_metric_to_pixel(transform, cfg.resolution_m) if prob is not None else None

    graph, report = heal_graph(
        graph,
        gap_max_m=cfg.gap_max_m,
        angle_max_deg=cfg.angle_max_deg,
        angle_penalty_factor=cfg.angle_penalty_factor,
        prob=prob,
        metric_to_pixel=metric_to_pixel,
        min_corridor_support=cfg.min_corridor_support,
        corridor_samples=cfg.corridor_samples,
    )

    # Stash the authoritative healing stats (measured now, pre-simplification) so
    # the evaluator reports the true connectivity ratio rather than re-deriving it.
    graph.graph["heal"] = {
        "connectivity_ratio_pct": round(report.connectivity_ratio, 2),
        "components_before": report.components_before,
        "components_after": report.components_after,
        "bridges_added": report.bridges_added,
        "bridges_rejected_crossing": report.bridges_rejected_crossing,
        "bridges_rejected_corridor": report.bridges_rejected_corridor,
    }

    # Carry the P1 provenance (checkpoint/threshold/commit) into the graph so the
    # graphml/geojson name the exact model that produced this network (§5A).
    from src.pipeline.p1_segment.provenance import read_provenance
    provenance = read_provenance(cfg.provenance_path)
    if provenance is not None:
        graph.graph["provenance"] = provenance

    simplify_report = None
    if cfg.simplify:
        simplify_report = simplify_graph(graph, min_stub_len_m=cfg.min_stub_len_m)
        graph.graph["simplify"] = {
            "nodes_before": simplify_report.nodes_before,
            "nodes_after": simplify_report.nodes_after,
            "node_reduction_pct": round(simplify_report.node_reduction_pct, 1),
            "stubs_pruned": simplify_report.stubs_pruned,
            "nodes_collapsed": simplify_report.nodes_collapsed,
        }

    consolidate_report = None
    if cfg.consolidate:
        consolidate_report = consolidate_graph(graph, tol_m=cfg.consolidate_tol_m)
        graph.graph["consolidate"] = {
            "nodes_before": consolidate_report.nodes_before,
            "nodes_after": consolidate_report.nodes_after,
            "nodes_merged": consolidate_report.nodes_merged,
        }

    polyline_report = None
    if cfg.simplify_geom:  # Douglas-Peucker in metric space, before reprojection
        polyline_report = simplify_polylines(graph, tol_m=cfg.geom_tol_m)
        graph.graph["polyline"] = {
            "vertices_before": polyline_report.vertices_before,
            "vertices_after": polyline_report.vertices_after,
            "vertex_reduction_pct": round(polyline_report.vertex_reduction_pct, 1),
        }

    # Per-edge confidence (bugs.md §9.3): mean P1 probability sampled along each
    # *final* edge's own geometry — the same corridor-support sampling healing
    # uses for candidate bridges, applied here to every surviving edge (bridged
    # edges too: their corridor support already stood in for confidence, so
    # sampling their drawn geometry like any other edge is exactly right). Runs
    # after simplify/consolidate/polyline-simplification settle the final
    # geometry, and before reprojection (metric_to_pixel expects the source
    # metric CRS, not lon/lat). No prob map -> no attribute at all (mask-only
    # input keeps producing the pre-§9.3 artifact, byte-for-byte).
    if prob is not None:
        for _, _, data in graph.edges(data=True):
            data["confidence"] = round(
                sample_prob_along_polyline(data["geometry"], prob, metric_to_pixel), 3
            )

    if crs is not None:
        reproject_graph_to_wgs84(graph, crs)

    save_graphml(graph, cfg.graphml_path)
    save_geojson(graph, cfg.geojson_path)

    simplify_line = ""
    if simplify_report is not None:
        simplify_line = (
            f"  simplify: nodes {simplify_report.nodes_before}->{simplify_report.nodes_after} "
            f"(-{simplify_report.node_reduction_pct:.0f}%), edges "
            f"{simplify_report.edges_before}->{simplify_report.edges_after} "
            f"| {simplify_report.stubs_pruned} stubs pruned, "
            f"{simplify_report.nodes_collapsed} degree-2 collapsed "
            f"| components preserved {simplify_report.components_before}->{simplify_report.components_after}\n"
        )
    if consolidate_report is not None:
        simplify_line += (
            f"  consolidate: nodes {consolidate_report.nodes_before}->{consolidate_report.nodes_after} "
            f"| {consolidate_report.nodes_merged} near-duplicate junctions merged "
            f"(tol {cfg.consolidate_tol_m:.0f} m)\n"
        )
    if polyline_report is not None:
        simplify_line += (
            f"  polyline: vertices {polyline_report.vertices_before}->{polyline_report.vertices_after} "
            f"(-{polyline_report.vertex_reduction_pct:.0f}%) via Douglas-Peucker (tol {cfg.geom_tol_m} m)\n"
        )
    print(
        f"[{cfg.aoi}] graph: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges | "
        f"components {report.components_before}->{report.components_after} | "
        f"+{report.bridges_added} bridges | "
        f"connectivity ratio +{report.connectivity_ratio:.1f}%\n"
        f"{simplify_line}"
        f"  -> {cfg.graphml_path}\n"
        f"  -> {cfg.geojson_path}"
    )
    return graph, report


def main() -> None:
    p = argparse.ArgumentParser(description="Build a healed routable graph from a mask.")
    p.add_argument("--aoi", required=True, help="AOI id (matches the mask filename)")
    p.add_argument("--gap-max-m", type=float, default=40.0, help="max bridge length (m)")
    p.add_argument("--angle-max-deg", type=float, default=60.0, help="max road turn (deg)")
    p.add_argument("--resolution-m", type=float, default=1.0, help="m/px (no-manifest fallback)")
    p.add_argument("--min-corridor-support", type=float, default=0.3,
                   help="reject a bridge if mean P1 prob-map support along it is below this "
                        "(0 disables; needs prob.png, bugs.md §4)")
    args = p.parse_args()

    cfg = GraphConfig(
        aoi=args.aoi,
        gap_max_m=args.gap_max_m,
        angle_max_deg=args.angle_max_deg,
        resolution_m=args.resolution_m,
        min_corridor_support=args.min_corridor_support,
    )
    build_graph(cfg)


if __name__ == "__main__":
    main()
