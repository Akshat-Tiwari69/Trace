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
from src.pipeline.p2_graph.healing import HealReport, heal_graph
from src.pipeline.p2_graph.simplify import simplify_graph
from src.pipeline.p2_graph.skeleton_graph import (
    mask_to_skeleton,
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

    skeleton = mask_to_skeleton(mask)
    graph = skeleton_to_graph(skeleton, transform=transform, resolution_m=cfg.resolution_m)
    prune_degenerate_edges(graph, cfg.min_edge_len_m)  # drop sub-pixel/self-loop edges

    graph, report = heal_graph(
        graph,
        gap_max_m=cfg.gap_max_m,
        angle_max_deg=cfg.angle_max_deg,
        angle_penalty_factor=cfg.angle_penalty_factor,
    )

    # Stash the authoritative healing stats (measured now, pre-simplification) so
    # the evaluator reports the true connectivity ratio rather than re-deriving it.
    graph.graph["heal"] = {
        "connectivity_ratio_pct": round(report.connectivity_ratio, 2),
        "components_before": report.components_before,
        "components_after": report.components_after,
        "bridges_added": report.bridges_added,
    }

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
    args = p.parse_args()

    cfg = GraphConfig(
        aoi=args.aoi,
        gap_max_m=args.gap_max_m,
        angle_max_deg=args.angle_max_deg,
        resolution_m=args.resolution_m,
    )
    build_graph(cfg)


if __name__ == "__main__":
    main()
