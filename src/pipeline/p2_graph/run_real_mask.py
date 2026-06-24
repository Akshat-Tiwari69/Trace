"""S2 driver: heal + analyse a **real predicted mask** (Tracker task S2).

S1 proved the graph/resilience engine on an OSM-derived mask. S2 runs the *same*
engine on the genuine output of the P1 segmentation model — the binary mask
``predict.py`` writes to ``data/interim/{aoi}_mask.png`` (``docs/Tracker.md`` §4).
There is **no new graph logic here**: S2 is literally ``build_graph`` → ``analyze``
pointed at the predicted mask, writing to ``data/processed/`` (not the committed
``data/sample/`` demo set S1 produced).

Coordinate space — important: a predicted mask is **pixel-space**. Unlike S1's
OSM mask, ``predict.py`` reads RGB imagery via OpenCV and drops any geo-transform,
so no alignment manifest accompanies the mask. The graph is therefore built in
pixels (``length_m`` = pixel length × ``--resolution-m``; node ``x, y`` are pixel
coordinates, not lon/lat). If the tile's geotransform is known, drop a manifest
at ``data/interim/{aoi}/manifest.json`` (same shape A3 writes) and ``build_graph``
will georeference automatically — that handoff is a coordination point with the
imagery/P1 lane, not faked here.

Example::

    # after: python -m src.pipeline.p1_segment.predict --image tile.tif \
    #            --checkpoint models/road.pt --aoi panaji
    python -m src.pipeline.p2_graph.run_real_mask --aoi panaji
"""

from __future__ import annotations

import argparse

from src.pipeline.p2_graph.build_graph import build_graph
from src.pipeline.p2_graph.config import GraphConfig
from src.pipeline.p3_analysis.analyze import analyze


def run(
    cfg: GraphConfig,
    critical_fraction: float = 0.10,
    k: int | None = None,
    curve_steps: int = 25,
) -> dict:
    """Build the healed graph from the predicted mask, then run criticality + resilience.

    Returns the ``analyze`` summary dict. Output artifacts land at the §4
    ``data/processed/`` paths via the existing ``build_graph`` / ``analyze``.
    """
    georef = "georeferenced (lon/lat)" if cfg.manifest_path.exists() else "pixel-space (no manifest)"
    print(f"[S2] AOI '{cfg.aoi}' from predicted mask {cfg.mask_path} | {georef}")

    print("[1/2] mask -> healed routable graph")
    build_graph(cfg)

    print("[2/2] betweenness + ablation + global-efficiency resilience")
    summary = analyze(cfg, critical_fraction=critical_fraction, k=k, curve_steps=curve_steps)

    print(f"\nS2 complete for '{cfg.aoi}' -> data/processed/")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="S2: heal + analyse a real predicted mask.")
    p.add_argument("--aoi", required=True, help="AOI id (matches data/interim/{aoi}_mask.png)")
    p.add_argument("--gap-max-m", type=float, default=40.0, help="max bridge length")
    p.add_argument("--angle-max-deg", type=float, default=60.0, help="max road turn (deg)")
    p.add_argument("--resolution-m", type=float, default=1.0,
                   help="m/px for pixel-space masks (sets length_m scale; ignored if a manifest exists)")
    p.add_argument("--critical-fraction", type=float, default=0.10, help="top fraction flagged critical")
    p.add_argument("--k", type=int, default=None, help="k-sample betweenness (large graphs)")
    p.add_argument("--curve-steps", type=int, default=25, help="ablation curve length")
    args = p.parse_args()

    cfg = GraphConfig(
        aoi=args.aoi,
        gap_max_m=args.gap_max_m,
        angle_max_deg=args.angle_max_deg,
        resolution_m=args.resolution_m,
    )
    run(cfg, critical_fraction=args.critical_fraction, k=args.k, curve_steps=args.curve_steps)


if __name__ == "__main__":
    main()
