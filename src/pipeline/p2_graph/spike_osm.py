"""S1 spike: end-to-end graph/resilience run on an OSM-derived graph.

This is task **S1** in ``docs/Tracker.md`` — exercise the *whole* P2+P3 pipeline
before the segmentation model exists, using OpenStreetMap as a stand-in for the
predicted mask. It runs::

    OSM roads → rasterise to a mask  (reusing P1's osm_mask tooling)
        → skeletonise → sknw graph → MST/Union-Find healing   (P2)
        → betweenness → node ablation → global-efficiency RI   (P3)
        → export small, real-shaped artifacts to data/sample/  (unblocks F1)

The output ``data/sample/{aoi}_graph.geojson`` + ``_criticality.csv`` are the
committed sample set that lets Saanvi's dashboard (F1) run with no GPU and no
prior pipeline run (``docs/Tracker.md`` §4 "sample set" contract). When the real
mask arrives (A4), task S2 runs the *same* ``build_graph`` + ``analyze`` on it —
this spike just proves the machinery on OSM first.

We *consume* P1's ``osm_mask`` helpers (read-only, cross-lane is fine — we don't
edit them); everything we write stays in Shaivi's P2/P3 lane + the shared sample.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from src.pipeline.p1_segment.osm_mask import (  # consume P1 tooling (read-only)
    MaskConfig,
    save_binary_png,
    save_qc_overlay,
)
from src.pipeline.p2_graph.build_graph import build_graph
from src.pipeline.p2_graph.config import GraphConfig
from src.pipeline.p3_analysis.analyze import analyze

# A small default AOI (a slice of Panaji, Goa) — keeps the spike fast on CPU and
# the committed sample artifacts small. Override with --bbox / --place / --aoi.
DEFAULT_AOI = "panaji_demo"
DEFAULT_BBOX = (73.823, 15.488, 73.842, 15.501)  # west, south, east, north


def simulate_occlusion(
    mask: np.ndarray,
    n_patches: int,
    patch_px: int,
    seed: int = 42,
) -> np.ndarray:
    """Punch ``n_patches`` road-centred holes in a binary mask (returns a copy).

    Raw OSM roads are already connected, so healing has nothing to do on them.
    A real *predicted* mask, by contrast, is broken by occlusion — trees,
    shadows, vehicles (``docs/PRD.md`` problem statement). To exercise (and
    demonstrate) MST/Union-Find healing on realistic geometry *before* the model
    exists, we mimic that occlusion here: zero out square patches centred on
    random road pixels, just like the CoarseDropout augmentation the segmentation
    lane will train against. Reproducible via ``seed``.
    """
    if n_patches <= 0:
        return mask.copy()
    rng = np.random.default_rng(seed)
    out = mask.copy()
    road_yx = np.argwhere(mask > 0)
    if len(road_yx) == 0:
        return out
    size = max(1, patch_px)
    half = size // 2
    for cy, cx in road_yx[rng.integers(0, len(road_yx), size=n_patches)]:
        # Anchor an exactly size×size window (clamped at the top/left edges) so
        # the hole is the requested patch size, not 2·half.
        y0 = max(0, cy - half)
        x0 = max(0, cx - half)
        out[y0 : y0 + size, x0 : x0 + size] = 0
    return out


def _resolve_bbox(args: argparse.Namespace) -> tuple[float, float, float, float]:
    """Get (west, south, east, north) from --bbox, --place, or the default."""
    if args.bbox:
        parts = [float(v) for v in args.bbox.split(",")]
        if len(parts) != 4:
            raise SystemExit("--bbox must be 'west,south,east,north' (4 numbers)")
        return tuple(parts)  # type: ignore[return-value]
    if args.place:
        import osmnx as ox

        west, south, east, north = ox.geocode_to_gdf(args.place).total_bounds
        return float(west), float(south), float(east), float(north)
    return DEFAULT_BBOX


def _export_sample(cfg: GraphConfig, sample_dir: Path) -> None:
    """Copy the processed graph + criticality into the committed sample set."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        (cfg.geojson_path, sample_dir / f"{cfg.aoi}_graph.geojson"),
        (cfg.processed_dir / f"{cfg.aoi}_criticality.csv", sample_dir / f"{cfg.aoi}_criticality.csv"),
    ]
    for src, dst in pairs:
        shutil.copyfile(src, dst)
        print(f"  -> {dst}")


def _apply_occlusion(cfg: GraphConfig, n_patches: int, patch_m: float, resolution_m: float) -> None:
    """Load the written mask, simulate occlusion, and overwrite it (+ QC overlay).

    The occluded mask stands in for the predicted mask S2 will consume, so the
    graph is built from gappy roads and healing has real work to do.
    """
    from PIL import Image

    mask = (np.asarray(Image.open(cfg.mask_path).convert("L")) > 0).astype(np.uint8)
    patch_px = max(1, int(round(patch_m / resolution_m)))
    occluded = simulate_occlusion(mask, n_patches=n_patches, patch_px=patch_px)
    save_binary_png(occluded, cfg.mask_path)
    save_qc_overlay(occluded, cfg.interim_dir / f"{cfg.aoi}_mask_occluded_qc.png")
    kept = occluded.sum() / max(1, mask.sum())
    print(f"      occlusion: {n_patches} patches (~{patch_m:.0f} m) | {kept:.0%} of road px kept")


def run_spike(
    aoi: str,
    bbox: tuple[float, float, float, float],
    resolution_m: float = 2.0,
    buffer_m: float = 6.0,
    gap_max_m: float = 40.0,
    angle_max_deg: float = 60.0,
    occlude_patches: int = 80,
    occlude_size_m: float = 22.0,
    sample_dir: Path = Path("data/sample"),
) -> None:
    """Run OSM→mask→(occlude)→graph→heal→criticality→resilience→sample for one AOI."""
    # 1 · OSM → aligned binary mask + manifest (reuse P1's osm_mask pipeline).
    from src.pipeline.p1_segment.build_dataset import build_aoi_masks

    print(f"[1/4] OSM -> mask for '{aoi}' {bbox}")
    mask_cfg = MaskConfig(aoi=aoi, resolution_m=resolution_m, buffer_m=buffer_m)
    build_aoi_masks(mask_cfg, bbox)

    graph_cfg = GraphConfig(
        aoi=aoi, resolution_m=resolution_m, gap_max_m=gap_max_m, angle_max_deg=angle_max_deg
    )
    if occlude_patches > 0:
        _apply_occlusion(graph_cfg, occlude_patches, occlude_size_m, resolution_m)

    # 2 · mask → skeleton → graph → MST/Union-Find healing  (P2)
    print("[2/4] mask -> healed routable graph")
    build_graph(graph_cfg)

    # 3 · criticality + resilience  (P3)
    print("[3/4] betweenness + ablation + global-efficiency resilience")
    analyze(graph_cfg)

    # 4 · export the committed sample artifacts that unblock the dashboard (F1)
    print("[4/4] export sample artifacts")
    _export_sample(graph_cfg, sample_dir)
    print(f"\nS1 spike complete for '{aoi}'.")


def main() -> None:
    p = argparse.ArgumentParser(description="S1 spike: OSM → graph → resilience → sample.")
    p.add_argument("--aoi", default=DEFAULT_AOI, help=f"AOI id (default {DEFAULT_AOI})")
    p.add_argument("--bbox", help="west,south,east,north in lon/lat (overrides default)")
    p.add_argument("--place", help="place name to geocode instead of a bbox")
    p.add_argument("--resolution-m", type=float, default=2.0, help="grid GSD (m/px)")
    p.add_argument("--gap-max-m", type=float, default=40.0, help="max bridge length (m)")
    p.add_argument("--angle-max-deg", type=float, default=60.0, help="max road turn (deg)")
    p.add_argument("--occlude-patches", type=int, default=80,
                   help="simulated occlusion holes (0 = clean OSM, no healing demo)")
    p.add_argument("--occlude-size-m", type=float, default=22.0, help="occlusion patch size (m)")
    args = p.parse_args()

    run_spike(
        aoi=args.aoi,
        bbox=_resolve_bbox(args),
        resolution_m=args.resolution_m,
        gap_max_m=args.gap_max_m,
        angle_max_deg=args.angle_max_deg,
        occlude_patches=args.occlude_patches,
        occlude_size_m=args.occlude_size_m,
    )


if __name__ == "__main__":
    main()
