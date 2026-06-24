"""CLI: build OSM road-mask labels for an AOI (task A3).

Orchestrates the Phase-0 data pipeline end to end for one AOI:
download/cache OSM roads → build an aligned metric grid → rasterise → tile →
QC. Writes the §4 contract artifact ``data/interim/{aoi}_mask.png`` plus tiles,
a QC overlay, and an alignment manifest.

Examples
--------
By bounding box (west,south,east,north in lon/lat)::

    python -m src.pipeline.p1_segment.build_dataset \
        --aoi panaji --bbox 73.80,15.47,73.86,15.51

By place name (geocoded via osmnx)::

    python -m src.pipeline.p1_segment.build_dataset --aoi panaji --place "Panaji, Goa, India"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.pipeline.p1_segment.osm_mask import (
    MaskConfig,
    build_grid,
    fetch_osm_roads,
    rasterize_roads,
    save_binary_png,
    save_qc_overlay,
    tile_array,
)


def _resolve_bbox(args: argparse.Namespace) -> tuple[float, float, float, float]:
    """Get (west, south, east, north) from --bbox or by geocoding --place."""
    if args.bbox:
        parts = [float(v) for v in args.bbox.split(",")]
        if len(parts) != 4:
            raise SystemExit("--bbox must be 'west,south,east,north' (4 numbers)")
        return tuple(parts)  # type: ignore[return-value]

    if args.place:
        import osmnx as ox

        gdf = ox.geocode_to_gdf(args.place)
        west, south, east, north = gdf.total_bounds
        return float(west), float(south), float(east), float(north)

    raise SystemExit("provide either --bbox or --place")


def build_aoi_masks(
    cfg: MaskConfig,
    bbox: tuple[float, float, float, float],
) -> dict:
    """Run the full OSM→mask→tile pipeline for one AOI. Returns a manifest dict."""
    interim = Path(cfg.interim_dir)
    cache_path = Path(cfg.raw_dir) / f"{cfg.aoi}_osm_roads.gpkg"

    # 1 · roads (cached) → 2 · grid → 3 · rasterise
    roads = fetch_osm_roads(bbox, network_type=cfg.network_type, cache_path=cache_path)
    transform, width, height, crs = build_grid(bbox, cfg.resolution_m)
    mask = rasterize_roads(roads, transform, width, height, crs, cfg.buffer_m)

    # full-AOI contract artifact + QC overlay
    full_mask_path = interim / f"{cfg.aoi}_mask.png"
    save_binary_png(mask, full_mask_path)
    save_qc_overlay(mask, interim / f"{cfg.aoi}_mask_qc.png")

    # 4 · tiles
    tiles = tile_array(mask, cfg.tile_size)
    tiles_dir = interim / cfg.aoi / "tiles"
    tile_records = []
    for t in tiles:
        name = f"{cfg.aoi}_r{t.row}_c{t.col}_mask.png"
        save_binary_png(t.data, tiles_dir / name)
        tile_records.append(
            {
                "file": f"tiles/{name}",
                "row": t.row,
                "col": t.col,
                "y0": t.y0,
                "x0": t.x0,
                "road_px": int(t.data.sum()),
            }
        )

    # QC: overlay the most road-dense tile so a human can eyeball alignment
    densest = max(tiles, key=lambda t: int(t.data.sum()))
    save_qc_overlay(densest.data, interim / f"{cfg.aoi}_qc_tile.png")

    road_fraction = float(mask.mean())
    manifest = {
        "aoi": cfg.aoi,
        "bbox_wgs84": list(bbox),
        "crs": str(crs),
        "transform": list(transform)[:6],
        "resolution_m": cfg.resolution_m,
        "buffer_m": cfg.buffer_m,
        "tile_size": cfg.tile_size,
        "width": width,
        "height": height,
        "road_pixel_fraction": road_fraction,
        "n_tiles": len(tiles),
        "tiles": tile_records,
    }
    manifest_path = interim / cfg.aoi / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        f"[{cfg.aoi}] grid {width}x{height}px @ {cfg.resolution_m} m/px | "
        f"roads {road_fraction:.2%} of pixels | {len(tiles)} tiles "
        f"({cfg.tile_size}px) | CRS {crs}\n"
        f"  -> {full_mask_path}\n"
        f"  -> {interim / f'{cfg.aoi}_qc_tile.png'} (QC: densest tile)\n"
        f"  -> {manifest_path}"
    )
    if road_fraction == 0:
        print("  [warn] no road pixels -- check the bbox covers a roaded area.")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Build OSM road-mask labels for an AOI.")
    p.add_argument("--aoi", required=True, help="short AOI id, e.g. 'panaji'")
    p.add_argument("--bbox", help="west,south,east,north in lon/lat")
    p.add_argument("--place", help="place name to geocode, e.g. 'Panaji, Goa, India'")
    p.add_argument("--resolution-m", type=float, default=1.0, help="grid GSD (m/px)")
    p.add_argument("--buffer-m", type=float, default=6.0, help="road width to paint (m)")
    p.add_argument("--tile-size", type=int, default=256, help="tile edge in pixels")
    p.add_argument("--network-type", default="drive", help="osmnx network filter")
    args = p.parse_args()

    cfg = MaskConfig(
        aoi=args.aoi,
        resolution_m=args.resolution_m,
        buffer_m=args.buffer_m,
        tile_size=args.tile_size,
        network_type=args.network_type,
    )
    bbox = _resolve_bbox(args)
    build_aoi_masks(cfg, bbox)


if __name__ == "__main__":
    main()
