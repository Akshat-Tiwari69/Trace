"""A16 data prep: convert SpaceNet-5 (Mumbai AOI 8) road chips → DeepGlobe-format pairs.

SpaceNet-5 ships **georeferenced GeoTIFF imagery** (PS-RGB, often 16-bit) + a
**GeoJSON of road centrelines** per chip — *genuinely Indian* 0.3 m data (the
in-domain prize from `docs/Research.md` → Post-A11 Build Plan). DeepGlobe is
0/255 raster masks at ~0.5 m. This:

1. reads each image's affine transform + CRS (rasterio),
2. rasterises the matching GeoJSON centrelines, buffered to road width
   (reuses A3's :func:`osm_mask.rasterize_roads` — same path the OSM corpus uses,
   so the combined corpus stays consistent),
3. 8-bit-normalises the imagery (percentile stretch — handles 16-bit),
4. optionally **down-samples 0.3 m → ~0.5 m** (``scale=0.6``) to scale-match
   DeepGlobe, tiles to 512, and keeps road-bearing tiles as DeepGlobe-format
   pairs (``{prefix}_{chip}_r{r}_c{c}_sat.jpg`` + ``_mask.png``, 0/255).

Pure CPU; reuses A3's rasteriser + tiler. License: SpaceNet CC BY-SA 4.0.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from src.pipeline.p1_segment.osm_mask import tile_array


def _chip_key(name: str) -> str:
    """The id shared by an image and its label (SpaceNet uses ``..._chip{N}``)."""
    m = re.search(r"chip[A-Za-z0-9]+", name)
    if m:
        return m.group(0)
    m = re.search(r"(\d+)(?:\.[^.]+)?$", name)  # fallback: trailing number
    return m.group(1) if m else Path(name).stem


def _to_rgb8(bands: np.ndarray) -> np.ndarray:
    """First 3 bands (C,H,W) → H×W×3 uint8 via per-channel 2–98% percentile stretch.

    Robust to 8-bit *or* 16-bit imagery; the stretch also tolerates SpaceNet's
    wide dynamic range without clipping detail to black/white.
    """
    rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1).astype(np.float32)
    out = np.empty(rgb.shape, dtype=np.uint8)
    for c in range(3):
        ch = rgb[..., c]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        if hi <= lo:
            hi = lo + 1.0
        out[..., c] = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


def _rasterize_roads_metric(roads, transform, width: int, height: int, crs, buffer_m: float) -> np.ndarray:
    """Burn road centrelines buffered to ``buffer_m`` **metres** onto the image grid.

    Unlike ``osm_mask.rasterize_roads`` (which buffers in the grid CRS — fine for a
    metric UTM grid), SpaceNet chips are georeferenced in **EPSG:4326 (degrees)**, so
    buffering in the grid CRS would apply ``buffer_m`` as *degrees* (~111 km/deg →
    all-road masks). We buffer in the data's estimated UTM (metres) and reproject the
    polygons back to the grid CRS before rasterising — correct for geographic *or*
    projected grids.
    """
    from rasterio.features import rasterize

    if roads.crs is None:
        roads = roads.set_crs("EPSG:4326")
    metric = roads.estimate_utm_crs()
    buffered = roads.to_crs(metric).buffer(buffer_m / 2.0).to_crs(crs)
    shapes = [(g, 1) for g in buffered if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros((height, width), dtype=np.uint8)
    burned = rasterize(shapes, out_shape=(height, width), transform=transform,
                       fill=0, all_touched=True, dtype=np.uint8)
    return (burned > 0).astype(np.uint8)


def convert_spacenet(img_dir: str | Path, label_dir: str | Path, out_dir: str | Path,
                     buffer_m: float = 6.0, scale: float = 0.6, tile_size: int = 512,
                     min_road_fraction: float = 0.005, max_tiles: int | None = None,
                     prefix: str = "sn5mum") -> int:
    """Tile + scale-match SpaceNet road chips into DeepGlobe-format pairs. Returns #kept."""
    import cv2
    import geopandas as gpd
    import rasterio
    from PIL import Image

    img_dir, label_dir, out = Path(img_dir), Path(label_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = {_chip_key(p.name): p for p in label_dir.glob("*.geojson")}

    kept = 0
    for img_p in sorted(img_dir.glob("*.tif*")):
        label = labels.get(_chip_key(img_p.name))
        if label is None:                            # no matching centreline file → skip
            continue
        with rasterio.open(img_p) as src:
            if src.count < 3:
                continue
            bands = src.read([1, 2, 3])              # SpaceNet PS-RGB band order = R,G,B
            transform, crs = src.transform, src.crs
            height, width = src.height, src.width

        roads = gpd.read_file(label)
        if len(roads) == 0:                          # empty centreline file → no road
            mask = np.zeros((height, width), dtype=np.uint8)
        else:
            mask = _rasterize_roads_metric(roads, transform, width, height, crs, buffer_m)

        img = _to_rgb8(bands)
        if scale != 1.0:                             # 0.3m -> ~0.5m to scale-match DeepGlobe
            nh, nw = max(1, int(height * scale)), max(1, int(width * scale))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            img = cv2.resize(img, (nw, nh), interpolation=interp)
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        chip = _chip_key(img_p.name)
        for ti, tm in zip(tile_array(img, tile_size), tile_array(mask, tile_size)):
            if float(tm.data.mean()) < min_road_fraction:   # skip near-empty tiles
                continue
            stem = f"{prefix}_{chip}_r{ti.row}_c{ti.col}"
            Image.fromarray(ti.data).save(out / f"{stem}_sat.jpg", quality=92)
            Image.fromarray((tm.data * 255).astype(np.uint8), mode="L").save(out / f"{stem}_mask.png")
            kept += 1
            if max_tiles and kept >= max_tiles:
                print(f"[spacenet] reached max_tiles={max_tiles} -> {out}")
                return kept

    print(f"[spacenet] {kept} road-bearing pairs -> {out}")
    return kept


def main() -> None:
    p = argparse.ArgumentParser(description="Convert SpaceNet-5 road chips to DeepGlobe-format pairs.")
    p.add_argument("--src", default="data/raw/spacenet/SN5_roads_train_AOI_8_Mumbai",
                   help="extracted tarball root (holds PS-RGB/ + geojson_roads_speed/)")
    p.add_argument("--img-subdir", default="PS-RGB")
    p.add_argument("--label-subdir", default="geojson_roads_speed")
    p.add_argument("--out", default="data/raw/spacenet/dg_format")
    p.add_argument("--buffer-m", type=float, default=6.0, help="painted road width in metres")
    p.add_argument("--scale", type=float, default=0.6, help="0.6 = 0.3m→0.5m scale-match to DeepGlobe")
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--min-road-fraction", type=float, default=0.005)
    p.add_argument("--max-tiles", type=int, default=None)
    args = p.parse_args()
    src = Path(args.src)
    convert_spacenet(src / args.img_subdir, src / args.label_subdir, args.out,
                     buffer_m=args.buffer_m, scale=args.scale, tile_size=args.tile_size,
                     min_road_fraction=args.min_road_fraction, max_tiles=args.max_tiles)


if __name__ == "__main__":
    main()
