"""A6 data prep: build (satellite, OSM-mask) training pairs for fine-tuning.

A3 (`build_dataset`) produces OSM road **masks** on a metric grid plus a manifest
holding that grid's CRS/transform. Fine-tuning the A4 model needs **image+mask
pairs**, so this:

1. builds the OSM mask for an AOI (reusing A3's `build_aoi_masks`),
2. fetches satellite imagery for the same AOI (keyless Esri World-Imagery XYZ),
3. **warps the imagery onto the mask's exact grid** so pixels align by
   construction,
4. tiles both with A3's `tile_array` and writes **DeepGlobe-format** pairs
   (`{aoi}_r{r}_c{c}_sat.jpg` + `_mask.png`) the A4 notebook reads unchanged,
   keeping only road-bearing tiles.

These are *weak* labels (OSM gaps/misalignment) — use as **fine-tune** data on
top of the DeepGlobe-pretrained checkpoint, with buffered/relaxed metrics
(`docs/Evaluation.md`). Output lands in `data/finetune/` (gitignored).

The run needs network (osmnx + imagery); the tile math and the warp are
unit-tested offline via an injectable ``tile_fetcher``.
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path
from typing import Callable

import numpy as np

from src.pipeline.p1_segment.build_dataset import build_aoi_masks
from src.pipeline.p1_segment.osm_mask import MaskConfig, tile_array

ESRI_TILE = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
             "World_Imagery/MapServer/tile/{z}/{y}/{x}")
WEB_MERCATOR = "EPSG:3857"
_EARTH_CIRCUMFERENCE = 2 * math.pi * 6378137.0  # metres, EPSG:3857

# Default domain-matched AOIs: drivable-road-dense Indian city patches.
# (west, south, east, north) in lon/lat — small (~3–4 km) so one run is quick.
# These 5 are the **held-out Indian eval set** (`data/finetune/`) — NEVER train on them.
DEFAULT_CITIES: dict[str, tuple[float, float, float, float]] = {
    "panaji": (73.80, 15.47, 73.84, 15.50),
    "margao": (73.94, 15.27, 73.98, 15.30),
    "mumbai_bandra": (72.82, 19.04, 72.86, 19.07),
    "bengaluru_indiranagar": (77.63, 12.96, 77.66, 12.99),
    "delhi_cp": (77.20, 28.62, 77.24, 28.65),
}

# A16(a): the in-domain Indian OSM **corpus** — a large, geographically *disjoint*
# roster (metros + tier-2/3 + east/NE) spanning Indian road morphology. These are
# **weak/noisy** labels for pretrain + self-training (A12) ONLY — they must never
# overlap the held-out eval AOIs above (a disjointness test enforces this). Built
# to a SEPARATE dir (`data/raw/indian_corpus/`). ~0.03°×0.025° (~3 km) boxes.
CORPUS_CITIES: dict[str, tuple[float, float, float, float]] = {
    "kolkata_park_st": (88.340, 22.540, 88.370, 22.565),
    "chennai_tnagar": (80.225, 13.035, 80.255, 13.060),
    "hyderabad_banjara": (78.430, 17.410, 78.460, 17.435),
    "ahmedabad_navrangpura": (72.550, 23.020, 72.580, 23.045),
    "pune_shivajinagar": (73.830, 18.520, 73.860, 18.545),
    "jaipur_cscheme": (75.790, 26.900, 75.820, 26.925),
    "lucknow_hazratganj": (80.930, 26.840, 80.960, 26.865),
    "kanpur_civillines": (80.330, 26.460, 80.360, 26.485),
    "surat_adajan": (72.790, 21.180, 72.820, 21.205),
    "nagpur_sitabuldi": (79.070, 21.130, 79.100, 21.155),
    "kochi_mgroad": (76.270, 9.970, 76.300, 9.995),
    "coimbatore_rspuram": (76.950, 11.000, 76.980, 11.025),
    "indore_vijaynagar": (75.890, 22.740, 75.920, 22.765),
    "bhopal_mpnagar": (77.420, 23.230, 77.450, 23.255),
    "visakhapatnam_dwaraka": (83.290, 17.720, 83.320, 17.745),
    "chandigarh_sec17": (76.770, 30.730, 76.800, 30.755),
    "guwahati_paltanbazaar": (91.740, 26.170, 91.770, 26.195),
    "patna_boring": (85.120, 25.600, 85.150, 25.625),
    "bhubaneswar_saheednagar": (85.820, 20.280, 85.850, 20.305),
    "dehradun_clocktower": (78.030, 30.320, 78.060, 30.345),
}


def deg2num(lat: float, lon: float, z: int) -> tuple[float, float]:
    """WGS84 lat/lon → fractional XYZ tile coordinates at zoom ``z``."""
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def num2merc(x: float, y: float, z: int) -> tuple[float, float]:
    """Fractional XYZ tile coordinates → Web-Mercator metres (EPSG:3857)."""
    mx = x / (2 ** z) * _EARTH_CIRCUMFERENCE - _EARTH_CIRCUMFERENCE / 2
    my = _EARTH_CIRCUMFERENCE / 2 - y / (2 ** z) * _EARTH_CIRCUMFERENCE
    return mx, my


def _default_tile_fetcher(z: int, x: int, y: int) -> bytes:
    """Fetch one Esri World-Imagery tile (keyless), retrying transient network errors.

    A multi-city corpus run issues thousands of tile requests; a single timed-out
    tile must NOT abort the whole run, so retry a few times with linear backoff.
    """
    import time
    import urllib.error
    import urllib.request

    req = urllib.request.Request(ESRI_TILE.format(z=z, x=x, y=y),
                                 headers={"User-Agent": "Mozilla/5.0 route-resilience-a6"})
    last: Exception | None = None
    for attempt in range(4):
        try:
            return urllib.request.urlopen(req, timeout=30).read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise last  # type: ignore[misc]


def fetch_imagery_mosaic(bbox: tuple[float, float, float, float], zoom: int,
                         tile_fetcher: Callable[[int, int, int], bytes] = _default_tile_fetcher):
    """Stitch the Esri tiles covering ``bbox`` → ``(rgb HxWx3 uint8, Affine 3857)``."""
    from affine import Affine
    from PIL import Image

    west, south, east, north = bbox
    x0, y0 = (int(math.floor(v)) for v in deg2num(north, west, zoom))  # top-left tile
    x1, y1 = (int(math.floor(v)) for v in deg2num(south, east, zoom))  # bottom-right tile
    nx, ny = x1 - x0 + 1, y1 - y0 + 1

    canvas = Image.new("RGB", (nx * 256, ny * 256))
    for ix in range(nx):
        for iy in range(ny):
            data = tile_fetcher(zoom, x0 + ix, y0 + iy)
            canvas.paste(Image.open(io.BytesIO(data)).convert("RGB"), (ix * 256, iy * 256))
    rgb = np.asarray(canvas)

    mx0, my0 = num2merc(x0, y0, zoom)            # top-left corner (metres)
    mx1, my1 = num2merc(x1 + 1, y1 + 1, zoom)    # bottom-right corner
    px = (mx1 - mx0) / (nx * 256)
    py = (my1 - my0) / (ny * 256)                # negative (north-up)
    return rgb, Affine(px, 0.0, mx0, 0.0, py, my0)


def warp_to_grid(rgb: np.ndarray, src_transform, dst_crs: str, dst_transform,
                 dst_shape: tuple[int, int], src_crs: str = WEB_MERCATOR) -> np.ndarray:
    """Reproject the mosaic onto the mask's grid so the pixels line up."""
    from affine import Affine
    from rasterio.warp import Resampling, reproject

    src = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))   # bands-first
    out = np.zeros((3, dst_shape[0], dst_shape[1]), np.uint8)
    reproject(
        src, out,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=Affine(*list(dst_transform)[:6]), dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    return np.transpose(out, (1, 2, 0))


def build_pairs(aoi: str, bbox: tuple[float, float, float, float], out_dir: str | Path,
                resolution_m: float = 0.5, tile_size: int = 512, zoom: int = 18,
                min_road_fraction: float = 0.005,
                tile_fetcher: Callable[[int, int, int], bytes] = _default_tile_fetcher,
                interim_dir: str | Path = "data/interim",
                raw_dir: str | Path = "data/raw") -> int:
    """Build aligned (sat, mask) tile pairs for one AOI. Returns #pairs kept."""
    from PIL import Image

    cfg = MaskConfig(aoi=aoi, resolution_m=resolution_m, tile_size=tile_size,
                     interim_dir=Path(interim_dir), raw_dir=Path(raw_dir))
    manifest = build_aoi_masks(cfg, bbox)                       # P0/A3: OSM mask + grid
    mask = (np.asarray(Image.open(Path(interim_dir) / f"{aoi}_mask.png").convert("L")) > 0).astype(np.uint8)

    rgb, src_transform = fetch_imagery_mosaic(bbox, zoom, tile_fetcher)
    sat = warp_to_grid(rgb, src_transform, manifest["crs"], manifest["transform"],
                       (manifest["height"], manifest["width"]))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mask_tiles = tile_array(mask, tile_size)
    sat_tiles = tile_array(sat, tile_size)
    kept = 0
    for mt, st in zip(mask_tiles, sat_tiles):
        if float(mt.data.mean()) < min_road_fraction:          # skip near-empty tiles
            continue
        stem = f"{aoi}_r{mt.row}_c{mt.col}"
        Image.fromarray(st.data).save(out / f"{stem}_sat.jpg", quality=92)
        # DeepGlobe convention is 0/255 (readers threshold at >127), not {0,1}.
        Image.fromarray((mt.data * 255).astype(np.uint8), mode="L").save(out / f"{stem}_mask.png")
        kept += 1
    print(f"[{aoi}] kept {kept}/{len(mask_tiles)} road-bearing pairs -> {out}")
    return kept


def main() -> None:
    p = argparse.ArgumentParser(description="A6: build OSM-labelled fine-tune pairs (image+mask).")
    p.add_argument("--aoi", help="single AOI id (with --bbox); omit to build a --cities set")
    p.add_argument("--bbox", help="west,south,east,north (lon/lat) for a single --aoi")
    p.add_argument("--cities", choices=["eval", "corpus"], default="eval",
                   help="eval = 5 held-out AOIs (data/finetune); "
                        "corpus = the disjoint Indian OSM corpus for A16a (data/raw/indian_corpus)")
    p.add_argument("--out-dir", default=None, help="override; otherwise defaults by --cities")
    p.add_argument("--resolution-m", type=float, default=0.5)
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--zoom", type=int, default=18, help="Esri XYZ zoom (~0.6 m/px at z18)")
    p.add_argument("--min-road-fraction", type=float, default=0.005)
    args = p.parse_args()

    if args.aoi:
        cities = {args.aoi: tuple(float(v) for v in args.bbox.split(","))}
        out_dir = args.out_dir or "data/finetune"
        label = args.aoi
    elif args.cities == "corpus":
        cities = CORPUS_CITIES                       # weak labels → SEPARATE dir, never data/finetune
        out_dir = args.out_dir or "data/raw/indian_corpus"
        label = "Indian OSM corpus (A16a, weak labels — pretrain/self-train only)"
    else:
        cities = DEFAULT_CITIES
        out_dir = args.out_dir or "data/finetune"
        label = "held-out eval set"

    total, built, skipped, failed = 0, 0, 0, []
    for aoi, bbox in cities.items():
        if list(Path(out_dir).glob(f"{aoi}_r*_sat.jpg")):     # resume: this city is already built
            print(f"[{aoi}] already present -> skip", flush=True)
            skipped += 1
            continue
        try:
            total += build_pairs(aoi, bbox, out_dir, resolution_m=args.resolution_m,
                                 tile_size=args.tile_size, zoom=args.zoom,
                                 min_road_fraction=args.min_road_fraction)
            built += 1
        except Exception as exc:                              # one city's failure must not abort the rest
            print(f"[{aoi}] FAILED ({type(exc).__name__}: {exc}) -> skipping", flush=True)
            failed.append(aoi)
    print(f"\n{label}: {total} pairs | built {built}, skipped(done) {skipped}, "
          f"failed {len(failed)} of {len(cities)} AOIs -> {out_dir}")
    if failed:
        print(f"  failed AOIs (re-run to retry — completed cities are skipped): {', '.join(failed)}")


if __name__ == "__main__":
    main()
