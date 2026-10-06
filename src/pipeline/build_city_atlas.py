"""City atlases for the web app's city picker: one analyzed area per city.

Each atlas is the ordinary P2/P3 artifact set (``{aoi}_graph.geojson``,
``_criticality.csv``, ``_resilience.csv``) plus ``{aoi}_atlas.json``, which tells
the app where the road network came from:

- ``--source osm``: OpenStreetMap roads rasterised to a mask (the same path as the
  Panaji sample), then the real P2 graph/healing and P3 analysis. Public data, so
  it is written to the committed ``data/atlas/``.
- ``--source imagery``: the segmentation checkpoint runs on local imagery
  (the ignored grid-corpus source), then the full pipeline. Imagery terms forbid
  redistribution, so it is written to the ignored ``data/atlas_private/`` and is
  only copied to the application host.

Every area is a square around a fixed neighbourhood. Re-running with a newer
checkpoint regenerates the imagery atlases; the record names the model.

    python -m src.pipeline.build_city_atlas --source osm
    python -m src.pipeline.build_city_atlas --source imagery --checkpoint models/road_v4.pt \\
        --release a4-roadseg-v4 --device cuda --area delhi_cp
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

from src.pipeline.p1_segment.build_finetune_data import CORPUS_CITIES, DEFAULT_CITIES

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DIR = ROOT / "data" / "atlas"
PRIVATE_DIR = ROOT / "data" / "atlas_private"
SIZE_M = 2000.0
OSM_RESOLUTION_M = 2.0      # the Panaji sample's grid
IMAGERY_RESOLUTION_M = 0.5  # the segmentation model's training resolution

# area -> (label, ISO 3166-2 region, seen by the training corpus?). Bounding boxes
# come from build_finetune_data. The grid corpus excludes DEFAULT_CITIES areas and
# Kolkata/Hyderabad are test-only cities, so those areas are unseen by the model.
AREAS: dict[str, tuple[str, str, bool]] = {
    "mumbai_bandra": ("Bandra, Mumbai", "IN-MH", False),
    "bengaluru_indiranagar": ("Indiranagar, Bengaluru", "IN-KA", False),
    "delhi_cp": ("Connaught Place, Delhi", "IN-DL", False),
    "kolkata_park_st": ("Park Street, Kolkata", "IN-WB", False),
    "hyderabad_banjara": ("Banjara Hills, Hyderabad", "IN-TG", False),
    "chennai_tnagar": ("T. Nagar, Chennai", "IN-TN", True),
    "ahmedabad_navrangpura": ("Navrangpura, Ahmedabad", "IN-GJ", True),
    "jaipur_cscheme": ("C-Scheme, Jaipur", "IN-RJ", True),
    "lucknow_hazratganj": ("Hazratganj, Lucknow", "IN-UP", True),
    "pune_shivajinagar": ("Shivajinagar, Pune", "IN-MH", True),
    "bhubaneswar_saheednagar": ("Saheed Nagar, Bhubaneswar", "IN-OR", True),
    "patna_boring": ("Boring Road, Patna", "IN-BR", True),
    "guwahati_paltanbazaar": ("Paltan Bazaar, Guwahati", "IN-AS", True),
}


def square_bbox(bbox: tuple[float, float, float, float], size_m: float = SIZE_M) -> tuple[float, float, float, float]:
    """A ``size_m`` square (lon/lat) centred on ``bbox``."""
    west, south, east, north = bbox
    lon, lat = (west + east) / 2, (south + north) / 2
    half_lat = size_m / 2 / 111_320.0
    half_lon = half_lat / math.cos(math.radians(lat))
    return (round(lon - half_lon, 6), round(lat - half_lat, 6), round(lon + half_lon, 6), round(lat + half_lat, 6))


def area_bbox(area: str, raw_dir: Path = ROOT / "data" / "raw") -> tuple[float, float, float, float]:
    """The area's square: ``SIZE_M``, or smaller when its cached OSM extract is
    smaller (some held-out areas), so the square always lies inside the extract.
    Both sources use this, so an area's OSM and imagery atlases cover the same ground."""
    meta = raw_dir / f"{area}_osm_roads.meta.json"
    if not meta.exists():
        return square_bbox({**DEFAULT_CITIES, **CORPUS_CITIES}[area])
    west, south, east, north = json.loads(meta.read_text(encoding="utf-8"))["bbox"]
    side_m = min((north - south) * 111_320.0,
                 (east - west) * 111_320.0 * math.cos(math.radians((south + north) / 2)))
    # Shrink by a metre so float rounding never puts the square outside the extract.
    return square_bbox((west, south, east, north), min(SIZE_M, side_m - 1.0))


def utm_crs(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def atlas_stats(aoi: str, out_dir: Path) -> dict:
    """Headline numbers for the picker, so listing cities never loads a graph."""
    features = json.loads((out_dir / f"{aoi}_graph.geojson").read_text(encoding="utf-8"))["features"]
    kinds = [feature["properties"].get("feature_type") for feature in features]
    with (out_dir / f"{aoi}_criticality.csv").open(newline="", encoding="utf-8") as handle:
        critical = sum(row["is_critical"].strip().lower() == "true" for row in csv.DictReader(handle))
    with (out_dir / f"{aoi}_resilience.csv").open(newline="", encoding="utf-8") as handle:
        first = next(row for row in csv.DictReader(handle) if int(row["n_removed"]) == 1)
    return {
        "node_count": kinds.count("node"),
        "edge_count": kinds.count("edge"),
        "critical_count": critical,
        "worst_single_loss": round(1.0 - float(first["targeted_resilience_index"]), 6),
    }


def write_record(aoi: str, out_dir: Path, record: dict) -> Path:
    """Write ``{aoi}_atlas.json`` for artifacts already in ``out_dir``."""
    record = {
        **record,
        "aoi": aoi,
        **atlas_stats(aoi, out_dir),
        "graph_sha256": _sha256(out_dir / f"{aoi}_graph.geojson"),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
    }
    path = out_dir / f"{aoi}_atlas.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def publish(aoi: str, processed_dir: Path, out_dir: Path, record: dict) -> Path:
    """Copy the P2/P3 artifacts into ``out_dir`` and record them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("_graph.geojson", "_criticality.csv", "_resilience.csv"):
        shutil.copyfile(processed_dir / f"{aoi}{suffix}", out_dir / f"{aoi}{suffix}")
    return write_record(aoi, out_dir, record)


def seed_osm_cache(area: str, aoi: str, bbox: tuple[float, float, float, float],
                   raw_dir: Path = ROOT / "data" / "raw") -> str | None:
    """Clip the area's cached OSM extract to ``bbox`` as ``aoi``'s road cache.

    Every area already has a cached extract for a larger box, so no Overpass
    request is needed. Returns the extract's snapshot date, or None when there
    is no covering extract (``fetch_osm_roads`` then queries Overpass).
    """
    import geopandas as gpd
    from shapely.geometry import box

    source, meta = raw_dir / f"{area}_osm_roads.gpkg", raw_dir / f"{area}_osm_roads.meta.json"
    if not (source.exists() and meta.exists()):
        return None
    signature = json.loads(meta.read_text(encoding="utf-8"))
    west, south, east, north = signature["bbox"]
    if not (west <= bbox[0] and south <= bbox[1] and east >= bbox[2] and north >= bbox[3]):
        return None
    roads = gpd.read_file(source).clip(box(*bbox))
    roads.to_file(raw_dir / f"{aoi}_osm_roads.gpkg", driver="GPKG")
    (raw_dir / f"{aoi}_osm_roads.meta.json").write_text(json.dumps(
        {"bbox": [round(v, 6) for v in bbox], "network_type": signature["network_type"]}))
    return datetime.date.fromtimestamp(source.stat().st_mtime).isoformat()


def build_osm(area: str, out_dir: Path = PUBLIC_DIR) -> Path:
    """OSM roads -> mask -> P2 graph/healing -> P3 analysis (no simulated occlusion)."""
    from src.pipeline.p1_segment.build_dataset import build_aoi_masks
    from src.pipeline.p1_segment.osm_mask import MaskConfig
    from src.pipeline.p2_graph.build_graph import build_graph
    from src.pipeline.p2_graph.config import GraphConfig
    from src.pipeline.p3_analysis.analyze import analyze

    aoi, bbox = f"{area}_osm", area_bbox(area)
    snapshot = seed_osm_cache(area, aoi, bbox)
    build_aoi_masks(MaskConfig(aoi=aoi, resolution_m=OSM_RESOLUTION_M, buffer_m=6.0), bbox)
    cfg = GraphConfig(aoi=aoi, resolution_m=OSM_RESOLUTION_M)
    build_graph(cfg)
    analyze(cfg)
    label, region, _ = AREAS[area]
    return publish(aoi, Path(cfg.processed_dir), out_dir, {
        "area": area, "label": label, "region": region, "bbox": list(bbox),
        "source": "osm", "model": None, "seen_in_training": None,
        "osm_snapshot": snapshot or datetime.date.today().isoformat(),
    })


def build_imagery(area: str, checkpoint: Path, release: str, device: str,
                  sources: Path = ROOT / "data" / "raw" / "grid_sources.json",
                  out_dir: Path = PRIVATE_DIR) -> Path:
    """Local imagery -> GeoTIFF -> the full P1-P3 pipeline with ``checkpoint``."""
    import rasterio
    from affine import Affine
    from rasterio.warp import transform_bounds

    from src.pipeline.p1_segment.build_grid_corpus import cached_fetcher, load_sources, render
    from src.pipeline.run_pipeline import run

    aoi, bbox = f"{area}_imagery", area_bbox(area)
    west, south, east, north = bbox
    crs = utm_crs((west + east) / 2, (south + north) / 2)
    left, bottom, right, top = transform_bounds("EPSG:4326", crs, *bbox)
    size = int(round(min(right - left, top - bottom) / IMAGERY_RESOLUTION_M))
    transform = Affine(IMAGERY_RESOLUTION_M, 0, left, 0, -IMAGERY_RESOLUTION_M, top)
    imagery_url, _ = load_sources(sources)
    fetch = cached_fetcher(imagery_url, ROOT / "data" / "raw" / "atlas_cache" / "sat")
    rgb = render(bbox, fetch, "RGB", crs, transform, (size, size))

    image = ROOT / "data" / "interim" / f"{aoi}.tif"
    image.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(image, "w", driver="GTiff", width=size, height=size, count=3,
                       dtype="uint8", crs=crs, transform=transform) as dst:
        dst.write(rgb.transpose(2, 0, 1))
    summary = run(image, checkpoint, aoi, resolution_m=IMAGERY_RESOLUTION_M, device=device)
    label, region, seen = AREAS[area]
    return publish(aoi, Path(summary["config"]["processed_dir"]), out_dir, {
        "area": area, "label": label, "region": region, "bbox": list(bbox),
        "source": "imagery",
        "model": {"release": release, "checkpoint": checkpoint.name, "sha256": _sha256(checkpoint)},
        "seen_in_training": seen,
    })


def main() -> None:
    p = argparse.ArgumentParser(description="Build city atlases for the web app's city picker.")
    p.add_argument("--source", choices=["osm", "imagery"], required=True)
    p.add_argument("--area", action="append", choices=sorted(AREAS), help="repeatable; default all")
    p.add_argument("--checkpoint", type=Path, help="segmentation checkpoint (imagery)")
    p.add_argument("--release", help="release name recorded with the model (imagery)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.source == "imagery" and not (args.checkpoint and args.release):
        p.error("--source imagery needs --checkpoint and --release")
    for area in args.area or sorted(AREAS):
        path = (build_osm(area) if args.source == "osm"
                else build_imagery(area, args.checkpoint, args.release, args.device))
        print(f"{area}: {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
