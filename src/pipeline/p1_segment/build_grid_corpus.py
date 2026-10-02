"""A50 data prep: an Indian city's (satellite, road-mask) corpus on a 1024 m grid.

Scripted, corrected version of a manual QGIS tiling project (built first for
Greater Mumbai): a 1024 m grid over a city boundary, with imagery and road labels
from a local XYZ tile source (an imagery layer and a road-label layer whose alpha
channel marks roads). The two URL templates live only in the local, ignored
``data/raw/grid_sources.json``; they are never committed.

1. **Grid** -- 1024 m cells in the city's local UTM zone, anchored at the
   boundary's top-left corner and intersecting it (529 cells for Greater Mumbai),
   rendered at 0.5 m/px plus a pad so centrelines do not fray at cell edges.
   The boundary is ``data/raw/<city>_grid/boundary.gpkg`` (fetched once from OSM
   with ``--osm-boundary`` when missing).
2. **Pinned zoom** -- both layers are fetched at one fixed XYZ zoom and warped
   onto the cell grid (QGIS picked a zoom per layer from the output scale).
3. **Clean mask** -- the road mask is the road layer's **alpha** channel (the old
   ``band1 < 250`` rule caught anti-alias haze and any non-white feature).
4. **Registration** -- v3.2 runs on the imagery and the label centrelines are
   moved by the integer offset (<= ``SHIFT_PX``) that maximises v3.2 probability
   under them; cells whose best agreement is too low are dropped.
5. **Width** -- the label layer's drawn stroke is skeletonised and re-buffered
   to ~6 m, the SpaceNet label convention (``build_spacenet_data``, buffer_m=6).
6. **Leakage** -- tiles within ``EXCLUDE_BUFFER_M`` of ANY SpaceNet-5 Mumbai chip
   or held-out Indian eval AOI are dropped, so no corpus pixel is near a test,
   validation or selection area.
7. **Filtering** -- tiles under 10% land or 0.5% road are dropped.

``--check`` scores the label centrelines *as if they were a prediction* on the
127 held-out chips (common-unit chip APLS vs SpaceNet vector GT, paired with
v3.2): it measures whether these labels agree with what the promotion gate grades.

Provenance: the source's terms restrict redistribution, so everything stays local
under ignored ``data/raw/`` and is never committed.

``--min-agreement 0`` keeps every cell: use it for test-only cities, where dropping
cells v3.2 disagrees with would bias the test toward v3.2.

    python -m src.pipeline.p1_segment.build_grid_corpus --check --v32 models/road_pan.pt
    python -m src.pipeline.p1_segment.build_grid_corpus --city mumbai --device cuda
    python -m src.pipeline.p1_segment.build_grid_corpus --city kolkata --device cuda \\
        --osm-boundary "Kolkata, West Bengal, India" --min-agreement 0
"""
from __future__ import annotations

import argparse
import datetime
import io
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.pipeline.p1_segment.build_finetune_data import (
    _default_tile_fetcher,
    deg2num,
    fetch_imagery_mosaic,
    warp_to_grid,
)

UTM = "EPSG:32643"
ZOOM = 19                 # ~0.28 m/px at Mumbai; both layers downsampled to GSD_M
GSD_M = 0.5               # v3.2's training/deployed resolution
CELL_M = 1024             # one grid cell = 2048 px
CELL_PX = int(CELL_M / GSD_M)
PAD_PX = 32               # rendered around each cell so skeletons don't fray
TILE_PX = 512
ROAD_HALF_WIDTH_PX = 6    # 13 px ~ 6.5 m, matching SpaceNet's 6 m buffered centrelines
SHIFT_PX = 10             # registration search radius (5 m)
EXCLUDE_BUFFER_M = 256.0  # one tile width around every SpaceNet chip
MIN_LAND = 0.10
MIN_ROAD = 0.005
FETCH_THREADS = 4         # 8 drew bursts of connection resets on a full-city build


def load_sources(path: Path) -> tuple[str, str]:
    """``(imagery_url, roads_url)`` XYZ templates from the local, ignored sources file."""
    if not Path(path).is_file():
        raise SystemExit(f"missing {path}: a local JSON with 'imagery_url' and 'roads_url' "
                         "XYZ templates ({z}/{x}/{y}); the road layer must mark roads in its "
                         "alpha channel")
    data = json.loads(Path(path).read_text())
    return data["imagery_url"], data["roads_url"]


def cached_fetcher(url: str, cache_dir: Path):
    """XYZ fetcher with an on-disk tile cache: a re-run never re-downloads a tile.

    Only bytes that decode as an image are cached, and each write is atomic, so a
    killed run or an error page can never poison the cache for later resumes."""
    def fetch(z: int, x: int, y: int) -> bytes:
        from PIL import Image

        path = cache_dir / str(z) / str(x) / f"{y}.bin"
        if path.is_file():
            data = path.read_bytes()
            try:
                Image.open(io.BytesIO(data)).verify()
                return data
            except Exception:          # corrupt/stale entry (e.g. pre-atomic cache): re-fetch
                path.unlink(missing_ok=True)
        data = _default_tile_fetcher(z, x, y, url=url)
        Image.open(io.BytesIO(data)).verify()          # raises on a non-image body
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return data
    return fetch


def render(bbox_lonlat, fetch, mode: str, dst_crs, dst_transform, shape) -> np.ndarray:
    """Fetch the ``ZOOM`` tiles covering ``bbox_lonlat`` and warp them onto a target grid."""
    west, south, east, north = bbox_lonlat
    x0, y0 = (int(math.floor(v)) for v in deg2num(north, west, ZOOM))
    x1, y1 = (int(math.floor(v)) for v in deg2num(south, east, ZOOM))
    tiles = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    with ThreadPoolExecutor(FETCH_THREADS) as pool:     # warm the cache in parallel
        list(pool.map(lambda xy: fetch(ZOOM, *xy), tiles))
    img, src_transform = fetch_imagery_mosaic(bbox_lonlat, ZOOM, fetch, mode=mode)
    return warp_to_grid(img, src_transform, dst_crs, dst_transform, shape)


def grid_cells(boundary, cell_m: float = CELL_M) -> list[tuple[int, int, object]]:
    """``(row, col, box)`` for cells anchored at the boundary's top-left that intersect it."""
    from shapely.geometry import box

    left, bottom, right, top = boundary.bounds
    cells = []
    for row in range(math.ceil((top - bottom) / cell_m)):
        for col in range(math.ceil((right - left) / cell_m)):
            cell = box(left + col * cell_m, top - (row + 1) * cell_m,
                       left + (col + 1) * cell_m, top - row * cell_m)
            if cell.intersects(boundary):
                cells.append((row, col, cell))
    return cells


def exclusion_zone(rgb_dir: Path, crs: str = UTM, buffer_m: float = EXCLUDE_BUFFER_M):
    """Buffered union of every SpaceNet chip footprint and every held-out Indian eval
    AOI (``build_finetune_data.DEFAULT_CITIES``), in ``crs``. Fails loud without the
    chips: a silently empty zone would leak the benchmark into training."""
    import rasterio
    from rasterio.warp import transform_bounds
    from shapely.geometry import box
    from shapely.ops import unary_union

    from src.pipeline.p1_segment.build_finetune_data import DEFAULT_CITIES

    tifs = sorted(Path(rgb_dir).glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"no SpaceNet chips in {rgb_dir}; refusing to build without the leakage guard")
    zones = []
    for tif in tifs:
        with rasterio.open(tif) as src:
            zones.append(box(*transform_bounds(src.crs, crs, *src.bounds)).buffer(buffer_m))
    for bbox in DEFAULT_CITIES.values():
        zones.append(box(*transform_bounds("EPSG:4326", crs, *bbox)).buffer(buffer_m))
    return unary_union(zones), len(tifs)


def best_shift(prob: np.ndarray, skeleton: np.ndarray, radius: int = SHIFT_PX):
    """Integer ``(dy, dx)`` that moves the centrelines onto the model's roads.

    Each shift is scored by the mean probability under the shifted skeleton;
    pixels shifted out of the window count as 0, so pushing hard-to-see roads
    off the edge can never raise the score.
    Returns ``(dy, dx, score, zero_shift_score)``; ties keep the zero shift.
    """
    rows, cols = np.nonzero(skeleton)
    if rows.size == 0:
        return 0, 0, 0.0, 0.0
    h, w = prob.shape
    zero = float(prob[rows, cols].sum(dtype=np.float64)) / rows.size
    best = (0, 0, zero)
    for dy in range(-radius, radius + 1):
        r = rows + dy
        for dx in range(-radius, radius + 1):
            c = cols + dx
            ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            score = float(prob[r[ok], c[ok]].sum(dtype=np.float64)) / rows.size
            if score > best[2] + 1e-9:
                best = (dy, dx, score)
    return best[0], best[1], best[2], zero


def shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate a 2-D mask by ``(dy, dx)``; pixels leaving the frame are dropped."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        mask[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def rebuffer(skeleton: np.ndarray, half_width_px: int = ROAD_HALF_WIDTH_PX) -> np.ndarray:
    """Centrelines -> uniform-width road mask (0/1 uint8)."""
    import cv2

    size = 2 * half_width_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(skeleton.astype(np.uint8), kernel)


def tile_drop_reason(tile_box, boundary, exclusion, mask: np.ndarray) -> str | None:
    """Why a tile is not written (``None`` = keep). Leakage is checked first."""
    if tile_box.intersects(exclusion):
        return "heldout"
    if tile_box.intersection(boundary).area / tile_box.area < MIN_LAND:
        return "land"
    if float(mask.mean()) < MIN_ROAD:
        return "road"
    return None


def _centerlines_geojson(skeleton: np.ndarray, transform, crs: str = UTM) -> dict:
    """Skeleton -> WGS84 LineString FeatureCollection (graph labels for A18-style work)."""
    from src.pipeline.p2_graph.skeleton_graph import reproject_graph_to_wgs84, skeleton_to_graph

    graph = skeleton_to_graph(skeleton, transform=transform)
    reproject_graph_to_wgs84(graph, crs)
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"length_m": round(float(d["length_m"]), 2)},
         "geometry": {"type": "LineString", "coordinates": d["geometry"]}}
        for _, _, d in graph.edges(data=True)]}


def build_cell(row: int, col: int, cell, boundary, exclusion, fetch_sat, fetch_roads,
               model, device: str, root: Path, min_agreement: float,
               crs: str = UTM, prefix: str = "mumbai") -> dict:
    """Render, register, re-label and tile one grid cell. Returns its manifest record.

    ``min_agreement <= 0`` keeps every cell (test cities: dropping cells where v3.2
    disagrees would bias the test toward v3.2)."""
    from affine import Affine
    from PIL import Image
    from rasterio.warp import transform_bounds
    from shapely.geometry import box
    from skimage.morphology import skeletonize

    from src.pipeline.p1_segment.model import predict_large_prob
    from src.pipeline.p1_segment.osm_mask import tile_array

    left, bottom, right, top = cell.bounds
    record = {"row": row, "col": col, "min_agreement": min_agreement,
              "land_fraction": round(cell.intersection(boundary).area / cell.area, 4)}
    tile_m = TILE_PX * GSD_M
    n = CELL_PX // TILE_PX
    boxes = {(r, c): box(left + c * tile_m, top - (r + 1) * tile_m,
                         left + (c + 1) * tile_m, top - r * tile_m)
             for r in range(n) for c in range(n)}
    # Geometry-only reasons first: a cell with no eligible tile is never downloaded.
    reasons = {rc: tile_drop_reason(b, boundary, exclusion, np.ones(1)) for rc, b in boxes.items()}
    if all(reasons.values()):
        return {**record, "status": "ineligible", "kept": 0}

    pad_m = PAD_PX * GSD_M
    size = CELL_PX + 2 * PAD_PX
    padded = Affine(GSD_M, 0.0, left - pad_m, 0.0, -GSD_M, top + pad_m)
    bbox = transform_bounds(crs, "EPSG:4326", left - pad_m, bottom - pad_m,
                            right + pad_m, top + pad_m)
    sat = render(bbox, fetch_sat, "RGB", crs, padded, (size, size))
    skeleton = skeletonize(render(bbox, fetch_roads, "RGBA", crs, padded, (size, size))[..., 3] >= 128)
    if not skeleton.any():
        return {**record, "status": "no_roads", "kept": 0}

    # ponytail: register on the road-densest half-cell window (9 v3.2 windows, not 36);
    # one rigid shift per cell -- per-tile shifts if cells prove non-rigid.
    win = CELL_PX // 2
    starts = range(PAD_PX, PAD_PX + CELL_PX - win + 1, win // 2)
    y0, x0 = max(((y, x) for y in starts for x in starts),
                 key=lambda s: skeleton[s[0]:s[0] + win, s[1]:s[1] + win].sum())
    prob = predict_large_prob(model, sat[y0:y0 + win, x0:x0 + win], tile_size=512, stride=384,
                              device=device, tta=False)
    dy, dx, score, score0 = best_shift(prob, skeleton[y0:y0 + win, x0:x0 + win])
    record.update({"shift_px": [dy, dx], "agreement": round(score, 4),
                   "agreement_unshifted": round(score0, 4)})
    if min_agreement > 0 and (score < min_agreement or max(abs(dy), abs(dx)) == SHIFT_PX):
        return {**record, "status": "low_agreement", "kept": 0}

    crop = slice(PAD_PX, PAD_PX + CELL_PX)
    skeleton = shift_mask(skeleton, dy, dx)[crop, crop]
    mask, sat = rebuffer(skeleton), sat[crop, crop]
    stem = f"{prefix}_{row:02d}_{col:02d}"
    pairs = root / "dg_format"
    pairs.mkdir(parents=True, exist_ok=True)
    dropped = {"heldout": 0, "land": 0, "road": 0}
    keep = np.zeros_like(skeleton)            # centrelines only where tiles survive
    for st, mt in zip(tile_array(sat, TILE_PX), tile_array(mask, TILE_PX)):
        reason = tile_drop_reason(boxes[mt.row, mt.col], boundary, exclusion, mt.data)
        if reason:
            dropped[reason] += 1
            continue
        name = f"{stem}_r{mt.row}_c{mt.col}"
        Image.fromarray(st.data).save(pairs / f"{name}_sat.jpg", quality=92)
        Image.fromarray((mt.data * 255).astype(np.uint8), mode="L").save(pairs / f"{name}_mask.png")
        keep[mt.y0:mt.y0 + TILE_PX, mt.x0:mt.x0 + TILE_PX] = True
    kept = int(keep.sum() // TILE_PX ** 2)
    if kept:   # the leakage guard applies to graph labels too
        lines = root / "centerlines" / f"{stem}.geojson"
        lines.parent.mkdir(parents=True, exist_ok=True)
        lines.write_text(json.dumps(_centerlines_geojson(
            skeleton & keep, Affine(GSD_M, 0.0, left, 0.0, -GSD_M, top), crs)))
    return {**record, "status": "ok", "kept": kept, "dropped": dropped}


def check_agreement(fetch_roads, v32: Path | None, device: str = "cpu",
                    n_chips: int | None = None, n_samples: int | None = None) -> dict:
    """Score label centrelines as a prediction on the held-out chips (common-unit APLS)."""
    import cv2
    import rasterio

    from src.pipeline.p1_segment import chip_apls_eval as ce
    from src.pipeline.p1_segment.stats import paired_bootstrap_ci

    n_samples = n_samples or ce.GATE_N_SAMPLES
    chips = ce._heldout_chips(n_chips, seed=7)
    ce._preflight_chip_inputs(chips)
    model = thr = None
    if v32 is not None:
        from src.pipeline.p1_segment.model import load_checkpoint
        model, meta = load_checkpoint(v32, map_location=device)
        model.to(device).eval()
        thr = float(meta.get("threshold", 0.44))

    labels, v32_scores, ceilings, per_chip = {}, {}, {}, {}
    for chip in chips:
        bands, transform, eff_x, eff_y = ce._read_chip(chip)
        gt = ce.adj_to_apls_graph(ce._geojson_adj(chip, transform), eff_x, eff_y)
        if gt.number_of_nodes() < 2:
            continue
        with rasterio.open(ce.SRC_RGB / f"SN5_roads_train_AOI_8_Mumbai_PS-RGB_{chip}.tif") as src:
            crs, bounds, shape = src.crs, src.bounds, src.shape
        from rasterio.warp import transform_bounds
        alpha = render(transform_bounds(crs, "EPSG:4326", *bounds), fetch_roads, "RGBA",
                       crs, transform, shape)[..., 3]
        mask = cv2.resize((alpha >= 128).astype(np.uint8), (ce.IMAGE_SIZE, ce.IMAGE_SIZE),
                          interpolation=cv2.INTER_NEAREST)
        ceilings[chip] = float(ce.chip_apls(gt, gt, n_samples))
        labels[chip] = float(ce.chip_apls(ce.mask_to_apls_graph_aniso(mask, eff_x, eff_y), gt, n_samples))
        per_chip[chip] = {"ceiling": ceilings[chip], "labels": labels[chip]}
        if model is not None:
            pred = ce.v32_chip_graph(model, bands, thr, eff_x, eff_y, device)
            v32_scores[chip] = float(ce.chip_apls(pred, gt, n_samples))
            per_chip[chip]["v32"] = v32_scores[chip]
        print(f"  {chip}: labels={labels[chip]:.3f} v32={per_chip[chip].get('v32', float('nan')):.3f} "
              f"ceiling={ceilings[chip]:.3f}", flush=True)

    report = {
        "protocol": "a50-label-agreement-v1 (common-unit chip APLS vs SpaceNet GT)",
        "zoom": ZOOM, "n_samples": n_samples,
        "n_scored": len(labels),
        "ceiling_mean": float(np.mean(list(ceilings.values()))),
        "labels_apls_mean": float(np.mean(list(labels.values()))),
        "labels_apls_norm_mean": ce._norm_mean(labels, ceilings),
        "per_chip": per_chip,
    }
    if v32_scores:
        common = sorted(v32_scores)
        ci = paired_bootstrap_ci([v32_scores[c] for c in common], [labels[c] for c in common])
        report.update({
            "v32_apls_mean": float(np.mean(list(v32_scores.values()))),
            "v32_apls_norm_mean": ce._norm_mean(v32_scores, ceilings),
            "labels_minus_v32": {"delta": ci.delta, "ci_low": ci.ci_low, "ci_high": ci.ci_high,
                                 "excludes_zero": ci.excludes_zero, "verdict": ci.verdict},
        })
        print(f"  grid labels vs v3.2: {ci.summary()}", flush=True)
    return report


PROVENANCE_KEYS = ("imagery_url", "roads_url", "zoom", "gsd_m", "cell_m", "crs",
                   "registration_model", "min_agreement", "exclude_buffer_m",
                   "min_land", "min_road", "road_width_px")


def record_provenance(path: Path, settings: dict) -> None:
    """Write the build settings, refusing to resume a root built with different ones.

    The tile cache and ``cells.jsonl`` are keyed only by position, so resuming
    under changed sources/parameters would silently mix two corpora."""
    if path.is_file():
        old = json.loads(path.read_text())
        changed = [k for k in PROVENANCE_KEYS if old.get(k) != settings.get(k)]
        if changed:
            raise SystemExit(f"{path} was built with different settings ({', '.join(changed)}); "
                             "use a new --root or remove the old outputs and cache")
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    path.write_text(json.dumps({**settings, "updated_utc": stamp}, indent=2))


def load_boundary(path: Path, osm_query: str | None = None):
    """``(boundary geometry, its local UTM CRS)``; fetched once from OSM if missing."""
    import geopandas as gpd

    path = Path(path)
    if not path.is_file():
        if not osm_query:
            raise SystemExit(f"missing {path}: pass --osm-boundary '<city, state, India>' to fetch it")
        import osmnx as ox

        path.parent.mkdir(parents=True, exist_ok=True)
        ox.geocode_to_gdf(osm_query)[["geometry"]].to_file(path, driver="GPKG")
    gdf = gpd.read_file(path)
    crs = gdf.estimate_utm_crs().to_string()
    return gdf.to_crs(crs).geometry.unary_union, crs


def main() -> None:
    p = argparse.ArgumentParser(description="A50: 1024 m grid corpus for an Indian city.")
    p.add_argument("--check", action="store_true",
                   help="score grid labels vs SpaceNet GT on the 127 held-out chips; build nothing")
    p.add_argument("--city", default="mumbai", help="city id: output data/raw/<city>_grid, file prefix")
    p.add_argument("--osm-boundary", default=None,
                   help="OSM geocoder query used once when <root>/boundary.gpkg is missing")
    p.add_argument("--v32", default="models/road_pan.pt", help="checkpoint for registration / check")
    p.add_argument("--device", default="cpu")
    p.add_argument("--root", default=None, help="output root (default data/raw/<city>_grid, ignored)")
    p.add_argument("--sources", default="data/raw/grid_sources.json", help="local XYZ sources JSON")
    p.add_argument("--boundary", default=None, help="boundary file (default <root>/boundary.gpkg)")
    p.add_argument("--cells", nargs="*", default=None, help="only these 'row,col' cells (pilot)")
    p.add_argument("--min-agreement", type=float, default=0.2,
                   help="drop a cell whose best registered v3.2 agreement is below this; "
                        "0 keeps every cell (use for test-only cities)")
    p.add_argument("--n-chips", type=int, default=None, help="--check: subsample chips")
    p.add_argument("--n-samples", type=int, default=None, help="--check: APLS samples per chip")
    p.add_argument("--out", default=".tmp/a50_label_check.json", help="--check report")
    args = p.parse_args()

    root = Path(args.root or f"data/raw/{args.city}_grid")
    imagery_url, roads_url = load_sources(Path(args.sources))
    fetch_roads = cached_fetcher(roads_url, root / "cache" / "roads")
    if args.check:
        from src.pipeline.p1_segment.chip_apls_eval import _write_report
        report = check_agreement(fetch_roads, Path(args.v32) if args.v32 else None,
                                 args.device, args.n_chips, args.n_samples)
        _write_report(args.out, report)
        print(f"-> {args.out}", flush=True)
        return

    from src.pipeline.p1_segment.chip_apls_eval import SRC_RGB
    from src.pipeline.p1_segment.model import load_checkpoint

    boundary, crs = load_boundary(Path(args.boundary or root / "boundary.gpkg"), args.osm_boundary)
    exclusion, n_chips = exclusion_zone(SRC_RGB, crs)
    cells = grid_cells(boundary)
    if args.cells:
        wanted = {tuple(int(v) for v in rc.split(",")) for rc in args.cells}
        cells = [c for c in cells if (c[0], c[1]) in wanted]
    model, _ = load_checkpoint(args.v32, map_location=args.device)
    model.to(args.device).eval()
    fetch_sat = cached_fetcher(imagery_url, root / "cache" / "sat")

    root.mkdir(parents=True, exist_ok=True)
    record_provenance(root / "provenance.json", {     # local-only, ignored
        "task": "A50", "city": args.city, "imagery_url": imagery_url, "roads_url": roads_url,
        "zoom": ZOOM, "gsd_m": GSD_M, "cell_m": CELL_M, "crs": crs,
        "registration_model": Path(args.v32).name,
        "min_agreement": args.min_agreement, "spacenet_chips_excluded": n_chips,
        "exclude_buffer_m": EXCLUDE_BUFFER_M, "min_land": MIN_LAND, "min_road": MIN_ROAD,
        "road_width_px": 2 * ROAD_HALF_WIDTH_PX + 1,
    })
    manifest = root / "cells.jsonl"
    done = set()
    if manifest.is_file():     # resume: every recorded non-failed cell is final
        for line in manifest.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:         # torn last line from a killed run: rebuild that cell
                continue
            if rec["status"] != "failed":
                done.add((rec["row"], rec["col"]))

    todo = [c for c in cells if (c[0], c[1]) not in done]
    print(f"A50 {args.city} ({crs}): {len(cells)} cells, {len(done)} done, {len(todo)} to build "
          f"({n_chips} SpaceNet chips + held-out eval AOIs excluded)", flush=True)
    failed = 0
    for i, (row, col, cell) in enumerate(todo, 1):
        try:
            rec = build_cell(row, col, cell, boundary, exclusion, fetch_sat, fetch_roads,
                             model, args.device, root, args.min_agreement, crs, args.city)
        except Exception as exc:          # one cell's failure must not abort the city
            rec = {"row": row, "col": col, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            failed += 1
        rec["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        with manifest.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{i}/{len(todo)}] r{row} c{col}: {rec['status']} kept={rec.get('kept', 0)} "
              f"shift={rec.get('shift_px')} agree={rec.get('agreement')}", flush=True)
    if todo and failed / len(todo) > 0.5:
        raise SystemExit(f"{failed}/{len(todo)} cells failed -- likely blocked or offline; re-run to resume")


if __name__ == "__main__":
    main()
