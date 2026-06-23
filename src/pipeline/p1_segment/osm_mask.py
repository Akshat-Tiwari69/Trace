"""OSM → road-mask label generation (Phase 0 / task A3).

Pull OpenStreetMap road vectors for an Area of Interest (AOI), rasterise them
onto a *metric* grid aligned to that AOI, and tile the result into fixed-size
binary masks. These masks are the **training labels** for the Phase I
segmentation model — the "zero-manual-labelling" route described in
``docs/Research.md`` → *Dataset Analysis* (osmnx pulls vectors, rasterio burns
them into a raster aligned with the imagery grid).

Output contract (``docs/Tracker.md`` §4):
    ``data/interim/{aoi}_mask.png``   binary {0,1}, aligned to the AOI grid.
We additionally write per-tile masks under ``data/interim/{aoi}/`` plus a
human-readable QC overlay, and a JSON manifest recording the CRS/transform so
the *same* grid can later tile the satellite imagery identically.

Design notes:
- Roads are buffered in metres (real-world road width) before rasterising, so
  thin centrelines survive at any resolution (``docs/Research.md`` recommends a
  3–5 px buffer; ``buffer_m`` makes that resolution-independent).
- The grid is built in an auto-selected UTM zone so distances are in metres and
  ``length_m`` downstream is correct; the AOI bbox defines the extent, not the
  road bounds, so imagery and labels share one grid.
- ``osmnx``/``geopandas`` are imported lazily: the pure-array helpers
  (``tile_array``) and tests stay importable without touching the network.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # import only for type checkers, never at runtime
    import geopandas as gpd
    from affine import Affine


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class MaskConfig:
    """Parameters for one AOI's label-mask build. Config, not hardcoded."""

    aoi: str
    resolution_m: float = 1.0          # ground sampling distance of the grid (m/px)
    buffer_m: float = 6.0              # total road width to paint (~3 px/side @1 m/px)
    tile_size: int = 256               # px; tiles fed to the segmentation model
    network_type: str = "drive"        # osmnx filter: drivable roads
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")


@dataclasses.dataclass
class TileRef:
    """One tile carved out of a full-AOI array."""

    row: int
    col: int
    y0: int       # top pixel offset into the full array
    x0: int       # left pixel offset into the full array
    data: np.ndarray


# --------------------------------------------------------------------------- #
# 1 · Fetch OSM roads (network; cached)
# --------------------------------------------------------------------------- #
def fetch_osm_roads(
    bbox: tuple[float, float, float, float],
    network_type: str = "drive",
    cache_path: Path | None = None,
) -> "gpd.GeoDataFrame":
    """Return drivable OSM road edges within ``bbox`` as a WGS84 GeoDataFrame.

    ``bbox`` is ``(west, south, east, north)`` in lon/lat (standard GIS
    minx,miny,maxx,maxy order). Results are cached to ``cache_path`` (GeoPackage)
    so repeated runs don't re-hit the Overpass API.
    """
    import geopandas as gpd  # lazy: heavy + only needed for the network path

    if cache_path is not None and Path(cache_path).exists():
        return gpd.read_file(cache_path)

    import osmnx as ox

    west, south, east, north = bbox
    # osmnx 1.9.x keyword form (north/south/east/west); retain_all keeps
    # disconnected stubs so the label covers every road in the box.
    graph = ox.graph_from_bbox(
        north=north,
        south=south,
        east=east,
        west=west,
        network_type=network_type,
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
    )
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    roads = edges[["geometry"]].reset_index(drop=True)

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        roads.to_file(cache_path, driver="GPKG")
    return roads


# --------------------------------------------------------------------------- #
# 2 · Build the metric grid from the AOI bbox
# --------------------------------------------------------------------------- #
def build_grid(
    bbox: tuple[float, float, float, float],
    resolution_m: float = 1.0,
) -> tuple["Affine", int, int, "object"]:
    """Build a north-up metric raster grid covering ``bbox``.

    Returns ``(transform, width, height, crs)`` where ``crs`` is the
    auto-selected UTM zone (metres) so ``resolution_m`` is a true ground
    sampling distance. ``bbox`` is ``(west, south, east, north)`` in lon/lat.
    """
    import geopandas as gpd
    from rasterio.transform import from_origin
    from shapely.geometry import box

    west, south, east, north = bbox
    aoi = gpd.GeoSeries([box(west, south, east, north)], crs="EPSG:4326")
    utm = aoi.estimate_utm_crs()
    minx, miny, maxx, maxy = aoi.to_crs(utm).total_bounds

    width = max(1, math.ceil((maxx - minx) / resolution_m))
    height = max(1, math.ceil((maxy - miny) / resolution_m))
    transform = from_origin(minx, maxy, resolution_m, resolution_m)
    return transform, width, height, utm


# --------------------------------------------------------------------------- #
# 3 · Rasterise roads onto the grid
# --------------------------------------------------------------------------- #
def rasterize_roads(
    roads_wgs84: "gpd.GeoDataFrame",
    transform: "Affine",
    width: int,
    height: int,
    crs: "object",
    buffer_m: float = 6.0,
) -> np.ndarray:
    """Burn buffered road geometries into a binary {0,1} ``uint8`` mask.

    Roads (WGS84) are reprojected to the grid ``crs`` (metres) and buffered by
    ``buffer_m / 2`` each side, so centrelines become roads of realistic width.
    """
    from rasterio.features import rasterize

    roads_proj = roads_wgs84.to_crs(crs)
    if buffer_m > 0:
        geoms = roads_proj.geometry.buffer(buffer_m / 2.0)
    else:
        geoms = roads_proj.geometry

    shapes = [(geom, 1) for geom in geoms if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros((height, width), dtype=np.uint8)

    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )
    return (mask > 0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 4 · Tile a full-AOI array (pure numpy — no geo/network deps)
# --------------------------------------------------------------------------- #
def tile_array(
    arr: np.ndarray,
    tile_size: int,
    pad_value: int = 0,
) -> list[TileRef]:
    """Split ``arr`` into ``tile_size``×``tile_size`` tiles, padding the edges.

    The last row/column is zero-padded (``pad_value``) up to ``tile_size`` so
    every tile is exactly square — what the segmentation model expects.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    height, width = arr.shape[:2]
    tiles: list[TileRef] = []
    for row, y0 in enumerate(range(0, height, tile_size)):
        for col, x0 in enumerate(range(0, width, tile_size)):
            patch = arr[y0 : y0 + tile_size, x0 : x0 + tile_size]
            ph, pw = patch.shape[:2]
            if ph < tile_size or pw < tile_size:
                padded = np.full((tile_size, tile_size), pad_value, dtype=arr.dtype)
                padded[:ph, :pw] = patch
                patch = padded
            tiles.append(TileRef(row=row, col=col, y0=y0, x0=x0, data=patch))
    return tiles


# --------------------------------------------------------------------------- #
# 5 · Small IO helpers
# --------------------------------------------------------------------------- #
def save_binary_png(mask01: np.ndarray, path: Path) -> None:
    """Save a {0,1} mask as a PNG holding pixel values 0/1 (per §4 contract)."""
    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask01.astype(np.uint8), mode="L").save(path)


def save_qc_overlay(mask01: np.ndarray, path: Path) -> None:
    """Save a human-viewable version of a {0,1} mask (roads = white 255)."""
    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask01.astype(np.uint8) * 255), mode="L").save(path)
