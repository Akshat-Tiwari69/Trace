"""Offline unit tests for the OSM→mask helpers (task A3).

These touch only the rasterise + tiling logic, so they run without any network
(no Overpass call). The osmnx fetch path is exercised manually via the CLI and
QC'd on a real tile (A3 done-criteria).
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from src.pipeline.p1_segment.osm_mask import (
    build_grid,
    rasterize_roads,
    tile_array,
)

# A small AOI near Panaji, Goa (west, south, east, north).
BBOX = (73.80, 15.47, 73.82, 15.49)


# --------------------------------------------------------------------------- #
# tile_array
# --------------------------------------------------------------------------- #
def test_tile_array_exact_multiple():
    arr = np.ones((512, 512), dtype=np.uint8)
    tiles = tile_array(arr, 256)
    assert len(tiles) == 4  # 2×2
    assert all(t.data.shape == (256, 256) for t in tiles)
    assert {(t.row, t.col) for t in tiles} == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_tile_array_pads_ragged_edges():
    arr = np.ones((300, 300), dtype=np.uint8)
    tiles = tile_array(arr, 256)
    assert len(tiles) == 4  # still 2×2, last row/col padded
    for t in tiles:
        assert t.data.shape == (256, 256)
    # bottom-right tile is mostly pad (zeros): only 44×44 real pixels
    br = next(t for t in tiles if (t.row, t.col) == (1, 1))
    assert br.data.sum() == 44 * 44


def test_tile_array_preserves_offsets():
    arr = np.arange(300 * 300, dtype=np.int32).reshape(300, 300)
    tiles = tile_array(arr, 256)
    top_left = next(t for t in tiles if (t.row, t.col) == (0, 0))
    assert top_left.y0 == 0 and top_left.x0 == 0
    assert top_left.data[0, 0] == arr[0, 0]
    second_col = next(t for t in tiles if (t.row, t.col) == (0, 1))
    assert second_col.x0 == 256


def test_tile_array_rejects_bad_size():
    with pytest.raises(ValueError):
        tile_array(np.zeros((10, 10)), 0)


# --------------------------------------------------------------------------- #
# build_grid
# --------------------------------------------------------------------------- #
def test_build_grid_is_metric_and_sized():
    transform, width, height, crs = build_grid(BBOX, resolution_m=1.0)
    assert width > 0 and height > 0
    # ~0.02° ≈ 2.1–2.2 km at this latitude → ~2000+ px at 1 m/px
    assert 1500 < width < 3000
    assert 1500 < height < 3000
    assert crs.is_projected  # UTM, metres
    assert transform.a == 1.0 and transform.e == -1.0  # 1 m/px, north-up


def test_build_grid_resolution_scales_size():
    _, w1, h1, _ = build_grid(BBOX, resolution_m=1.0)
    _, w2, h2, _ = build_grid(BBOX, resolution_m=2.0)
    assert w2 == pytest.approx(w1 / 2, abs=2)
    assert h2 == pytest.approx(h1 / 2, abs=2)


# --------------------------------------------------------------------------- #
# rasterize_roads
# --------------------------------------------------------------------------- #
def _diagonal_road_gdf() -> gpd.GeoDataFrame:
    """One diagonal road spanning the test bbox, in WGS84."""
    west, south, east, north = BBOX
    line = LineString([(west, south), (east, north)])
    return gpd.GeoDataFrame({"geometry": [line]}, crs="EPSG:4326")


def test_rasterize_roads_is_binary_and_right_shape():
    transform, width, height, crs = build_grid(BBOX, resolution_m=2.0)
    mask = rasterize_roads(_diagonal_road_gdf(), transform, width, height, crs, buffer_m=6.0)
    assert mask.shape == (height, width)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})  # strictly binary
    assert mask.sum() > 0  # the road got painted


def test_rasterize_roads_buffer_widens():
    transform, width, height, crs = build_grid(BBOX, resolution_m=2.0)
    thin = rasterize_roads(_diagonal_road_gdf(), transform, width, height, crs, buffer_m=2.0)
    wide = rasterize_roads(_diagonal_road_gdf(), transform, width, height, crs, buffer_m=20.0)
    assert wide.sum() > thin.sum()  # bigger buffer → more road pixels


def test_rasterize_roads_empty_input():
    transform, width, height, crs = build_grid(BBOX, resolution_m=4.0)
    empty = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    mask = rasterize_roads(empty, transform, width, height, crs)
    assert mask.shape == (height, width)
    assert mask.sum() == 0
