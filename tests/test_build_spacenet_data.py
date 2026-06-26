"""Test the SpaceNet-5 → DeepGlobe-format converter (A16). CPU, synthetic geo."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


def _write_chip(img_dir, lbl_dir, chip: str, with_road: bool, size: int = 512):
    """Write a synthetic SpaceNet-style GeoTIFF (16-bit, UTM 0.3 m) + GeoJSON road."""
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import LineString

    crs = "EPSG:32643"  # UTM 43N (Mumbai)
    res, ox, oy = 0.3, 500_000.0, 2_000_000.0
    transform = from_origin(ox, oy, res, res)
    data = (np.random.rand(3, size, size) * 4000).astype(np.uint16)  # 16-bit imagery
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(img_dir / f"SN5_AOI_8_Mumbai_PS-RGB_{chip}.tif", "w", driver="GTiff",
                       height=size, width=size, count=3, dtype="uint16",
                       crs=crs, transform=transform) as dst:
        dst.write(data)

    if with_road:                                    # a horizontal centreline across the chip
        ymid = oy - (size // 2) * res
        geom = [LineString([(ox + 10 * res, ymid), (ox + (size - 10) * res, ymid)])]
    else:
        geom = []
    gpd.GeoDataFrame({"geometry": geom}, crs=crs).to_file(
        lbl_dir / f"SN5_AOI_8_Mumbai_geojson_roads_speed_{chip}.geojson", driver="GeoJSON")


def test_convert_produces_deepglobe_pairs(tmp_path):
    from src.pipeline.p1_segment.build_spacenet_data import convert_spacenet

    img_dir, lbl_dir, out = tmp_path / "PS-RGB", tmp_path / "geojson_roads_speed", tmp_path / "out"
    _write_chip(img_dir, lbl_dir, "chip1", with_road=True)

    n = convert_spacenet(img_dir, lbl_dir, out, buffer_m=8.0, scale=1.0,
                         tile_size=512, min_road_fraction=0.001)
    assert n >= 1
    sats = sorted(out.glob("*_sat.jpg"))
    masks = sorted(out.glob("*_mask.png"))
    assert len(sats) == len(masks) == n
    # mask is true DeepGlobe 0/255 and actually contains road
    arr = np.asarray(Image.open(masks[0]).convert("L"))
    assert set(np.unique(arr).tolist()) <= {0, 255} and (arr == 255).any()
    # imagery is 8-bit RGB JPG of the tile size
    sat = np.asarray(Image.open(sats[0]).convert("RGB"))
    assert sat.shape == (512, 512, 3) and sat.dtype == np.uint8


def test_road_free_chip_is_skipped(tmp_path):
    from src.pipeline.p1_segment.build_spacenet_data import convert_spacenet

    img_dir, lbl_dir, out = tmp_path / "PS-RGB", tmp_path / "geojson_roads_speed", tmp_path / "out"
    _write_chip(img_dir, lbl_dir, "chip1", with_road=False)
    n = convert_spacenet(img_dir, lbl_dir, out, buffer_m=8.0, scale=1.0, min_road_fraction=0.005)
    assert n == 0


def test_scale_downsamples_to_match_deepglobe(tmp_path):
    """0.3 m → 0.5 m: scale=0.6 shrinks a 512 chip below one 512 tile (→ padded to 1)."""
    from src.pipeline.p1_segment.build_spacenet_data import convert_spacenet

    img_dir, lbl_dir, out = tmp_path / "PS-RGB", tmp_path / "geojson_roads_speed", tmp_path / "out"
    _write_chip(img_dir, lbl_dir, "chip1", with_road=True)
    n = convert_spacenet(img_dir, lbl_dir, out, buffer_m=8.0, scale=0.6, min_road_fraction=0.001)
    assert n == 1                                    # 307×307 → one padded 512 tile
    arr = np.asarray(Image.open(next(out.glob("*_mask.png"))).convert("L"))
    assert (arr == 255).any()


def test_missing_geojson_is_skipped(tmp_path):
    """An image with no matching geojson is skipped, not crashed on."""
    from src.pipeline.p1_segment.build_spacenet_data import convert_spacenet

    img_dir, lbl_dir, out = tmp_path / "PS-RGB", tmp_path / "geojson_roads_speed", tmp_path / "out"
    _write_chip(img_dir, lbl_dir, "chip1", with_road=True)
    # a second image with no label
    _write_chip(img_dir, tmp_path / "_throwaway", "chip2", with_road=True)
    n = convert_spacenet(img_dir, lbl_dir, out, buffer_m=8.0, scale=1.0, min_road_fraction=0.001)
    assert n >= 1                                    # chip1 converted, chip2 (no label) skipped
