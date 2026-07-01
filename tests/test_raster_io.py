"""Tests for A26 georeferenced/PAN inference reader (`p1_segment/raster_io.py`)."""
import json
import numpy as np
from src.pipeline.p1_segment.raster_io import read_image_any, write_manifest


def _geotiff(path, bands, dtype=np.uint16):
    import rasterio
    from rasterio.transform import from_origin
    transform = from_origin(72.8, 19.1, 0.5, 0.5)
    data = np.random.default_rng(0).integers(0, 4000, (bands, 32, 40), dtype=dtype)
    with rasterio.open(path, "w", driver="GTiff", height=32, width=40, count=bands,
                       dtype=dtype, crs="EPSG:4326", transform=transform) as dst:
        dst.write(data)
    return transform


def test_read_geotiff_rgb_keeps_crs_transform(tmp_path):
    p = tmp_path / "rgb.tif"; _geotiff(p, 3)
    rgb, transform, crs = read_image_any(p)
    assert rgb.shape == (32, 40, 3) and rgb.dtype == np.uint8
    assert crs == "EPSG:4326" and transform is not None


def test_read_geotiff_pan_becomes_grey_3ch(tmp_path):
    p = tmp_path / "pan.tif"; _geotiff(p, 1)
    rgb, _, crs = read_image_any(p)
    assert rgb.shape == (32, 40, 3) and crs == "EPSG:4326"
    assert np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])


def test_non_geotiff_falls_back_to_pixels(tmp_path):
    import cv2
    p = tmp_path / "x.png"; cv2.imwrite(str(p), np.zeros((10, 12, 3), np.uint8))
    rgb, transform, crs = read_image_any(p)
    assert transform is None and crs is None and rgb.shape == (10, 12, 3)


def test_manifest_roundtrips_via_p2_affine(tmp_path):
    from affine import Affine
    from rasterio.transform import from_origin
    t = from_origin(72.8, 19.1, 0.5, 0.5)
    m = write_manifest("mytile", tmp_path, t, "EPSG:4326")
    meta = json.loads(m.read_text())
    assert meta["crs"] == "EPSG:4326"
    assert Affine(*meta["transform"]) == t         # exactly how P2's _load_alignment rebuilds it
    assert write_manifest("x", tmp_path, None, None) is None
