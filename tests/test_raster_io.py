"""Tests for A26 georeferenced/PAN inference reader (`p1_segment/raster_io.py`)."""
import json
import numpy as np
import pytest
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
    m = write_manifest("mytile", tmp_path, t, "EPSG:4326", width=40, height=32)
    meta = json.loads(m.read_text())
    assert meta["crs"] == "EPSG:4326"
    assert (meta["width"], meta["height"]) == (40, 32)
    assert Affine(*meta["transform"]) == t         # exactly how P2's _load_alignment rebuilds it
    assert write_manifest("x", tmp_path, None, None, width=40, height=32) is None


def test_rotated_manifest_has_nonzero_scale_and_loads_in_p2(tmp_path):
    from affine import Affine

    from src.pipeline.p2_graph.build_graph import _load_alignment

    transform = Affine(0, -1, 100, 1, 0, 200)
    manifest = write_manifest(
        "rotated", tmp_path, transform, "EPSG:32643", width=40, height=32
    )
    payload = json.loads(manifest.read_text())
    loaded, crs = _load_alignment(manifest, (32, 40))
    assert payload["resolution_m"] == pytest.approx(1.0)
    assert loaded == transform and crs == "EPSG:32643"
