"""A50 Mumbai grid corpus: offline checks of grid, registration, relabel and filters."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon, box

from src.pipeline.p1_segment import build_grid_corpus as bg
from src.pipeline.p1_segment.build_finetune_data import fetch_imagery_mosaic, warp_to_grid


def test_grid_cells_anchor_top_left_and_keep_only_intersecting():
    # L-shaped boundary: two top-row cells of its 3x2 bounding grid lie clear of it
    # (edge-touching cells DO count as intersecting, like QGIS; land filtering drops them later).
    boundary = Polygon([(0, 0), (3000, 0), (3000, 900), (900, 900), (900, 2000), (0, 2000)])
    cells = bg.grid_cells(boundary, cell_m=1000)
    assert {(r, c) for r, c, _ in cells} == {(0, 0), (1, 0), (1, 1), (1, 2)}
    assert dict(((r, c), g.bounds) for r, c, g in cells)[(0, 0)] == (0, 1000, 1000, 2000)


def test_best_shift_moves_centrelines_onto_the_roads():
    prob = np.zeros((100, 100), np.float32)
    prob[50, :] = prob[:, 40] = 1.0                       # model sees an L of roads
    skeleton = np.zeros((100, 100), bool)
    skeleton[53, :] = skeleton[:, 38] = True              # labels 3 px low, 2 px left
    dy, dx, score, zero = bg.best_shift(prob, skeleton, radius=5)
    assert (dy, dx) == (-3, 2)
    assert score > 0.9 > zero


def test_best_shift_cannot_win_by_pushing_roads_out_of_frame():
    prob = np.zeros((100, 100), np.float32)
    prob[20:, :] = 0.6                                    # model misses the top strip only
    skeleton = np.zeros((100, 100), bool)
    skeleton[3, :] = skeleton[50, :] = True               # one unseen road near the edge
    dy, dx, score, zero = bg.best_shift(prob, skeleton, radius=5)
    assert (dy, dx) == (0, 0) and score == pytest.approx(zero)   # old mean picked dy=-5


def test_best_shift_keeps_zero_on_ties_and_empty_skeleton():
    assert bg.best_shift(np.zeros((20, 20)), np.ones((20, 20), bool), radius=3)[:2] == (0, 0)
    assert bg.best_shift(np.ones((20, 20)), np.zeros((20, 20), bool)) == (0, 0, 0.0, 0.0)


def test_shift_mask_translates_without_wrapping():
    mask = np.zeros((5, 5), np.uint8)
    mask[0, 0] = mask[4, 4] = 1
    out = bg.shift_mask(mask, 1, 1)
    assert out[1, 1] == 1 and out.sum() == 1               # (4,4) left the frame, not wrapped


def test_rebuffer_gives_uniform_spacenet_width():
    skeleton = np.zeros((60, 60), bool)
    skeleton[30, 10:50] = True
    mask = bg.rebuffer(skeleton)
    assert mask[:, 30].sum() == 2 * bg.ROAD_HALF_WIDTH_PX + 1   # 13 px ~ 6.5 m at 0.5 m/px


def test_tile_drop_reason_checks_leakage_first():
    tile = box(0, 0, 256, 256)
    road = np.ones((4, 4), np.uint8)
    land = box(-1000, -1000, 1000, 1000)
    far = box(5000, 5000, 5001, 5001)
    assert bg.tile_drop_reason(tile, land, box(200, 200, 300, 300), np.zeros((4, 4))) == "heldout"
    assert bg.tile_drop_reason(tile, box(0, 0, 256, 20), far, road) == "land"      # ~8% land
    assert bg.tile_drop_reason(tile, land, far, np.zeros((4, 4))) == "road"
    assert bg.tile_drop_reason(tile, land, far, road) is None


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (4, 4)).save(buf, format="PNG")
    return buf.getvalue()


def test_cached_fetcher_downloads_each_tile_once(tmp_path, monkeypatch):
    calls = []
    png = _png()
    monkeypatch.setattr(bg, "_default_tile_fetcher",
                        lambda z, x, y, url: calls.append((z, x, y, url)) or png)
    fetch = bg.cached_fetcher("u/{z}/{x}/{y}", tmp_path)
    assert fetch(19, 1, 2) == fetch(19, 1, 2) == png
    assert calls == [(19, 1, 2, "u/{z}/{x}/{y}")]
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == ["2.bin"]   # no .tmp left


def test_cached_fetcher_refetches_a_corrupt_cache_entry(tmp_path, monkeypatch):
    png = _png()
    monkeypatch.setattr(bg, "_default_tile_fetcher", lambda z, x, y, url: png)
    bad = tmp_path / "19" / "1" / "2.bin"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(png[:20])                              # truncated by an old, killed run
    assert bg.cached_fetcher("u", tmp_path)(19, 1, 2) == png
    assert bad.read_bytes() == png


def test_cached_fetcher_never_caches_a_non_image(tmp_path, monkeypatch):
    monkeypatch.setattr(bg, "_default_tile_fetcher", lambda z, x, y, url: b"<html>quota</html>")
    with pytest.raises(Exception):
        bg.cached_fetcher("u", tmp_path)(19, 1, 2)
    assert not any(p.is_file() for p in tmp_path.rglob("*"))


def test_load_boundary_picks_the_local_utm_zone(tmp_path):
    import geopandas as gpd

    path = tmp_path / "boundary.gpkg"
    gpd.GeoDataFrame(geometry=[box(88.30, 22.50, 88.40, 22.60)], crs="EPSG:4326").to_file(path)
    geom, crs = bg.load_boundary(path)                          # Kolkata -> UTM 45N
    assert crs == "EPSG:32645" and geom.area > 1e8
    with pytest.raises(SystemExit, match="osm-boundary"):
        bg.load_boundary(tmp_path / "missing.gpkg")


def test_load_sources_reads_local_file_and_fails_clearly_without_it(tmp_path):
    with pytest.raises(SystemExit, match="imagery_url"):
        bg.load_sources(tmp_path / "sources.json")
    (tmp_path / "sources.json").write_text('{"imagery_url": "a/{z}", "roads_url": "b/{z}"}')
    assert bg.load_sources(tmp_path / "sources.json") == ("a/{z}", "b/{z}")


def test_exclusion_zone_refuses_to_run_without_chips(tmp_path):
    with pytest.raises(FileNotFoundError, match="leakage guard"):
        bg.exclusion_zone(tmp_path)


def test_exclusion_zone_covers_held_out_indian_eval_aois(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    from src.pipeline.p1_segment.build_finetune_data import DEFAULT_CITIES

    with rasterio.open(tmp_path / "chip1.tif", "w", driver="GTiff", width=2, height=2, count=1,
                       dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(72.90, 19.10, 0.001, 0.001)) as dst:
        dst.write(np.zeros((1, 2, 2), np.uint8))
    crs = "EPSG:32643"                                          # Bengaluru and Delhi share UTM 43N
    zone, n = bg.exclusion_zone(tmp_path, crs)
    assert n == 1
    for aoi in ("bengaluru_indiranagar", "delhi_cp", "mumbai_bandra"):
        aoi_box = box(*transform_bounds("EPSG:4326", crs, *DEFAULT_CITIES[aoi]))
        assert zone.contains(aoi_box.centroid), aoi


def test_rgba_mosaic_and_warp_keep_the_overlay_alpha():
    tile = np.zeros((256, 256, 4), np.uint8)
    tile[100:110, :, 3] = 255                             # one opaque road row, rest transparent
    buf = io.BytesIO()
    Image.fromarray(tile, mode="RGBA").save(buf, format="PNG")
    rgba, transform = fetch_imagery_mosaic((0.0, 0.0, 0.001, 0.001), 19,
                                           lambda z, x, y: buf.getvalue(), mode="RGBA")
    assert rgba.shape[2] == 4 and rgba[..., 3].max() == 255 and rgba[..., 3].min() == 0
    out = warp_to_grid(rgba, transform, "EPSG:3857", transform, rgba.shape[:2], src_crs="EPSG:3857")
    assert out.shape == rgba.shape
    assert np.array_equal(out[..., 3] >= 128, rgba[..., 3] >= 128)
