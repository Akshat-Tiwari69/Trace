"""A6 data-prep tests: tile math + imagery warp, fully offline (no network).

The network parts (osmnx + Esri tiles) are integration-only; here we inject a
synthetic ``tile_fetcher`` and check the mosaic geometry + an identity reproject.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from src.pipeline.p1_segment.build_finetune_data import (
    CORPUS_CITIES,
    DEFAULT_CITIES,
    deg2num,
    fetch_imagery_mosaic,
    num2merc,
    warp_to_grid,
)


def _overlaps(a, b) -> bool:
    """True if two (w, s, e, n) bboxes overlap."""
    w1, s1, e1, n1 = a
    w2, s2, e2, n2 = b
    return w1 < e2 and w2 < e1 and s1 < n2 and s2 < n1


def _solid_tile_bytes(color=(120, 80, 40)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (256, 256), color).save(buf, format="JPEG")
    return buf.getvalue()


def test_deg2num_num2merc_consistent():
    # a tile's corner round-trips: deg2num then num2merc lands in Web-Mercator range
    x, y = deg2num(15.49, 73.82, 18)
    mx, my = num2merc(x, y, 18)
    assert abs(mx) < 2.0037509e7 and abs(my) < 2.0037509e7   # within the 3857 world
    # x increases eastward, y increases southward (tile space)
    x_e, _ = deg2num(15.49, 73.83, 18)
    assert x_e > x


def test_fetch_imagery_mosaic_shape_and_transform():
    calls = []

    def fake_fetcher(z, x, y):
        calls.append((z, x, y))
        return _solid_tile_bytes()

    bbox = (73.80, 15.47, 73.84, 15.50)
    rgb, transform = fetch_imagery_mosaic(bbox, zoom=18, tile_fetcher=fake_fetcher)

    assert rgb.ndim == 3 and rgb.shape[2] == 3
    assert rgb.shape[0] % 256 == 0 and rgb.shape[1] % 256 == 0   # whole tiles
    assert len(calls) == (rgb.shape[0] // 256) * (rgb.shape[1] // 256)
    assert transform.a > 0 and transform.e < 0                   # north-up, +x east
    assert np.allclose(rgb.reshape(-1, 3).mean(0), [120, 80, 40], atol=8)  # solid colour (JPEG-tolerant)


def test_warp_to_grid_identity():
    # warping onto the SAME grid/CRS must return (≈) the input image
    from affine import Affine

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, size=(128, 160, 3), dtype=np.uint8)
    t = Affine(0.6, 0.0, 8_200_000.0, 0.0, -0.6, 1_740_000.0)   # arbitrary 3857 grid
    out = warp_to_grid(rgb, t, "EPSG:3857", t, (128, 160), src_crs="EPSG:3857")

    assert out.shape == rgb.shape
    # bilinear identity is near-exact in the interior (edges may differ slightly)
    assert np.abs(out[2:-2, 2:-2].astype(int) - rgb[2:-2, 2:-2].astype(int)).mean() < 1.0


def test_default_cities_are_well_formed():
    assert len(DEFAULT_CITIES) >= 3
    for aoi, (w, s, e, n) in DEFAULT_CITIES.items():
        assert w < e and s < n, aoi          # valid bbox
        assert 68 < w < 98 and 6 < s < 36     # roughly within India's lon/lat span


def test_corpus_cities_are_well_formed():
    assert len(CORPUS_CITIES) >= 15           # a real corpus, not a handful
    for aoi, (w, s, e, n) in CORPUS_CITIES.items():
        assert w < e and s < n, aoi
        assert 68 < w < 98 and 6 < s < 36, aoi


def test_corpus_is_disjoint_from_held_out_eval():
    """The corpus must NOT overlap the eval AOIs — else we'd train on our test set."""
    assert set(CORPUS_CITIES) & set(DEFAULT_CITIES) == set()        # no shared ids
    for c_aoi, c_box in CORPUS_CITIES.items():
        for e_aoi, e_box in DEFAULT_CITIES.items():
            assert not _overlaps(c_box, e_box), f"{c_aoi} overlaps eval {e_aoi}"


def test_tile_fetcher_retries_then_succeeds(monkeypatch):
    """A couple of transient timeouts must NOT abort the fetch — it retries."""
    import urllib.error
    import urllib.request

    from src.pipeline.p1_segment import build_finetune_data as bf

    calls = {"n": 0}

    class _Resp:
        def read(self):
            return b"TILEBYTES"

    def flaky(req, timeout=30):
        calls["n"] += 1
        if calls["n"] < 3:                       # fail twice, then succeed
            raise urllib.error.URLError("timed out")
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    monkeypatch.setattr("time.sleep", lambda *_: None)   # no real backoff wait
    assert bf._default_tile_fetcher(18, 1, 1) == b"TILEBYTES"
    assert calls["n"] == 3


def test_tile_fetcher_gives_up_after_retries(monkeypatch):
    import urllib.error
    import urllib.request

    from src.pipeline.p1_segment import build_finetune_data as bf

    def always_fail(req, timeout=30):
        raise urllib.error.URLError("host down")

    monkeypatch.setattr(urllib.request, "urlopen", always_fail)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    with pytest.raises(urllib.error.URLError):
        bf._default_tile_fetcher(18, 1, 1)
