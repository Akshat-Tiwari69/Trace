"""A26 — georeferenced + PAN-aware image reader for inference (Cartosat prep).

`predict.py`/`run_pipeline` previously read imagery via OpenCV, dropping the
GeoTIFF CRS/transform → the graph fell back to pixel space and routing distances
were a guess. Cartosat-3 final data is georeferenced (and often **panchromatic**,
1-band). This module reads GeoTIFFs with rasterio, handles PAN/RGB/multispectral,
and writes the alignment manifest P2's `build_graph` already consumes
(`data/interim/{aoi}/manifest.json`: `{"transform": [6], "crs": str}`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_GEO_SUFFIXES = {".tif", ".tiff"}


def to_rgb8(bands: np.ndarray) -> np.ndarray:
    """First 3 bands (C,H,W) → H×W×3 uint8 via per-channel 2–98% percentile stretch.

    Robust to 8-bit *or* 16-bit imagery; the stretch also tolerates SpaceNet's
    wide dynamic range without clipping detail to black/white.
    """
    rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1).astype(np.float32)
    out = np.empty(rgb.shape, dtype=np.uint8)
    for c in range(3):
        ch = rgb[..., c]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        if hi <= lo:
            hi = lo + 1.0
        out[..., c] = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


def imread_rgb(path: str | Path) -> np.ndarray:
    """Read an image as RGB uint8, raising (with the path) instead of cv2's silent ``None``."""
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cv2.imread returned None for {str(path)!r} (missing or unreadable file)")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def imread_gray(path: str | Path) -> np.ndarray:
    """Read an image as single-channel uint8, raising instead of cv2's silent ``None``."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"cv2.imread returned None for {str(path)!r} (missing or unreadable file)")
    return img


def read_image_any(path: str | Path) -> tuple[np.ndarray, object, str | None]:
    """Read imagery → ``(rgb_uint8 HxWx3, transform, crs)``.

    GeoTIFF via rasterio: **1-band PAN** → 2–98 % percentile-stretch replicated to
    3 channels; **≥3-band** RGB/pan-sharpened/MX → first 3 bands stretched
    (:func:`to_rgb8`). ``transform``/``crs`` come from the file. Non-GeoTIFF falls back
    to OpenCV with ``(None, None)`` — the pipeline still runs in pixel space.
    """
    path = Path(path)
    if path.suffix.lower() in _GEO_SUFFIXES:
        import rasterio

        with rasterio.open(path) as src:
            bands = src.read()  # (C, H, W)
            transform, crs = src.transform, (str(src.crs) if src.crs else None)
        if bands.shape[0] == 1:                       # PAN → 3ch grey
            rgb = to_rgb8(np.repeat(bands, 3, axis=0))
        else:                                         # RGB / pan-sharpened / MX
            rgb = to_rgb8(bands)
        return rgb, transform, crs

    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None, None


def write_manifest(aoi: str, interim_dir: str | Path, transform, crs) -> Path | None:
    """Write P2's alignment manifest if the source is georeferenced; else no-op.

    ``transform`` is a rasterio/affine ``Affine`` (its first 6 params are stored,
    which P2 rebuilds via ``Affine(*meta["transform"])``).
    """
    if transform is None or crs is None:
        return None
    out_dir = Path(interim_dir) / aoi
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "crs": str(crs),
        "transform": list(transform)[:6],
        "resolution_m": abs(float(transform[0])),
    }, indent=2))
    return manifest
