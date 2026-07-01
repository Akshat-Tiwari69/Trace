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

from src.pipeline.p1_segment.build_spacenet_data import _to_rgb8

_GEO_SUFFIXES = {".tif", ".tiff"}


def read_image_any(path: str | Path) -> tuple[np.ndarray, object, str | None]:
    """Read imagery → ``(rgb_uint8 HxWx3, transform, crs)``.

    GeoTIFF via rasterio: **1-band PAN** → 2–98 % percentile-stretch replicated to
    3 channels; **≥3-band** RGB/pan-sharpened/MX → first 3 bands stretched (A16's
    `_to_rgb8`). ``transform``/``crs`` come from the file. Non-GeoTIFF falls back
    to OpenCV with ``(None, None)`` — the pipeline still runs in pixel space.
    """
    path = Path(path)
    if path.suffix.lower() in _GEO_SUFFIXES:
        import rasterio

        with rasterio.open(path) as src:
            bands = src.read()  # (C, H, W)
            transform, crs = src.transform, (str(src.crs) if src.crs else None)
        if bands.shape[0] == 1:                       # PAN → 3ch grey
            rgb = _to_rgb8(np.repeat(bands, 3, axis=0))
        else:                                         # RGB / pan-sharpened / MX
            rgb = _to_rgb8(bands)
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
