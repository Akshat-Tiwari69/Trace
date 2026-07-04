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
import os
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


def raster_dimensions(path: str | Path) -> tuple[int, int]:
    """Return ``(height, width)`` from image metadata **without loading pixels**.

    Used to decide whether an AOI is large enough to need the windowed inference
    path (bugs.md §5H) before we ever allocate a full-resolution array.
    """
    path = Path(path)
    if path.suffix.lower() in _GEO_SUFFIXES:
        import rasterio

        with rasterio.open(path) as src:
            return int(src.height), int(src.width)
    from PIL import Image

    with Image.open(path) as im:  # PIL reads the header only, not the pixels
        w, h = im.size
    return int(h), int(w)


def raster_georef(path: str | Path) -> tuple[object, str | None]:
    """Return ``(transform, crs)`` from metadata without loading pixels.

    Lets the windowed inference path (bugs.md §5H) write the same alignment
    manifest as the whole-image path, without ever materialising the raster.
    """
    path = Path(path)
    if path.suffix.lower() in _GEO_SUFFIXES:
        import rasterio

        with rasterio.open(path) as src:
            return src.transform, (str(src.crs) if src.crs else None)
    return None, None


def iter_windows(
    path: str | Path, window_px: int = 2048, overlap_px: int = 256
):
    """Yield overlapping RGB windows of a (possibly huge) raster, lazily.

    Reads one ``window_px`` tile at a time — GeoTIFFs via rasterio windowed reads
    (never materialising the whole image), other formats via a PIL crop — so a
    100 km² AOI streams through inference instead of OOMing the reader
    (bugs.md §5H). Yields ``(rgb_uint8 HxWx3, row_off, col_off)``; the caller
    stitches them into the full-size (binary, 1-byte/px) output mask. Windows
    overlap by ``overlap_px`` so the per-window Hann blending seams are absorbed.
    """
    path = Path(path)
    height, width = raster_dimensions(path)
    step = max(1, window_px - overlap_px)
    row_offs = sorted({*range(0, max(1, height - window_px + 1), step), max(0, height - window_px)})
    col_offs = sorted({*range(0, max(1, width - window_px + 1), step), max(0, width - window_px)})

    if path.suffix.lower() in _GEO_SUFFIXES:
        import rasterio
        from rasterio.windows import Window

        with rasterio.open(path) as src:
            for r0 in row_offs:
                for c0 in col_offs:
                    h = min(window_px, height - r0)
                    w = min(window_px, width - c0)
                    bands = src.read(window=Window(c0, r0, w, h))  # (C, h, w)
                    rgb = to_rgb8(np.repeat(bands, 3, axis=0) if bands.shape[0] == 1 else bands)
                    yield rgb, r0, c0
        return

    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        for r0 in row_offs:
            for c0 in col_offs:
                h = min(window_px, height - r0)
                w = min(window_px, width - c0)
                crop = im.crop((c0, r0, c0 + w, r0 + h))
                yield np.asarray(crop), r0, c0


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
    payload = json.dumps({
        "crs": str(crs),
        "transform": list(transform)[:6],
        "resolution_m": abs(float(transform[0])),
    }, indent=2)
    # Atomic (A36): temp + os.replace, so a crash mid-write can't leave a
    # truncated manifest that silently drops P2 into pixel space.
    tmp = manifest.with_name(manifest.name + ".tmp")
    tmp.write_text(payload)
    os.replace(tmp, manifest)
    return manifest
