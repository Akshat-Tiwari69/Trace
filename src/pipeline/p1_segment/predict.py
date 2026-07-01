"""CLI: run the trained segmentation model on imagery → road mask (P1 inference).

The inference half of P1: load a fine-tuned checkpoint (from the A4 notebook),
predict a binary road mask for an image, and write it at the §4 contract path
``data/interim/{aoi}_mask.png`` that P2 (Shaivi) consumes.

Example
-------
    python -m src.pipeline.p1_segment.predict \
        --image data/raw/panaji_tile.tif --checkpoint models/segformer_mit_b2_deepglobe.pt --aoi panaji

Reads common RGB images plus georeferenced GeoTIFFs. Single-band PAN GeoTIFFs
(such as Cartosat-style panchromatic imagery) are percentile-stretched and
stacked to 3 channels, and the CRS/transform is written beside the mask so P2 can
build a metric/georeferenced graph instead of falling back to pixel space.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.pipeline.p1_segment.model import load_checkpoint, predict_large
from src.pipeline.p1_segment.osm_mask import save_binary_png
from src.pipeline.p1_segment.postprocess import postprocess_mask


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert H×W, C×H×W, or H×W×C imagery to H×W×3 uint8 RGB.

    Cartosat-style PAN can arrive as a single high-bit-depth GeoTIFF band; model
    inference expects 3 channels, so PAN is contrast-stretched and stacked.
    """
    if arr.ndim == 2:
        rgb = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[0] in {1, 3, 4}:  # rasterio C,H,W
        if arr.shape[0] == 1:
            rgb = np.repeat(arr[0][..., None], 3, axis=2)
        else:
            rgb = np.transpose(arr[:3], (1, 2, 0))
    elif arr.ndim == 3 and arr.shape[2] in {1, 3, 4}:  # cv2/PIL H,W,C
        if arr.shape[2] == 1:
            rgb = np.repeat(arr[..., :1], 3, axis=2)
        else:
            rgb = arr[..., :3]
    else:
        raise ValueError(f"unsupported image shape for RGB conversion: {arr.shape}")

    if rgb.dtype == np.uint8:
        return rgb
    rgb = rgb.astype(np.float32)
    out = np.empty(rgb.shape, dtype=np.uint8)
    for c in range(3):
        ch = rgb[..., c]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        if hi <= lo:
            hi = lo + 1.0
        out[..., c] = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


def read_image_with_manifest(path: str | Path) -> tuple[np.ndarray, dict | None]:
    """Read RGB/PAN imagery and return ``(rgb_uint8, manifest_or_none)``.

    For GeoTIFFs, preserve CRS and affine transform so P2 can build a metric
    graph instead of falling back to pixel-space distances. For JPG/PNG, return
    no manifest.
    """
    import cv2

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        import rasterio

        with rasterio.open(path) as src:
            bands = src.read()
            image = _to_uint8_rgb(bands)
            manifest = {
                "source_image": str(path),
                "crs": src.crs.to_string() if src.crs else None,
                "transform": list(src.transform)[:6],
                "width": int(src.width),
                "height": int(src.height),
                "note": "written by p1_segment.predict; mask shares this grid",
            }
        return image, manifest

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None


def write_manifest(manifest: dict | None, interim_dir: str | Path, aoi: str) -> None:
    """Write ``data/interim/{aoi}/manifest.json`` when georeference exists."""
    if not manifest or not manifest.get("crs"):
        return
    out_dir = Path(interim_dir) / aoi
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def resolve_inference_params(
    meta: dict,
    tile_size: int | None,
    threshold: float | None,
) -> tuple[int, float]:
    """Resolve CLI overrides against checkpoint metadata.

    This guards the deployment path from silently falling back to stale defaults
    like 256px windows or threshold 0.5 when the checkpoint records a better
    trained/evaluated deploy setting.
    """
    resolved_tile = int(tile_size if tile_size is not None else meta.get("image_size", 512))
    resolved_threshold = float(threshold if threshold is not None else meta.get("threshold", 0.5))
    return resolved_tile, resolved_threshold


def main() -> None:
    p = argparse.ArgumentParser(description="Predict a road mask from imagery using a trained checkpoint.")
    p.add_argument("--image", required=True, help="RGB image/tile or GeoTIFF/PAN tile")
    p.add_argument("--checkpoint", required=True, help="trained .pt checkpoint from the A4/A23 notebook")
    p.add_argument("--aoi", required=True, help="short AOI id -> data/interim/{aoi}_mask.png")
    p.add_argument("--tile-size", type=int, default=None,
                   help="window size; default = checkpoint meta image_size or 512")
    p.add_argument("--stride", type=int, default=None,
                   help="optional overlap stride; e.g. 384 for 512px windows")
    p.add_argument("--threshold", type=float, default=None,
                   help="override threshold; default = checkpoint meta threshold")
    p.add_argument("--tta", action="store_true", help="D4 test-time augmentation (8× compute, usually off)")
    p.add_argument("--postprocess", action="store_true",
                   help="A10 mask cleanup: drop tiny false components (+ optional close)")
    p.add_argument("--min-component-size", type=int, default=50,
                   help="min connected-component size in px to keep (with --postprocess)")
    p.add_argument("--pp-close-radius", type=int, default=0,
                   help="binary-close disk radius to bridge pin-hole gaps (with --postprocess)")
    p.add_argument("--pp-fill-holes", type=int, default=0,
                   help="fill background holes up to this area in px (with --postprocess)")
    p.add_argument("--interim-dir", default="data/interim")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    image, manifest = read_image_with_manifest(args.image)
    model, meta = load_checkpoint(args.checkpoint, map_location=args.device)
    tile_size, threshold = resolve_inference_params(meta, args.tile_size, args.threshold)
    mask = predict_large(model, image, tile_size=tile_size, stride=args.stride,
                         device=args.device, threshold=threshold, tta=args.tta)

    if args.postprocess:
        roads_before = mask.mean()
        mask = postprocess_mask(mask, min_size=args.min_component_size,
                                close_radius=args.pp_close_radius,
                                fill_holes=args.pp_fill_holes)
        print(f"[{args.aoi}] postprocess: roads {roads_before:.2%} -> {mask.mean():.2%}")

    out = Path(args.interim_dir) / f"{args.aoi}_mask.png"
    save_binary_png(mask, out)
    write_manifest(manifest, args.interim_dir, args.aoi)
    print(f"[{args.aoi}] {image.shape[1]}x{image.shape[0]}px "
          f"(encoder {meta.get('encoder', '?')}, tile {tile_size}, "
          f"stride {args.stride or tile_size}, thr {threshold:.2f}) "
          f"-> roads {mask.mean():.2%} of pixels -> {out}")
    if manifest and manifest.get("crs"):
        print(f"[{args.aoi}] wrote georeference manifest -> {Path(args.interim_dir) / args.aoi / 'manifest.json'}")


if __name__ == "__main__":
    main()
