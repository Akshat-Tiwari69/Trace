"""CLI: run the trained segmentation model on imagery → road mask (P1 inference).

The inference half of P1: load a fine-tuned checkpoint, predict a binary road
mask for an image, and write it at the §4 contract path
``data/interim/{aoi}_mask.png`` that P2 (Shaivi) consumes.

:func:`run_inference` is the single shared path (A36) — both this CLI and
``run_pipeline.segment()`` call it, so the two entry points can't drift.

Example
-------
    python -m src.pipeline.p1_segment.predict \
        --image data/raw/panaji_tile.tif --checkpoint models/road_pan.pt --aoi panaji

Reads jpg/png via OpenCV and GeoTIFF via rasterio (``read_image_any``, A26):
1-band PAN is percentile-stretched to 3-channel grey, ≥3-band imagery uses the
first three bands, and CRS/transform are kept for P2's alignment manifest.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.pipeline.p1_segment.model import (
    DEPLOYED_RELEASE,
    load_checkpoint,
    predict_large,
    predict_large_prob,
    predict_large_raster,
)
from src.pipeline.p1_segment.osm_mask import save_binary_png, save_prob_png
from src.pipeline.p1_segment.postprocess import add_postprocess_args, postprocess_mask

_AOI_RE = re.compile(r"^[a-z0-9_-]{1,64}$")

# Above this size (either dimension), stream the raster in windows rather than
# loading it whole — the whole-RGB-float image is the OOM risk at pilot scale (§5H).
WINDOWED_THRESHOLD_PX = 4096


def validate_aoi(aoi: str) -> str:
    """Reject AOI ids that could produce weird/unsafe artifact paths (A36)."""
    if not _AOI_RE.match(aoi):
        raise SystemExit(
            f"invalid --aoi {aoi!r}: must match ^[a-z0-9_-]{{1,64}}$ "
            "(lowercase letters, digits, '_' and '-' only)"
        )
    return aoi


def run_inference(
    image_path: str | Path,
    checkpoint: str | Path,
    aoi: str,
    interim_dir: str | Path = "data/interim",
    *,
    tile_size: int | None = None,
    threshold: float | None = None,
    blend: bool = True,
    stride: int | None = None,
    tta: bool = False,
    device: str = "cpu",
    postprocess: bool = False,
    min_component_size: int = 50,
    pp_open_radius: int = 0,
    pp_close_radius: int = 0,
    fill_holes: int = 0,
    windowed: bool | None = None,
    window_px: int = 2048,
) -> tuple[Path, float]:
    """Shared P1 inference: image + checkpoint → ``data/interim/{aoi}_mask.png``.

    ``tile_size``/``threshold`` default to the checkpoint's deploy settings
    (its ``meta``), so a good model isn't hobbled by the wrong CLI values.
    ``blend`` (default, A27) runs Hann-blended overlapping windows — no
    tile-seam breaks; ``blend=False`` is the older non-overlapping tiling.
    ``postprocess`` runs the A10 cleanup before writing.

    ``windowed`` streams a large raster off disk in ``window_px`` tiles instead
    of loading it whole (bugs.md §5H) — the only full-size array held is the
    binary mask. ``None`` (default) auto-enables it once the raster exceeds
    ``WINDOWED_THRESHOLD_PX`` in either dimension. Returns
    ``(mask_path, road_pixel_fraction)``.
    """
    validate_aoi(aoi)
    from src.pipeline.p1_segment.raster_io import (
        raster_dimensions,
        raster_georef,
        read_image_any,
        write_manifest,
    )

    model, meta = load_checkpoint(checkpoint, map_location=device)
    tile_size = tile_size if tile_size is not None else int(meta.get("image_size", 512))
    threshold = threshold if threshold is not None else float(meta.get("threshold", 0.5))

    height, width = raster_dimensions(image_path)
    if windowed is None:
        windowed = max(height, width) > WINDOWED_THRESHOLD_PX

    prob = None  # only the whole-image blended path below ever fills this in
    if windowed:
        # Stream windows off disk — never materialise the full RGB image. Each
        # window's prob map is thresholded and discarded immediately (§5H's
        # memory bound), so there is no full-size prob array to persist here.
        transform, crs = raster_georef(image_path)
        mask = predict_large_raster(model, image_path, tile_size=tile_size, threshold=threshold,
                                    device=device, tta=tta, window_px=window_px)
        print(f"[{aoi}] windowed inference over {width}x{height}px in {window_px}px tiles")
    else:
        # A26: rasterio for GeoTIFFs (keeps CRS/transform; handles 1-band PAN), else cv2
        image, transform, crs = read_image_any(image_path)
        if blend:
            prob = predict_large_prob(model, image, tile_size=tile_size, stride=stride,
                                      device=device, tta=tta)
            mask = (prob >= threshold).astype("uint8")
        else:
            mask = predict_large(model, image, tile_size=tile_size,
                                 device=device, threshold=threshold, tta=tta)

    if postprocess:
        roads_before = mask.mean()
        mask = postprocess_mask(mask, min_size=min_component_size,
                                open_radius=pp_open_radius,
                                close_radius=pp_close_radius, fill_holes=fill_holes)
        print(f"[{aoi}] postprocess: roads {roads_before:.2%} -> {mask.mean():.2%}")

    out = Path(interim_dir) / f"{aoi}_mask.png"
    save_binary_png(mask, out)

    # Persist the P1 probability map (bugs.md §4): the blended path computes it
    # then used to discard it after thresholding. P2's healing needs it to tell
    # a sub-threshold occluded road from terrain with no road signal at all.
    prob_path = None
    if prob is not None:
        prob_path = Path(interim_dir) / aoi / "prob.png"
        save_prob_png(prob, prob_path)

    manifest = write_manifest(aoi, interim_dir, transform, crs, prob_png=prob_path is not None)  # A26: georef for P2

    # Provenance (bugs.md §5A): record which checkpoint/threshold/commit made this
    # mask, alongside it, so P2/P3 can carry the lineage into every artifact.
    from src.pipeline.p1_segment.provenance import build_provenance, write_provenance
    prov = build_provenance(checkpoint, meta, threshold)
    write_provenance(Path(interim_dir) / aoi / "provenance.json", prov)

    geo = f" · georeferenced ({crs}) -> {manifest}" if manifest else " · pixel-space (no CRS)"
    prob_note = f" · prob map -> {prob_path}" if prob_path else ""
    print(f"[{aoi}] {width}x{height}px "
          f"(encoder {meta.get('encoder', '?')}) -> roads {mask.mean():.2%} of pixels -> {out}{geo}{prob_note}")
    return out, float(mask.mean())


def main() -> None:
    p = argparse.ArgumentParser(description="Predict a road mask from imagery using a trained checkpoint.")
    p.add_argument("--image", required=True, help="imagery (jpg/png/GeoTIFF incl. 1-band PAN)")
    p.add_argument("--checkpoint", required=True,
                   help=f"trained .pt checkpoint (deployed: Release {DEPLOYED_RELEASE})")
    p.add_argument("--aoi", required=True, help="short AOI id -> data/interim/{aoi}_mask.png")
    p.add_argument("--tile-size", type=int, default=None, help="default: checkpoint meta image_size")
    p.add_argument("--threshold", type=float, default=None, help="default: checkpoint meta threshold")
    p.add_argument("--tta", action="store_true", help="D4 test-time augmentation (8× compute, ~+IoU)")
    p.add_argument("--no-blend", action="store_true",
                   help="disable A27 Hann-blended inference (default ON); falls back to hard tiling")
    p.add_argument("--blend", action="store_true", help=argparse.SUPPRESS)  # legacy no-op: blend is the default
    p.add_argument("--stride", type=int, default=None, help="blend window stride (default 75%% overlap)")
    p.add_argument("--windowed", dest="windowed", action="store_true", default=None,
                   help=f"stream the raster in windows (auto-enabled above {WINDOWED_THRESHOLD_PX}px, §5H)")
    p.add_argument("--no-windowed", dest="windowed", action="store_false",
                   help="force whole-image inference even for a large raster")
    p.add_argument("--window-px", type=int, default=2048, help="window size for --windowed inference")
    add_postprocess_args(p)
    p.add_argument("--interim-dir", default="data/interim")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    run_inference(
        args.image, args.checkpoint, validate_aoi(args.aoi), args.interim_dir,
        tile_size=args.tile_size, threshold=args.threshold,
        blend=not args.no_blend, stride=args.stride, tta=args.tta, device=args.device,
        postprocess=args.postprocess, min_component_size=args.min_component_size,
        pp_open_radius=args.pp_open_radius, pp_close_radius=args.pp_close_radius,
        fill_holes=args.fill_holes, windowed=args.windowed, window_px=args.window_px,
    )


if __name__ == "__main__":
    main()
