"""CLI: run the trained segmentation model on imagery → road mask (P1 inference).

The inference half of P1: load a fine-tuned checkpoint (from the A4 notebook),
predict a binary road mask for an image, and write it at the §4 contract path
``data/interim/{aoi}_mask.png`` that P2 (Shaivi) consumes.

Example
-------
    python -m src.pipeline.p1_segment.predict \
        --image data/raw/panaji_tile.tif --checkpoint models/segformer_mit_b2_deepglobe.pt --aoi panaji

Reads a 3-channel RGB image (jpg/png/3-band tif via OpenCV). Multiband GeoTIFF
imagery would need a band-selection reader — out of scope here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.p1_segment.model import load_checkpoint, predict_large, predict_large_prob
from src.pipeline.p1_segment.osm_mask import save_binary_png
from src.pipeline.p1_segment.postprocess import postprocess_mask


def main() -> None:
    p = argparse.ArgumentParser(description="Predict a road mask from imagery using a trained checkpoint.")
    p.add_argument("--image", required=True, help="RGB image/tile (jpg/png/3-band tif)")
    p.add_argument("--checkpoint", required=True, help="trained .pt checkpoint from the A4 notebook")
    p.add_argument("--aoi", required=True, help="short AOI id -> data/interim/{aoi}_mask.png")
    p.add_argument("--tile-size", type=int, default=None, help="default: checkpoint meta image_size")
    p.add_argument("--threshold", type=float, default=None, help="default: checkpoint meta threshold")
    p.add_argument("--tta", action="store_true", help="D4 test-time augmentation (8× compute, ~+IoU)")
    p.add_argument("--blend", action="store_true", help="A27: overlapped Hann-blended inference (no tile-seam breaks)")
    p.add_argument("--stride", type=int, default=None, help="blend window stride (default 75%% overlap)")
    p.add_argument("--postprocess", action="store_true",
                   help="A10 mask cleanup: drop tiny false components (+ optional close)")
    p.add_argument("--min-component-size", type=int, default=50,
                   help="min connected-component size in px to keep (with --postprocess)")
    p.add_argument("--pp-close-radius", type=int, default=0,
                   help="binary-close disk radius to bridge pin-hole gaps (with --postprocess)")
    p.add_argument("--interim-dir", default="data/interim")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    from src.pipeline.p1_segment.raster_io import read_image_any, write_manifest

    # A26: rasterio for GeoTIFFs (keeps CRS/transform; handles 1-band PAN), else cv2
    image, transform, crs = read_image_any(args.image)

    model, meta = load_checkpoint(args.checkpoint, map_location=args.device)
    # fall back to the checkpoint's deploy settings (like run_pipeline) so a good
    # model isn't hobbled by the wrong CLI threshold/resolution
    tile_size = args.tile_size if args.tile_size is not None else int(meta.get("image_size", 512))
    threshold = args.threshold if args.threshold is not None else float(meta.get("threshold", 0.5))
    if args.blend:
        prob = predict_large_prob(model, image, tile_size=tile_size, stride=args.stride,
                                  device=args.device, tta=args.tta)
        mask = (prob >= threshold).astype("uint8")
    else:
        mask = predict_large(model, image, tile_size=tile_size,
                             device=args.device, threshold=threshold, tta=args.tta)

    if args.postprocess:
        roads_before = mask.mean()
        mask = postprocess_mask(mask, min_size=args.min_component_size,
                                close_radius=args.pp_close_radius)
        print(f"[{args.aoi}] postprocess: roads {roads_before:.2%} -> {mask.mean():.2%}")

    out = Path(args.interim_dir) / f"{args.aoi}_mask.png"
    save_binary_png(mask, out)
    manifest = write_manifest(args.aoi, args.interim_dir, transform, crs)  # A26: georef for P2
    geo = f" · georeferenced ({crs}) -> {manifest}" if manifest else " · pixel-space (no CRS)"
    print(f"[{args.aoi}] {image.shape[1]}x{image.shape[0]}px "
          f"(encoder {meta.get('encoder', '?')}) -> roads {mask.mean():.2%} of pixels -> {out}{geo}")


if __name__ == "__main__":
    main()
