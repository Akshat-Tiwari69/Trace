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

from src.pipeline.p1_segment.model import load_checkpoint, predict_large
from src.pipeline.p1_segment.osm_mask import save_binary_png


def main() -> None:
    p = argparse.ArgumentParser(description="Predict a road mask from imagery using a trained checkpoint.")
    p.add_argument("--image", required=True, help="RGB image/tile (jpg/png/3-band tif)")
    p.add_argument("--checkpoint", required=True, help="trained .pt checkpoint from the A4 notebook")
    p.add_argument("--aoi", required=True, help="short AOI id -> data/interim/{aoi}_mask.png")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tta", action="store_true", help="D4 test-time augmentation (8× compute, ~+IoU)")
    p.add_argument("--interim-dir", default="data/interim")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    import cv2

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"could not read image: {args.image}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    model, meta = load_checkpoint(args.checkpoint, map_location=args.device)
    mask = predict_large(model, image, tile_size=args.tile_size,
                         device=args.device, threshold=args.threshold, tta=args.tta)

    out = Path(args.interim_dir) / f"{args.aoi}_mask.png"
    save_binary_png(mask, out)
    print(f"[{args.aoi}] {image.shape[1]}x{image.shape[0]}px "
          f"(encoder {meta.get('encoder', '?')}) -> roads {mask.mean():.2%} of pixels -> {out}")


if __name__ == "__main__":
    main()
