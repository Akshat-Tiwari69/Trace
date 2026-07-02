"""A11 data prep: convert Massachusetts Roads tiles → DeepGlobe-format pairs.

Massachusetts is 1500×1500 RGB **@1 m** with 0/255 road masks (`{stem}.tiff`
image / `{stem}.tif` label). DeepGlobe is **@0.5 m**, so road widths differ ~2×.
To train a *scale-consistent* combined corpus we upsample Massachusetts 2× (to
~0.5 m), tile to 512, and keep road-bearing tiles as DeepGlobe-format pairs
(`{prefix}_{stem}_r{r}_c{c}_sat.jpg` + `_mask.png`, 0/255). Reuses A3's tiler.

.. note::
    **A11 was run and rejected** — the resulting combined model lost on the
    Indian deployment target (`docs/Tracker.md` §6 A11). Kept as the reference
    for scale-matched corpus conversion; the production data path is
    `build_spacenet_data.py` (A16/A23).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.pipeline.p1_segment.osm_mask import tile_array


def convert_massachusetts(img_dir: str | Path, label_dir: str | Path, out_dir: str | Path,
                          upsample: float = 2.0, tile_size: int = 512,
                          min_road_fraction: float = 0.005, max_tiles: int | None = None,
                          prefix: str = "mass") -> int:
    """Tile + scale-match Massachusetts into DeepGlobe-format pairs. Returns #kept."""
    import cv2
    from PIL import Image

    img_dir, label_dir, out = Path(img_dir), Path(label_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kept = 0
    for img_p in sorted(img_dir.glob("*.tif*")):
        label = next(iter(label_dir.glob(img_p.stem + ".tif*")), None)
        if label is None:
            continue
        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(label), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if upsample != 1.0:
            nh, nw = int(img.shape[0] * upsample), int(img.shape[1] * upsample)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)

        for ti, tm in zip(tile_array(img, tile_size), tile_array(mask, tile_size)):
            if float(tm.data.mean()) < min_road_fraction:   # skip near-empty tiles
                continue
            stem = f"{prefix}_{img_p.stem}_r{ti.row}_c{ti.col}"
            Image.fromarray(ti.data).save(out / f"{stem}_sat.jpg", quality=92)
            Image.fromarray((tm.data * 255).astype(np.uint8), mode="L").save(out / f"{stem}_mask.png")
            kept += 1
            if max_tiles and kept >= max_tiles:
                print(f"[massachusetts] reached max_tiles={max_tiles} -> {out}")
                return kept
    print(f"[massachusetts] {kept} road-bearing pairs -> {out}")
    return kept


def main() -> None:
    p = argparse.ArgumentParser(description="Convert Massachusetts Roads to DeepGlobe-format pairs.")
    p.add_argument("--src", default="data/raw/massachusetts/tiff", help="dir with <split>/ + <split>_labels/")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="data/raw/massachusetts/dg_format")
    p.add_argument("--upsample", type=float, default=2.0, help="2.0 = 1m→0.5m scale-match to DeepGlobe")
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--min-road-fraction", type=float, default=0.005)
    p.add_argument("--max-tiles", type=int, default=None)
    args = p.parse_args()
    src = Path(args.src)
    convert_massachusetts(src / args.split, src / f"{args.split}_labels", args.out,
                          upsample=args.upsample, tile_size=args.tile_size,
                          min_road_fraction=args.min_road_fraction, max_tiles=args.max_tiles)


if __name__ == "__main__":
    main()
