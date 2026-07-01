"""A17 (topology half) — APLS on the held-out SpaceNet-Mumbai split.

IoU rewards pixel overlap; **APLS** rewards getting the road *network* right (can
you still route A→B?), which is what the resilience goal actually needs. Per tile
we skeletonise the predicted mask and the GT mask into graphs and score their
routing similarity with S7's `p3_analysis.apls.apls`.

Coordinate note: `skeleton_to_graph(resolution_m=GSD)` yields **metric** x,y +
`length_m`; S7's `apls` expects lon/lat node coords (it projects with `_to_metres`),
so we rescale x,y into the equivalent degrees (length_m stays metric). No edits to
Shaivi's apls.

    python -m src.pipeline.p1_segment.apls_eval \
        --checkpoints models/road_spacenet.pt models/deepglobe_mit_b3_scse_512px_best.pt \
        --device cuda --n-tiles 80
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.pipeline.p1_segment.eval_spacenet import (
    DEFAULT_CORPUS, DEFAULT_MANIFEST, chip_of, heldout_pairs, load_or_make_heldout)

GSD_M = 0.5            # SpaceNet dg_format ground sampling distance
_DEG_X = 111_320.0     # metres per degree lon near the equator
_DEG_Y = 110_540.0     # metres per degree lat


def mask_to_apls_graph(mask01: np.ndarray, gsd_m: float = GSD_M):
    """Skeletonise a binary mask into an apls-ready graph (metric length_m,
    node x,y rescaled to the degrees S7's apls projects back to metres)."""
    from skimage.morphology import skeletonize

    from src.pipeline.p2_graph.skeleton_graph import skeleton_to_graph

    g = skeleton_to_graph(skeletonize(mask01.astype(bool)), transform=None, resolution_m=gsd_m)
    for _, d in g.nodes(data=True):
        d["x"] = d["x"] / _DEG_X
        d["y"] = d["y"] / _DEG_Y
    return g


def tile_apls(pred01: np.ndarray, gt01: np.ndarray, n_samples: int = 200, tol_m: float = 15.0) -> float:
    """APLS between one predicted and one GT mask (1.0 = identical routing)."""
    from src.pipeline.p3_analysis.apls import apls

    if gt01.sum() == 0:
        return float("nan")             # no GT roads on this tile -> undefined, skip
    gt_g = mask_to_apls_graph(gt01)
    pred_g = mask_to_apls_graph(pred01)
    if gt_g.number_of_nodes() < 2:
        return float("nan")             # GT skeletonised to ~nothing -> skip
    if pred_g.number_of_nodes() < 2:
        return 0.0                      # GT has roads, prediction has none -> worst APLS
    return apls(gt_g, pred_g, n_samples=n_samples, tol_m=tol_m)["apls"]


def apls_on_heldout(checkpoint: Path, n_tiles: int | None = 80, threshold: float = 0.44,
                    device: str = "cpu", seed: int = 7) -> dict:
    """Mean per-tile APLS of a checkpoint on the held-out SpaceNet-Mumbai tiles."""
    import cv2

    from src.pipeline.p1_segment.model import load_checkpoint, predict_mask

    corpus = Path(DEFAULT_CORPUS)
    chips = sorted({chip_of(p.name) for p in corpus.glob("*_sat.jpg")})
    pairs = heldout_pairs(corpus, load_or_make_heldout(chips, DEFAULT_MANIFEST))
    if n_tiles and n_tiles < len(pairs):
        pairs = random.Random(seed).sample(pairs, n_tiles)

    model, _ = load_checkpoint(checkpoint, map_location=device)
    model.to(device).eval()
    scores = []
    for sat, mask_path in pairs:
        image = cv2.cvtColor(cv2.imread(str(sat)), cv2.COLOR_BGR2RGB)
        gt = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        pred = predict_mask(model, image, device=device, threshold=threshold)
        s = tile_apls(pred, gt)
        if not np.isnan(s):
            scores.append(s)
    return {"checkpoint": Path(checkpoint).name, "n_scored": len(scores),
            "apls_mean": float(np.mean(scores)) if scores else 0.0}


def main() -> None:
    p = argparse.ArgumentParser(description="A17 APLS (topology) on held-out SpaceNet-Mumbai.")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--n-tiles", type=int, default=80, help="random held-out tiles to score (None=all)")
    p.add_argument("--threshold", type=float, default=0.44)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="data/sample/spacenet_mumbai_apls.json")
    args = p.parse_args()

    results = []
    for ckpt in args.checkpoints:
        r = apls_on_heldout(Path(ckpt), n_tiles=args.n_tiles, threshold=args.threshold, device=args.device)
        results.append(r)
        print(f"  {r['checkpoint']:42s} APLS {r['apls_mean']:.4f}  (n={r['n_scored']})", flush=True)
    Path(args.out).write_text(json.dumps({"n_tiles": args.n_tiles, "models": results}, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
