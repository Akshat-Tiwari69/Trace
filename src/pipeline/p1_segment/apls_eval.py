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


def _write_report(path: Path | str, report: dict) -> None:
    """Write JSON after ensuring a caller-supplied output directory exists."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))


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


def apls_on_heldout(checkpoint: Path, n_tiles: int | None = 80, threshold: float | None = None,
                    device: str = "cpu", seed: int = 7, per_tile: bool = False) -> dict:
    """Mean per-tile APLS of a checkpoint on the held-out SpaceNet-Mumbai tiles.

    ``threshold=None`` (default) uses the checkpoint's own deployed ``meta``
    threshold (falling back to 0.44); pass a float to force a shared threshold.
    ``per_tile=True`` also returns ``per_tile`` — an ordered ``{tile_name: score}``
    map (NaN-tiles omitted) so two checkpoints can be paired tile-by-tile for a
    bootstrap CI (see :func:`compare_checkpoints_apls`).
    """
    from src.pipeline.p1_segment.model import load_checkpoint, predict_mask
    from src.pipeline.p1_segment.raster_io import imread_gray, imread_rgb

    corpus = Path(DEFAULT_CORPUS)
    chips = sorted({chip_of(p.name) for p in corpus.glob("*_sat.jpg")})
    pairs = heldout_pairs(corpus, load_or_make_heldout(chips, DEFAULT_MANIFEST))
    if n_tiles and n_tiles < len(pairs):
        pairs = random.Random(seed).sample(pairs, n_tiles)

    model, meta = load_checkpoint(checkpoint, map_location=device)
    model.to(device).eval()
    thr = threshold if threshold is not None else float(meta.get("threshold", 0.44))
    scores = []
    per_tile_scores: dict[str, float] = {}
    for sat, mask_path in pairs:
        image = imread_rgb(sat)
        gt = (imread_gray(mask_path) > 127).astype(np.uint8)
        pred = predict_mask(model, image, device=device, threshold=thr)
        s = tile_apls(pred, gt)
        if not np.isnan(s):
            scores.append(s)
            per_tile_scores[sat.name] = float(s)
    out = {"checkpoint": Path(checkpoint).name, "n_scored": len(scores),
           "apls_mean": float(np.mean(scores)) if scores else 0.0, "threshold": thr}
    if per_tile:
        out["per_tile"] = per_tile_scores
    return out


def compare_checkpoints_apls(
    checkpoint_a: Path, checkpoint_b: Path, n_tiles: int | None = 80,
    threshold: float | None = None, device: str = "cpu", seed: int = 7,
) -> dict:
    """Paired-bootstrap comparison of two checkpoints' APLS (bugs.md §3).

    Scores both checkpoints on the **same** held-out tiles, pairs them by tile,
    and returns a 95% CI on the APLS delta (b − a). A promotion is only justified
    when the CI excludes zero — a headline gain that straddles zero is within
    sampling noise (the trap that made the A12 OSM-agreement metric misleading).
    """
    from src.pipeline.p1_segment.stats import paired_bootstrap_ci

    a = apls_on_heldout(checkpoint_a, n_tiles=n_tiles, threshold=threshold,
                        device=device, seed=seed, per_tile=True)
    b = apls_on_heldout(checkpoint_b, n_tiles=n_tiles, threshold=threshold,
                        device=device, seed=seed, per_tile=True)
    # Pair only on tiles both checkpoints scored (a NaN tile for either drops out).
    common = [t for t in a["per_tile"] if t in b["per_tile"]]
    scores_a = [a["per_tile"][t] for t in common]
    scores_b = [b["per_tile"][t] for t in common]
    if not common:
        return {
            "checkpoint_a": a["checkpoint"], "checkpoint_b": b["checkpoint"],
            "apls_a": a["apls_mean"], "apls_b": b["apls_mean"], "n_paired": 0,
            "delta": None, "ci_low": None, "ci_high": None, "p_two_sided": None,
            "excludes_zero": False, "verdict": "no paired tiles",
        }
    ci = paired_bootstrap_ci(scores_a, scores_b)
    # ascii only: redirected stdout on Windows is cp1252 and dies on "→" —
    # this print crashed the A38 run AFTER the CI was computed (log 2026-07-09)
    print(f"  {a['checkpoint']} -> {b['checkpoint']}: {ci.summary()}", flush=True)
    return {
        "checkpoint_a": a["checkpoint"], "checkpoint_b": b["checkpoint"],
        "apls_a": a["apls_mean"], "apls_b": b["apls_mean"], "n_paired": len(common),
        "delta": ci.delta, "ci_low": ci.ci_low, "ci_high": ci.ci_high,
        "p_two_sided": ci.p_two_sided, "excludes_zero": ci.excludes_zero,
        "verdict": ci.verdict,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="A17 APLS (topology) on held-out SpaceNet-Mumbai.")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--n-tiles", type=int, default=80, help="random held-out tiles to score (None=all)")
    p.add_argument("--threshold", type=float, default=None,
                   help="shared override; default = each checkpoint's deployed meta threshold")
    p.add_argument("--device", default="cpu")
    p.add_argument("--compare", action="store_true",
                   help="bugs.md §3: paired-bootstrap CI on the APLS delta between the first two checkpoints")
    p.add_argument("--out", default="data/sample/spacenet_mumbai_apls.json")
    args = p.parse_args()

    if args.compare:
        if len(args.checkpoints) < 2:
            raise SystemExit("--compare needs at least two --checkpoints (a then b)")
        rep = compare_checkpoints_apls(Path(args.checkpoints[0]), Path(args.checkpoints[1]),
                                       n_tiles=args.n_tiles, threshold=args.threshold, device=args.device)
        _write_report(args.out, rep); print(f"-> {args.out}"); return

    results = []
    for ckpt in args.checkpoints:
        r = apls_on_heldout(Path(ckpt), n_tiles=args.n_tiles, threshold=args.threshold, device=args.device)
        results.append(r)
        print(f"  {r['checkpoint']:42s} APLS {r['apls_mean']:.4f}  (n={r['n_scored']})", flush=True)
    _write_report(args.out, {"n_tiles": args.n_tiles, "models": results})
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
