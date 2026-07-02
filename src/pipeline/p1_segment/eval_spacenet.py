"""A17 — truthful Indian benchmark on a frozen held-out SpaceNet-5 Mumbai split.

SpaceNet-5 Mumbai (built in A16) is real, human-drawn Indian road ground truth.
This module freezes a **chip-level** held-out TEST split (so a future supervised
fine-tune — A23 — can never train on it) and scores checkpoints on it with
**real-GT IoU/Dice** at a fixed threshold. That replaces the *misleading*
OSM-agreement metric (A12): for the first time we measure on real Indian GT.

Single source of truth for the split lives in the committed manifest
``data/sample/spacenet_mumbai_heldout_chips.json`` so it stays frozen forever.

    python -m src.pipeline.p1_segment.eval_spacenet \
        --checkpoints models/deepglobe_mit_b3_scse_512px_best.pt models/road_v2.pt \
        --device cuda --image-size 384

APLS (topology metric) is the planned follow-up — it needs metre-correct
mask->graph plumbing (reuse S7 `p3_analysis/apls.py`); IoU is delivered here.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

CHIP_RE = re.compile(r"(chip\d+)")
DEFAULT_CORPUS = Path("data/raw/spacenet/dg_format")
DEFAULT_MANIFEST = Path("data/sample/spacenet_mumbai_heldout_chips.json")


def chip_of(name: str) -> str:
    """Return the SpaceNet chip id embedded in a tile name (e.g. ``chip12``)."""
    m = CHIP_RE.search(name)
    if not m:
        raise ValueError(f"no chip id in {name!r}")
    return m.group(1)


def make_split(chips: list[str], test_frac: float = 0.2, seed: int = 17) -> tuple[list[str], list[str]]:
    """Deterministic chip-level train/test split (sorted then seeded-shuffled)."""
    ordered = sorted(set(chips))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_test = max(1, round(len(ordered) * test_frac))
    test = sorted(ordered[:n_test], key=lambda c: int(c[4:]))
    train = sorted(set(ordered) - set(test), key=lambda c: int(c[4:]))
    return train, test


def load_or_make_heldout(
    chips: list[str], manifest: Path, test_frac: float = 0.2, seed: int = 17
) -> list[str]:
    """Return the frozen held-out TEST chips, creating the manifest on first use."""
    manifest = Path(manifest)
    if manifest.is_file():
        return json.loads(manifest.read_text())["test_chips"]
    _, test = make_split(chips, test_frac=test_frac, seed=seed)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(
        {"dataset": "spacenet5_mumbai_aoi8", "test_frac": test_frac, "seed": seed,
         "n_test_chips": len(test), "test_chips": test}, indent=2))
    return test


def heldout_pairs(corpus: Path, test_chips: list[str]) -> list[tuple[Path, Path]]:
    """(sat, mask) pairs whose chip is in the held-out TEST set."""
    return _pairs_by_chip(corpus, set(test_chips), keep_in_set=True)


def train_pairs(corpus: Path, test_chips: list[str]) -> list[tuple[Path, Path]]:
    """(sat, mask) pairs whose chip is NOT held out — the A23 supervised train set."""
    return _pairs_by_chip(corpus, set(test_chips), keep_in_set=False)


def _pairs_by_chip(corpus: Path, chips: set[str], keep_in_set: bool) -> list[tuple[Path, Path]]:
    out = []
    for sat in sorted(Path(corpus).glob("*_sat.jpg")):
        if (chip_of(sat.name) in chips) == keep_in_set:
            mask = sat.with_name(sat.name.replace("_sat.jpg", "_mask.png"))
            if mask.exists():
                out.append((sat, mask))
    return out


def evaluate_checkpoints(
    checkpoints: list[Path],
    corpus: Path = DEFAULT_CORPUS,
    manifest: Path = DEFAULT_MANIFEST,
    threshold: float | None = None,
    image_size: int = 384,
    device: str = "cpu",
    grayscale: bool = False,
) -> dict:
    """Score each checkpoint on the frozen held-out SpaceNet-Mumbai split (IoU/Dice).

    ``threshold=None`` (default) scores each checkpoint at its **own deployed
    threshold** (checkpoint ``meta``, falling back to 0.44) — so v3.2 @0.52 isn't
    scored at v1's operating point; pass a float to force a shared threshold.
    ``grayscale=True`` desaturates the input — a **Cartosat-3 PAN** proxy (A24) to
    measure the sensor-modality gap vs the RGB number.
    """
    from torch.utils.data import DataLoader

    from src.pipeline.p1_segment.dataset import RoadTileDataset, build_val_transform
    from src.pipeline.p1_segment.model import load_checkpoint
    from src.pipeline.p1_segment.train import evaluate

    all_chips = sorted({chip_of(p.name) for p in Path(corpus).glob("*_sat.jpg")})
    test_chips = load_or_make_heldout(all_chips, manifest)
    pairs = heldout_pairs(corpus, test_chips)
    mode = "GRAYSCALE (Cartosat-PAN proxy)" if grayscale else "RGB"
    print(f"held-out SpaceNet-Mumbai TEST [{mode}]: {len(test_chips)} chips / {len(pairs)} tiles "
          f"(of {len(all_chips)} chips)", flush=True)

    loader = DataLoader(RoadTileDataset(pairs, build_val_transform(image_size, grayscale=grayscale)),
                        batch_size=4, shuffle=False, num_workers=0)
    results = {}
    for ckpt in checkpoints:
        model, meta = load_checkpoint(ckpt, map_location=device)
        model.to(device).eval()
        thr = threshold if threshold is not None else float(meta.get("threshold", 0.44))
        m = evaluate(model, loader, device, threshold=thr)
        results[Path(ckpt).name] = {"iou": m["iou"], "dice": m["dice"], "threshold": thr}
        print(f"  {Path(ckpt).name:42s} real-GT IoU {m['iou']:.4f}  Dice {m['dice']:.4f}  (thr {thr:.2f})",
              flush=True)
        del model
    return {"n_test_chips": len(test_chips), "n_test_tiles": len(pairs),
            "threshold": threshold, "image_size": image_size, "grayscale": grayscale,
            "models": results}


def threshold_sweep(
    checkpoints, corpus=DEFAULT_CORPUS, manifest=DEFAULT_MANIFEST,
    thresholds=None, image_size: int = 512, device: str = "cpu", grayscale: bool = False,
) -> dict:
    """Sweep thresholds per model (one inference per tile, cheap threshold loop).

    Reports each model's best threshold + its IoU there, plus the IoU at a shared
    0.44 for continuity — so the "best model" verdict doesn't hinge on a threshold
    picked for the older v1/A4 checkpoints (A21 / Codex audit #3)."""
    import cv2
    import numpy as np

    from src.pipeline.p1_segment.model import load_checkpoint, predict_prob
    from src.pipeline.p1_segment.raster_io import imread_gray, imread_rgb

    if thresholds is None:
        thresholds = [round(0.20 + 0.02 * i, 2) for i in range(26)]  # 0.20..0.70
    corpus = Path(corpus)
    chips = sorted({chip_of(p.name) for p in corpus.glob("*_sat.jpg")})
    pairs = heldout_pairs(corpus, load_or_make_heldout(chips, manifest))
    mode = "GRAYSCALE (Cartosat-PAN proxy)" if grayscale else "RGB"
    print(f"threshold sweep [{mode}] over {len(pairs)} held-out tiles, {len(thresholds)} thresholds", flush=True)

    results = {}
    for ckpt in checkpoints:
        model, _ = load_checkpoint(ckpt, map_location=device)
        model.to(device).eval()
        eps = 1e-7
        inter = {t: 0.0 for t in thresholds}
        union = {t: 0.0 for t in thresholds}
        for sat, mask_path in pairs:
            img = imread_rgb(sat)
            if img.shape[0] != image_size:
                img = cv2.resize(img, (image_size, image_size))
            if grayscale:
                g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = np.stack([g, g, g], -1)
            gt = imread_gray(mask_path) > 127
            if gt.shape[0] != image_size:
                gt = cv2.resize(gt.astype(np.uint8), (image_size, image_size),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
            prob = predict_prob(model, img, device=device)
            gt_sum = float(gt.sum())  # constant across thresholds — hoisted out of the loop
            for t in thresholds:
                pred = prob >= t
                i = float(np.logical_and(pred, gt).sum())
                inter[t] += i
                union[t] += float(pred.sum()) + gt_sum - i
        iou = {t: (inter[t] + eps) / (union[t] + eps) for t in thresholds}
        best_t = max(iou, key=iou.get)
        results[Path(ckpt).name] = {"best_threshold": best_t, "best_iou": round(iou[best_t], 4),
                                    "iou_at_0.44": round(iou.get(0.44, 0.0), 4),
                                    "iou_by_threshold": {t: round(v, 4) for t, v in iou.items()}}
        print(f"  {Path(ckpt).name:42s} best thr {best_t:.2f} IoU {iou[best_t]:.4f}  (@0.44 {iou.get(0.44,0):.4f})", flush=True)
    return {"grayscale": grayscale, "models": results}


def main() -> None:
    p = argparse.ArgumentParser(description="A17: eval on held-out SpaceNet-5 Mumbai (real Indian GT).")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--threshold", type=float, default=None,
                   help="shared override; default = each checkpoint's deployed meta threshold")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--device", default="cpu")
    p.add_argument("--sweep", action="store_true", help="A21: sweep thresholds 0.20-0.70 and report each model's best")
    p.add_argument("--grayscale", action="store_true",
                   help="desaturate input (Cartosat-3 PAN proxy) to measure the sensor-modality gap")
    p.add_argument("--out", default="data/sample/spacenet_mumbai_eval.json")
    args = p.parse_args()

    if args.sweep:
        rep = threshold_sweep([Path(c) for c in args.checkpoints], Path(args.corpus), Path(args.manifest),
                              image_size=args.image_size, device=args.device, grayscale=args.grayscale)
        Path(args.out).write_text(json.dumps(rep, indent=2)); print(f"-> {args.out}"); return

    report = evaluate_checkpoints([Path(c) for c in args.checkpoints], Path(args.corpus),
                                  Path(args.manifest), args.threshold, args.image_size, args.device,
                                  grayscale=args.grayscale)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
