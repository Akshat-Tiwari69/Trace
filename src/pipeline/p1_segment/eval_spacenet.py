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


def _write_report(path: Path | str, report: dict) -> None:
    """Write JSON after ensuring a caller-supplied output directory exists."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))


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


def split_heldout_chips(
    test_chips: list[str], selection_frac: float = 0.5, seed: int = 23
) -> tuple[list[str], list[str]]:
    """Split the frozen held-out chips into a threshold-SELECTION and REPORT half.

    Tuning the deploy threshold on the same chips used for the reported number is
    the classic A17 "optimise on the test set" bias: the winning IoU is
    inflated by having picked the operating point on that very data. This carves
    the *already-frozen* held-out set into two disjoint halves — select the
    threshold on ``selection``, report the final number on ``report`` — without
    touching the committed manifest (so no split is ever re-drawn). Deterministic
    under ``seed`` and disjoint by construction.
    """
    ordered = sorted(set(test_chips), key=lambda c: int(c[4:]))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_sel = max(1, round(len(ordered) * selection_frac))
    selection = sorted(ordered[:n_sel], key=lambda c: int(c[4:]))
    report = sorted(ordered[n_sel:], key=lambda c: int(c[4:]))
    return selection, report


def heldout_pairs(corpus: Path, test_chips: list[str]) -> list[tuple[Path, Path]]:
    """(sat, mask) pairs whose chip is in the held-out TEST set."""
    return _pairs_by_chip(corpus, set(test_chips), keep_in_set=True)


def train_pairs(corpus: Path, test_chips: list[str],
                min_road_fraction: float = 0.005) -> list[tuple[Path, Path]]:
    """Non-heldout road-bearing train pairs; heldout eval retains negatives."""
    pairs = _pairs_by_chip(corpus, set(test_chips), keep_in_set=False)
    if min_road_fraction <= 0:
        return pairs
    from PIL import Image
    import numpy as np

    kept = []
    for pair in pairs:
        try:
            fraction = float((np.asarray(Image.open(pair[1]).convert("L")) > 127).mean())
        except Exception:
            kept.append(pair)  # tolerate legacy/test placeholder files
            continue
        if fraction >= min_road_fraction:
            kept.append(pair)
    return kept


def _pairs_by_chip(corpus: Path, chips: set[str], keep_in_set: bool) -> list[tuple[Path, Path]]:
    out = []
    for sat in sorted(Path(corpus).glob("*_sat.jpg")):
        if (chip_of(sat.name) in chips) == keep_in_set:
            mask = sat.with_name(sat.name.replace("_sat.jpg", "_mask.png"))
            if mask.exists():
                out.append((sat, mask))
    return out


def _evaluate_with_chip_stats(model, loader, pairs, device: str, threshold: float) -> dict:
    """Global IoU/Dice plus chip-clustered IoU from one inference pass."""
    import torch

    eps = 1e-7
    total_inter = total_pred = total_target = 0.0
    chip_counts: dict[str, list[float]] = {}
    offset = 0
    with torch.inference_mode():
        for images, masks in loader:
            probs = torch.sigmoid(model(images.to(device))).cpu()
            pred = probs >= threshold
            target = masks >= 0.5
            for j in range(len(images)):
                p, t = pred[j], target[j]
                inter = float((p & t).sum())
                pred_sum, target_sum = float(p.sum()), float(t.sum())
                total_inter += inter
                total_pred += pred_sum
                total_target += target_sum
                chip = chip_of(pairs[offset + j][0].name)
                counts = chip_counts.setdefault(chip, [0.0, 0.0, 0.0])
                counts[0] += inter
                counts[1] += pred_sum
                counts[2] += target_sum
            offset += len(images)
    union = total_pred + total_target - total_inter
    return {
        "iou": (total_inter + eps) / (union + eps),
        "dice": (2 * total_inter + eps) / (total_pred + total_target + eps),
        "per_chip_iou": {
            chip: (inter + eps) / (pred_sum + target_sum - inter + eps)
            for chip, (inter, pred_sum, target_sum) in chip_counts.items()
        },
    }


def evaluate_checkpoints(
    checkpoints: list[Path],
    corpus: Path = DEFAULT_CORPUS,
    manifest: Path = DEFAULT_MANIFEST,
    threshold: float | None = None,
    image_size: int = 512,
    device: str = "cpu",
    grayscale: bool = False,
    chips: list[str] | None = None,
) -> dict:
    """Score each checkpoint on the frozen held-out SpaceNet-Mumbai split (IoU/Dice).

    ``threshold=None`` (default) scores each checkpoint at its **own deployed
    threshold** (checkpoint ``meta``, falling back to 0.44) — so v3.2 @0.52 isn't
    scored at v1's operating point; pass a float to force a shared threshold.
    ``grayscale=True`` desaturates the input — a **Cartosat-3 PAN** proxy (A24) to
    measure the sensor-modality gap vs the RGB number. ``chips`` restricts scoring
    to a subset of the held-out chips (used by :func:`honest_threshold_eval` to
    report on the half not used for threshold selection).
    """
    from torch.utils.data import DataLoader

    from src.pipeline.p1_segment.dataset import RoadTileDataset, build_val_transform
    from src.pipeline.p1_segment.model import load_checkpoint

    all_chips = sorted({chip_of(p.name) for p in Path(corpus).glob("*_sat.jpg")})
    test_chips = chips if chips is not None else load_or_make_heldout(all_chips, manifest)
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
        m = _evaluate_with_chip_stats(model, loader, pairs, device, thr)
        results[Path(ckpt).name] = {
            "iou": m["iou"], "dice": m["dice"], "threshold": thr,
            "per_chip_iou": m["per_chip_iou"],
        }
        print(f"  {Path(ckpt).name:42s} real-GT IoU {m['iou']:.4f}  Dice {m['dice']:.4f}  (thr {thr:.2f})",
              flush=True)
        del model
    report = {"n_test_chips": len(test_chips), "n_test_tiles": len(pairs),
              "threshold": threshold, "image_size": image_size, "grayscale": grayscale,
              "models": results}
    if len(checkpoints) >= 2:
        from src.pipeline.p1_segment.stats import paired_bootstrap_ci

        a_name, b_name = Path(checkpoints[0]).name, Path(checkpoints[1]).name
        a_scores = results[a_name]["per_chip_iou"]
        b_scores = results[b_name]["per_chip_iou"]
        common = sorted(set(a_scores) & set(b_scores))
        if common:
            ci = paired_bootstrap_ci([a_scores[c] for c in common],
                                     [b_scores[c] for c in common])
            report["paired_chip_iou"] = {
                "checkpoint_a": a_name, "checkpoint_b": b_name,
                "n_paired_chips": len(common), "delta_b_minus_a": ci.delta,
                "ci_low": ci.ci_low, "ci_high": ci.ci_high,
                "excludes_zero": ci.excludes_zero, "verdict": ci.verdict,
            }
    return report


def threshold_sweep(
    checkpoints, corpus=DEFAULT_CORPUS, manifest=DEFAULT_MANIFEST,
    thresholds=None, image_size: int = 512, device: str = "cpu", grayscale: bool = False,
    chips: list[str] | None = None,
) -> dict:
    """Sweep thresholds per model (one inference per tile, cheap threshold loop).

    Reports each model's best threshold + its IoU there, plus the IoU at a shared
    0.44 for continuity — so the "best model" verdict doesn't hinge on a threshold
    picked for the older v1/A4 checkpoints (A21). ``chips``
    restricts the sweep to a subset of the held-out chips (used by
    :func:`honest_threshold_eval` to select the threshold on a held-out half that
    is disjoint from the reporting half)."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from src.pipeline.p1_segment.dataset import RoadTileDataset, build_val_transform
    from src.pipeline.p1_segment.model import load_checkpoint

    if thresholds is None:
        thresholds = [round(0.20 + 0.02 * i, 2) for i in range(26)]  # 0.20..0.70
    corpus = Path(corpus)
    all_chips = sorted({chip_of(p.name) for p in corpus.glob("*_sat.jpg")})
    test_chips = chips if chips is not None else load_or_make_heldout(all_chips, manifest)
    pairs = heldout_pairs(corpus, test_chips)
    loader = DataLoader(
        RoadTileDataset(pairs, build_val_transform(image_size, grayscale=grayscale)),
        batch_size=16, shuffle=False, num_workers=0)
    mode = "GRAYSCALE (Cartosat-PAN proxy)" if grayscale else "RGB"
    print(f"threshold sweep [{mode}] over {len(pairs)} held-out tiles, {len(thresholds)} thresholds", flush=True)

    results = {}
    for ckpt in checkpoints:
        model, _ = load_checkpoint(ckpt, map_location=device)
        model.to(device).eval()
        eps = 1e-7
        inter = {t: 0.0 for t in thresholds}
        union = {t: 0.0 for t in thresholds}
        # Use the exact same centre-crop/normalise transform as report evaluation;
        # selecting a threshold under a different resize protocol is invalid.
        with torch.inference_mode():
            for images, masks in loader:
                probs = torch.sigmoid(model(images.to(device))).cpu().numpy()[:, 0]
                gts = masks.numpy()[:, 0] > 0.5
                for prob, gt in zip(probs, gts):
                    gt_sum = float(gt.sum())
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


def honest_threshold_eval(
    checkpoints, corpus=DEFAULT_CORPUS, manifest=DEFAULT_MANIFEST,
    thresholds=None, image_size: int = 512, device: str = "cpu", grayscale: bool = False,
    selection_frac: float = 0.5, seed: int = 23,
) -> dict:
    """Bias-free threshold + report: select the operating point on one held-out
    half, report IoU on the disjoint other half (A17).

    Fixes the "threshold tuned on the reported set" leak: the sweep runs only on
    the SELECTION chips, and the final IoU is measured on the REPORT chips the
    threshold never saw — so the reported number is not inflated by having chosen
    the operating point on that very data.
    """
    corpus = Path(corpus)
    all_chips = sorted({chip_of(p.name) for p in corpus.glob("*_sat.jpg")})
    test_chips = load_or_make_heldout(all_chips, manifest)
    sel_chips, rep_chips = split_heldout_chips(test_chips, selection_frac, seed)
    print(f"honest threshold eval: {len(sel_chips)} selection / {len(rep_chips)} report chips "
          f"(disjoint halves of {len(test_chips)} held-out)", flush=True)

    swept = threshold_sweep(checkpoints, corpus, manifest, thresholds, image_size, device,
                            grayscale, chips=sel_chips)
    models = {}
    for ckpt in checkpoints:
        name = Path(ckpt).name
        best_t = swept["models"][name]["best_threshold"]
        rep = evaluate_checkpoints([ckpt], corpus, manifest, threshold=best_t,
                                   image_size=image_size, device=device, grayscale=grayscale,
                                   chips=rep_chips)
        report_iou = rep["models"][name]["iou"]
        models[name] = {
            "threshold_selected_on": "selection_half",
            "threshold": best_t,
            "selection_iou_at_threshold": swept["models"][name]["best_iou"],
            "report_iou": report_iou,  # the honest, un-inflated number
        }
        print(f"  {name:42s} thr {best_t:.2f} (sel IoU {swept['models'][name]['best_iou']:.4f}) "
              f"→ REPORT IoU {report_iou:.4f}", flush=True)
    return {"grayscale": grayscale, "selection_chips": sel_chips, "report_chips": rep_chips,
            "models": models}


def main() -> None:
    p = argparse.ArgumentParser(description="A17: eval on held-out SpaceNet-5 Mumbai (real Indian GT).")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--threshold", type=float, default=None,
                   help="shared override; default = each checkpoint's deployed meta threshold")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--device", default="cpu")
    p.add_argument("--sweep", action="store_true", help="A21: sweep thresholds 0.20-0.70 and report each model's best")
    p.add_argument("--honest-threshold", action="store_true",
                   help="A17: select the threshold on one held-out half, report IoU on the disjoint other half")
    p.add_argument("--grayscale", action="store_true",
                   help="desaturate input (Cartosat-3 PAN proxy) to measure the sensor-modality gap")
    p.add_argument("--out", default="data/sample/spacenet_mumbai_eval.json")
    args = p.parse_args()

    if args.honest_threshold:
        rep = honest_threshold_eval([Path(c) for c in args.checkpoints], Path(args.corpus), Path(args.manifest),
                                    image_size=args.image_size, device=args.device, grayscale=args.grayscale)
        _write_report(args.out, rep); print(f"-> {args.out}"); return

    if args.sweep:
        rep = threshold_sweep([Path(c) for c in args.checkpoints], Path(args.corpus), Path(args.manifest),
                              image_size=args.image_size, device=args.device, grayscale=args.grayscale)
        _write_report(args.out, rep); print(f"-> {args.out}"); return

    report = evaluate_checkpoints([Path(c) for c in args.checkpoints], Path(args.corpus),
                                  Path(args.manifest), args.threshold, args.image_size, args.device,
                                  grayscale=args.grayscale)
    _write_report(args.out, report)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
