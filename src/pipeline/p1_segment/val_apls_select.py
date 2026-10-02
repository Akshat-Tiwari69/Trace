"""A51: pick a training run's release candidate by routing APLS on SpaceNet
*validation* chips, among candidates that do not invent roads or forget DeepGlobe
relative to the deployed model.

``finetune`` picks epochs inside a stage by pixel IoU, but routing is what the
promotion gate grades and better pixels have scored worse routes before. APLS skips
road-free chips, so it cannot see invented roads; two paired checks against the
deployed checkpoint therefore gate eligibility first:

- **road-free**: per-tile share of pixels predicted as road on the validation tiles
  with no labelled road. Fails if the candidate is confidently worse than deployed by
  more than ``FP_TOL`` of a tile. Reported per kind -- chip-edge no-data, open water,
  and the hand-labelled **land** tiles (roofs, sheds, fields, forest, shore
  structures; ``data/sample/a51_road_free_land_tiles.json``), the hard negatives.
- **DeepGlobe**: per-tile IoU on the run's held-out DeepGlobe validation tiles (the
  deployed model's v1 lineage may have seen some, which only biases this against the
  candidate). Fails if confidently worse by more than the stage-2 tolerance.

The pick is the highest mean common-unit chip APLS among eligible candidates
(stage 1 + every kept stage-2 epoch, so stage 1 survives a re-anchor that does not
help) -- never the frozen 127 held-out chips, which stay for the final comparison.

    python -m modal volume get trace-train-runs a51-mit_b5 models/a51
    python -m src.pipeline.p1_segment.val_apls_select models/a51
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LAND_TILES = ROOT / "data" / "sample" / "a51_road_free_land_tiles.json"
FP_TOL = 0.001           # 0.1% of a tile's pixels predicted as road where there is none
INVENTED_PX = 200        # a tile "has an invented road" at this many false-positive pixels


def validation_split(record: dict, corpus: Path | None = None, deepglobe: Path | None = None,
                     manifest: Path | None = None) -> tuple[list[str], list, list]:
    """The run's validation chips, Indian validation tiles and DeepGlobe validation
    tiles, rebuilt from local data exactly as ``deploy/modal_train.py`` built them;
    refuses if the local data is not what the run trained on (the split would differ)."""
    from src.pipeline.p1_segment.eval_spacenet import (
        DEFAULT_CORPUS, DEFAULT_MANIFEST, chip_of, load_or_make_heldout, train_pairs)
    from src.pipeline.p1_segment.finetune import FineTuneConfig, gather_pairs
    from src.pipeline.p1_segment.provenance import dir_fingerprint

    corpus = corpus or ROOT / DEFAULT_CORPUS
    deepglobe = deepglobe or ROOT / "data/raw/deepglobe/train"
    for name, path in (("spacenet", corpus), ("deepglobe", deepglobe)):
        if dir_fingerprint(path) != record["data"][name]:
            raise SystemExit(f"local {name} data differs from the run's; the validation split would not match")
    spec = record["recipe"]
    chips = sorted({chip_of(p.name) for p in Path(corpus).glob("*_sat.jpg")})
    held = load_or_make_heldout(chips, manifest or ROOT / DEFAULT_MANIFEST)
    pairs = train_pairs(corpus, held, min_road_fraction=0.0)
    if spec["pilot_cap"]:
        pairs = pairs[: spec["pilot_cap"]]
    _, indian_val, dg_val = gather_pairs(FineTuneConfig(
        init_checkpoint="", finetune_pairs=pairs, deepglobe_dir=deepglobe, deepglobe_subset=0,
        deepglobe_val=spec["common"]["deepglobe_val"], seed=spec["seed"]))
    return sorted({chip_of(Path(p[0]).name) for p in indian_val}), indian_val, dg_val


def road_free_tiles(indian_val: list, land_names: set[str]) -> list[tuple[tuple, str]]:
    """``((sat, mask), kind)`` for validation tiles with no labelled road;
    kind is 'no-data' (< 25% valid pixels), 'land' (hand-labelled) or 'water'."""
    from src.pipeline.p1_segment.finetune import _read_val_pair

    out = []
    for sat, mask in indian_val:
        image, gt = _read_val_pair(str(sat), str(mask))
        if gt.any():
            continue
        kind = ("no-data" if (image.max(axis=2) >= 8).mean() < 0.25 else
                "land" if Path(sat).name in land_names else "water")
        out.append(((sat, mask), kind))
    return out


def false_positives(model, thr: float, tiles: list, image_size: int, device: str) -> tuple[list, list]:
    """Per road-free tile: share and count of pixels predicted as road (all false)."""
    from src.pipeline.p1_segment.finetune import _read_val_pair
    from src.pipeline.p1_segment.model import predict_large_prob

    shares, pixels = [], []
    for (sat, mask), _ in tiles:
        image, _gt = _read_val_pair(str(sat), str(mask))
        pred = predict_large_prob(model, image, tile_size=image_size, device=device) >= thr
        shares.append(float(pred.mean()))
        pixels.append(int(pred.sum()))
    return shares, pixels


def chip_apls_scores(models: dict, chips: list[str], device: str) -> tuple[dict, dict]:
    """Common-unit chip APLS (each checkpoint at its own threshold) per candidate."""
    from src.pipeline.p1_segment import chip_apls_eval as ce

    ce._preflight_chip_inputs(chips)
    scores: dict[str, dict[str, float]] = {name: {} for name in models}
    ceilings: dict[str, float] = {}
    for i, chip in enumerate(chips, 1):
        bands, transform, eff_x, eff_y = ce._read_chip(chip)
        gt = ce.adj_to_apls_graph(ce._geojson_adj(chip, transform), eff_x, eff_y)
        if gt.number_of_nodes() < 2:
            continue   # road-free chip: APLS undefined (the road-free check covers these)
        ceilings[chip] = float(ce.chip_apls(gt, gt))
        for name, (model, thr) in models.items():
            scores[name][chip] = float(ce.chip_apls(
                ce.v32_chip_graph(model, bands, thr, eff_x, eff_y, device), gt))
        print(f"[{i}/{len(chips)}] {chip} " + " ".join(f"{n}={s[chip]:.3f}" for n, s in scores.items()),
              flush=True)
    return scores, ceilings


def paired(deployed: list[float], candidate: list[float]) -> dict:
    from src.pipeline.p1_segment.stats import paired_bootstrap_ci

    ci = paired_bootstrap_ci(deployed, candidate)
    return {"deployed": float(np.mean(deployed)), "candidate": float(np.mean(candidate)),
            "delta": ci.delta, "ci_low": ci.ci_low, "ci_high": ci.ci_high}


def road_free_report(deployed: tuple[list, list], candidate: tuple[list, list], kinds: list[str]) -> dict:
    """All road-free tiles (the gate) plus each kind on its own (land = hard negatives).
    ``deployed``/``candidate`` are ``(shares, pixels)`` from ``false_positives``."""
    report: dict = {}
    for kind in ("all", "land", "water", "no-data"):
        idx = [i for i, k in enumerate(kinds) if kind in ("all", k)]
        if not idx:
            continue
        report[kind] = {
            "n_tiles": len(idx),
            "fp_share": paired([deployed[0][i] for i in idx], [candidate[0][i] for i in idx]),
            "tiles_with_invented_road": {
                "deployed": float(np.mean([deployed[1][i] >= INVENTED_PX for i in idx])),
                "candidate": float(np.mean([candidate[1][i] >= INVENTED_PX for i in idx]))}}
    report["passes"] = report["all"]["fp_share"]["ci_low"] <= FP_TOL
    return report


def main() -> None:
    import torch

    from src.pipeline.p1_segment.finetune import _iou_scores_on_pairs
    from src.pipeline.p1_segment.model import DEPLOYED_CHECKPOINT, load_checkpoint

    p = argparse.ArgumentParser(description="A51: pick a run's candidate by validation-chip APLS.")
    p.add_argument("run_dir", help="a downloaded run directory (run.json, candidates.json, checkpoints)")
    run_dir = Path(p.parse_args().run_dir)
    record = json.loads((run_dir / "run.json").read_text())
    names = json.loads((run_dir / "candidates.json").read_text())["candidates"]
    spec, device = record["recipe"], "cuda" if torch.cuda.is_available() else "cpu"
    size, dg_tol = spec["common"]["image_size"], spec["stage2"]["deepglobe_iou_tolerance"]

    chips, indian_val, dg_val = validation_split(record)
    tiles = road_free_tiles(indian_val, set(json.loads(LAND_TILES.read_text())["land_tiles"]))
    kinds = [kind for _, kind in tiles]
    models = {}
    for name, path in [("deployed", ROOT / DEPLOYED_CHECKPOINT)] + [(n, run_dir / n) for n in names]:
        model, meta = load_checkpoint(path, map_location=device)
        models[name] = (model.to(device).eval(), float(meta["threshold"]))
    fp = {n: false_positives(m, t, tiles, size, device) for n, (m, t) in models.items()}
    dg = {n: _iou_scores_on_pairs(m, dg_val, size, device, t) for n, (m, t) in models.items()}
    scores, ceilings = chip_apls_scores({n: models[n] for n in names}, chips, device)

    candidates = {}
    for n in names:
        dg_check = paired(dg["deployed"], dg[n])
        dg_check["passes"] = dg_check["ci_low"] >= -dg_tol
        rf = road_free_report(fp["deployed"], fp[n], kinds)
        candidates[n] = {"val_apls_mean": float(np.mean(list(scores[n].values()))),
                         "road_free": rf, "deepglobe": dg_check, "eligible": rf["passes"] and dg_check["passes"]}
    eligible = [n for n in names if candidates[n]["eligible"]]
    # pre-registered: highest mean validation APLS among eligible candidates; ties -> earliest
    pick = max(eligible, key=lambda n: candidates[n]["val_apls_mean"]) if eligible else None
    report = {"n_val_chips": len(chips), "n_apls_scored": len(ceilings),
              "ceiling_mean": float(np.mean(list(ceilings.values()))),
              "n_road_free_tiles": {k: kinds.count(k) for k in ("land", "water", "no-data")},
              "deployed": DEPLOYED_CHECKPOINT, "fp_tolerance": FP_TOL, "deepglobe_tolerance": dg_tol,
              "candidates": candidates, "pick": pick}
    (run_dir / "val_apls_selection.json").write_text(json.dumps(report, indent=2))
    for n, c in candidates.items():
        rf, land = c["road_free"]["all"]["fp_share"], c["road_free"].get("land", {}).get("fp_share", {})
        print(f"{n:>16}: val APLS {c['val_apls_mean']:.4f} | road-free FP {rf['candidate']:.5f} "
              f"(deployed {rf['deployed']:.5f}) land {land.get('candidate', float('nan')):.5f} "
              f"(deployed {land.get('deployed', float('nan')):.5f}) | DeepGlobe {c['deepglobe']['candidate']:.4f} "
              f"(deployed {c['deepglobe']['deployed']:.4f}) | {'eligible' if c['eligible'] else 'NOT eligible'}"
              f"{'  <- pick' if n == pick else ''}")
    if pick is None:
        print("No candidate passes both checks against the deployed model -- nothing to promote.")


if __name__ == "__main__":
    main()
