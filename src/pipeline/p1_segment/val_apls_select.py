"""A51: pick a training run's release candidate by routing APLS on SpaceNet
*validation* chips, then check DeepGlobe against the deployed model.

``finetune`` picks epochs inside a stage by pixel IoU, but routing is what the
promotion gate grades and better pixels have scored worse routes before. So the
pick among a run's ``candidates.json`` (stage 1 + every kept stage-2 epoch, so
stage 1 survives a re-anchor that does not help) is the highest mean common-unit
chip APLS on the validation chips finetune held out -- never the frozen 127
held-out chips, which stay for the one final comparison. The pick must then not
forget DeepGlobe relative to the deployed checkpoint: paired bootstrap on the
run's held-out DeepGlobe validation tiles (the deployed model's v1 lineage may
have seen some of them, which only biases this check against the candidate).

    python -m modal volume get trace-train-runs a51-mit_b5 models/a51
    python -m src.pipeline.p1_segment.val_apls_select models/a51
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def validation_split(record: dict, corpus: Path | None = None, deepglobe: Path | None = None,
                     manifest: Path | None = None) -> tuple[list[str], list]:
    """The run's validation chips and DeepGlobe validation tiles, rebuilt from local
    data exactly as ``deploy/modal_train.py`` built them; refuses if the local data
    is not what the run trained on (the split would silently differ)."""
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
    return sorted({chip_of(Path(p[0]).name) for p in indian_val}), dg_val


def score_candidates(paths: list[Path], chips: list[str], device: str) -> tuple[dict, dict]:
    """Common-unit chip APLS (each checkpoint at its own threshold) per candidate."""
    from src.pipeline.p1_segment import chip_apls_eval as ce
    from src.pipeline.p1_segment.model import load_checkpoint

    models = {}
    for path in paths:
        model, meta = load_checkpoint(path, map_location=device)
        models[path.name] = (model.to(device).eval(), float(meta["threshold"]))
    ce._preflight_chip_inputs(chips)
    scores: dict[str, dict[str, float]] = {name: {} for name in models}
    ceilings: dict[str, float] = {}
    for i, chip in enumerate(chips, 1):
        bands, transform, eff_x, eff_y = ce._read_chip(chip)
        gt = ce.adj_to_apls_graph(ce._geojson_adj(chip, transform), eff_x, eff_y)
        if gt.number_of_nodes() < 2:
            continue   # road-free chip: APLS undefined (training's pooled val IoU covers these)
        ceilings[chip] = float(ce.chip_apls(gt, gt))
        for name, (model, thr) in models.items():
            scores[name][chip] = float(ce.chip_apls(
                ce.v32_chip_graph(model, bands, thr, eff_x, eff_y, device), gt))
        print(f"[{i}/{len(chips)}] {chip} " + " ".join(f"{n}={s[chip]:.3f}" for n, s in scores.items()),
              flush=True)
    return scores, ceilings


def deepglobe_vs_deployed(pick: Path, dg_val: list, image_size: int, device: str, tolerance: float) -> dict:
    """Paired per-tile DeepGlobe IoU: candidate minus the deployed checkpoint."""
    from src.pipeline.p1_segment.finetune import _iou_scores_on_pairs
    from src.pipeline.p1_segment.model import DEPLOYED_CHECKPOINT, load_checkpoint
    from src.pipeline.p1_segment.stats import paired_bootstrap_ci

    per = {}
    for name, path in (("deployed", ROOT / DEPLOYED_CHECKPOINT), ("pick", pick)):
        model, meta = load_checkpoint(path, map_location=device)
        per[name] = _iou_scores_on_pairs(model.to(device).eval(), dg_val, image_size, device,
                                         float(meta["threshold"]))
    ci = paired_bootstrap_ci(per["deployed"], per["pick"])
    return {"deployed": DEPLOYED_CHECKPOINT, "n_tiles": len(dg_val),
            "deployed_iou": sum(per["deployed"]) / len(dg_val), "pick_iou": sum(per["pick"]) / len(dg_val),
            "delta": ci.delta, "ci_low": ci.ci_low, "ci_high": ci.ci_high,
            "tolerance": tolerance, "keeps_deepglobe": ci.ci_low >= -tolerance}


def main() -> None:
    import numpy as np
    import torch

    p = argparse.ArgumentParser(description="A51: pick a run's candidate by validation-chip APLS.")
    p.add_argument("run_dir", help="a downloaded run directory (run.json, candidates.json, checkpoints)")
    run_dir = Path(p.parse_args().run_dir)
    record = json.loads((run_dir / "run.json").read_text())
    names = json.loads((run_dir / "candidates.json").read_text())["candidates"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    chips, dg_val = validation_split(record)
    scores, ceilings = score_candidates([run_dir / n for n in names], chips, device)
    means = {n: float(np.mean(list(scores[n].values()))) for n in names}
    pick = max(names, key=means.__getitem__)             # pre-registered: highest mean; ties -> earliest
    spec = record["recipe"]
    dg = deepglobe_vs_deployed(run_dir / pick, dg_val, spec["common"]["image_size"], device,
                               spec["stage2"]["deepglobe_iou_tolerance"])
    report = {"n_val_chips": len(chips), "n_scored": len(ceilings),
              "ceiling_mean": float(np.mean(list(ceilings.values()))),
              "val_apls_mean": means, "pick": pick, "deepglobe_vs_deployed": dg}
    (run_dir / "val_apls_selection.json").write_text(json.dumps(report, indent=2))
    for n in names:
        print(f"{n:>16}: val APLS {means[n]:.4f}{'  <- pick' if n == pick else ''}")
    print(f"DeepGlobe {pick} {dg['pick_iou']:.4f} vs deployed {dg['deployed_iou']:.4f} "
          f"(delta {dg['delta']:+.4f}, CI [{dg['ci_low']:+.4f},{dg['ci_high']:+.4f}]) -> "
          f"{'keeps' if dg['keeps_deepglobe'] else 'FORGETS'} DeepGlobe")


if __name__ == "__main__":
    main()
