"""A51: train the road segmenter on Modal (no laptop needed once launched).

Two stages with ``finetune.finetune`` on an A100:

1. **Foundation** -- ImageNet-initialised encoder (default MiT-B5 + SCSE U-Net),
   DeepGlobe + SpaceNet-5 Mumbai train split (x2, road-free tiles included) + the
   local city-grid corpora, warm-up + cosine LR, gradient clipping, grayscale/gamma
   augmentation.
2. **Re-anchor** -- SpaceNet-only fine-tune with a DeepGlobe anchor, so the final
   road definition matches what the promotion gate grades. Every DeepGlobe-keeping
   epoch is kept (``stage2.epNN.pt``).

The release pick is made afterwards, locally, by routing APLS on the SpaceNet
validation chips among ``candidates.json`` (stage 1 stays a candidate in case
re-anchoring does not help) -- ``src.pipeline.p1_segment.val_apls_select``.

Validation and checkpoint selection use the SpaceNet validation split (the frozen
127 held-out chips are never seen); test-only cities are never uploaded. Data
lives in the Modal Volume ``trace-train-data`` (uploaded once with
``modal volume put``); checkpoints, ``run.json`` (recipe, data fingerprints, code
revision per launch), per-stage summaries and done-markers go to
``trace-train-runs``. A CPU pre-flight validates the data and refuses to reuse a
run directory recorded with a different recipe or data before any GPU is billed;
a restarted run resumes.

    python -m modal run deploy/modal_train.py --pilot                # -> a51-mit_b5-pilot (~cost/speed check)
    python -m modal run --detach deploy/modal_train.py               # -> a51-mit_b5 (full run)
    python -m modal volume get trace-train-runs a51-mit_b5 models/a51
"""
from __future__ import annotations

import modal

GRID_CITIES = ("mumbai", "bengaluru")       # training cities (test-only cities never listed)
SOURCES = {"spacenet": "/data/spacenet/dg_format", "deepglobe": "/data/deepglobe/train",
           **{c: f"/data/{c}_grid/dg_format" for c in GRID_CITIES}}
data = modal.Volume.from_name("trace-train-data", create_if_missing=True)
runs = modal.Volume.from_name("trace-train-runs", create_if_missing=True)


def recipe(encoder: str = "mit_b5", pilot: bool = False) -> dict:
    """Everything that defines the experiment; one run directory holds one recipe."""
    return {
        "encoder": encoder, "pilot": pilot, "seed": 2026, "grid_cities": list(GRID_CITIES),
        "pilot_cap": 400 if pilot else None,      # tiles per source: speed/cost check only
        "common": {"deepglobe_val": 200, "image_size": 512, "batch_size": 8, "num_workers": 8,
                   "grayscale_p": 0.7, "occlusion": True, "cldice_weight": 0.0,
                   "cosine": True, "max_grad_norm": 1.0},
        # From ImageNet the DeepGlobe guard is relative to an untrained init, so it cannot
        # bind; DeepGlobe is a training source here (all of train minus the val tiles).
        "stage1": {"finetune_oversample": 2, "deepglobe_subset": 100_000, "deepglobe_iou_tolerance": 1.0,
                   "lr": 2.0e-4, "encoder_lr_scale": 0.35, "epochs": 1 if pilot else 12, "warmup_epochs": 1},
        # SpaceNet re-anchor with a DeepGlobe anchor and a real forget guard.
        "stage2": {"finetune_oversample": 1, "deepglobe_subset": 2000, "deepglobe_iou_tolerance": 0.03,
                   "lr": 1.0e-4, "encoder_lr_scale": 0.1, "epochs": 1 if pilot else 5, "warmup_epochs": 0,
                   "save_every_epoch": True},
    }


def _cache_imagenet_weights() -> None:
    import segmentation_models_pytorch as smp

    smp.Unet("mit_b5", encoder_weights="imagenet", classes=1)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "segmentation-models-pytorch==0.3.4", "timm==0.9.7",
        "albumentations==2.0.8", "opencv-python-headless==4.10.0.84", "numpy==1.26.4",
        "pillow==10.4.0", "scipy==1.13.1", "rasterio==1.3.10",
    )
    .run_function(_cache_imagenet_weights)
    .add_local_dir("src", "/root/src")
    .add_local_file("data/sample/spacenet_mumbai_heldout_chips.json",
                    "/root/data/sample/spacenet_mumbai_heldout_chips.json")
)
app = modal.App("trace-train", image=image)


@app.function(cpu=1, memory=2048, timeout=1800, volumes={"/data": data, "/runs": runs})
def prepare(run: str, spec: dict, code: dict) -> None:
    """CPU pre-flight (no GPU billed, no retries): every source must have pairs, and an
    existing run directory must have been recorded with this exact recipe and data."""
    import sys
    from datetime import datetime, timezone

    sys.path.insert(0, "/root")
    from src.pipeline.p1_segment.provenance import dir_fingerprint, record_run

    fingerprints = {name: dir_fingerprint(path) for name, path in SOURCES.items()}
    empty = [name for name, fp in fingerprints.items() if not fp["pairs"]]
    if empty:
        raise RuntimeError(f"no training pairs on the data volume for: {empty}")
    record_run(f"/runs/{run}", {"recipe": spec, "data": fingerprints},
               {**code, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    runs.commit()
    print(f"{run}: data OK " + " | ".join(f"{k} {v['pairs']}" for k, v in fingerprints.items()), flush=True)


# Sized from a live a51-b5 profile: GPU ~85% busy, 12 loader workers ~1 core total, 12 GB RAM.
# Modal bills the reservation, so more cores/RAM only cost money.
@app.function(gpu="A100-40GB", cpu=4, memory=32768, timeout=24 * 3600,
              volumes={"/data": data, "/runs": runs}, retries=modal.Retries(max_retries=2))
def train(run: str, spec: dict) -> None:
    import json
    import os
    import sys
    import threading
    from pathlib import Path

    os.chdir("/root")
    sys.path.insert(0, "/root")
    import torch

    from src.pipeline.p1_segment.dataset import pair_deepglobe
    from src.pipeline.p1_segment.eval_spacenet import chip_of, load_or_make_heldout, train_pairs
    from src.pipeline.p1_segment.finetune import FineTuneConfig, finetune
    from src.pipeline.p1_segment.model import build_model, save_checkpoint

    out = Path("/runs") / run
    record = json.loads((out / "run.json").read_text())       # written by prepare()
    stop = threading.Event()

    def commit_every_5_min() -> None:          # checkpoints survive a crash or preemption
        while not stop.wait(300):
            runs.commit()

    threading.Thread(target=commit_every_5_min, daemon=True).start()

    sn_dir = Path(SOURCES["spacenet"])
    chips = sorted({chip_of(p.name) for p in sn_dir.glob("*_sat.jpg")})
    held = load_or_make_heldout(chips, Path("/root/data/sample/spacenet_mumbai_heldout_chips.json"))
    # Road-free tiles stay in (train and validation): the model must learn what no road
    # looks like, and validation must be able to see invented roads.
    sp_train = train_pairs(sn_dir, held, min_road_fraction=0.0)
    cities = {c: pair_deepglobe(f"/data/{c}_grid/dg_format") for c in spec["grid_cities"]}
    if spec["pilot_cap"]:
        sp_train = sp_train[: spec["pilot_cap"]]
        cities = {c: p[: spec["pilot_cap"]] for c, p in cities.items()}
    # Balance cities roughly equally (the smaller is repeated), as in A50e.
    n_max = max(len(p) for p in cities.values())
    extra = [pair for p in cities.values() for pair in p * max(1, round(n_max / len(p)))]
    print(f"A51 {run}: encoder {spec['encoder']} | SpaceNet train {len(sp_train)} (held-out {len(held)} reserved) | "
          + " | ".join(f"{c} {len(p)}" for c, p in cities.items()) + f" | grid total {len(extra)}", flush=True)

    init = out / "init_imagenet.pt"
    if not init.exists():
        torch.manual_seed(spec["seed"])      # the random decoder init is part of the recipe
        save_checkpoint(build_model(encoder=spec["encoder"], encoder_weights="imagenet",
                                    decoder_attention_type="scse"),
                        init, meta={"encoder": spec["encoder"], "arch": "unet", "decoder_attention_type": "scse",
                                    "image_size": 512, "threshold": 0.5, "init": "imagenet"})
    common = dict(finetune_pairs=sp_train, deepglobe_dir=SOURCES["deepglobe"], device="cuda",
                  seed=spec["seed"], **spec["common"])

    def stage(name: str, **kwargs) -> None:
        if (out / f"{name}.done").exists():
            print(f"{name} already done -> skip", flush=True)
            return
        last = out / f"{name}.last.pt"
        summary = finetune(FineTuneConfig(**common, **kwargs, out_path=str(out / f"{name}.pt"),
                                          resume=str(last) if last.exists() else None))
        if not summary.get("best"):
            raise RuntimeError(f"{name} kept no epoch")
        summary["run"] = record                # recipe, data fingerprints, code revision(s)
        (out / f"{name}.json").write_text(json.dumps(summary, default=str, indent=2))
        (out / f"{name}.done").touch()
        runs.commit()

    stage("stage1", init_checkpoint=str(init), extra_train_pairs=extra, **spec["stage1"])
    stage("stage2", init_checkpoint=str(out / "stage1.pt"), **spec["stage2"])
    candidates = ["stage1.pt"] + sorted(p.name for p in out.glob("stage2.ep*.pt"))
    (out / "candidates.json").write_text(json.dumps(
        {"select_by": "SpaceNet validation-chip APLS (src.pipeline.p1_segment.val_apls_select)",
         "candidates": candidates}, indent=2))
    stop.set()
    runs.commit()
    print(f"A51 {run} DONE -> candidates {candidates}", flush=True)


@app.local_entrypoint()
def main(run: str = "", encoder: str = "mit_b5", pilot: bool = False) -> None:
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    run = run or f"a51-{encoder}"
    if pilot and not run.endswith("-pilot"):      # a pilot never shares a full run's directory
        run += "-pilot"
    spec = recipe(encoder, pilot)
    code = {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "src", "deploy"))}
    prepare.remote(run, spec, code)
    train.remote(run, spec)
