"""A51: train the road segmenter on Modal (no laptop needed once launched).

Two stages with ``finetune.finetune`` on an A100:

1. **Foundation** -- ImageNet-initialised encoder (default MiT-B5 + SCSE U-Net),
   DeepGlobe + SpaceNet-5 Mumbai train split (x2) + the local city-grid corpora,
   warm-up + cosine LR, gradient clipping, grayscale/gamma augmentation.
2. **Re-anchor** -- SpaceNet-only fine-tune with a DeepGlobe anchor, so the final
   road definition matches what the promotion gate grades.

Validation and checkpoint selection use the SpaceNet validation split (the frozen
127 held-out chips are never seen); test-only cities are never uploaded. Data
lives in the Modal Volume ``trace-train-data`` (uploaded once with
``modal volume put``); checkpoints, per-stage summaries and done-markers go to
``trace-train-runs`` and survive restarts (a restarted run resumes).

    python -m modal run deploy/modal_train.py --run a51-pilot --pilot      # ~cost/speed check
    python -m modal run --detach deploy/modal_train.py --run a51-b5        # full run
    python -m modal volume get trace-train-runs a51-b5/stage2.pt models/road_a51.pt
"""
from __future__ import annotations

import modal

GRID_CITIES = ("mumbai", "bengaluru")       # training cities (test-only cities never listed)
data = modal.Volume.from_name("trace-train-data", create_if_missing=True)
runs = modal.Volume.from_name("trace-train-runs", create_if_missing=True)


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


# Sized from a live a51-b5 profile: GPU ~85% busy, 12 loader workers ~1 core total, 12 GB RAM.
# Modal bills the reservation, so more cores/RAM only cost money.
@app.function(gpu="A100-40GB", cpu=4, memory=32768, timeout=24 * 3600,
              volumes={"/data": data, "/runs": runs}, retries=modal.Retries(max_retries=2))
def train(run: str, encoder: str = "mit_b5", epochs1: int = 12, epochs2: int = 5,
          batch: int = 8, workers: int = 8, pilot: bool = False) -> None:
    import json
    import os
    import sys
    import threading
    from pathlib import Path

    os.chdir("/root")
    sys.path.insert(0, "/root")
    from src.pipeline.p1_segment.dataset import pair_deepglobe
    from src.pipeline.p1_segment.eval_spacenet import chip_of, load_or_make_heldout, train_pairs
    from src.pipeline.p1_segment.finetune import FineTuneConfig, finetune
    from src.pipeline.p1_segment.model import build_model, save_checkpoint

    out = Path("/runs") / run
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def commit_every_5_min() -> None:          # checkpoints survive a crash or preemption
        while not stop.wait(300):
            runs.commit()

    threading.Thread(target=commit_every_5_min, daemon=True).start()

    sn_dir = Path("/data/spacenet/dg_format")
    chips = sorted({chip_of(p.name) for p in sn_dir.glob("*_sat.jpg")})
    held = load_or_make_heldout(chips, Path("/root/data/sample/spacenet_mumbai_heldout_chips.json"))
    sp_train = train_pairs(sn_dir, held)
    cities = {c: pair_deepglobe(f"/data/{c}_grid/dg_format") for c in GRID_CITIES}
    if pilot:   # a few hundred tiles per source, one epoch per stage: measures speed and cost
        sp_train = sp_train[:400]
        cities = {c: p[:400] for c, p in cities.items()}
        epochs1 = epochs2 = 1
    # Balance cities roughly equally (the smaller is repeated), as in A50e.
    n_max = max(len(p) for p in cities.values())
    extra = [pair for p in cities.values() for pair in p * max(1, round(n_max / len(p)))]
    print(f"A51 {run}: encoder {encoder} | SpaceNet train {len(sp_train)} (held-out {len(held)} reserved) | "
          + " | ".join(f"{c} {len(p)}" for c, p in cities.items()) + f" | grid total {len(extra)}", flush=True)

    init = out / "init_imagenet.pt"
    if not init.exists():
        save_checkpoint(build_model(encoder=encoder, encoder_weights="imagenet", decoder_attention_type="scse"),
                        init, meta={"encoder": encoder, "arch": "unet", "decoder_attention_type": "scse",
                                    "image_size": 512, "threshold": 0.5, "init": "imagenet"})
    common = dict(finetune_pairs=sp_train, deepglobe_dir="/data/deepglobe/train", deepglobe_val=40,
                  image_size=512, batch_size=batch, grayscale_p=0.7, occlusion=True, cldice_weight=0.0,
                  num_workers=workers, device="cuda", seed=2026, cosine=True, max_grad_norm=1.0)

    def stage(name: str, **kwargs) -> None:
        if (out / f"{name}.done").exists():
            print(f"{name} already done -> skip", flush=True)
            return
        last = out / f"{name}.last.pt"
        summary = finetune(FineTuneConfig(**common, **kwargs, out_path=str(out / f"{name}.pt"),
                                          resume=str(last) if last.exists() else None))
        if not summary.get("best"):
            raise RuntimeError(f"{name} kept no epoch")
        (out / f"{name}.json").write_text(json.dumps(summary, default=str, indent=2))
        (out / f"{name}.done").touch()
        runs.commit()

    # Stage 1: from ImageNet. The DeepGlobe guard is relative to the untrained init, so
    # it cannot bind; DeepGlobe is a training source here (all of train minus 40 val).
    stage("stage1", init_checkpoint=str(init), extra_train_pairs=extra, finetune_oversample=2,
          deepglobe_subset=100_000, deepglobe_iou_tolerance=1.0, lr=2.0e-4, encoder_lr_scale=0.35,
          epochs=epochs1, warmup_epochs=1)
    # Stage 2: SpaceNet re-anchor with a DeepGlobe anchor and a real forget guard.
    stage("stage2", init_checkpoint=str(out / "stage1.pt"), finetune_oversample=1,
          deepglobe_subset=2000, deepglobe_iou_tolerance=0.03, lr=1.0e-4, encoder_lr_scale=0.1,
          epochs=epochs2, warmup_epochs=0)
    stop.set()
    runs.commit()
    print(f"A51 {run} DONE -> /runs/{run}/stage2.pt", flush=True)


@app.local_entrypoint()
def main(run: str = "a51-b5", encoder: str = "mit_b5", pilot: bool = False) -> None:
    train.remote(run, encoder=encoder, pilot=pilot)
