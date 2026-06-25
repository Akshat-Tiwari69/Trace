"""A6 step 3: fine-tune the released checkpoint on domain-matched pairs.

Starts from `a4-roadseg-v1` (never from scratch, per §2) and adapts it to the
Indian OSM pairs from `build_finetune_data.py` (`data/finetune/`) **without
forgetting DeepGlobe**. The first naive run (full-model fine-tune, small anchor)
adapted to India (+0.13 IoU) but regressed DeepGlobe (−0.07) — fine-tuning the
encoder on weak OSM labels corrupted its general features. So this harness:

- **freezes the encoder by default** (`encoder_lr_scale=0`), adapting only the
  decoder — preserving SegFormer's general road features;
- mixes a **large clean DeepGlobe anchor** with the oversampled Indian pairs;
- **validates on held-out DeepGlobe *and* Indian each epoch**, and saves the
  checkpoint that **maximises Indian IoU subject to not regressing DeepGlobe**
  (the actual release criterion) — so a saved `v2` is releasable by construction.

Reuses `RoadTileDataset` + occlusion aug, `ComboLoss`, and `train.train_one_epoch`.
Training is a GPU step; the orchestration + selection logic are CPU-tested.

Example (local 3070 Ti)::

    python -m src.pipeline.p1_segment.finetune \
        --init models/deepglobe_mit_b3_scse_512px_best.pt \
        --finetune-dir data/finetune --deepglobe-dir data/raw/deepglobe/train \
        --deepglobe-subset 2000 --epochs 12 --device cuda --out models/road_v2.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.pipeline.p1_segment.dataset import (
    RoadTileDataset,
    build_train_transform,
    pair_deepglobe,
)
from src.pipeline.p1_segment.losses import ComboLoss
from src.pipeline.p1_segment.model import load_checkpoint, predict_large, save_checkpoint
from src.pipeline.p1_segment.train import train_one_epoch


@dataclasses.dataclass
class FineTuneConfig:
    init_checkpoint: str | Path          # the v1 checkpoint to start from
    finetune_dir: str | Path             # data/finetune (DeepGlobe-format Indian pairs)
    deepglobe_dir: str | Path | None = None   # mix in DeepGlobe to avoid forgetting
    deepglobe_subset: int = 2000         # clean DeepGlobe anchor size
    deepglobe_val: int = 40              # held-out DeepGlobe tiles (forget-check)
    out_path: str | Path = "models/road_v2.pt"
    image_size: int = 512
    batch_size: int = 2
    lr: float = 1.0e-4                   # decoder LR (encoder frozen by default)
    encoder_lr_scale: float = 0.0        # 0 = freeze encoder; >0 = discriminative
    epochs: int = 12
    finetune_oversample: int = 3
    crops_per_image: int = 1
    val_fraction: float = 0.15
    deepglobe_iou_tolerance: float = 0.005   # max allowed DeepGlobe drop vs v1
    device: str = "cpu"
    seed: int = 2026


def gather_pairs(cfg: FineTuneConfig) -> tuple[list, list, list]:
    """Build (train, indian_val, deepglobe_val). Val sets are disjoint from train."""
    rng = random.Random(cfg.seed)
    indian = pair_deepglobe(cfg.finetune_dir)
    if not indian:
        raise SystemExit(f"no DeepGlobe-format pairs in {cfg.finetune_dir} — run build_finetune_data first.")
    rng.shuffle(indian)
    n_val = max(1, round(len(indian) * cfg.val_fraction))
    indian_val = indian[:n_val]
    train = indian[n_val:] * cfg.finetune_oversample

    deepglobe_val: list = []
    if cfg.deepglobe_dir:
        dg = pair_deepglobe(cfg.deepglobe_dir)
        rng.shuffle(dg)
        deepglobe_val = dg[: cfg.deepglobe_val]                      # held-out forget-check
        train += dg[cfg.deepglobe_val : cfg.deepglobe_val + cfg.deepglobe_subset]  # anchor (disjoint)
    rng.shuffle(train)
    return train, indian_val, deepglobe_val


def _build_optimizer(model: torch.nn.Module, cfg: FineTuneConfig) -> torch.optim.Optimizer:
    """Freeze the encoder (default) or give it a low discriminative LR."""
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    dec = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    groups = [{"params": dec, "lr": cfg.lr}]
    if cfg.encoder_lr_scale <= 0.0:
        for p in enc:
            p.requires_grad_(False)                                  # freeze
    else:
        groups.append({"params": enc, "lr": cfg.lr * cfg.encoder_lr_scale})
    return torch.optim.AdamW(groups, weight_decay=1.0e-4)


@torch.no_grad()
def _iou_on_pairs(model, pairs, tile_size, device, thr) -> float:
    """Mean IoU over (sat, mask) pairs via full-image sliding prediction."""
    import cv2

    if not pairs:
        return float("nan")
    total = 0.0
    for sat_path, mask_path in pairs:
        img = cv2.cvtColor(cv2.imread(str(sat_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
        pred = predict_large(model, img, tile_size=tile_size, device=device, threshold=thr) > 0
        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        total += inter / max(union, 1)
    return total / len(pairs)


def finetune(cfg: FineTuneConfig) -> dict:
    """Fine-tune from v1; save the checkpoint that best adapts to India while
    preserving DeepGlobe. Returns a summary dict."""
    torch.manual_seed(cfg.seed)
    model, meta = load_checkpoint(cfg.init_checkpoint, map_location=cfg.device)
    model = model.to(cfg.device)
    thr = float(meta.get("threshold", 0.5))

    train_pairs, indian_val, deepglobe_val = gather_pairs(cfg)
    base_dg = _iou_on_pairs(model, deepglobe_val, cfg.image_size, cfg.device, thr)
    base_ind = _iou_on_pairs(model, indian_val, cfg.image_size, cfg.device, thr)
    frozen = cfg.encoder_lr_scale <= 0.0
    print(f"v1 baseline | DeepGlobe IoU {base_dg:.4f} | Indian IoU {base_ind:.4f} | "
          f"encoder {'FROZEN' if frozen else f'lr×{cfg.encoder_lr_scale}'} | "
          f"train {len(train_pairs)} (anchor incl.) | val dg {len(deepglobe_val)}/ind {len(indian_val)}")

    train_ds = RoadTileDataset(train_pairs, build_train_transform(cfg.image_size, occlusion=True),
                               crops_per_image=cfg.crops_per_image)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0)
    loss_fn = ComboLoss(bce_weight=0.4, dice_weight=0.4, lovasz_weight=0.2, cldice_weight=0.1)
    optimizer = _build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.device != "cpu"))

    out = Path(cfg.out_path)
    best_score, best_row, history = -1e9, None, []
    keep_floor = (base_dg - cfg.deepglobe_iou_tolerance) if deepglobe_val else -1.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, cfg.device, scaler)
        ind_iou = _iou_on_pairs(model, indian_val, cfg.image_size, cfg.device, thr)
        dg_iou = _iou_on_pairs(model, deepglobe_val, cfg.image_size, cfg.device, thr)
        keeps_dg = (not deepglobe_val) or (dg_iou >= keep_floor)
        score = ind_iou if keeps_dg else -1e9
        row = {"epoch": epoch, "train_loss": train_loss, "indian_iou": ind_iou,
               "deepglobe_iou": dg_iou, "keeps_deepglobe": keeps_dg}
        history.append(row)
        print(f"epoch {epoch:02d} | loss {train_loss:.4f} | Indian {ind_iou:.4f} "
              f"(v1 {base_ind:.4f}) | DeepGlobe {dg_iou:.4f} (v1 {base_dg:.4f}) | "
              f"{'KEEPS dg' if keeps_dg else 'regresses dg'}")
        if score > best_score:
            best_score, best_row = score, row
            save_checkpoint(model, out, meta={
                **{k: meta.get(k) for k in ("encoder", "arch", "decoder_attention_type", "image_size", "threshold")},
                "finetuned_from": str(cfg.init_checkpoint), "encoder_frozen": frozen,
                "indian_val_iou": float(ind_iou), "deepglobe_val_iou": float(dg_iou),
                "v1_indian_val_iou": float(base_ind), "v1_deepglobe_val_iou": float(base_dg),
                "epoch": epoch,
            })
            print(f"  saved new best -> {out} (Indian {ind_iou:.4f}, DeepGlobe {dg_iou:.4f})")

    if best_row is None:
        print("\nNo epoch preserved DeepGlobe within tolerance — nothing saved. Loosen the "
              "anchor/epochs or raise --deepglobe-tol.")
    else:
        print(f"\nbest: epoch {best_row['epoch']} | Indian {best_row['indian_iou']:.4f} "
              f"(v1 {base_ind:.4f}, {best_row['indian_iou']-base_ind:+.4f}) | "
              f"DeepGlobe {best_row['deepglobe_iou']:.4f} (v1 {base_dg:.4f}, "
              f"{best_row['deepglobe_iou']-base_dg:+.4f}) -> {out}")
    return {"best": best_row, "v1_deepglobe": base_dg, "v1_indian": base_ind, "history": history,
            "n_train": len(train_pairs)}


def main() -> None:
    p = argparse.ArgumentParser(description="A6: fine-tune v1 on Indian pairs without forgetting DeepGlobe.")
    p.add_argument("--init", required=True, help="v1 checkpoint to start from")
    p.add_argument("--finetune-dir", default="data/finetune", help="DeepGlobe-format Indian pairs")
    p.add_argument("--deepglobe-dir", default=None, help="DeepGlobe train dir (anchor + forget-check)")
    p.add_argument("--deepglobe-subset", type=int, default=2000, help="clean DeepGlobe anchor size")
    p.add_argument("--deepglobe-val", type=int, default=40, help="held-out DeepGlobe tiles for the forget-check")
    p.add_argument("--out", default="models/road_v2.pt")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1.0e-4)
    p.add_argument("--encoder-lr-scale", type=float, default=0.0, help="0 = freeze encoder; >0 = discriminative")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--oversample", type=int, default=3)
    p.add_argument("--crops-per-image", type=int, default=1)
    p.add_argument("--deepglobe-tol", type=float, default=0.005, help="max allowed DeepGlobe IoU drop vs v1")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    finetune(FineTuneConfig(
        init_checkpoint=args.init, finetune_dir=args.finetune_dir, deepglobe_dir=args.deepglobe_dir,
        deepglobe_subset=args.deepglobe_subset, deepglobe_val=args.deepglobe_val, out_path=args.out,
        image_size=args.image_size, batch_size=args.batch_size, lr=args.lr,
        encoder_lr_scale=args.encoder_lr_scale, epochs=args.epochs, finetune_oversample=args.oversample,
        crops_per_image=args.crops_per_image, deepglobe_iou_tolerance=args.deepglobe_tol, device=args.device,
    ))


if __name__ == "__main__":
    main()
