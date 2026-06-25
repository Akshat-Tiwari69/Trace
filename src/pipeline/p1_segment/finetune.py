"""A6 step 3: fine-tune the released checkpoint on domain-matched pairs.

Starts from `a4-roadseg-v1` (never from scratch, per §2) and continues training
on the Indian OSM pairs from `build_finetune_data.py` (`data/finetune/`),
**mixed with DeepGlobe** (oversample the Indian set) so the model adapts to the
domain without forgetting. Reuses the existing `RoadTileDataset` + occlusion aug,
`ComboLoss`, and `train.train_one_epoch` / `evaluate`; saves a checkpoint whose
`meta` carries the same `encoder`/`arch`/`decoder_attention_type`/`threshold`, so
`predict.py` / `evaluate.py` deploy it unchanged.

This is a **lightweight** fine-tune harness (short, low-LR nudge). Training is a
GPU step (Kaggle/Colab); the orchestration is CPU-smoke-tested. After the run,
re-evaluate with `p1_segment/evaluate.py` and release v2 **only if it wins**.

Example (Kaggle GPU)::

    python -m src.pipeline.p1_segment.finetune \
        --init models/deepglobe_mit_b3_scse_512px_best.pt \
        --finetune-dir data/finetune --deepglobe-dir /kaggle/input/.../train \
        --epochs 10 --lr 4e-5 --out models/road_v2.pt --device cuda
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.pipeline.p1_segment.dataset import (
    RoadTileDataset,
    build_train_transform,
    build_val_transform,
    pair_deepglobe,
)
from src.pipeline.p1_segment.losses import ComboLoss
from src.pipeline.p1_segment.model import load_checkpoint, save_checkpoint
from src.pipeline.p1_segment.train import evaluate, train_one_epoch


@dataclasses.dataclass
class FineTuneConfig:
    init_checkpoint: str | Path          # the v1 checkpoint to start from
    finetune_dir: str | Path             # data/finetune (DeepGlobe-format Indian pairs)
    deepglobe_dir: str | Path | None = None   # mix in DeepGlobe to avoid forgetting
    out_path: str | Path = "models/road_v2.pt"
    image_size: int = 512
    batch_size: int = 2
    lr: float = 4.0e-5                   # ~A4 base_lr ÷5 — a gentle nudge
    epochs: int = 10
    finetune_oversample: int = 4         # repeat the small Indian set per epoch
    crops_per_image: int = 4
    val_fraction: float = 0.15
    device: str = "cpu"
    seed: int = 2026


def gather_pairs(cfg: FineTuneConfig) -> tuple[list, list]:
    """Build the (train, val) pair lists: oversampled Indian pairs (+ DeepGlobe)."""
    import random

    indian = pair_deepglobe(cfg.finetune_dir)
    if not indian:
        raise SystemExit(f"no DeepGlobe-format pairs in {cfg.finetune_dir} — run build_finetune_data first.")
    rng = random.Random(cfg.seed)
    rng.shuffle(indian)
    n_val = max(1, round(len(indian) * cfg.val_fraction))
    val = indian[:n_val]                          # validate on held-out Indian pairs
    train = indian[n_val:] * cfg.finetune_oversample
    if cfg.deepglobe_dir:
        train += pair_deepglobe(cfg.deepglobe_dir)   # mix DeepGlobe in to retain it
    rng.shuffle(train)
    return train, val


def finetune(cfg: FineTuneConfig) -> dict:
    """Fine-tune from the v1 checkpoint; save the best-val checkpoint. Returns a summary."""
    torch.manual_seed(cfg.seed)
    model, meta = load_checkpoint(cfg.init_checkpoint, map_location=cfg.device)
    model = model.to(cfg.device)

    train_pairs, val_pairs = gather_pairs(cfg)
    train_ds = RoadTileDataset(train_pairs, build_train_transform(cfg.image_size, occlusion=True),
                               crops_per_image=cfg.crops_per_image)
    val_ds = RoadTileDataset(val_pairs, build_val_transform(cfg.image_size))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    loss_fn = ComboLoss(bce_weight=0.4, dice_weight=0.4, lovasz_weight=0.2, cldice_weight=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.device != "cpu"))

    out = Path(cfg.out_path)
    best_iou, history = -1.0, []
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, cfg.device, scaler)
        val = evaluate(model, val_loader, cfg.device, threshold=float(meta.get("threshold", 0.5)))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_iou": val["iou"], "val_dice": val["dice"]})
        print(f"epoch {epoch:02d} | loss {train_loss:.4f} | val IoU {val['iou']:.4f} | Dice {val['dice']:.4f}")
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            save_checkpoint(model, out, meta={
                **{k: meta.get(k) for k in ("encoder", "arch", "decoder_attention_type", "image_size", "threshold")},
                "finetuned_from": str(cfg.init_checkpoint),
                "finetune_pairs": len(train_pairs), "finetune_val_iou": float(best_iou), "epoch": epoch,
            })
            print(f"  saved new best -> {out} (val IoU {best_iou:.4f})")

    print(f"\nfine-tune done. best val IoU {best_iou:.4f} -> {out}\n"
          f"  next: evaluate.py on a proper held-out set; release v2 only if it beats v1.")
    return {"best_val_iou": best_iou, "out": str(out), "history": history,
            "n_train": len(train_pairs), "n_val": len(val_pairs)}


def main() -> None:
    p = argparse.ArgumentParser(description="A6: fine-tune the v1 checkpoint on Indian pairs.")
    p.add_argument("--init", required=True, help="v1 checkpoint to start from")
    p.add_argument("--finetune-dir", default="data/finetune", help="DeepGlobe-format Indian pairs")
    p.add_argument("--deepglobe-dir", default=None, help="DeepGlobe train dir to mix in (recommended)")
    p.add_argument("--out", default="models/road_v2.pt")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=4.0e-5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--oversample", type=int, default=4)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    finetune(FineTuneConfig(
        init_checkpoint=args.init, finetune_dir=args.finetune_dir, deepglobe_dir=args.deepglobe_dir,
        out_path=args.out, image_size=args.image_size, batch_size=args.batch_size, lr=args.lr,
        epochs=args.epochs, finetune_oversample=args.oversample, device=args.device,
    ))


if __name__ == "__main__":
    main()
