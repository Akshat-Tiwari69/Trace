"""From-scratch combined-corpus retrain (A11 / the "3+4" path).

Trains the full A4 recipe on a **multi-dataset** corpus (DeepGlobe + Massachusetts
+ …) — the principled way to break the recipe plateau the A7–A9 fine-tunes hit.
Unlike those fine-tunes, this trains from the **ImageNet-pretrained encoder**
(per §2 — never random init) so clDice + heavy occlusion can help from epoch 0.

Recipe: SegFormer MiT + SCSE U-Net, **EMA** weights (evaluated + saved),
ComboLoss (BCE+Dice+Lovász+clDice), discriminative encoder/decoder LR, warmup +
cosine, AMP, road-aware-ish multi-crop. Saves the best-val **EMA** checkpoint.

GPU step; the orchestration is CPU-smoke-tested.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import math
import random
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
from src.pipeline.p1_segment.model import build_model, save_checkpoint
from src.pipeline.p1_segment.train import evaluate


class ModelEMA:
    """Exponential moving average of weights — evaluated and exported."""

    def __init__(self, net: torch.nn.Module, decay: float = 0.9998) -> None:
        self.module = copy.deepcopy(net).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, net: torch.nn.Module) -> None:
        src = net.state_dict()
        for k, v in self.module.state_dict().items():
            s = src[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(s, alpha=1.0 - self.decay)
            else:
                v.copy_(s)


@dataclasses.dataclass
class TrainConfig:
    data_dirs: list[str]                 # DeepGlobe-format dirs (DeepGlobe, Massachusetts, …)
    out_path: str | Path = "models/road_combined.pt"
    encoder: str = "mit_b3"
    encoder_weights: str | None = "imagenet"
    image_size: int = 512
    batch_size: int = 4
    lr: float = 2.0e-4
    encoder_lr_scale: float = 0.35
    epochs: int = 25
    warmup_epochs: int = 2
    crops_per_image: int = 2
    val_fraction: float = 0.06
    occlusion: bool | str = "heavy"
    cldice_weight: float = 0.1
    ema_decay: float = 0.9998
    threshold: float = 0.44
    max_per_dir: int | None = None       # cap pairs per source (balance)
    num_workers: int = 0                 # >0 = parallel data loading (much faster on GPU)
    device: str = "cpu"
    seed: int = 2026


def gather(cfg: TrainConfig) -> tuple[list, list]:
    """Concatenate DeepGlobe-format pairs from all dirs; split train/val."""
    rng = random.Random(cfg.seed)
    pairs: list = []
    for d in cfg.data_dirs:
        ps = pair_deepglobe(d)
        rng.shuffle(ps)
        if cfg.max_per_dir:
            ps = ps[: cfg.max_per_dir]
        pairs += ps
        print(f"  + {len(ps):>6} pairs from {d}")
    rng.shuffle(pairs)
    n_val = max(1, round(len(pairs) * cfg.val_fraction))
    return pairs[n_val:], pairs[:n_val]


def _optimizer(net, cfg: TrainConfig):
    enc = [p for n, p in net.named_parameters() if n.startswith("encoder.")]
    dec = [p for n, p in net.named_parameters() if not n.startswith("encoder.")]
    return torch.optim.AdamW(
        [{"params": dec, "lr": cfg.lr},
         {"params": enc, "lr": cfg.lr * cfg.encoder_lr_scale}],
        betas=(0.9, 0.999), weight_decay=1e-4)


def train_combined(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = cfg.device
    model = build_model(encoder=cfg.encoder, encoder_weights=cfg.encoder_weights,
                        decoder_attention_type="scse").to(device)
    ema = ModelEMA(model, cfg.ema_decay)

    train_pairs, val_pairs = gather(cfg)
    train_ds = RoadTileDataset(train_pairs, build_train_transform(cfg.image_size, occlusion=cfg.occlusion),
                               crops_per_image=cfg.crops_per_image)
    val_ds = RoadTileDataset(val_pairs, build_val_transform(cfg.image_size))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                              num_workers=cfg.num_workers,
                              persistent_workers=cfg.num_workers > 0,
                              prefetch_factor=2 if cfg.num_workers > 0 else None,
                              pin_memory=(cfg.device != "cpu"))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    print(f"train {len(train_ds)} samples ({len(train_pairs)} pairs × {cfg.crops_per_image}) | val {len(val_pairs)}")

    loss_fn = ComboLoss(bce_weight=0.4, dice_weight=0.4, lovasz_weight=0.2, cldice_weight=cfg.cldice_weight)
    optimizer = _optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=(device != "cpu"))
    steps_per_epoch = max(1, len(train_loader))
    total = steps_per_epoch * cfg.epochs
    warmup = steps_per_epoch * cfg.warmup_epochs

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        phase = (step - warmup) / max(1, total - warmup)
        return 0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * min(1.0, phase)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    out = Path(cfg.out_path)
    best_iou, history = -1.0, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda" if device != "cpu" else "cpu", enabled=(device != "cpu")):
                loss = loss_fn(model(images), masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            running += float(loss.detach())
        val = evaluate(ema.module, val_loader, device, threshold=cfg.threshold)  # evaluate the EMA
        history.append({"epoch": epoch, "loss": running / steps_per_epoch, "val_iou": val["iou"]})
        print(f"epoch {epoch:02d} | loss {running/steps_per_epoch:.4f} | EMA val IoU {val['iou']:.4f} "
              f"| Dice {val['dice']:.4f}", flush=True)
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            save_checkpoint(ema.module, out, meta={
                "encoder": cfg.encoder, "arch": "unet", "decoder_attention_type": "scse",
                "image_size": cfg.image_size, "threshold": cfg.threshold,
                "trained_on": cfg.data_dirs, "val_iou": float(best_iou), "epoch": epoch,
                "recipe": "from-scratch combined (EMA, ComboLoss+clDice, heavy occ)",
            })
            print(f"  saved best EMA -> {out} (val IoU {best_iou:.4f})", flush=True)

    print(f"\ndone. best EMA val IoU {best_iou:.4f} -> {out}")
    return {"best_val_iou": best_iou, "out": str(out), "history": history,
            "n_train": len(train_pairs), "n_val": len(val_pairs)}


def main() -> None:
    p = argparse.ArgumentParser(description="From-scratch combined-corpus retrain (A11).")
    p.add_argument("--data-dirs", nargs="+", required=True, help="DeepGlobe-format dirs")
    p.add_argument("--out", default="models/road_combined.pt")
    p.add_argument("--encoder", default="mit_b3")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.0e-4)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--crops-per-image", type=int, default=2)
    p.add_argument("--occlusion", choices=["standard", "heavy", "none"], default="heavy")
    p.add_argument("--cldice-weight", type=float, default=0.1)
    p.add_argument("--max-per-dir", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    occ = {"standard": True, "heavy": "heavy", "none": False}[args.occlusion]
    train_combined(TrainConfig(
        data_dirs=args.data_dirs, out_path=args.out, encoder=args.encoder, image_size=args.image_size,
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, crops_per_image=args.crops_per_image,
        occlusion=occ, cldice_weight=args.cldice_weight, max_per_dir=args.max_per_dir,
        num_workers=args.num_workers, device=args.device,
    ))


if __name__ == "__main__":
    main()
