"""A12 — topology-aware self-training on the in-domain Indian corpus.

The post-A11 lever (`docs/Research.md` → Post-A11 Build Plan): improve the Indian
deployment target by **mean-teacher self-training** on lots of unlabeled Indian
tiles, the way the published cross-domain road work does.

Recipe (a pragmatic UniMatch/FixMatch-style core, per `docs/Rules.md` "simplicity
first"):
- **Labeled** stream = real labels (DeepGlobe + SpaceNet-5 Mumbai): supervised
  ComboLoss on the student.
- **Unlabeled** stream = the Indian OSM corpus *images* (OSM masks ignored — too
  noisy to trust directly): an **EMA teacher** pseudo-labels a *weakly*-augmented
  view; the **student** must match it on a *strongly*-augmented (same-geometry)
  view — weak→strong consistency.
- **Connectivity-refined pseudo-labels**: drop small disconnected pseudo-road
  blobs (noise) so self-training doesn't amplify OSM/​prediction noise — the
  key safeguard from the topology-aware UDA papers.
- Confidence-gated: only train consistency where the teacher is confident.

GPU step; the orchestration is CPU-smoke-tested. Validate on a held-out Indian
split + DeepGlobe; release only if it beats v2 (same honest bar).
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.pipeline.p1_segment.dataset import (
    RoadTileDataset,
    build_train_transform,
    build_val_transform,
    pair_deepglobe,
)
from src.pipeline.p1_segment.losses import ComboLoss
from src.pipeline.p1_segment.model import IMAGENET_MEAN, IMAGENET_STD, build_model, load_checkpoint, save_checkpoint
from src.pipeline.p1_segment.train import evaluate
from src.pipeline.p1_segment.train_combined import ModelEMA


# --------------------------------------------------------------------------- #
# Unlabeled stream: weak + strong views that share the SAME geometry
# --------------------------------------------------------------------------- #
def _geometric_transform(size: int):
    """Flips/rotate/crop only (no colour, no normalise) → uint8 H×W×3."""
    import albumentations as A

    return A.Compose([
        A.RandomCrop(size, size) if size else A.NoOp(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])


def _weak_transform():
    """Weak view = normalise only (teacher sees clean pixels)."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


def _strong_transform(size: int):
    """Strong view = photometric jitter + blur + cutout, then normalise.

    Geometry is already fixed by ``_geometric_transform`` so the student's strong
    prediction aligns pixel-for-pixel with the teacher's weak pseudo-label.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    hole = max(1, size // 6)
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
        A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=18, val_shift_limit=12, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(hole // 2, hole),
                        hole_width_range=(hole // 2, hole), fill=0, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def list_images(dirs: list[str]) -> list[Path]:
    """All ``*_sat.jpg`` images under the given DeepGlobe-format dirs (masks ignored)."""
    imgs: list[Path] = []
    for d in dirs:
        imgs += sorted(Path(d).glob("*_sat.jpg"))
    return imgs


class UnlabeledTileDataset(Dataset):
    """Yields ``(weak, strong)`` tensor views of one unlabeled tile (same geometry)."""

    def __init__(self, images: list[Path], size: int = 512) -> None:
        if not images:
            raise ValueError("UnlabeledTileDataset got no images")
        self.images = images
        self.geo = _geometric_transform(size)
        self.weak = _weak_transform()
        self.strong = _strong_transform(size)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        import cv2

        bgr = cv2.imread(str(self.images[idx]), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        geo = self.geo(image=rgb)["image"]            # uint8 H×W×3, geometry fixed
        return self.weak(image=geo)["image"], self.strong(image=geo)["image"]


# --------------------------------------------------------------------------- #
# Connectivity refinement + consistency loss
# --------------------------------------------------------------------------- #
def refine_pseudo(pseudo: torch.Tensor, min_size: int = 64) -> torch.Tensor:
    """Drop small disconnected pseudo-road blobs (noise); keep connected roads.

    ``pseudo`` is ``(B,1,H,W)`` in {0,1}; returns the same shape with components
    smaller than ``min_size`` pixels zeroed — the "drop discontinuous pseudo-labels"
    safeguard so self-training doesn't reinforce speckle.
    """
    from scipy import ndimage

    out = pseudo.clone()
    arr = pseudo.squeeze(1).bool().cpu().numpy()       # (B,H,W)
    for i in range(arr.shape[0]):
        lab, n = ndimage.label(arr[i])
        if n == 0:
            out[i, 0] = 0.0
            continue
        counts = np.bincount(lab.ravel())
        keep = np.isin(lab, [j for j in range(1, len(counts)) if counts[j] >= min_size])
        out[i, 0] = torch.from_numpy(keep.astype(np.float32)).to(pseudo.device)
    return out


def consistency_loss(student_logits: torch.Tensor, pseudo: torch.Tensor,
                     conf_mask: torch.Tensor) -> torch.Tensor:
    """Confidence-masked BCE between the student's strong-view logits and the
    teacher's pseudo-labels (mean over confident pixels only)."""
    import torch.nn.functional as F

    per_px = F.binary_cross_entropy_with_logits(student_logits, pseudo, reduction="none")
    return (per_px * conf_mask).sum() / conf_mask.sum().clamp_min(1.0)


@dataclasses.dataclass
class SelfTrainConfig:
    labeled_dirs: list[str]              # real labels (DeepGlobe + SpaceNet-5 Mumbai)
    unlabeled_dirs: list[str]            # Indian OSM corpus (images only)
    val_dirs: list[str]                  # held-out split for selection (NOT the final test)
    init_checkpoint: str | None = None   # warm-start (v1/v2); None = ImageNet encoder
    out_path: str | Path = "models/road_selftrain.pt"
    encoder: str = "mit_b3"
    encoder_weights: str | None = "imagenet"
    decoder_attention_type: str | None = "scse"
    image_size: int = 512
    labeled_batch_size: int = 4
    unlabeled_batch_size: int = 4
    lr: float = 1.0e-4
    encoder_lr_scale: float = 0.35
    epochs: int = 20
    warmup_epochs: int = 1
    consistency_weight: float = 1.0      # λ ceiling
    consistency_rampup_epochs: int = 5   # ramp λ 0→ceiling (don't trust early pseudo-labels)
    conf_threshold: float = 0.7
    min_pseudo_size: int = 64
    connectivity_refine: bool = True
    cldice_weight: float = 0.1
    ema_decay: float = 0.999
    threshold: float = 0.44
    num_workers: int = 0
    device: str = "cpu"
    seed: int = 2026


def _optimizer(net, cfg: SelfTrainConfig):
    enc = [p for n, p in net.named_parameters() if n.startswith("encoder.")]
    dec = [p for n, p in net.named_parameters() if not n.startswith("encoder.")]
    return torch.optim.AdamW(
        [{"params": dec, "lr": cfg.lr},
         {"params": enc, "lr": cfg.lr * cfg.encoder_lr_scale}],
        betas=(0.9, 0.999), weight_decay=1e-4)


def _build_student(cfg: SelfTrainConfig):
    if cfg.init_checkpoint:
        student, _ = load_checkpoint(cfg.init_checkpoint, map_location="cpu")   # warm-start
        return student
    return build_model(encoder=cfg.encoder, encoder_weights=cfg.encoder_weights,
                       decoder_attention_type=cfg.decoder_attention_type)


def train_selftrain(cfg: SelfTrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = cfg.device
    student = _build_student(cfg).to(device)
    teacher = ModelEMA(student, cfg.ema_decay)         # EMA teacher generates pseudo-labels

    labeled_pairs: list = []
    for d in cfg.labeled_dirs:
        labeled_pairs += pair_deepglobe(d)
    val_pairs: list = []
    for d in cfg.val_dirs:
        val_pairs += pair_deepglobe(d)
    unlabeled_imgs = list_images(cfg.unlabeled_dirs)
    print(f"labeled {len(labeled_pairs)} | unlabeled {len(unlabeled_imgs)} | val {len(val_pairs)}", flush=True)

    lab_ds = RoadTileDataset(labeled_pairs, build_train_transform(cfg.image_size, occlusion=True))
    unl_ds = UnlabeledTileDataset(unlabeled_imgs, cfg.image_size)
    val_ds = RoadTileDataset(val_pairs, build_val_transform(cfg.image_size))
    lab_loader = DataLoader(lab_ds, batch_size=cfg.labeled_batch_size, shuffle=True, drop_last=True,
                            num_workers=cfg.num_workers, pin_memory=(device != "cpu"))
    unl_loader = DataLoader(unl_ds, batch_size=cfg.unlabeled_batch_size, shuffle=True, drop_last=True,
                            num_workers=cfg.num_workers, pin_memory=(device != "cpu"))
    val_loader = DataLoader(val_ds, batch_size=cfg.labeled_batch_size, shuffle=False, num_workers=0)

    loss_fn = ComboLoss(bce_weight=0.4, dice_weight=0.4, lovasz_weight=0.2, cldice_weight=cfg.cldice_weight)
    optimizer = _optimizer(student, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=(device != "cpu"))
    steps_per_epoch = max(1, len(lab_loader))
    out = Path(cfg.out_path)
    best_iou, history = -1.0, []

    for epoch in range(1, cfg.epochs + 1):
        student.train()
        lam = cfg.consistency_weight * min(1.0, epoch / max(1, cfg.consistency_rampup_epochs))
        unl_iter = itertools.cycle(unl_loader)         # cycle unlabeled to match labeled length
        run_sup = run_cons = 0.0
        for images, masks in lab_loader:
            images, masks = images.to(device), masks.to(device)
            weak, strong = next(unl_iter)
            weak, strong = weak.to(device), strong.to(device)

            with torch.no_grad():                      # teacher pseudo-labels the weak view
                prob = torch.sigmoid(teacher.module(weak))
                pseudo = (prob > 0.5).float()
                conf = torch.where(pseudo > 0.5, prob, 1.0 - prob)
                conf_mask = (conf > cfg.conf_threshold).float()
                if cfg.connectivity_refine:
                    refined = refine_pseudo(pseudo, cfg.min_pseudo_size)
                    dropped = ((pseudo > 0.5) & (refined < 0.5)).float()   # speckle → don't supervise
                    conf_mask = conf_mask * (1.0 - dropped)
                    pseudo = refined

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda" if device != "cpu" else "cpu", enabled=(device != "cpu")):
                sup = loss_fn(student(images), masks)
                cons = consistency_loss(student(strong), pseudo, conf_mask)
                loss = sup + lam * cons
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            teacher.update(student)
            run_sup += float(sup.detach())
            run_cons += float(cons.detach())

        val = evaluate(teacher.module, val_loader, device, threshold=cfg.threshold)  # evaluate the teacher
        history.append({"epoch": epoch, "sup": run_sup / steps_per_epoch,
                        "cons": run_cons / steps_per_epoch, "lambda": lam, "val_iou": val["iou"]})
        print(f"epoch {epoch:02d} | sup {run_sup/steps_per_epoch:.4f} | cons {run_cons/steps_per_epoch:.4f} "
              f"(λ {lam:.2f}) | teacher val IoU {val['iou']:.4f}", flush=True)
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            save_checkpoint(teacher.module, out, meta={
                "encoder": cfg.encoder, "arch": "unet", "decoder_attention_type": cfg.decoder_attention_type,
                "image_size": cfg.image_size, "threshold": cfg.threshold, "val_iou": float(best_iou),
                "epoch": epoch, "recipe": "A12 mean-teacher self-training (weak→strong + connectivity refine)",
                "labeled_on": cfg.labeled_dirs, "unlabeled_on": cfg.unlabeled_dirs,
            })
            print(f"  saved best teacher -> {out} (val IoU {best_iou:.4f})", flush=True)

    print(f"\ndone. best teacher val IoU {best_iou:.4f} -> {out}")
    return {"best_val_iou": best_iou, "out": str(out), "history": history}


def main() -> None:
    p = argparse.ArgumentParser(description="A12 topology-aware self-training on the Indian corpus.")
    p.add_argument("--labeled-dirs", nargs="+", required=True, help="real-label dirs (DeepGlobe + SpaceNet)")
    p.add_argument("--unlabeled-dirs", nargs="+", required=True, help="Indian OSM corpus dirs (images)")
    p.add_argument("--val-dirs", nargs="+", required=True, help="held-out split for selection (NOT final test)")
    p.add_argument("--init-checkpoint", default=None, help="warm-start checkpoint (v1/v2)")
    p.add_argument("--out", default="models/road_selftrain.pt")
    p.add_argument("--encoder", default="mit_b3")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--conf-threshold", type=float, default=0.7)
    p.add_argument("--consistency-weight", type=float, default=1.0)
    p.add_argument("--no-refine", action="store_true", help="disable connectivity refinement")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    train_selftrain(SelfTrainConfig(
        labeled_dirs=args.labeled_dirs, unlabeled_dirs=args.unlabeled_dirs, val_dirs=args.val_dirs,
        init_checkpoint=args.init_checkpoint, out_path=args.out, encoder=args.encoder, epochs=args.epochs,
        conf_threshold=args.conf_threshold, consistency_weight=args.consistency_weight,
        connectivity_refine=not args.no_refine, num_workers=args.num_workers, device=args.device,
    ))


if __name__ == "__main__":
    main()
