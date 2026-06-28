"""Dataset + augmentation for road segmentation (task A4).

``RoadTileDataset`` yields ``(image, mask)`` tensor pairs for training. The
training augmentation includes **CoarseDropout** — the cheap occlusion
simulation from ``docs/Research.md`` that randomly blanks patches so the model
learns to "see through" trees/vehicles. The same dropout boxes are returned as
an *occlusion mask* so ``occlusion_recall`` can score recovery under occlusion.

Works with any ``(image, mask)`` tile pairs; ``pair_deepglobe`` discovers the
DeepGlobe layout (``{id}_sat.jpg`` / ``{id}_mask.png``), our primary corpus.

``albumentations`` is imported lazily so importing this module (and the model /
metrics / loss modules) doesn't require it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.pipeline.p1_segment.model import IMAGENET_MEAN, IMAGENET_STD


def pair_deepglobe(root: str | Path) -> list[tuple[Path, Path]]:
    """Find ``(sat_image, mask)`` pairs in a DeepGlobe-style directory."""
    root = Path(root)
    pairs: list[tuple[Path, Path]] = []
    for sat in sorted(root.glob("*_sat.jpg")):
        mask = sat.with_name(sat.name.replace("_sat.jpg", "_mask.png"))
        if mask.exists():
            pairs.append((sat, mask))
    return pairs


def build_train_transform(size: int = 256, occlusion: bool | str = True) -> Any:
    """Training augmentation: flips/rotate/colour (+ optional occlusion), norm.

    ``occlusion`` controls the occlusion-simulation stack (task A8):
    ``True`` = the A4-baseline CoarseDropout boxes; ``"heavy"`` = a stronger stack
    (more/larger CoarseDropout + RandomShadow + hue/sat jitter) to push
    Occlusion-Recall; ``False`` = none.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    heavy = occlusion == "heavy"
    augs = [
        A.RandomCrop(size, size) if size else A.NoOp(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3 if heavy else 0.2,
                                   contrast_limit=0.3 if heavy else 0.2,
                                   p=0.4 if heavy else 0.3),
    ]
    if occlusion:
        # blank boxes (~1/16–1/6 of the tile) to mimic trees/buildings/vehicles;
        # "heavy" uses more, larger boxes more often (albumentations 2.x API).
        hole = max(1, size // (6 if heavy else 8))
        lo = max(1, size // 16)
        augs.append(
            A.CoarseDropout(
                num_holes_range=(1, 12 if heavy else 8),
                hole_height_range=(lo, hole),
                hole_width_range=(lo, hole),
                fill=0,
                fill_mask=None,  # keep the road label under the hole
                p=0.6 if heavy else 0.5,
            )
        )
    if heavy:
        augs += [
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.RandomShadow(p=0.3),  # cast shadows — a real occlusion mode
        ]
    augs += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(augs)


def build_val_transform(size: int = 256, grayscale: bool = False) -> Any:
    """Validation transform: a native-resolution centre crop (same scale as the
    train RandomCrop) + normalise. NOT a full-image resize — resizing a 1024px
    DeepGlobe tile down to 256 shrinks thin roads ~4x, evaluating the model at a
    scale it never trained on and pinning val IoU artificially low.

    ``grayscale=True`` desaturates the image (3-channel grey) — a proxy for
    **Cartosat-3 panchromatic** input, to measure the sensor-modality gap (A24).
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    steps = [
        A.PadIfNeeded(size, size, border_mode=0),  # guard images smaller than the crop
        A.CenterCrop(size, size),
    ]
    if grayscale:
        steps.append(A.ToGray(p=1.0))
    steps += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(steps)


class RoadTileDataset(Dataset):
    """``(image, mask)`` road-tile pairs with albumentations transforms.

    Returns ``image`` as a ``(3,H,W)`` float tensor and ``mask`` as a
    ``(1,H,W)`` float tensor in {0,1}.

    ``crops_per_image`` > 1 makes each source image appear that many times per
    epoch — with a random-crop train transform that yields a different 256px
    window each time, so a 1024px DeepGlobe tile (16 non-overlapping windows)
    is actually used instead of sampling just one crop per epoch. Big data-
    efficiency win; keep it at 1 for deterministic val (centre crop).
    """

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        transform: Any | None = None,
        crops_per_image: int = 1,
    ) -> None:
        if not pairs:
            raise ValueError("RoadTileDataset got no (image, mask) pairs")
        self.pairs = pairs
        self.transform = transform
        self.crops_per_image = max(1, crops_per_image)

    def __len__(self) -> int:
        return len(self.pairs) * self.crops_per_image

    def _read(self, image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)  # DeepGlobe masks are 0/255
        return image, mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._read(*self.pairs[idx % len(self.pairs)])
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
            mask = mask.unsqueeze(0).float()
            return image, mask
        # no transform → return plain tensors (mainly for tests)
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return image_t, mask_t
