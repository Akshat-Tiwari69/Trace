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


def build_train_transform(size: int = 256, occlusion: bool = True) -> Any:
    """Training augmentation: flips/rotate/colour (+ optional occlusion), norm."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    augs = [
        A.RandomCrop(size, size) if size else A.NoOp(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
    ]
    if occlusion:
        # blank up to 8 boxes (~1/16–1/8 of the tile each) to mimic
        # trees/buildings/vehicles (albumentations 2.x CoarseDropout API)
        hole = max(1, size // 8)
        lo = max(1, size // 16)
        augs.append(
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(lo, hole),
                hole_width_range=(lo, hole),
                fill=0,
                fill_mask=None,  # keep the road label under the hole
                p=0.5,
            )
        )
    augs += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(augs)


def build_val_transform(size: int = 256) -> Any:
    """Validation transform: deterministic resize + normalise (no occlusion)."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class RoadTileDataset(Dataset):
    """``(image, mask)`` road-tile pairs with albumentations transforms.

    Returns ``image`` as a ``(3,H,W)`` float tensor and ``mask`` as a
    ``(1,H,W)`` float tensor in {0,1}.
    """

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        transform: Any | None = None,
    ) -> None:
        if not pairs:
            raise ValueError("RoadTileDataset got no (image, mask) pairs")
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def _read(self, image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)  # DeepGlobe masks are 0/255
        return image, mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._read(*self.pairs[idx])
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
            mask = mask.unsqueeze(0).float()
            return image, mask
        # no transform → return plain tensors (mainly for tests)
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return image_t, mask_t
