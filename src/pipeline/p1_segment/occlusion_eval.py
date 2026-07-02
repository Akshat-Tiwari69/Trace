"""Occlusion-Recall evaluation (tasks A8 / E3).

The project's headline metric is **Occlusion-Recall** — of the true road pixels
deliberately hidden behind boxes, how many does the model still predict? This
module occludes held-out tiles, runs inference on the *occluded* image, and
measures recovery inside the boxes (clean IoU is reported alongside as the cost).
Used to test whether heavier occlusion augmentation (A8) actually helps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.pipeline.p1_segment.model import predict_large


def occlude(image: np.ndarray, rng: np.random.Generator,
            max_boxes: int = 6, frac: tuple[float, float] = (1 / 16, 1 / 7)
            ) -> tuple[np.ndarray, np.ndarray]:
    """Black out a few random boxes; return ``(occluded_image, occlusion_mask)``."""
    out = image.copy()
    h, w = out.shape[:2]
    occ = np.zeros((h, w), np.uint8)
    for _ in range(int(rng.integers(1, max_boxes + 1))):
        bh = int(rng.integers(max(1, int(h * frac[0])), max(2, int(h * frac[1]))))
        bw = int(rng.integers(max(1, int(w * frac[0])), max(2, int(w * frac[1]))))
        y = int(rng.integers(0, max(1, h - bh)))
        x = int(rng.integers(0, max(1, w - bw)))
        out[y:y + bh, x:x + bw] = 0
        occ[y:y + bh, x:x + bw] = 1
    return out, occ


def evaluate_occlusion_recall(model, pairs: list[tuple[Path, Path]], device: str = "cpu",
                              threshold: float = 0.5, tile_size: int = 512, seed: int = 0) -> dict:
    """Mean Occlusion-Recall + clean IoU over (sat, mask) pairs.

    Occlusion-Recall: of true-road pixels inside the boxes, the fraction predicted.
    Clean IoU: IoU on the *un*-occluded image (the cost side of the trade-off).
    """
    from src.pipeline.p1_segment.raster_io import imread_gray, imread_rgb

    rng = np.random.default_rng(seed)
    hidden_road = recovered = 0
    inter = union = 0
    for sat_path, mask_path in pairs:
        img = imread_rgb(sat_path)
        gt = imread_gray(mask_path) > 127

        occ_img, occ = occlude(img, rng)
        pred_occ = predict_large(model, occ_img, tile_size=tile_size, device=device, threshold=threshold) > 0
        hidden = gt & occ.astype(bool)
        hidden_road += int(hidden.sum())
        recovered += int((pred_occ & hidden).sum())

        pred_clean = predict_large(model, img, tile_size=tile_size, device=device, threshold=threshold) > 0
        inter += int(np.logical_and(pred_clean, gt).sum())
        union += int(np.logical_or(pred_clean, gt).sum())

    return {
        "occlusion_recall": recovered / max(hidden_road, 1),
        "clean_iou": inter / max(union, 1),
        "n_pairs": len(pairs),
    }
