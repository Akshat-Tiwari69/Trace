"""CPU smoke tests for the A12 mean-teacher self-training harness."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def _write_pairs(folder, n, size=64):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        rng = np.random.default_rng(i + abs(hash(str(folder))) % 1000)
        Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(folder / f"t{i}_sat.jpg")
        m = np.zeros((size, size), np.uint8)
        m[size // 2 - 2:size // 2 + 2, :] = 255
        Image.fromarray(m, mode="L").save(folder / f"t{i}_mask.png")


def test_unlabeled_dataset_yields_two_perturbed_views(tmp_path):
    from src.pipeline.p1_segment.train_selftrain import UnlabeledTileDataset, list_images

    d = tmp_path / "corpus"
    _write_pairs(d, 3, size=80)
    imgs = list_images([str(d)])
    assert len(imgs) == 3                              # globs *_sat.jpg (masks ignored)

    ds = UnlabeledTileDataset(imgs, size=64)
    weak, strong = ds[0]
    assert weak.shape == (3, 64, 64) and strong.shape == (3, 64, 64)
    assert not torch.allclose(weak, strong)           # strong view is photometrically perturbed


def test_refine_pseudo_drops_small_blobs():
    from src.pipeline.p1_segment.train_selftrain import refine_pseudo

    p = torch.zeros(1, 1, 32, 32)
    p[0, 0, 5:25, 12:15] = 1.0                         # a ~60-px road strip → keep
    p[0, 0, 0:2, 0:2] = 1.0                            # a 4-px speckle → drop
    out = refine_pseudo(p, min_size=20)
    assert out[0, 0, 5:25, 12:15].sum() > 0
    assert out[0, 0, 0:2, 0:2].sum() == 0


def test_consistency_loss_ignores_unconfident_pixels():
    from src.pipeline.p1_segment.train_selftrain import consistency_loss

    logits = torch.zeros(1, 1, 4, 4)                   # sigmoid 0.5 everywhere
    pseudo = torch.ones(1, 1, 4, 4)
    assert consistency_loss(logits, pseudo, torch.zeros(1, 1, 4, 4)).item() == 0.0   # all masked → 0
    assert consistency_loss(logits, pseudo, torch.ones(1, 1, 4, 4)).item() > 0.0     # confident → >0


def test_train_selftrain_runs_and_saves(tmp_path):
    from src.pipeline.p1_segment.model import load_checkpoint
    from src.pipeline.p1_segment.train_selftrain import SelfTrainConfig, train_selftrain

    lab, unl, val = tmp_path / "lab", tmp_path / "unl", tmp_path / "val"
    _write_pairs(lab, 4); _write_pairs(unl, 4); _write_pairs(val, 2)
    out = tmp_path / "selftrain.pt"
    cfg = SelfTrainConfig(
        labeled_dirs=[str(lab)], unlabeled_dirs=[str(unl)], val_dirs=[str(val)],
        out_path=out, encoder="mit_b0", encoder_weights=None, image_size=64,
        labeled_batch_size=2, unlabeled_batch_size=2, epochs=1, warmup_epochs=0,
        consistency_rampup_epochs=1, device="cpu",
    )
    summary = train_selftrain(cfg)
    assert out.exists() and summary["best_val_iou"] >= 0.0
    _, meta = load_checkpoint(out)
    assert "self-training" in meta["recipe"]
    assert meta["unlabeled_on"] == [str(unl)]
