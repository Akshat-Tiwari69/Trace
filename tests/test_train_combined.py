"""CPU smoke test for the from-scratch combined-retrain harness (A11)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.pipeline.p1_segment.model import load_checkpoint
from src.pipeline.p1_segment.train_combined import ModelEMA, TrainConfig, gather, train_combined
from src.pipeline.p1_segment.model import build_model


def _write_pairs(folder, n, size=64):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        rng = np.random.default_rng(i + abs(hash(str(folder))) % 1000)
        Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(folder / f"t{i}_sat.jpg")
        m = np.zeros((size, size), np.uint8)
        m[size // 2 - 2:size // 2 + 2, :] = 255
        Image.fromarray(m, mode="L").save(folder / f"t{i}_mask.png")


def test_gather_concatenates_dirs(tmp_path):
    a, b = tmp_path / "deepglobe", tmp_path / "mass"
    _write_pairs(a, 6); _write_pairs(b, 4)
    cfg = TrainConfig(data_dirs=[str(a), str(b)], val_fraction=0.2)
    train, val = gather(cfg)
    assert len(train) + len(val) == 10
    assert len(val) == 2


def test_ema_tracks_weights():
    net = build_model(encoder_weights=None, decoder_attention_type="scse")
    ema = ModelEMA(net, decay=0.5)
    # change a param, update EMA, confirm EMA moved toward it
    with __import__("torch").no_grad():
        for p in net.parameters():
            p.add_(1.0); break
    before = next(iter(ema.module.parameters())).clone()
    ema.update(net)
    after = next(iter(ema.module.parameters()))
    assert not after.equal(before)


def test_train_combined_runs_and_saves(tmp_path):
    a, b = tmp_path / "deepglobe", tmp_path / "mass"
    _write_pairs(a, 4); _write_pairs(b, 4)
    out = tmp_path / "combined.pt"
    cfg = TrainConfig(
        data_dirs=[str(a), str(b)], out_path=out, encoder="mit_b0", encoder_weights=None,
        image_size=64, batch_size=2, epochs=1, warmup_epochs=0, crops_per_image=1,
        val_fraction=0.25, occlusion=True, device="cpu",
    )
    summary = train_combined(cfg)
    assert out.exists() and summary["best_val_iou"] >= 0.0
    _, meta = load_checkpoint(out)
    assert meta["decoder_attention_type"] == "scse"
    assert "trained_on" in meta and len(meta["trained_on"]) == 2
