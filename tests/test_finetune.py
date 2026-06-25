"""A6 fine-tune harness test — runs the orchestration on CPU, no GPU, tiny model."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.pipeline.p1_segment.finetune import FineTuneConfig, finetune, gather_pairs
from src.pipeline.p1_segment.model import build_model, load_checkpoint, save_checkpoint


def _write_pair(folder, stem, size=80):
    """Write one DeepGlobe-format pair (0/255 mask, the convention readers expect)."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(stem)) % 2**32)
    Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(folder / f"{stem}_sat.jpg")
    mask = np.zeros((size, size), np.uint8)
    mask[size // 2 - 2 : size // 2 + 2, :] = 255          # a horizontal road
    Image.fromarray(mask, mode="L").save(folder / f"{stem}_mask.png")


def _tiny_v1_checkpoint(path):
    model = build_model(encoder_weights=None, decoder_attention_type="scse")
    save_checkpoint(model, path, meta={"encoder": "mit_b0", "arch": "unet",
                                        "decoder_attention_type": "scse", "image_size": 64, "threshold": 0.44})


def test_gather_pairs_oversamples_and_splits(tmp_path):
    ft = tmp_path / "finetune"
    for i in range(6):
        _write_pair(ft, f"city_{i}")
    cfg = FineTuneConfig(init_checkpoint="x", finetune_dir=ft, finetune_oversample=3, val_fraction=0.25)
    train, val = gather_pairs(cfg)
    assert len(val) == max(1, round(6 * 0.25))            # held-out Indian pairs
    assert len(train) == (6 - len(val)) * 3               # remainder oversampled ×3
    assert set(p[0] for p in val).isdisjoint(p[0] for p in train)  # no leakage


def test_finetune_runs_and_saves_a_better_metaed_checkpoint(tmp_path):
    ft = tmp_path / "finetune"
    for i in range(5):
        _write_pair(ft, f"city_{i}")
    init = tmp_path / "v1.pt"
    _tiny_v1_checkpoint(init)
    out = tmp_path / "v2.pt"

    cfg = FineTuneConfig(init_checkpoint=init, finetune_dir=ft, out_path=out,
                         image_size=64, batch_size=2, epochs=1, finetune_oversample=2,
                         crops_per_image=2, device="cpu")
    summary = finetune(cfg)

    assert out.exists()
    assert summary["best_val_iou"] >= 0.0
    # the saved checkpoint reloads and carries fine-tune provenance + the v1 arch
    model, meta = load_checkpoint(out)
    assert meta["decoder_attention_type"] == "scse"
    assert meta["finetuned_from"] == str(init)
    assert meta["threshold"] == 0.44
