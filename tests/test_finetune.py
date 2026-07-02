"""A6 fine-tune harness test — orchestration + anti-forget selection on CPU."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.pipeline.p1_segment.finetune import (
    FineTuneConfig,
    _build_optimizer,
    finetune,
    gather_pairs,
)
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


def test_gather_pairs_3way_split_disjoint(tmp_path):
    ft, dg = tmp_path / "ft", tmp_path / "dg"
    for i in range(10):
        _write_pair(ft, f"c{i}")
    for i in range(30):
        _write_pair(dg, f"d{i}")
    cfg = FineTuneConfig(init_checkpoint="x", finetune_dir=ft, deepglobe_dir=dg,
                         deepglobe_subset=8, deepglobe_val=5, finetune_oversample=2, val_fraction=0.2)
    train, ind_val, dg_val = gather_pairs(cfg)
    assert len(ind_val) == 2                       # round(10 * 0.2)
    assert len(dg_val) == 5                         # held-out DeepGlobe forget-check
    assert len(train) == (10 - 2) * 2 + 8          # indian ×2 + anchor
    # the forget-check tiles must NOT leak into the training anchor
    assert {p[0] for p in dg_val}.isdisjoint(p[0] for p in train)


def test_freeze_encoder_disables_encoder_grads(tmp_path):
    model = build_model(encoder_weights=None, decoder_attention_type="scse")
    cfg = FineTuneConfig(init_checkpoint="x", finetune_dir=tmp_path, encoder_lr_scale=0.0)
    _build_optimizer(model, cfg)
    enc = [p.requires_grad for n, p in model.named_parameters() if n.startswith("encoder.")]
    dec = [p.requires_grad for n, p in model.named_parameters() if not n.startswith("encoder.")]
    assert not any(enc) and all(dec)               # encoder frozen, decoder trainable


def test_finetune_selects_and_saves_releasable_checkpoint(tmp_path):
    ft, dg = tmp_path / "ft", tmp_path / "dg"
    for i in range(5):
        _write_pair(ft, f"c{i}")
    for i in range(8):
        _write_pair(dg, f"d{i}")
    init = tmp_path / "v1.pt"
    _tiny_v1_checkpoint(init)
    out = tmp_path / "v2.pt"
    cfg = FineTuneConfig(init_checkpoint=init, finetune_dir=ft, deepglobe_dir=dg,
                         deepglobe_subset=4, deepglobe_val=2, out_path=out, image_size=64,
                         batch_size=2, epochs=1, finetune_oversample=2, crops_per_image=1,
                         deepglobe_iou_tolerance=1.0, device="cpu")  # big tol → always "keeps"
    summary = finetune(cfg)

    assert out.exists() and summary["best"] is not None
    model, meta = load_checkpoint(out)
    assert meta["encoder_frozen"] is True
    assert "indian_val_iou" in meta and "deepglobe_val_iou" in meta


def test_finetune_grayscale_tracks_pan_proxy(tmp_path):
    """A24: grayscale_p>0 records the Cartosat-PAN (grayscale) IoU each epoch + in meta."""
    ft, dg = tmp_path / "ft", tmp_path / "dg"
    for i in range(5):
        _write_pair(ft, f"c{i}")
    for i in range(6):
        _write_pair(dg, f"d{i}")
    init = tmp_path / "v1.pt"
    _tiny_v1_checkpoint(init)
    out = tmp_path / "v32.pt"
    cfg = FineTuneConfig(init_checkpoint=init, finetune_dir=ft, deepglobe_dir=dg,
                         deepglobe_subset=3, deepglobe_val=2, out_path=out, image_size=64,
                         batch_size=2, epochs=1, finetune_oversample=2, grayscale_p=0.5,
                         deepglobe_iou_tolerance=1.0, device="cpu")
    summary = finetune(cfg)
    # per-epoch grayscale IoU is a real number (not the nan sentinel used when off)
    assert np.isfinite(summary["history"][0]["indian_gray_iou"])
    _, meta = load_checkpoint(out)
    assert "indian_gray_val_iou" in meta and np.isfinite(meta["indian_gray_val_iou"])
