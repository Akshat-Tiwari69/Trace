"""Unit tests for the seg-lane E1 evaluator. CPU, no real checkpoint needed."""

from __future__ import annotations

import json

from src.pipeline.p1_segment.evaluate import seg_report, _load_meta
from src.pipeline.p1_segment.model import build_model, save_checkpoint

# A meta block shaped like the real A4 checkpoint's.
META = {
    "encoder": "mit_b3", "arch": "unet", "decoder_attention_type": "scse",
    "image_size": 512, "epoch": 30,
    "final_full_resolution_iou": 0.6698520785889119,
    "best_full_resolution_iou": 0.6637514999623423,
    "threshold": 0.44, "clean_iou_at_selected_threshold": 0.6617056272464931,
    "occlusion_recall": 0.7927011592853116,
    "threshold_selection": "max occlusion recall with <= 0.010 clean-IoU drop",
    "final_flip_tta": True, "final_tta_scales": [1.0, 1.25],
}


def test_seg_report_pulls_validation_metrics():
    r = seg_report(META)
    assert r["model"]["encoder"] == "mit_b3"
    assert r["model"]["decoder_attention_type"] == "scse"
    assert r["validation"]["iou_flip_multiscale_tta"] == 0.6699   # rounded to 4dp
    assert r["validation"]["deploy_threshold"] == 0.44
    assert r["validation"]["occlusion_recall"] == 0.7927
    assert r["dataset"].startswith("DeepGlobe")


def test_seg_report_tolerates_missing_keys():
    r = seg_report({"encoder": "mit_b0"})          # sparse meta must not crash
    assert r["model"]["encoder"] == "mit_b0"
    assert r["validation"]["iou_flip_multiscale_tta"] is None
    assert r["validation"]["occlusion_recall"] is None


def test_main_refuses_null_report(tmp_path, monkeypatch):
    """A checkpoint whose meta has no validation metrics (e.g. the deployed
    fine-tune) must NOT overwrite the committed report with nulls — main()
    exits loudly instead (the guard added after exactly that clobber)."""
    import pytest

    from src.pipeline.p1_segment import evaluate as ev

    model = build_model(encoder_weights=None, decoder_attention_type="scse")
    ckpt = tmp_path / "no_metrics.pt"
    # threshold present (like road_pan.pt) but zero eval metrics
    save_checkpoint(model, ckpt, meta={"encoder": "mit_b3", "arch": "unet",
                                       "decoder_attention_type": "scse",
                                       "image_size": 512, "threshold": 0.52})
    out = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["evaluate", "--checkpoint", str(ckpt),
                                     "--out", str(out)])
    with pytest.raises(SystemExit, match="no validation metrics"):
        ev.main()
    assert not out.exists()


def test_load_meta_roundtrips_from_a_checkpoint(tmp_path):
    # write a real (tiny, random) checkpoint with our meta, read meta back
    model = build_model(encoder_weights=None, decoder_attention_type="scse")
    ckpt = tmp_path / "tiny.pt"
    save_checkpoint(model, ckpt, meta=META)
    meta = _load_meta(ckpt)
    assert meta["occlusion_recall"] == META["occlusion_recall"]
    # and the report serialises cleanly to JSON
    json.dumps(seg_report(meta))
