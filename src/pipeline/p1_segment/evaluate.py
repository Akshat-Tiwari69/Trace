"""CLI: package the A4 segmentation evaluation numbers (task E1, seg lane).

The seg-lane analogue of ``p3_analysis/evaluate.py``. The headline IoU /
Occlusion-Recall were computed on the held-out DeepGlobe validation set during
the A4 Kaggle run and **baked into the checkpoint ``meta``** — this reads them
back, writes a small committed JSON report, and prints a readable summary, so
the numbers are reproducible/inspectable without re-running training.

Optionally (``--image``) it runs a live inference demo on one tile and reports
the predicted road coverage + saves a red overlay — a qualitative real-world
check (no ground truth, so it's coverage/visual only, not an IoU).

Example::

    python -m src.pipeline.p1_segment.evaluate            # report from checkpoint meta
    python -m src.pipeline.p1_segment.evaluate --image data/raw/tile.jpg --aoi demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_CKPT = "models/deepglobe_mit_b3_scse_512px_best.pt"
DEFAULT_OUT = "data/sample/segmentation_eval.json"


def _round(meta: dict, key: str, ndigits: int = 4):
    """Return ``meta[key]`` rounded, or ``None`` if absent."""
    return round(float(meta[key]), ndigits) if key in meta and meta[key] is not None else None


def seg_report(meta: dict[str, Any]) -> dict[str, Any]:
    """Build the segmentation-evaluation summary from a checkpoint's ``meta``."""
    return {
        "model": {
            "encoder": meta.get("encoder"),
            "arch": meta.get("arch"),
            "decoder_attention_type": meta.get("decoder_attention_type"),
            "image_size": meta.get("image_size"),
            "epochs": meta.get("epoch"),
        },
        "dataset": "DeepGlobe Road Extraction (held-out validation)",
        "validation": {
            # full-resolution sliding-window (Hann) validation, EMA weights
            "iou_flip_multiscale_tta": _round(meta, "final_full_resolution_iou"),
            "iou_best_single_view": _round(meta, "best_full_resolution_iou"),
            "deploy_threshold": meta.get("threshold"),
            "clean_iou_at_deploy_threshold": _round(meta, "clean_iou_at_selected_threshold"),
            "occlusion_recall": _round(meta, "occlusion_recall"),
            "threshold_selection": meta.get("threshold_selection"),
            "tta": {"flip": meta.get("final_flip_tta"), "scales": meta.get("final_tta_scales")},
        },
    }


def _load_meta(checkpoint: Path) -> dict[str, Any]:
    """Read just the ``meta`` block (no model build needed for the report)."""
    import torch

    if not checkpoint.exists():
        raise SystemExit(
            f"checkpoint not found: {checkpoint}\n"
            "  Download it from the GitHub Release a4-roadseg-v1 into models/."
        )
    return torch.load(checkpoint, map_location="cpu", weights_only=False).get("meta", {})


def _live_demo(checkpoint: Path, image_path: Path, aoi: str, out_dir: Path) -> dict[str, Any]:
    """Run inference on one real tile; report coverage + save an overlay (no GT)."""
    import cv2
    import numpy as np

    from src.pipeline.p1_segment.model import load_checkpoint, predict_large

    model, meta = load_checkpoint(checkpoint)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"could not read image: {image_path}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    thr = float(meta.get("threshold", 0.5))
    mask = predict_large(model, image, tile_size=512, threshold=thr)

    overlay = image.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.25 + np.array([255, 40, 40]) * 0.75).astype(np.uint8)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / f"{aoi}_seg_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return {
        "aoi": aoi,
        "image": str(image_path),
        "size_px": [int(image.shape[1]), int(image.shape[0])],
        "threshold": thr,
        "road_pixel_fraction": round(float(mask.mean()), 4),
        "overlay": str(overlay_path),
        "note": "qualitative real-world inference — no ground truth, coverage/visual only",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Package the A4 segmentation evaluation numbers.")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT, help="trained .pt (Release a4-roadseg-v1)")
    p.add_argument("--out", default=DEFAULT_OUT, help="where to write the JSON report")
    p.add_argument("--image", default=None, help="optional tile for a live inference demo")
    p.add_argument("--aoi", default="demo", help="AOI id for the demo overlay filename")
    p.add_argument("--interim-dir", default="data/interim", help="dir for the demo overlay")
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    report = seg_report(_load_meta(checkpoint))
    if args.image:
        report["live_demo"] = _live_demo(checkpoint, Path(args.image), args.aoi, Path(args.interim_dir))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    v = report["validation"]
    m = report["model"]
    print(f"\n=== Segmentation evaluation — A4 ({m['encoder']} + {m['arch']}/{m['decoder_attention_type']}) ===")
    print(f"Dataset:      {report['dataset']}")
    print(f"IoU:          {v['iou_flip_multiscale_tta']} (flip+multi-scale TTA) | "
          f"{v['iou_best_single_view']} (best single-view EMA)")
    print(f"Deploy:       thr {v['deploy_threshold']} -> clean IoU {v['clean_iou_at_deploy_threshold']} / "
          f"Occlusion-Recall {v['occlusion_recall']}")
    print(f"Selection:    {v['threshold_selection']}")
    if "live_demo" in report:
        d = report["live_demo"]
        print(f"Live demo:    {d['aoi']} {d['size_px'][0]}x{d['size_px'][1]} -> "
              f"roads {d['road_pixel_fraction']:.2%} -> {d['overlay']}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
