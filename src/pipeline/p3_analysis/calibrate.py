"""A46 (graph-first) — routing-first threshold/radius calibration.

A46 asks for **existing-checkpoint APLS selection and threshold/radius
calibration first**, before any training. This is the graph lane's half of that:
find the decode settings that maximise *routing* quality (APLS) on an existing
checkpoint — no retraining, no new data.

Two free levers the current decode path leaves on the table:

* **Threshold.** v3.2 deploys the checkpoint's meta threshold, which was selected
  on *pixel* criteria (clean IoU / occlusion-recall). The threshold that maximises
  pixels is not generally the one that maximises routing: a slightly lower cut
  can close a gap that reconnects a whole component (large APLS gain, tiny IoU
  cost), and a higher cut can drop speckle that adds spurious junctions.
* **Healing radius.** ``chip_apls_eval.mask_to_apls_graph_aniso`` scores a **raw
  skeleton** — the S1 MST/Union-Find healing never runs. Bridging genuine gaps is
  precisely what APLS rewards, so a healing radius is a second, independent knob.

Design note — **predict once, threshold many times.** Inference dominates cost;
thresholding a cached probability map is nearly free. So the sweep consumes
*probability* maps and re-thresholds them, rather than re-running the model per
candidate.

Nothing here promotes anything: it reports a table plus a **paired bootstrap CI**
of the best setting against the current default, per the A46 promotion protocol
(``docs/Evaluation.md`` — the interval must exclude zero in the candidate's
favour). Selection must run on the 102-chip validation split; the 127-chip
comparison stays closed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class Setting:
    """One decode candidate: a probability threshold + an optional heal radius."""

    threshold: float
    gap_max_m: float = 0.0  # 0 = no healing (the current raw-skeleton behaviour)

    def label(self) -> str:
        return f"thr={self.threshold:g},heal={self.gap_max_m:g}m"


def graph_from_prob(
    prob: np.ndarray,
    setting: Setting,
    eff_x: float,
    eff_y: float,
    angle_max_deg: float = 60.0,
    frame_size: int | None = None,
):
    """Decode a probability map into an APLS-ready graph under ``setting``.

    Mirrors ``chip_apls_eval.mask_to_apls_graph_aniso`` (anisotropic metres, then
    rescaled to apls' degree convention) but adds the optional healing step —
    applied in **metric** space, before the degree rescale, so ``gap_max_m`` is
    true metres.

    ``frame_size`` reproduces the v3.2 contract: threshold at the model's native
    scale, *then* nearest-resize the **mask** into the common frame (never resize
    the probability — that would smooth across the decision boundary and change
    what a threshold means).
    """
    from affine import Affine
    from skimage.morphology import skeletonize

    from src.pipeline.p1_segment.apls_eval import _DEG_X, _DEG_Y
    from src.pipeline.p2_graph.healing import heal_graph
    from src.pipeline.p2_graph.skeleton_graph import skeleton_to_graph

    mask = np.asarray(prob) >= setting.threshold
    if frame_size is not None and mask.shape != (frame_size, frame_size):
        import cv2

        mask = cv2.resize(mask.astype(np.uint8), (frame_size, frame_size),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    graph = skeleton_to_graph(
        skeletonize(mask), transform=Affine(eff_x, 0.0, 0.0, 0.0, eff_y, 0.0)
    )
    if setting.gap_max_m > 0 and graph.number_of_nodes() >= 2:
        graph, _ = heal_graph(
            graph, gap_max_m=setting.gap_max_m, angle_max_deg=angle_max_deg
        )
    for _, data in graph.nodes(data=True):  # metres -> apls' degree convention
        data["x"] = data["x"] / _DEG_X
        data["y"] = data["y"] / _DEG_Y
    return graph


def score_setting(
    chips: dict,
    setting: Setting,
    score_fn,
    frame_size: int | None = None,
) -> dict[str, float]:
    """Per-chip APLS for one ``setting``.

    ``chips`` maps ``chip_id -> (prob, gt_graph, eff_x, eff_y)``. ``score_fn`` takes
    ``(pred_graph, gt_graph)`` and returns an APLS float (inject
    ``chip_apls_eval.chip_apls`` in production; a stub in tests). NaN scores (GT
    too small to score) are dropped, matching the evaluator's guard.
    """
    scores: dict[str, float] = {}
    for chip, (prob, gt_graph, eff_x, eff_y) in chips.items():
        pred = graph_from_prob(prob, setting, eff_x, eff_y, frame_size=frame_size)
        value = float(score_fn(pred, gt_graph))
        if not np.isnan(value):
            scores[chip] = value
    return scores


def sweep(chips: dict, settings: list[Setting], score_fn,
          frame_size: int | None = None) -> list[dict]:
    """Score every ``setting`` over ``chips``; returns rows sorted best-mean-first."""
    rows = []
    for setting in settings:
        scores = score_setting(chips, setting, score_fn, frame_size=frame_size)
        rows.append({
            "label": setting.label(),
            "threshold": setting.threshold,
            "gap_max_m": setting.gap_max_m,
            "n_chips": len(scores),
            "mean_apls": round(float(np.mean(list(scores.values()))), 4) if scores else 0.0,
            "scores": scores,
        })
    return sorted(rows, key=lambda r: r["mean_apls"], reverse=True)


def paired_gain(baseline: dict[str, float], candidate: dict[str, float]) -> dict:
    """Paired bootstrap CI of ``candidate - baseline`` over the chips both scored.

    Reuses the segmentation lane's ``paired_bootstrap_ci`` so A46's statistics are
    one implementation, not two. Promotion needs the interval to exclude zero in
    the candidate's favour (protocol step 6).
    """
    from src.pipeline.p1_segment.stats import paired_bootstrap_ci

    common = sorted(set(baseline) & set(candidate))
    if not common:
        return {"n_paired": 0, "delta": None, "ci_low": None, "ci_high": None,
                "excludes_zero": False, "verdict": "no paired chips"}
    ci = paired_bootstrap_ci([baseline[c] for c in common],
                             [candidate[c] for c in common])
    return {"n_paired": len(common), "delta": ci.delta, "ci_low": ci.ci_low,
            "ci_high": ci.ci_high, "p_two_sided": ci.p_two_sided,
            "excludes_zero": ci.excludes_zero, "verdict": ci.verdict}


def calibrate(chips: dict, settings: list[Setting], default: Setting, score_fn,
              frame_size: int | None = None) -> dict:
    """Sweep ``settings``, pick the routing-best, and pair it against ``default``.

    Returns the full table plus the paired CI of best-vs-default. Reports only —
    promotion remains a human gate under the A46 protocol.
    """
    rows = sweep(chips, settings, score_fn, frame_size=frame_size)
    best = rows[0]
    default_row = next((r for r in rows if r["label"] == default.label()), None)
    gain = (paired_gain(default_row["scores"], best["scores"])
            if default_row else {"verdict": "default not in sweep"})
    return {
        "n_settings": len(rows),
        "default": default.label(),
        "default_mean_apls": default_row["mean_apls"] if default_row else None,
        "best": best["label"],
        "best_mean_apls": best["mean_apls"],
        "paired_best_vs_default": gain,
        "table": [{k: v for k, v in r.items() if k != "scores"} for r in rows],
    }


def write_report(report: dict, path: Path) -> None:
    """Write the calibration report JSON (report-only; no promotion side effects)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


def format_report(report: dict) -> str:
    """Readable summary of a :func:`calibrate` report."""
    lines = [
        f"routing-first calibration — {report['n_settings']} settings",
        f"  default {report['default']}: mean APLS {report['default_mean_apls']}",
        f"  best    {report['best']}: mean APLS {report['best_mean_apls']}",
    ]
    gain = report["paired_best_vs_default"]
    if gain.get("n_paired"):
        lines.append(
            f"  paired best-vs-default (n={gain['n_paired']}): delta {gain['delta']:+.4f} "
            f"CI [{gain['ci_low']:+.4f}, {gain['ci_high']:+.4f}] "
            f"excludes_zero={gain['excludes_zero']} -> {gain['verdict']}"
        )
    lines.append("  top settings:")
    for row in report["table"][:5]:
        lines.append(f"    {row['label']:<24} mean APLS {row['mean_apls']:.4f} "
                     f"(n={row['n_chips']})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI — run the sweep against v3.2 on the frozen heldout chips
# --------------------------------------------------------------------------- #
def _v32_prob(model, bands: np.ndarray, device: str = "cpu") -> np.ndarray:
    """v3.2's probability map at its **deployed** scale — predicted once per chip.

    Same path as ``chip_apls_eval.v32_chip_graph`` up to (but not including) the
    threshold, so the sweep re-thresholds exactly what v3.2 would have decoded,
    without paying for inference per candidate.
    """
    import cv2

    from src.pipeline.p1_segment.chip_apls_eval import V32_SCALE
    from src.pipeline.p1_segment.model import predict_large_prob
    from src.pipeline.p1_segment.raster_io import to_rgb8

    rgb = to_rgb8(bands)
    nh = max(1, int(round(bands.shape[1] * V32_SCALE)))
    nw = max(1, int(round(bands.shape[2] * V32_SCALE)))
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return predict_large_prob(model, rgb, tile_size=512, stride=384,
                              device=device, tta=False)


def load_v32_chips(model, n_chips: int | None, device: str = "cpu", seed: int = 7) -> dict:
    """Build ``{chip: (prob, gt_graph, eff_x, eff_y)}`` for the heldout chips.

    Requires the licensed SpaceNet SN5-Mumbai chips on disk (see
    ``chip_apls_eval.SRC_RGB``). Chips whose GT is too small to score are skipped,
    matching the evaluator's guard.
    """
    from src.pipeline.p1_segment.chip_apls_eval import (
        _geojson_adj,
        _heldout_chips,
        _read_chip,
        adj_to_apls_graph,
    )

    chips: dict = {}
    for chip in _heldout_chips(n_chips, seed):
        bands, transform, eff_x, eff_y = _read_chip(chip)
        gt = adj_to_apls_graph(_geojson_adj(chip, transform), eff_x, eff_y)
        if gt.number_of_nodes() < 2:
            continue  # unscorable GT
        chips[chip] = (_v32_prob(model, bands, device), gt, eff_x, eff_y)
        print(f"  cached prob for chip {chip}", flush=True)
    return chips


def main() -> None:
    from src.pipeline.p1_segment.chip_apls_eval import IMAGE_SIZE, chip_apls

    p = argparse.ArgumentParser(
        description="A46 routing-first threshold/radius calibration on an existing checkpoint."
    )
    p.add_argument("--ckpt", required=True, help="v3.2 checkpoint (.pt)")
    p.add_argument("--n-chips", type=int, default=None,
                   help="heldout chips to use (default: all; use the 102-chip split)")
    p.add_argument("--thresholds", default="0.36,0.40,0.44,0.48,0.52,0.56,0.60",
                   help="probability thresholds to sweep")
    p.add_argument("--radii", default="0,10,20,30",
                   help="healing gap radii in metres (0 = raw skeleton, today's behaviour)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="data/processed/a46_routing_calibration.json")
    args = p.parse_args()

    from src.pipeline.p1_segment.model import load_checkpoint

    model, meta = load_checkpoint(args.ckpt, map_location=args.device)
    default_thr = float(meta.get("threshold", 0.52))

    thresholds = [float(t) for t in args.thresholds.split(",")]
    radii = [float(r) for r in args.radii.split(",")]
    settings = [Setting(t, r) for t in thresholds for r in radii]
    default = Setting(default_thr, 0.0)  # today: meta threshold, no healing
    if default not in settings:
        settings.append(default)

    print(f"caching probabilities (predict once, sweep {len(settings)} settings)...", flush=True)
    chips = load_v32_chips(model, args.n_chips, args.device, args.seed)
    if not chips:
        raise SystemExit("no scorable chips found — check the SpaceNet data paths")

    report = calibrate(chips, settings, default, chip_apls, frame_size=IMAGE_SIZE)
    report["checkpoint"] = str(args.ckpt)
    report["default_threshold_from_meta"] = default_thr
    write_report(report, Path(args.out))
    print("\n" + format_report(report))
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
