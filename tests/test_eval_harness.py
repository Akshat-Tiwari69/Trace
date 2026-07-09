"""Golden + statistical tests for the model-promotion harness (bugs.md §3).

The promotion evals (`eval_spacenet`, `apls_eval`) have historically been run by
hand and transcribed into the Tracker — a regression *in the harness itself*
(a threshold-split leak, a bad IoU accumulation, a bootstrap that never excludes
zero) would go unnoticed. These tests pin the harness's behaviour end-to-end on
synthetic inputs with hand-checked answers, no GPU/checkpoints required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.pipeline.p1_segment.apls_eval import tile_apls
from src.pipeline.p1_segment.eval_spacenet import split_heldout_chips
from src.pipeline.p1_segment.stats import paired_bootstrap_ci
from src.pipeline.p1_segment.train import evaluate


# --------------------------------------------------------------------------- #
# Paired bootstrap CI
# --------------------------------------------------------------------------- #
def test_bootstrap_identical_scores_is_inconclusive():
    scores = [0.4, 0.5, 0.6, 0.55, 0.45, 0.5]
    ci = paired_bootstrap_ci(scores, scores)
    assert ci.delta == pytest.approx(0.0)
    assert not ci.excludes_zero
    assert "inconclusive" in ci.verdict


def test_bootstrap_clear_improvement_excludes_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(0.40, 0.02, size=60)
    b = a + 0.10  # a uniform, unambiguous +0.10 per-tile gain
    ci = paired_bootstrap_ci(a, b)
    assert ci.excludes_zero
    assert ci.delta == pytest.approx(0.10, abs=1e-9)
    assert ci.ci_low > 0
    assert ci.verdict == "b wins"


def test_bootstrap_noise_sized_gap_is_inconclusive():
    # A tiny mean gap swamped by per-tile variance must NOT be called a win —
    # this is the exact A17/A24 trap the CI exists to catch.
    rng = np.random.default_rng(1)
    a = rng.normal(0.40, 0.15, size=40)
    b = a - a.mean() + 0.404 + rng.normal(0, 0.15, size=40)  # ~+0.004 mean, big noise
    ci = paired_bootstrap_ci(a, b)
    assert not ci.excludes_zero


def test_bootstrap_rejects_mismatched_or_empty():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([0.1, 0.2], [0.1])
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], [])


# --------------------------------------------------------------------------- #
# Honest threshold split
# --------------------------------------------------------------------------- #
def test_threshold_split_is_disjoint_and_covers_all():
    chips = [f"chip{i}" for i in range(20)]
    sel, rep = split_heldout_chips(chips, selection_frac=0.5, seed=23)
    assert set(sel).isdisjoint(rep)
    assert set(sel) | set(rep) == set(chips)
    assert len(sel) == 10 and len(rep) == 10


def test_threshold_split_is_deterministic():
    chips = [f"chip{i}" for i in range(17)]
    assert split_heldout_chips(chips, seed=23) == split_heldout_chips(chips, seed=23)
    # a different seed should generally reshuffle the halves
    assert split_heldout_chips(chips, seed=23) != split_heldout_chips(chips, seed=99)


# --------------------------------------------------------------------------- #
# Golden IoU/Dice math (train.evaluate) — stub model, hand-checked answer
# --------------------------------------------------------------------------- #
class _FixedLogits(torch.nn.Module):
    """A stub 'model' whose output logits are supplied directly (bypasses weights)."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits

    def forward(self, _x):  # noqa: D401 - test stub
        return self._logits


def test_evaluate_global_iou_is_exact_on_known_masks():
    # GT = top half roads; prediction = a fixed pattern with a known overlap.
    gt = torch.zeros(1, 1, 4, 4)
    gt[..., :2, :] = 1.0                      # 8 positive px
    logits = torch.full((1, 1, 4, 4), -5.0)   # sigmoid ~0 → predict background
    logits[..., :1, :] = 5.0                  # predict row 0 (4 px), all true positives
    loader = [(torch.zeros(1, 1, 4, 4), gt)]
    model = _FixedLogits(logits)

    m = evaluate(model, loader, device="cpu", threshold=0.5)
    # inter=4, pred=4, target=8 → IoU 4/8=0.5, Dice 8/12≈0.6667
    assert m["iou"] == pytest.approx(0.5, abs=1e-4)
    assert m["dice"] == pytest.approx(2 / 3, abs=1e-4)


# --------------------------------------------------------------------------- #
# Golden APLS (tile_apls) on constructed masks
# --------------------------------------------------------------------------- #
def test_tile_apls_perfect_match_scores_high():
    mask = np.zeros((64, 64), np.uint8)
    mask[32, 4:60] = 1          # one horizontal road
    mask[4:60, 32] = 1          # one vertical road → a '+' with a junction
    assert tile_apls(mask, mask.copy(), n_samples=100) > 0.95


def test_tile_apls_prediction_with_no_roads_scores_zero():
    gt = np.zeros((64, 64), np.uint8)
    gt[32, 4:60] = 1
    gt[4:60, 32] = 1
    empty = np.zeros((64, 64), np.uint8)
    assert tile_apls(empty, gt) == 0.0  # GT has roads, pred has none → worst APLS


def test_tile_apls_no_gt_is_nan_and_skipped():
    empty = np.zeros((64, 64), np.uint8)
    pred = np.zeros((64, 64), np.uint8)
    pred[10, 4:60] = 1
    assert np.isnan(tile_apls(pred, empty))  # no GT roads → undefined, skip
