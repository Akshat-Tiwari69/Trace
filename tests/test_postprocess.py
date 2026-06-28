"""Tests for A10 mask post-processing (`p1_segment/postprocess.py`).

The core contract (Tracker A10): tiny false components drop on a mask without
IoU loss > 0.005 against the clean target.
"""
import numpy as np

from src.pipeline.p1_segment.postprocess import (
    morphological_cleanup,
    otsu_threshold,
    postprocess_mask,
    prune_skeleton_spurs,
    remove_small_components,
)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = (a | b).sum()
    return 1.0 if union == 0 else float((a & b).sum() / union)


def _road_mask() -> np.ndarray:
    """A clean 64x64 mask: one thick horizontal road + one thick vertical road."""
    m = np.zeros((64, 64), np.uint8)
    m[30:33, 4:60] = 1   # horizontal road (3 px thick)
    m[4:60, 30:33] = 1   # vertical road
    return m


def test_remove_small_components_drops_specks_keeps_roads():
    mask = _road_mask()
    noisy = mask.copy()
    noisy[5, 50] = 1                  # 1-px speck
    noisy[10:12, 10:12] = 1           # 4-px blob
    noisy[50:53, 8:11] = 1            # 9-px blob
    cleaned = remove_small_components(noisy, min_size=20)
    assert set(np.unique(cleaned)).issubset({0, 1})
    assert cleaned[5, 50] == 0 and cleaned[10, 10] == 0 and cleaned[50, 8] == 0
    # the real roads are untouched -> identical to the clean mask
    assert np.array_equal(cleaned, mask)


def test_remove_small_components_keeps_large_blob():
    mask = _road_mask()
    out = remove_small_components(mask, min_size=20)
    assert np.array_equal(out, mask)  # nothing removed


def test_morphological_cleanup_open_removes_speck_close_fills_gap():
    m = _road_mask()
    m[5, 5] = 1                       # isolated speck -> opening removes it
    opened = morphological_cleanup(m, open_radius=1, close_radius=0)
    assert opened[5, 5] == 0
    # closing bridges a 1-px gap punched in a road
    g = _road_mask()
    g[31, 40] = 0
    closed = morphological_cleanup(g, open_radius=0, close_radius=1)
    assert closed[31, 40] == 1


def test_morphological_cleanup_is_binary_and_noop_when_radii_zero():
    m = _road_mask()
    assert np.array_equal(morphological_cleanup(m, open_radius=0, close_radius=0), m)


def test_otsu_threshold_bimodal_and_degenerate():
    prob = np.full((16, 16), 0.1)
    prob[4:12, 4:12] = 0.9           # clear foreground cluster
    out = otsu_threshold(prob)
    assert out[8, 8] == 1 and out[0, 0] == 0
    assert set(np.unique(out)).issubset({0, 1})
    # degenerate (all equal) -> all background, no crash
    assert otsu_threshold(np.full((8, 8), 0.5)).sum() == 0


def test_prune_skeleton_spurs_removes_short_spur_keeps_line():
    skel = np.zeros((20, 20), np.uint8)
    skel[10, 2:18] = 1               # main horizontal line
    skel[11:15, 10] = 1              # 4-px spur down from the line
    pruned = prune_skeleton_spurs(skel, max_spur_len=5)
    assert pruned[10, 2:18].all()    # main line preserved
    assert pruned[12:15, 10].sum() == 0  # deep spur removed (junction nub may remain)


def test_prune_skeleton_spurs_keeps_long_branch():
    skel = np.zeros((30, 30), np.uint8)
    skel[15, 2:28] = 1
    skel[16:28, 14] = 1              # 12-px branch (longer than max) -> kept
    pruned = prune_skeleton_spurs(skel, max_spur_len=5)
    assert pruned[16:28, 14].sum() == 12


def test_postprocess_mask_preserves_iou_within_tolerance():
    """Core A10 criterion: cleanup removes false specks without IoU loss > 0.005."""
    gt = _road_mask()
    noisy = gt.copy()
    rng = np.random.default_rng(0)
    for _ in range(40):              # scatter tiny false-positive specks
        r, c = rng.integers(0, 64, 2)
        noisy[r, c] = 1
    cleaned = postprocess_mask(noisy, min_size=20)
    assert set(np.unique(cleaned)).issubset({0, 1})
    # cleanup should not LOSE IoU vs the noisy input (it removes false positives)
    assert _iou(cleaned, gt) >= _iou(noisy, gt) - 0.005
    # and on an already-clean mask it must be (near) lossless
    assert _iou(postprocess_mask(gt, min_size=20), gt) >= 1.0 - 0.005


def test_postprocess_mask_default_is_iou_safe_on_thin_roads():
    """Default cleanup must not erode thin (1-px) roads -> no IoU collapse."""
    thin = np.zeros((32, 32), np.uint8)
    thin[16, 2:30] = 1               # 1-px-thick road
    out = postprocess_mask(thin, min_size=5)
    assert _iou(out, thin) >= 1.0 - 0.005
