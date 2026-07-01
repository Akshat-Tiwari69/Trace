"""Tests for A17 APLS-on-masks (`p1_segment/apls_eval.py`)."""
import math
import numpy as np
from src.pipeline.p1_segment.apls_eval import mask_to_apls_graph, tile_apls


def _cross() -> np.ndarray:
    m = np.zeros((64, 64), np.uint8)
    m[30:33, 5:60] = 1   # horizontal road
    m[5:60, 30:33] = 1   # vertical road
    return m


def test_mask_to_apls_graph_has_nodes_and_edges():
    g = mask_to_apls_graph(_cross())
    assert g.number_of_nodes() > 0 and g.number_of_edges() > 0
    # coords were rescaled to ~degrees (tiny), length_m stays metric (>0)
    assert all(abs(d["x"]) < 1 and abs(d["y"]) < 1 for _, d in g.nodes(data=True))


def test_identical_masks_score_near_one():
    m = _cross()
    assert tile_apls(m, m, n_samples=100) >= 0.95


def test_empty_gt_is_nan_and_skipped():
    assert math.isnan(tile_apls(_cross(), np.zeros((64, 64), np.uint8)))
