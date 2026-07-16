"""Unit tests for A46 routing-first threshold/radius calibration.

Offline: synthetic probability maps + a stub scorer, so the sweep/selection/CI
logic is tested without the licensed SpaceNet chips or a checkpoint.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.pipeline.p3_analysis.calibrate import (
    Setting,
    calibrate,
    graph_from_prob,
    paired_gain,
    sweep,
)


def _prob_with_gap() -> np.ndarray:
    """A horizontal road at p=0.9 with a 10-px gap in the middle."""
    prob = np.zeros((40, 100), dtype=float)
    prob[19:22, 5:45] = 0.9
    prob[19:22, 55:95] = 0.9
    return prob


def _node_count_score(pred, _gt) -> float:
    """Stub scorer: more decoded structure = better (deterministic per setting)."""
    return pred.number_of_nodes() / 100.0


# --------------------------------------------------------------------------- #
# graph_from_prob — thresholding + healing
# --------------------------------------------------------------------------- #
def test_threshold_gates_the_decode():
    prob = _prob_with_gap()
    empty = graph_from_prob(prob, Setting(threshold=0.95), 1.0, 1.0)
    assert empty.number_of_nodes() == 0          # nothing clears 0.95
    decoded = graph_from_prob(prob, Setting(threshold=0.5), 1.0, 1.0)
    assert decoded.number_of_nodes() > 0


def test_healing_radius_reconnects_the_gap():
    """The core A46 lever: healing bridges a gap the raw skeleton leaves open."""
    prob = _prob_with_gap()
    raw = graph_from_prob(prob, Setting(0.5, gap_max_m=0.0), 1.0, 1.0)
    healed = graph_from_prob(prob, Setting(0.5, gap_max_m=20.0), 1.0, 1.0)
    assert nx.number_connected_components(raw) == 2          # gap splits the road
    assert nx.number_connected_components(healed) == 1       # healing closes it


# --------------------------------------------------------------------------- #
# sweep / calibrate
# --------------------------------------------------------------------------- #
def test_sweep_sorts_best_mean_first():
    chips = {"c1": (_prob_with_gap(), None, 1.0, 1.0)}
    rows = sweep(chips, [Setting(0.95), Setting(0.5)], _node_count_score)
    assert rows[0]["mean_apls"] >= rows[-1]["mean_apls"]      # sorted descending
    assert rows[0]["threshold"] == 0.5                        # 0.95 decodes nothing


def test_calibrate_pairs_best_against_default():
    prob = _prob_with_gap()
    chips = {f"c{i}": (prob, None, 1.0, 1.0) for i in range(5)}
    default = Setting(0.95)
    report = calibrate(chips, [Setting(0.5), default], default, _node_count_score)

    assert report["best"] == Setting(0.5).label()
    assert report["default"] == default.label()
    assert report["best_mean_apls"] > report["default_mean_apls"]
    gain = report["paired_best_vs_default"]
    assert gain["n_paired"] == 5
    assert gain["delta"] > 0                                  # candidate beats default
    assert [r["label"] for r in report["table"]]              # table has no raw scores
    assert "scores" not in report["table"][0]


def test_paired_gain_without_overlap():
    assert paired_gain({"a": 0.1}, {"b": 0.2})["n_paired"] == 0
