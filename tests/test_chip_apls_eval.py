"""Unit tests for the A18 common-unit chip-level APLS evaluator.

Pure-function tests (no raw data / model needed): coordinate scaling, identical/
empty graphs, the paired-bootstrap sign convention, and strict-coverage failure.
Run: .venv-gpu/Scripts/python.exe -m pytest tests/test_chip_apls_eval.py -q
"""
from __future__ import annotations

import networkx as nx

from src.pipeline.p1_segment.apls_eval import _DEG_X, _DEG_Y
from src.pipeline.p1_segment.chip_apls_eval import (
    a18_pred_path, adj_to_apls_graph, chip_apls, coverage, strict_exit_code)
from src.pipeline.p1_segment.stats import paired_bootstrap_ci


def test_coordinate_scaling_is_anisotropic():
    # edge (0,0)->(2,0): dr=2, dc=0 -> length uses eff_y only
    g = adj_to_apls_graph({(0, 0): [(2, 0)]}, eff_x=2.0, eff_y=3.0)
    (u, v, d), = g.edges(data=True)
    assert abs(d["length_m"] - 6.0) < 1e-9          # 2 rows * eff_y=3
    # node (2,0): x from col=0, y from row=2*eff_y, each rescaled to degrees
    xy = {(round(n["x"] * _DEG_X, 6), round(n["y"] * _DEG_Y, 6)) for _, n in g.nodes(data=True)}
    assert (0.0, 0.0) in xy and (0.0, 6.0) in xy    # (col*eff_x, row*eff_y)


def test_horizontal_edge_uses_eff_x():
    g = adj_to_apls_graph({(0, 0): [(0, 5)]}, eff_x=2.0, eff_y=3.0)
    (u, v, d), = g.edges(data=True)
    assert abs(d["length_m"] - 10.0) < 1e-9         # 5 cols * eff_x=2


def _square():
    # a small connected loop so apls has >=2 reachable nodes
    return adj_to_apls_graph(
        {(0, 0): [(0, 20), (20, 0)], (0, 20): [(0, 0), (20, 20)],
         (20, 0): [(0, 0), (20, 20)], (20, 20): [(0, 20), (20, 0)]},
        eff_x=1.0, eff_y=1.0)


def test_identical_graph_scores_one():
    g = _square()
    assert chip_apls(g, g, n_samples=50) == 1.0


def test_empty_prediction_scores_zero():
    assert chip_apls(nx.Graph(), _square(), n_samples=50) == 0.0


def test_no_gt_returns_nan():
    import math
    assert math.isnan(chip_apls(_square(), nx.Graph(), n_samples=50))


def test_bootstrap_sign_is_a18_minus_v32():
    # paired_bootstrap_ci(v32, a18) -> delta = a18 - v32; A18 clearly better -> positive
    v32 = [0.30, 0.32, 0.28, 0.31, 0.29]
    a18 = [0.50, 0.52, 0.48, 0.51, 0.49]
    ci = paired_bootstrap_ci(v32, a18)
    assert ci.delta > 0 and ci.excludes_zero


def test_coverage_flags_missing_a18():
    cov = coverage(["a", "b", "c"], {"a": .5, "b": .4, "c": .3}, {"a": .6, "c": .5},
                   want_v32=True, want_a18=True)
    assert not cov["complete"]
    assert cov["missing_a18"] == ["b"] and cov["missing_v32"] == []


def test_coverage_complete_when_all_present():
    cov = coverage(["a", "b"], {"a": .5, "b": .4}, {"a": .6, "b": .5},
                   want_v32=True, want_a18=True)
    assert cov["complete"]


def test_a18_pred_path_contract():
    p = a18_pred_path("out", "chip42")
    assert p.parts[-2:] == ("graph", "mumbai_chip42.p")


def test_missing_artifact_fails_strict_with_nonzero_exit(tmp_path):
    # One real artifact present, one deliberately missing -> the loader sees a
    # coverage gap, and a strict gate must report GATE FAIL with exit code 2.
    import pickle
    (tmp_path / "graph").mkdir()
    a18 = {}
    for chip in ("chip1", "chip2"):
        # both chips are scorable (v3.2 scored both); only chip1 has an A18 artifact
        if chip == "chip1":
            a18_pred_path(tmp_path, chip).write_bytes(pickle.dumps({(0, 0): [(0, 5)]}))
        if a18_pred_path(tmp_path, chip).exists():
            a18[chip] = 0.4
    v32 = {"chip1": 0.5, "chip2": 0.5}
    cov = coverage(["chip1", "chip2"], v32, a18, want_v32=True, want_a18=True)
    assert not cov["complete"] and cov["missing_a18"] == ["chip2"]
    assert strict_exit_code({"compare": {"verdict": "GATE FAIL: incomplete coverage"}}) == 2


def test_strict_exit_code_encodes_promotion_not_completion():
    # real BootstrapCI.verdict strings; only complete-coverage + A18 win -> 0
    assert strict_exit_code({"compare": {"verdict": "b wins"}}) == 0            # A18 wins -> promote
    assert strict_exit_code({"compare": {"verdict": "a wins"}}) == 3            # v3.2 wins -> regression
    assert strict_exit_code(
        {"compare": {"verdict": "inconclusive (CI straddles 0 — within sampling noise)"}}) == 3
    assert strict_exit_code({"compare": {"verdict": "GATE FAIL: incomplete coverage"}}) == 2
    assert strict_exit_code({}) == 3                                            # missing comparison
