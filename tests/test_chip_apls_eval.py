"""Unit tests for the A18 common-unit chip-level APLS evaluator.

Pure-function tests (no raw data / model needed): coordinate scaling, identical/
empty graphs, the paired-bootstrap sign convention, and strict-coverage failure.
Run: .venv-gpu/Scripts/python.exe -m pytest tests/test_chip_apls_eval.py -q
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx

from src.pipeline.p1_segment.chip_apls_eval import (
    MIN_MATERIAL_APLS_DELTA, MIN_NORMALIZED_APLS,
    _candidate_sort_key, _graph_artifact_digest, _graph_diagnostics,
    _load_chip_ids, _native_coord_to_frame,
    _load_adjacency, _paired_comparison, _promotion_gate,
    _prediction_provenance, _preflight_chip_inputs,
    _require_registered_comparison, _require_registered_selection,
    _select_complete_candidate, _write_report, a18_pred_path,
    adj_to_apls_graph, chip_apls, coverage, strict_exit_code)
from src.pipeline.p1_segment.stats import paired_bootstrap_ci


def test_coordinate_scaling_is_anisotropic():
    # edge (0,0)->(2,0): dr=2, dc=0 -> length uses eff_y only
    g = adj_to_apls_graph({(0, 0): [(2, 0)]}, eff_x=2.0, eff_y=3.0)
    (u, v, d), = g.edges(data=True)
    assert abs(d["length_m"] - 6.0) < 1e-9          # 2 rows * eff_y=3
    # Node coordinates stay in the declared projected-metre frame.
    xy = {(round(n["x"], 6), round(n["y"], 6)) for _, n in g.nodes(data=True)}
    assert (0.0, 0.0) in xy and (0.0, 6.0) in xy    # (col*eff_x, row*eff_y)


def test_horizontal_edge_uses_eff_x():
    g = adj_to_apls_graph({(0, 0): [(0, 5)]}, eff_x=2.0, eff_y=3.0)
    (u, v, d), = g.edges(data=True)
    assert abs(d["length_m"] - 10.0) < 1e-9         # 5 cols * eff_x=2


def test_native_border_rounding_clamps_to_valid_frame():
    assert _native_coord_to_frame(0) == 0
    assert _native_coord_to_frame(1299) == 399
    assert _native_coord_to_frame(1300) == 399
    assert _native_coord_to_frame(-1) == 0


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


def test_exploratory_comparison_reports_zero_overlap_without_bootstrap():
    rep = _paired_comparison(["a", "b"], {"a": 0.4}, {"b": 0.5}, False)
    assert rep["n_paired"] == 0
    assert rep["verdict"] == "no paired chips"
    assert rep["delta_a18_minus_v32"] is None
    assert not rep["excludes_zero"]


def test_report_writer_creates_nested_parent(tmp_path):
    out = tmp_path / "new" / "nested" / "report.json"
    _write_report(out, {"ok": True})
    assert out.is_file()


def test_explicit_chip_manifest_is_deterministic(tmp_path):
    manifest = tmp_path / "split.json"
    manifest.write_text('{"validation": ["chip9", "chip2", "chip5"]}')
    assert _load_chip_ids(manifest, "validation", None, seed=99) == [
        "chip2", "chip5", "chip9"]
    assert _load_chip_ids(manifest, "validation", 2, seed=7) == ["chip2", "chip9"]


def test_chip_manifest_rejects_duplicate_ids(tmp_path):
    manifest = tmp_path / "split.json"
    manifest.write_text('{"validation": ["chip2", "chip2"]}')
    import pytest
    with pytest.raises(ValueError, match="duplicate"):
        _load_chip_ids(manifest, "validation", None, seed=7)


def test_a46_selection_manifest_is_frozen_and_disjoint_from_comparison():
    root = Path(__file__).resolve().parents[1]
    selection = json.loads(
        (root / "data/sample/a46_selection_chips.json").read_text())["validation"]
    comparison = json.loads(
        (root / "data/sample/spacenet_mumbai_heldout_chips.json").read_text())["test_chips"]
    assert len(selection) == len(set(selection)) == 102
    assert set(selection).isdisjoint(comparison)


def test_a46_calibration_plan_freezes_selection_only_search():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(
        (root / "data/sample/a46_calibration_plan.json").read_text())
    assert plan["checkpoint"]["epoch"] == 22
    assert plan["chip_count"] == 102
    assert plan["apls_samples_per_chip"] == 600
    assert plan["stage_1"] == {
        "variable": "topology_threshold",
        "values": [0.6, 0.64, 0.67, 0.69, 0.72, 0.75, 0.78],
        "all_other_values": "baseline",
    }
    assert plan["selection_rule"]["promotion_floors"] == {
        "minimum_raw_delta_vs_incumbent": 0.02,
        "paired_ci_low_must_exceed": 0.0,
        "minimum_normalized_apls": 0.25,
    }
    assert "forbidden" in plan["comparison_split_access"]


def test_a46_registered_result_is_internally_consistent_and_not_deployable():
    root = Path(__file__).resolve().parents[1]
    registration = json.loads(
        (root / "data/sample/a46_comparison_preregistration.json").read_text())
    result = json.loads(
        (root / "data/sample/a46_comparison_result.json").read_text())

    assert registration["candidate"]["thresholds"]["topology"] == 0.6
    assert result["comparison_manifest_sha256"] == registration["comparison"]["manifest_sha256"]
    assert result["chip_count"] == registration["comparison"]["chip_count"] == 127
    assert result["candidate"]["checkpoint_sha256"] == registration["candidate"]["checkpoint_sha256"]
    raw_delta = result["candidate"]["raw_apls"] - result["incumbent"]["raw_apls"]
    assert abs(raw_delta - result["paired_candidate_minus_incumbent"]["raw_apls_delta"]) < 1e-12
    assert result["metric_gate"]["passed"] is True
    assert result["paired_candidate_minus_incumbent"]["raw_apls_ci95"][0] > 0
    assert result["candidate"]["normalized_apls"] >= 0.25
    assert result["production_decision"]["promoted"] is False


def test_a46_run_manifest_schema_covers_decision_provenance():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (root / "data/sample/a46_run_manifest.schema.json").read_text())
    assert set(schema["required"]) == {
        "schema_version", "run_id", "purpose", "redistribution_allowed",
        "licenses", "code", "data", "model", "environment", "inference",
        "promotion_contract",
    }
    for section in (
            "code", "data", "model", "environment", "inference",
            "promotion_contract"):
        nested = schema["properties"][section]
        assert set(nested["required"]) <= set(nested["properties"])
        assert all(
            "type" in nested["properties"][key]
            or "const" in nested["properties"][key]
            for key in nested["required"])


def test_selection_cli_rejects_closed_comparison_manifest():
    root = Path(__file__).resolve().parents[1]
    comparison = root / "data/sample/spacenet_mumbai_heldout_chips.json"
    import pytest
    with pytest.raises(ValueError, match="registered 102-chip"):
        _require_registered_selection(comparison, "test_chips", None, 600)


def test_strict_cli_rejects_subsets_and_weak_sampling():
    root = Path(__file__).resolve().parents[1]
    comparison = root / "data/sample/spacenet_mumbai_heldout_chips.json"
    import pytest
    with pytest.raises(ValueError, match="all registered comparison chips"):
        _require_registered_comparison(comparison, "test_chips", 10, 600, None)
    with pytest.raises(ValueError, match="600 APLS samples"):
        _require_registered_comparison(comparison, "test_chips", None, 100, None)
    with pytest.raises(ValueError, match="threshold override"):
        _require_registered_comparison(comparison, "test_chips", None, 600, .4)


def test_graph_diagnostics_report_fragmentation_and_metric_length():
    g = nx.Graph()
    g.add_edge(0, 1, length_m=3.0)
    g.add_node(2)
    d = _graph_diagnostics(g)
    assert d == {
        "nodes": 3,
        "edges": 1,
        "components": 2,
        "isolates": 1,
        "largest_component_node_fraction": 2 / 3,
        "reachable_pair_fraction": 1 / 3,
        "total_edge_length_m": 3.0,
    }


def test_candidate_ranking_uses_apls_then_connectivity_tiebreakers():
    strong = {
        "label": "strong", "apls_mean": .2, "apls_norm_mean": .3,
        "diagnostics_mean": {"reachable_pair_fraction": .4,
                             "largest_component_node_fraction": .5,
                             "components": 3},
    }
    connected = {
        "label": "connected", "apls_mean": .1, "apls_norm_mean": .2,
        "diagnostics_mean": {"reachable_pair_fraction": .8,
                             "largest_component_node_fraction": .9,
                             "components": 1},
    }
    assert sorted([connected, strong], key=_candidate_sort_key)[0]["label"] == "strong"

    tied = {**strong, "label": "tied", "diagnostics_mean": {
        **strong["diagnostics_mean"], "reachable_pair_fraction": .6}}
    assert sorted([strong, tied], key=_candidate_sort_key)[0]["label"] == "tied"

    incomplete = {**strong, "label": "incomplete", "apls_mean": .9,
                  "coverage_complete": False, "missing_chips": ["hard"]}
    complete = {**connected, "coverage_complete": True, "missing_chips": []}
    assert sorted([incomplete, complete], key=_candidate_sort_key)[0]["label"] == "connected"
    assert _select_complete_candidate([incomplete]) is None
    assert _select_complete_candidate([complete]) == "connected"


def test_prediction_pickle_loader_rejects_globals_and_malformed_adjacency(tmp_path):
    import pytest
    unsafe = tmp_path / "unsafe.p"
    unsafe.write_bytes(pickle.dumps(range(3)))
    with pytest.raises(pickle.UnpicklingError):
        _load_adjacency(unsafe)

    malformed = tmp_path / "malformed.p"
    malformed.write_bytes(pickle.dumps({(0, 0): ["not-a-node"]}))
    with pytest.raises(ValueError, match="neighbor"):
        _load_adjacency(malformed)

    valid = tmp_path / "valid.p"
    graph = {(0, 0): [(0, 4)], (0, 4): [(0, 0)]}
    valid.write_bytes(pickle.dumps(graph))
    assert _load_adjacency(valid) == graph


def test_chip_input_preflight_reports_missing_rgb_and_gt(tmp_path):
    import pytest
    rgb = tmp_path / "rgb"
    geo = tmp_path / "geo"
    rgb.mkdir()
    geo.mkdir()
    with pytest.raises(FileNotFoundError, match="missing RGB.*chip1.*missing GT.*chip1"):
        _preflight_chip_inputs(["chip1"], rgb, geo)


def test_prediction_provenance_verifies_scored_graph_digest(tmp_path):
    pred = tmp_path / "pred"
    graph_dir = pred / "graph"
    graph_dir.mkdir(parents=True)
    (pred / "config.yaml").write_text("threshold: 1\n")
    root = Path(__file__).resolve().parents[1]
    selection_path = root / "data/sample/a46_selection_chips.json"
    selection_ids = json.loads(selection_path.read_text())["validation"]
    for chip in selection_ids:
        (graph_dir / f"mumbai_{chip}.p").write_bytes(pickle.dumps({}))
    digest, count = _graph_artifact_digest(pred)

    import hashlib
    config_sha = hashlib.sha256((pred / "config.yaml").read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1, "run_id": "test", "purpose": "selection",
        "redistribution_allowed": False, "licenses": {},
        "code": {"trace_git_sha": "a", "samroadplus_upstream_git_sha": "b",
                 "samroadplus_local_patch_sha256": "c"},
        "data": {"selection_manifest_sha256": hashlib.sha256(
                     selection_path.read_bytes()).hexdigest(),
                 "selection_key": "validation",
                 "chip_count": len(selection_ids), "chip_ids": selection_ids,
                 "coordinate_contract": "x=column,y=row"},
        "model": {"checkpoint_sha256": "e", "epoch": 1,
                  "config_sha256": config_sha,
                  "effective_config_sha256": config_sha,
                  "sam_checkpoint_sha256": "f"},
        "environment": {"python": "3", "platform": "test", "torch": "2",
                        "lightning": "2", "cuda_runtime": "12", "cudnn": 9,
                        "gpu": "test", "deterministic_algorithms": False,
                        "historical_checkpoint_reproducibility": "limited"},
        "inference": {"command": ["python"], "thresholds": {},
                      "wall_seconds_inference": 1.0,
                      "graph_count": count, "graphs_sha256": digest},
        "promotion_contract": {"selection_split_only": True,
                               "min_candidate_minus_incumbent_raw_apls": .02,
                               "min_candidate_normalized_apls": .25,
                               "comparison_split_runs_allowed_after_preregistration": 1},
    }
    (pred / "run_manifest.json").write_text(json.dumps(manifest))
    assert _prediction_provenance(pred)["valid"]

    (graph_dir / f"mumbai_{selection_ids[0]}.p").write_bytes(
        pickle.dumps({(0, 0): []}))
    invalid = _prediction_provenance(pred)
    assert not invalid["valid"]
    assert any("graph digest" in error for error in invalid["errors"])


def test_coverage_flags_missing_a18():
    cov = coverage(["a", "b", "c"], {"a": .5, "b": .4, "c": .3}, {"a": .6, "c": .5},
                   want_v32=True, want_a18=True)
    assert not cov["complete"]
    assert cov["missing_a18"] == ["b"] and cov["missing_v32"] == []


def test_coverage_complete_when_all_present():
    cov = coverage(["a", "b"], {"a": .5, "b": .4}, {"a": .6, "b": .5},
                   want_v32=True, want_a18=True)
    assert cov["complete"]


def test_coverage_includes_requested_incumbent():
    cov = coverage(
        ["a", "b"], {"a": .2, "b": .2}, {"a": .4, "b": .4},
        want_v32=True, want_a18=True,
        incumbent_scores={"a": .3}, want_incumbent=True)
    assert not cov["complete"]
    assert cov["missing_incumbent"] == ["b"]


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
    # A relative win alone is not a promotion. The material and absolute gates
    # must also pass against the incumbent A18 checkpoint.
    assert strict_exit_code({"promotion_gate": {"passed": True}}) == 0
    assert strict_exit_code({"promotion_gate": {"passed": False}}) == 3
    assert strict_exit_code({"compare": {"verdict": "GATE FAIL: incomplete coverage"}}) == 2
    assert strict_exit_code({}) == 3


def test_promotion_gate_requires_paired_material_and_absolute_gain():
    base = {
        "coverage": {"complete": True},
        "compare_incumbent": {
            "verdict": "b wins", "excludes_zero": True,
            "delta_a18_minus_incumbent": MIN_MATERIAL_APLS_DELTA,
        },
        "a18": {"apls_norm_mean": MIN_NORMALIZED_APLS},
        "prediction_provenance": {
            "incumbent": {"valid": True}, "candidate": {"valid": True}},
    }
    assert _promotion_gate(base)["passed"]

    too_small = {**base, "compare_incumbent": {
        **base["compare_incumbent"],
        "delta_a18_minus_incumbent": MIN_MATERIAL_APLS_DELTA - 1e-6,
    }}
    assert not _promotion_gate(too_small)["passed"]

    too_fragmented = {**base, "a18": {
        "apls_norm_mean": MIN_NORMALIZED_APLS - 1e-6,
    }}
    assert not _promotion_gate(too_fragmented)["passed"]
