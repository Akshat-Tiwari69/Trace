"""A51 experiment record: run.json identity, data fingerprints, validation-split rebuild."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.pipeline.p1_segment.provenance import dir_fingerprint, record_run


def _pair(folder, stem):
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(folder / f"{stem}_sat.jpg")
    Image.fromarray(np.zeros((8, 8), np.uint8), mode="L").save(folder / f"{stem}_mask.png")


def test_dir_fingerprint_counts_pairs_and_changes_with_the_data(tmp_path):
    _pair(tmp_path, "a")
    _pair(tmp_path, "b")
    before = dir_fingerprint(tmp_path)
    assert before["pairs"] == 2
    _pair(tmp_path, "c")
    assert dir_fingerprint(tmp_path) != before


def test_dir_fingerprint_counts_only_complete_non_empty_pairs(tmp_path):
    # A51: counting images let a half-uploaded source pass the CPU pre-flight.
    _pair(tmp_path, "whole")
    _pair(tmp_path, "no_mask")
    (tmp_path / "no_mask_mask.png").unlink()
    _pair(tmp_path, "empty_mask")
    (tmp_path / "empty_mask_mask.png").write_bytes(b"")
    (tmp_path / "orphan_mask.png").write_bytes(b"x")
    fp = dir_fingerprint(tmp_path)
    assert fp["pairs"] == 1 and fp["incomplete"] == 3


def test_road_free_gate_fails_invented_roads_and_reports_land_separately():
    from src.pipeline.p1_segment.val_apls_select import FP_TOL, INVENTED_PX, road_free_report

    kinds = ["land"] * 10 + ["water"] * 30
    clean = ([0.0] * 40, [0] * 40)
    same = road_free_report(clean, clean, kinds)
    assert same["passes"] and same["land"]["n_tiles"] == 10 and same["water"]["n_tiles"] == 30
    inventing = ([0.02] * 10 + [0.0] * 30, [INVENTED_PX * 20] * 10 + [0] * 30)   # roads on every land tile
    bad = road_free_report(clean, inventing, kinds)
    assert not bad["passes"] and bad["all"]["fp_share"]["ci_low"] > FP_TOL
    assert bad["land"]["tiles_with_invented_road"]["candidate"] == 1.0
    assert bad["water"]["tiles_with_invented_road"]["candidate"] == 0.0


def test_land_negatives_have_their_own_gate():
    # Better on 30 water tiles, worse on the 10 land tiles: the overall average passes,
    # the land limit must still fail it.
    from src.pipeline.p1_segment.val_apls_select import INVENTED_PX, road_free_report

    kinds = ["land"] * 10 + ["water"] * 30
    deployed = ([0.0] * 10 + [0.01] * 30, [0] * 10 + [INVENTED_PX * 10] * 30)
    candidate = ([0.006] * 10 + [0.0] * 30, [INVENTED_PX * 5] * 10 + [0] * 30)
    report = road_free_report(deployed, candidate, kinds)
    assert report["passes_all"] and not report["passes_land"] and not report["passes"]


def test_rejected_stage2_falls_back_to_stage1(tmp_path):
    pytest.importorskip("modal")
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
    from modal_train import finish_stage

    rejected = finish_stage(tmp_path, "stage2", {"best": None, "history": []}, {"recipe": {}}, required=False)
    saved = json.loads((tmp_path / "stage2.json").read_text())
    assert rejected and saved["rejected"] and (tmp_path / "stage2.done").exists()   # retries skip it
    assert not finish_stage(tmp_path, "stage1", {"best": {"epoch": 3}}, {}, required=True)
    with pytest.raises(RuntimeError, match="kept no epoch"):
        finish_stage(tmp_path, "stage1", {"best": None}, {}, required=True)          # nothing to fall back on


def test_candidate_list_publishes_stage1_alone_when_stage2_is_rejected(tmp_path):
    # The list train() publishes, not just finish_stage's return value.
    pytest.importorskip("modal")
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
    from modal_train import publish_candidates

    (tmp_path / "stage1.pt").touch()
    (tmp_path / "stage2.ep01.pt").touch()
    assert publish_candidates(tmp_path, stage2_rejected=True) == ["stage1.pt"]
    assert json.loads((tmp_path / "candidates.json").read_text())["candidates"] == ["stage1.pt"]
    assert publish_candidates(tmp_path, stage2_rejected=False) == ["stage1.pt", "stage2.ep01.pt"]


def test_claim_run_allows_one_launch_per_run(tmp_path):
    pytest.importorskip("modal")
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
    from modal_train import claim_run, release_run

    class Store(dict):          # the Modal Dict surface claim_run uses: atomic put-if-absent
        def put(self, key, value, skip_if_exists=False):
            if skip_if_exists and key in self:
                return False
            self[key] = value
            return True

    store = Store()
    claim_run(store, "a51", {"launch": 1})
    with pytest.raises(SystemExit, match="locked by an earlier launch"):
        claim_run(store, "a51", {"launch": 2})                     # a second concurrent launch
    claim_run(store, "a51", {"launch": 3}, takeover=True)         # deliberate takeover of a dead launch
    assert store["a51"] == {"launch": 3}
    release_run(store, "a51")
    release_run(store, "a51")                                     # releasing twice is harmless
    claim_run(store, "a51", {"launch": 4})


def test_selector_refuses_unsafe_candidates_and_changed_assets(tmp_path):
    from src.pipeline.p1_segment.val_apls_select import candidate_paths, check_assets

    assert candidate_paths(tmp_path, ["stage1.pt"]) == [(tmp_path / "stage1.pt").resolve()]
    for bad in ("../outside.pt", "sub/stage1.pt", "stage1.pkl", str(tmp_path / "abs.pt")):
        with pytest.raises(SystemExit, match="refusing candidate"):
            candidate_paths(tmp_path, [bad])
    import hashlib

    asset = tmp_path / "land.json"
    asset.write_text('{"land_tiles": []}')
    record = {"assets": {"land_tiles": hashlib.sha256(asset.read_bytes()).hexdigest()}}
    check_assets(record, {"land_tiles": asset})                   # unchanged: fine
    asset.write_text('{"land_tiles": ["x"]}')
    with pytest.raises(SystemExit, match="differs from the run"):
        check_assets(record, {"land_tiles": asset})
    check_assets({}, {"land_tiles": asset})                       # older runs: note only


def test_road_free_tiles_kinds(tmp_path):
    from src.pipeline.p1_segment.val_apls_select import road_free_tiles

    def tile(stem, value, road=False, black=False):
        img = np.full((16, 16, 3), value, np.uint8)
        if black:
            img[:, 2:] = 0                       # mostly chip-edge padding
        mask = np.zeros((16, 16), np.uint8)
        if road:
            mask[8, :] = 255
        Image.fromarray(img).save(tmp_path / f"{stem}_sat.png")
        Image.fromarray(mask, mode="L").save(tmp_path / f"{stem}_mask.png")
        return tmp_path / f"{stem}_sat.png", tmp_path / f"{stem}_mask.png"

    pairs = [tile("roof", 120), tile("sea", 90), tile("edge", 90, black=True), tile("street", 90, road=True)]
    kinds = {Path(p[0]).stem: k for p, k in road_free_tiles(pairs, {"roof_sat.png"})}
    assert kinds == {"roof_sat": "land", "sea_sat": "water", "edge_sat": "no-data"}   # road tile excluded


def test_record_run_refuses_a_different_recipe_or_unrecorded_artifacts(tmp_path):
    run = tmp_path / "a51-mit_b5"
    full = {"recipe": {"encoder": "mit_b5", "pilot": False}, "data": {"spacenet": {"pairs": 2}}}
    record_run(run, full, {"commit": "abc", "dirty": False})
    again = record_run(run, full, {"commit": "abc", "dirty": False})   # same recipe and code: a resume
    assert [launch["commit"] for launch in again["launches"]] == ["abc", "abc"]
    for changed in ({**full, "recipe": {"encoder": "mit_b5", "pilot": True}},   # pilot vs full
                    {**full, "recipe": {"encoder": "mit_b3", "pilot": False}},  # another encoder
                    {**full, "data": {"spacenet": {"pairs": 3}}}):              # data changed
        with pytest.raises(RuntimeError, match="different recipe or data"):
            record_run(run, changed, {"commit": "abc", "dirty": False})
    # Changed (or uncommitted) code must not silently reuse the run's checkpoints...
    for code in ({"commit": "def", "dirty": False}, {"commit": "abc", "dirty": True}):
        with pytest.raises(RuntimeError, match="code changed"):
            record_run(run, full, code)
    # ...unless that is a deliberate, recorded choice.
    resumed = record_run(run, full, {"commit": "def", "dirty": False}, allow_code_change=True)
    assert resumed["launches"][-1] == {"commit": "def", "dirty": False, "allow_code_change": True}
    assert not list(run.glob("*.tmp"))                                  # written via an atomic replace
    legacy = tmp_path / "a51-b5"
    legacy.mkdir()
    (legacy / "stage1.done").touch()
    with pytest.raises(RuntimeError, match="no run.json"):
        record_run(legacy, full, {"commit": "abc"})
    assert json.loads((run / "run.json").read_text())["recipe"] == full["recipe"]


def test_validation_split_refuses_local_data_that_differs_from_the_run(tmp_path):
    from src.pipeline.p1_segment.val_apls_select import validation_split

    sn, dg = tmp_path / "sn", tmp_path / "dg"
    for chip in range(10):
        _pair(sn, f"sn5mum_chip{chip}_r0_c0")
    for i in range(6):
        _pair(dg, f"d{i}")
    record = {"recipe": {"seed": 2026, "pilot_cap": None, "common": {"deepglobe_val": 2}},
              "data": {"spacenet": dir_fingerprint(sn), "deepglobe": dir_fingerprint(dg)}}
    chips, _, dg_val = validation_split(record, sn, dg, tmp_path / "heldout.json")
    held = json.loads((tmp_path / "heldout.json").read_text())["test_chips"]
    assert chips and not set(chips) & set(held) and len(dg_val) == 2
    _pair(dg, "d_new")
    with pytest.raises(SystemExit, match="deepglobe data differs"):
        validation_split(record, sn, dg, tmp_path / "heldout.json")
