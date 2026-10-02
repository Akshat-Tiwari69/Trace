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
    record_run(run, full, {"commit": "abc"})
    again = record_run(run, full, {"commit": "def"})                   # same recipe: a resume
    assert [launch["commit"] for launch in again["launches"]] == ["abc", "def"]
    for changed in ({**full, "recipe": {"encoder": "mit_b5", "pilot": True}},   # pilot vs full
                    {**full, "recipe": {"encoder": "mit_b3", "pilot": False}},  # another encoder
                    {**full, "data": {"spacenet": {"pairs": 3}}}):              # data changed
        with pytest.raises(RuntimeError, match="different recipe or data"):
            record_run(run, changed, {"commit": "ghi"})
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
