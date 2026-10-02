"""A51 experiment record: run.json identity, data fingerprints, validation-split rebuild."""

from __future__ import annotations

import json

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
    chips, dg_val = validation_split(record, sn, dg, tmp_path / "heldout.json")
    held = json.loads((tmp_path / "heldout.json").read_text())["test_chips"]
    assert chips and not set(chips) & set(held) and len(dg_val) == 2
    _pair(dg, "d_new")
    with pytest.raises(SystemExit, match="deepglobe data differs"):
        validation_split(record, sn, dg, tmp_path / "heldout.json")
