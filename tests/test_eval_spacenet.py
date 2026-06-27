"""Tests for A17 SpaceNet-Mumbai held-out benchmark (`p1_segment/eval_spacenet.py`).

Split must be (a) chip-level (no tiles from one chip leaking across train/test),
(b) deterministic/frozen, so a future fine-tune (A23) can trust the held-out set.
"""
import json

from src.pipeline.p1_segment.eval_spacenet import (
    chip_of,
    load_or_make_heldout,
    make_split,
)


def test_chip_of_extracts_chip_id():
    assert chip_of("sn5mum_chip12_r0_c3_sat.jpg") == "chip12"
    assert chip_of("sn5mum_chip0_r1_c0") == "chip0"


def test_make_split_is_deterministic_disjoint_and_chip_level():
    chips = [f"chip{i}" for i in range(100)]
    train_a, test_a = make_split(chips, test_frac=0.2, seed=17)
    train_b, test_b = make_split(chips, test_frac=0.2, seed=17)
    assert test_a == test_b                      # deterministic
    assert set(train_a).isdisjoint(test_a)       # no leakage
    assert set(train_a) | set(test_a) == set(chips)
    assert abs(len(test_a) - 20) <= 1            # ~20%


def test_make_split_changes_with_seed():
    chips = [f"chip{i}" for i in range(100)]
    _, test_a = make_split(chips, test_frac=0.2, seed=1)
    _, test_b = make_split(chips, test_frac=0.2, seed=2)
    assert test_a != test_b


def test_load_or_make_heldout_freezes_to_manifest(tmp_path):
    chips = [f"chip{i}" for i in range(50)]
    manifest = tmp_path / "heldout.json"
    first = load_or_make_heldout(chips, manifest, test_frac=0.2, seed=17)
    assert manifest.is_file()
    # second call ignores inputs and returns the frozen set
    second = load_or_make_heldout([f"chip{i}" for i in range(999)], manifest, test_frac=0.5, seed=99)
    assert first == second
    assert set(json.loads(manifest.read_text())["test_chips"]) == set(first)
