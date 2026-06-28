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


def test_train_and_heldout_pairs_are_disjoint_by_chip(tmp_path):
    """A23 train pairs must never include a held-out chip (no leakage)."""
    from src.pipeline.p1_segment.eval_spacenet import heldout_pairs, train_pairs
    corpus = tmp_path / "dg"
    corpus.mkdir()
    for chip in range(6):
        for r in range(2):
            (corpus / f"sn5mum_chip{chip}_r{r}_c0_sat.jpg").write_bytes(b"x")
            (corpus / f"sn5mum_chip{chip}_r{r}_c0_mask.png").write_bytes(b"x")
    test_chips = ["chip0", "chip3"]
    held = {chip_of(p[0].name) for p in heldout_pairs(corpus, test_chips)}
    train = {chip_of(p[0].name) for p in train_pairs(corpus, test_chips)}
    assert held == {"chip0", "chip3"}
    assert train.isdisjoint(held)
    assert held | train == {f"chip{i}" for i in range(6)}


def test_grayscale_val_transform_desaturates_colour_only():
    """A24: the grayscale (Cartosat-PAN proxy) transform desaturates a colour
    image but is a no-op on an already-grey one."""
    import numpy as np

    from src.pipeline.p1_segment.dataset import build_val_transform

    rng = np.random.default_rng(0)
    mask = np.zeros((32, 32), np.uint8)
    colour = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    rgb = build_val_transform(32, grayscale=False)(image=colour, mask=mask)["image"]
    grey = build_val_transform(32, grayscale=True)(image=colour, mask=mask)["image"]
    assert grey.shape == rgb.shape
    assert not np.allclose(grey.numpy(), rgb.numpy())          # ToGray changed the colour image
    g = rng.integers(0, 255, (32, 32), dtype=np.uint8)
    already_grey = np.stack([g, g, g], -1)
    a = build_val_transform(32, grayscale=False)(image=already_grey, mask=mask)["image"]
    b = build_val_transform(32, grayscale=True)(image=already_grey, mask=mask)["image"]
    assert np.allclose(a.numpy(), b.numpy(), atol=1e-5)        # no-op on a grey image


def test_load_or_make_heldout_freezes_to_manifest(tmp_path):
    chips = [f"chip{i}" for i in range(50)]
    manifest = tmp_path / "heldout.json"
    first = load_or_make_heldout(chips, manifest, test_frac=0.2, seed=17)
    assert manifest.is_file()
    # second call ignores inputs and returns the frozen set
    second = load_or_make_heldout([f"chip{i}" for i in range(999)], manifest, test_frac=0.5, seed=99)
    assert first == second
    assert set(json.loads(manifest.read_text())["test_chips"]) == set(first)
