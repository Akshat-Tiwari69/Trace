"""A6 fine-tune harness test — orchestration + anti-forget selection on CPU."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.pipeline.p1_segment.finetune import (
    FineTuneConfig,
    _build_optimizer,
    _iou_scores_on_pairs,
    _select_threshold,
    finetune,
    gather_pairs,
)
from src.pipeline.p1_segment.model import build_model, load_checkpoint, save_checkpoint


def _write_pair(folder, stem, size=80):
    """Write one DeepGlobe-format pair (0/255 mask, the convention readers expect)."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(stem)) % 2**32)
    Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(folder / f"{stem}_sat.jpg")
    mask = np.zeros((size, size), np.uint8)
    mask[size // 2 - 2 : size // 2 + 2, :] = 255          # a horizontal road
    Image.fromarray(mask, mode="L").save(folder / f"{stem}_mask.png")


def _tiny_v1_checkpoint(path):
    model = build_model(encoder_weights=None, decoder_attention_type="scse")
    save_checkpoint(model, path, meta={"encoder": "mit_b0", "arch": "unet",
                                        "decoder_attention_type": "scse", "image_size": 64, "threshold": 0.44})


def test_model_selection_scores_share_hann_blended_probabilities(monkeypatch):
    import src.pipeline.p1_segment.finetune as finetune_module

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    ground_truth = np.zeros((4, 4), dtype=bool)
    ground_truth[:, :2] = True
    probability = ground_truth.astype(np.float32)
    blended_calls = []

    monkeypatch.setattr(finetune_module, "_read_val_pair", lambda *_: (image, ground_truth))
    monkeypatch.setattr(
        finetune_module,
        "predict_large_prob",
        lambda *args, **kwargs: blended_calls.append((args, kwargs)) or probability,
    )
    monkeypatch.setattr(
        finetune_module,
        "predict_large",
        lambda *args, **kwargs: np.zeros_like(ground_truth, dtype=np.uint8),
        raising=False,
    )

    _, selected_iou = _select_threshold(None, [("sat", "mask")], 4, "cpu", (0.5,))
    reported_iou = _iou_scores_on_pairs(None, [("sat", "mask")], 4, "cpu", 0.5)

    assert reported_iou == [selected_iou] == [1.0]
    assert len(blended_calls) == 2


def test_resume_rejects_scores_from_an_unlabelled_inference_protocol(monkeypatch):
    import src.pipeline.p1_segment.finetune as finetune_module

    monkeypatch.setattr(
        finetune_module,
        "load_checkpoint",
        lambda *args, **kwargs: (torch.nn.Conv2d(3, 1, 1), {"threshold": 0.5}),
    )
    monkeypatch.setattr(
        finetune_module,
        "load_train_state",
        lambda *args, **kwargs: {
            "v1_deepglobe": 0.5,
            "v1_indian": 0.5,
            "v1_deepglobe_scores": [0.5],
        },
    )
    pair = ("sat.jpg", "mask.png")
    monkeypatch.setattr(finetune_module, "gather_pairs", lambda cfg: ([pair], [pair], [pair]))

    cfg = FineTuneConfig(init_checkpoint="init.pt", resume="old.last.pt", deepglobe_dir="dg")
    with pytest.raises(ValueError, match="inference protocol"):
        finetune(cfg)


def test_gather_pairs_3way_split_disjoint(tmp_path):
    ft, dg = tmp_path / "ft", tmp_path / "dg"
    for i in range(10):
        _write_pair(ft, f"c{i}")
    for i in range(30):
        _write_pair(dg, f"d{i}")
    cfg = FineTuneConfig(init_checkpoint="x", finetune_dir=ft, deepglobe_dir=dg,
                         deepglobe_subset=8, deepglobe_val=5, finetune_oversample=2, val_fraction=0.2)
    train, ind_val, dg_val = gather_pairs(cfg)
    assert len(ind_val) == 2                       # round(10 * 0.2)
    assert len(dg_val) == 5                         # held-out DeepGlobe forget-check
    assert len(train) == (10 - 2) * 2 + 8          # indian ×2 + anchor
    # the forget-check tiles must NOT leak into the training anchor
    assert {p[0] for p in dg_val}.isdisjoint(p[0] for p in train)


def test_gather_pairs_keeps_spacenet_chip_tiles_in_one_split(tmp_path):
    dg = tmp_path / "dg"
    dg.mkdir()
    for i in range(8):
        _write_pair(dg, f"dg{i}")
    indian = []
    for chip in range(5):
        for tile in range(3):
            sat = tmp_path / f"sn5mum_chip{chip}_r0_c{tile}_sat.jpg"
            mask = tmp_path / f"sn5mum_chip{chip}_r0_c{tile}_mask.png"
            indian.append((sat, mask))
    cfg = FineTuneConfig(
        init_checkpoint="x", finetune_pairs=indian, deepglobe_dir=dg,
        deepglobe_subset=2, deepglobe_val=2, finetune_oversample=1,
        val_fraction=0.4,
    )

    train, val, _ = gather_pairs(cfg)
    train_chips = {p[0].name.split("_")[1] for p in train if "sn5mum_" in p[0].name}
    val_chips = {p[0].name.split("_")[1] for p in val}
    assert train_chips.isdisjoint(val_chips)


def test_gather_pairs_extra_pairs_are_train_only(tmp_path):
    ft = tmp_path / "ft"
    for i in range(10):
        _write_pair(ft, f"c{i}")
    extra = [(tmp_path / f"mgrid_{i}_sat.jpg", tmp_path / f"mgrid_{i}_mask.png") for i in range(4)]
    cfg = FineTuneConfig(init_checkpoint="x", finetune_dir=ft, extra_train_pairs=extra,
                         finetune_oversample=2, val_fraction=0.2)
    train, val, _ = gather_pairs(cfg)
    assert set(extra) <= set(train)
    assert sum(pair in extra for pair in train) == 4          # added once, never oversampled
    assert not set(extra) & set(val)                          # never used for checkpoint selection


def test_select_threshold_pools_iou_so_invented_roads_count(monkeypatch):
    # A51: a per-tile mean scores a road-free tile 0 whatever is predicted; pooled IoU
    # charges its false positives.
    import src.pipeline.p1_segment.finetune as finetune_module

    road = np.zeros((4, 4), bool)
    road[:, :2] = True
    tiles = {"road": (np.zeros((4, 4, 3), np.uint8), road),
             "empty": (np.ones((4, 4, 3), np.uint8), np.zeros((4, 4), bool))}
    monkeypatch.setattr(finetune_module, "_read_val_pair", lambda sat, mask: tiles[sat])

    def iou_with(invented):
        monkeypatch.setattr(finetune_module, "predict_large_prob",
                            lambda model, image, **k: road.astype(np.float32)
                            if image is tiles["road"][0] else invented)
        return _select_threshold(None, [("road", "m"), ("empty", "m")], 4, "cpu", (0.5,))[1]

    assert iou_with(np.zeros((4, 4), np.float32)) == 1.0
    assert iou_with(np.ones((4, 4), np.float32)) == pytest.approx(8 / 24)  # 8 hits / (8 road + 16 invented)


def test_finetune_save_every_epoch_and_flags_not_beating_init(tmp_path):
    ft, dg = tmp_path / "ft", tmp_path / "dg"
    for i in range(5):
        _write_pair(ft, f"c{i}")
    for i in range(6):
        _write_pair(dg, f"d{i}")
    init = tmp_path / "v1.pt"
    _tiny_v1_checkpoint(init)
    out = tmp_path / "v2.pt"
    summary = finetune(FineTuneConfig(
        init_checkpoint=init, finetune_dir=ft, deepglobe_dir=dg, deepglobe_subset=3, deepglobe_val=2,
        out_path=out, image_size=64, batch_size=2, epochs=2, finetune_oversample=2,
        deepglobe_iou_tolerance=1.0, device="cpu", save_every_epoch=True))
    assert (tmp_path / "v2.ep01.pt").exists() and (tmp_path / "v2.ep02.pt").exists()
    assert load_checkpoint(tmp_path / "v2.ep02.pt")[1]["epoch"] == 2
    assert summary["beats_init"] == (summary["best"]["indian_iou"] > summary["v1_indian"])


def test_per_group_sampler_draws_a_fixed_fresh_share_per_city():
    from src.pipeline.p1_segment.finetune import _PerGroupEpochSampler

    # base 0..9, big city 10..109 (100 pairs), small city 110..114 (5 pairs), 12 per city
    sampler = _PerGroupEpochSampler(10, [100, 5], 12, seed=1)
    epochs = []
    for epoch in (1, 2, 3):
        sampler.set_epoch(epoch)
        epochs.append(list(sampler))
    first, second = epochs[0], epochs[1]
    assert len(first) == len(sampler) == 10 + 2 * 12
    for drawn in (first, second):
        assert sorted(i for i in drawn if i < 10) == list(range(10))          # base: every pair once
        assert sum(10 <= i < 110 for i in drawn) == 12                         # fixed share per city
        small = sorted(i for i in drawn if i >= 110)
        assert len(small) == 12 and set(small) == set(range(110, 115))        # small city repeats
    assert {i for i in first if 10 <= i < 110} != {i for i in second if 10 <= i < 110}   # fresh draw
    # A resume builds a new sampler (and loader) at epoch 3: it must replay the uninterrupted
    # run's epoch 3 whatever else consumed random numbers in between.
    torch.randperm(1000, generator=torch.Generator().manual_seed(1))
    torch.rand(100)
    resumed = _PerGroupEpochSampler(10, [100, 5], 12, seed=1)
    resumed.set_epoch(3)
    assert list(resumed) == epochs[2]
    plain = _PerGroupEpochSampler(7, [], 0, seed=1)                            # no groups: a shuffle
    assert sorted(plain) == list(range(7))


def test_finetune_trains_on_per_city_draws(tmp_path):
    ft, dg = tmp_path / "ft", tmp_path / "dg"
    for i in range(5):
        _write_pair(ft, f"c{i}")
    for i in range(6):
        _write_pair(dg, f"d{i}")
    cities = {}
    for city, n in (("big", 6), ("small", 2)):
        for i in range(n):
            _write_pair(tmp_path / city, f"{city}{i}")
        cities[city] = [(tmp_path / city / f"{city}{i}_sat.jpg", tmp_path / city / f"{city}{i}_mask.png")
                        for i in range(n)]
    init = tmp_path / "v1.pt"
    _tiny_v1_checkpoint(init)
    common = dict(init_checkpoint=init, finetune_dir=ft, deepglobe_dir=dg, deepglobe_subset=2,
                  deepglobe_val=2, out_path=tmp_path / "v2.pt", image_size=64, batch_size=2, epochs=1,
                  finetune_oversample=1, deepglobe_iou_tolerance=1.0, device="cpu")
    summary = finetune(FineTuneConfig(extra_train_groups=cities, extra_per_group=3, **common))
    assert summary["best"] is not None
    with pytest.raises(ValueError, match="extra_per_group"):
        finetune(FineTuneConfig(extra_train_groups=cities, extra_per_group=0, **common))


def test_cosine_scheduler_warms_up_decays_and_resumes_on_the_curve():
    from src.pipeline.p1_segment.finetune import _cosine_scheduler

    def make_optimizer():
        net = torch.nn.Linear(2, 1)
        return torch.optim.AdamW([{"params": [net.weight]}, {"params": [net.bias]}], lr=1.0)

    cfg = FineTuneConfig(init_checkpoint="x", lr=1.0, encoder_lr_scale=0.5, epochs=4,
                         warmup_epochs=1, cosine=True)
    opt = make_optimizer()
    scheduler = _cosine_scheduler(opt, cfg, steps_per_epoch=10)
    lrs = []
    for _ in range(40):
        lrs.append(opt.param_groups[0]["lr"])
        assert opt.param_groups[1]["lr"] == pytest.approx(0.5 * lrs[-1])   # encoder keeps its scale
        opt.step()
        scheduler.step()
    assert lrs[0] == pytest.approx(0.1) and lrs[10] == pytest.approx(1.0)  # warm-up, then peak
    assert 0.03 <= lrs[-1] < 0.05                                          # decayed to the floor
    resumed = make_optimizer()
    _cosine_scheduler(resumed, cfg, steps_per_epoch=10, start_epoch=3)
    assert resumed.param_groups[0]["lr"] == pytest.approx(lrs[20])          # same point on the curve


def test_train_one_epoch_steps_scheduler_per_batch_and_clips():
    from src.pipeline.p1_segment.train import train_one_epoch

    net = torch.nn.Conv2d(3, 1, 1)
    batches = [(torch.randn(2, 3, 8, 8), torch.rand(2, 1, 8, 8).round()) for _ in range(3)]
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
    loss = train_one_epoch(net, batches, opt, torch.nn.functional.binary_cross_entropy_with_logits,
                           "cpu", None, scheduler=scheduler, max_grad_norm=1e-3)
    assert scheduler.last_epoch == 3 and loss > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP step skipping needs CUDA")
def test_amp_skipped_step_does_not_advance_the_scheduler():
    # A51: AMP skips the optimizer step on an overflowing batch; stepping the scheduler
    # anyway drew PyTorch's "lr_scheduler.step() before optimizer.step()" warning.
    import warnings

    from src.pipeline.p1_segment.train import train_one_epoch

    net = torch.nn.Conv2d(3, 1, 1).cuda()
    batches = [(torch.randn(2, 3, 8, 8), torch.rand(2, 1, 8, 8).round()) for _ in range(3)]
    calls = iter([float("inf"), 1.0, 1.0])                  # the first batch overflows

    def loss_fn(logits, target):
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, target) * next(calls)

    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train_one_epoch(net, batches, opt, loss_fn, "cuda", torch.amp.GradScaler("cuda"), scheduler=scheduler)
    assert scheduler.last_epoch == 2                         # two real steps, the skipped one not counted
    assert not [w for w in caught if "lr_scheduler.step()" in str(w.message)]


def test_freeze_encoder_disables_encoder_grads(tmp_path):
    model = build_model(encoder_weights=None, decoder_attention_type="scse")
    cfg = FineTuneConfig(init_checkpoint="x", finetune_dir=tmp_path, encoder_lr_scale=0.0)
    _build_optimizer(model, cfg)
    enc = [p.requires_grad for n, p in model.named_parameters() if n.startswith("encoder.")]
    dec = [p.requires_grad for n, p in model.named_parameters() if not n.startswith("encoder.")]
    assert not any(enc) and all(dec)               # encoder frozen, decoder trainable


def test_finetune_selects_and_saves_releasable_checkpoint(tmp_path):
    ft, dg = tmp_path / "ft", tmp_path / "dg"
    for i in range(5):
        _write_pair(ft, f"c{i}")
    for i in range(8):
        _write_pair(dg, f"d{i}")
    init = tmp_path / "v1.pt"
    _tiny_v1_checkpoint(init)
    out = tmp_path / "v2.pt"
    cfg = FineTuneConfig(init_checkpoint=init, finetune_dir=ft, deepglobe_dir=dg,
                         deepglobe_subset=4, deepglobe_val=2, out_path=out, image_size=64,
                         batch_size=2, epochs=1, finetune_oversample=2, crops_per_image=1,
                         deepglobe_iou_tolerance=1.0, device="cpu")  # big tol → always "keeps"
    summary = finetune(cfg)

    assert out.exists() and summary["best"] is not None
    model, meta = load_checkpoint(out)
    assert meta["encoder_frozen"] is True
    assert "indian_val_iou" in meta and "deepglobe_val_iou" in meta
    assert meta["threshold"] in cfg.selection_thresholds
    assert meta["threshold"] == summary["best"]["threshold"]
    assert "deepglobe_delta_ci_low" in meta
    assert "deepglobe_delta_ci_high" in meta
    assert meta["validation_inference_protocol"] == "hann_blended_probability_v1"
    assert summary["validation_inference_protocol"] == meta["validation_inference_protocol"]


def test_finetune_resume_continues_from_next_epoch(tmp_path):
    """A19: a --resume run picks up from the rolling `.last` checkpoint's epoch
    (restoring optimizer/scaler/best/history) instead of restarting at epoch 1."""
    from src.pipeline.p1_segment.finetune import _last_path
    ft, dg = tmp_path / "ft", tmp_path / "dg"
    for i in range(5):
        _write_pair(ft, f"c{i}")
    for i in range(6):
        _write_pair(dg, f"d{i}")
    init = tmp_path / "v1.pt"
    _tiny_v1_checkpoint(init)
    out = tmp_path / "v2.pt"
    common = dict(init_checkpoint=init, finetune_dir=ft, deepglobe_dir=dg,
                  deepglobe_subset=3, deepglobe_val=2, out_path=out, image_size=64,
                  batch_size=2, finetune_oversample=2, deepglobe_iou_tolerance=1.0, device="cpu")

    first = finetune(FineTuneConfig(epochs=1, **common))     # run A: 1 epoch → rolling .last @1
    assert first["start_epoch"] == 1
    last = _last_path(out)
    assert last.exists()

    resumed = finetune(FineTuneConfig(epochs=3, resume=last, **common))   # run B: resume → epochs 2,3
    assert resumed["start_epoch"] == 2                       # did NOT restart at epoch 1
    assert [r["epoch"] for r in resumed["history"]] == [1, 2, 3]  # epoch-1 row carried over


def test_finetune_resume_hands_cuda_rng_states_back_on_cpu(tmp_path, monkeypatch):
    """A51: on a GPU run the `.last` is loaded with map_location="cuda", which puts the
    CUDA RNG states on the GPU; set_rng_state_all only takes CPU ByteTensors (this
    crashed the first Modal resume). Simulated so it runs on CPU-only machines."""
    import src.pipeline.p1_segment.finetune as ft_mod
    from src.pipeline.p1_segment.finetune import _last_path

    class OnGpu:  # a ByteTensor that map_location="cuda" moved to the GPU
        def __init__(self, t):
            self.t = t

        def cpu(self):
            return self.t

    def set_states(states):
        assert all(isinstance(s, torch.Tensor) for s in states), "RNG state must be a torch.ByteTensor"
        received.extend(states)

    def load_on_gpu(path, map_location="cpu"):
        state = real_load(path)
        state["cuda_rng_states"] = [OnGpu(s) for s in state["cuda_rng_states"]]
        return state

    received, real_load = [], ft_mod.load_train_state
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [torch.zeros(4, dtype=torch.uint8)])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", set_states)
    monkeypatch.setattr(ft_mod, "load_train_state", load_on_gpu)
    ft, dg = tmp_path / "ft", tmp_path / "dg"
    for i in range(5):
        _write_pair(ft, f"c{i}")
    for i in range(6):
        _write_pair(dg, f"d{i}")
    init = tmp_path / "v1.pt"
    _tiny_v1_checkpoint(init)
    out = tmp_path / "v2.pt"
    common = dict(init_checkpoint=init, finetune_dir=ft, deepglobe_dir=dg,
                  deepglobe_subset=3, deepglobe_val=2, out_path=out, image_size=64,
                  batch_size=2, finetune_oversample=2, deepglobe_iou_tolerance=1.0, device="cpu")

    finetune(FineTuneConfig(epochs=1, **common))
    resumed = finetune(FineTuneConfig(epochs=2, resume=_last_path(out), **common))
    assert resumed["start_epoch"] == 2
    assert len(received) == 1 and received[0].dtype == torch.uint8


def test_finetune_sdt_bce_weight_runs_end_to_end(tmp_path):
    """A41: --sdt-bce forwards to ComboLoss and the run completes on CPU."""
    ft, dg = tmp_path / "ft", tmp_path / "dg"
    for i in range(5):
        _write_pair(ft, f"c{i}")
    for i in range(6):
        _write_pair(dg, f"d{i}")
    init = tmp_path / "v1.pt"
    _tiny_v1_checkpoint(init)
    out = tmp_path / "v_sdt.pt"
    cfg = FineTuneConfig(init_checkpoint=init, finetune_dir=ft, deepglobe_dir=dg,
                         deepglobe_subset=3, deepglobe_val=2, out_path=out, image_size=64,
                         batch_size=2, epochs=1, finetune_oversample=2,
                         sdt_bce_weight=1.0, deepglobe_iou_tolerance=1.0, device="cpu")
    summary = finetune(cfg)
    assert out.exists() and summary["best"] is not None


def test_finetune_grayscale_tracks_pan_proxy(tmp_path):
    """A24: grayscale_p>0 records the Cartosat-PAN (grayscale) IoU each epoch + in meta."""
    ft, dg = tmp_path / "ft", tmp_path / "dg"
    for i in range(5):
        _write_pair(ft, f"c{i}")
    for i in range(6):
        _write_pair(dg, f"d{i}")
    init = tmp_path / "v1.pt"
    _tiny_v1_checkpoint(init)
    out = tmp_path / "v32.pt"
    cfg = FineTuneConfig(init_checkpoint=init, finetune_dir=ft, deepglobe_dir=dg,
                         deepglobe_subset=3, deepglobe_val=2, out_path=out, image_size=64,
                         batch_size=2, epochs=1, finetune_oversample=2, grayscale_p=0.5,
                         deepglobe_iou_tolerance=1.0, device="cpu")
    summary = finetune(cfg)
    # per-epoch grayscale IoU is a real number (not the nan sentinel used when off)
    assert np.isfinite(summary["history"][0]["indian_gray_iou"])
    _, meta = load_checkpoint(out)
    assert "indian_gray_val_iou" in meta and np.isfinite(meta["indian_gray_val_iou"])
