"""A17 — truthful Indian benchmark on a frozen held-out SpaceNet-5 Mumbai split.

SpaceNet-5 Mumbai (built in A16) is real, human-drawn Indian road ground truth.
This module freezes a **chip-level** held-out TEST split (so a future supervised
fine-tune — A23 — can never train on it) and scores checkpoints on it with
**real-GT IoU/Dice** at a fixed threshold. That replaces the *misleading*
OSM-agreement metric (A12): for the first time we measure on real Indian GT.

Single source of truth for the split lives in the committed manifest
``data/sample/spacenet_mumbai_heldout_chips.json`` so it stays frozen forever.

    python -m src.pipeline.p1_segment.eval_spacenet \
        --checkpoints models/deepglobe_mit_b3_scse_512px_best.pt models/road_v2.pt \
        --device cuda --image-size 384

APLS (topology metric) is the planned follow-up — it needs metre-correct
mask->graph plumbing (reuse S7 `p3_analysis/apls.py`); IoU is delivered here.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

CHIP_RE = re.compile(r"(chip\d+)")
DEFAULT_CORPUS = Path("data/raw/spacenet/dg_format")
DEFAULT_MANIFEST = Path("data/sample/spacenet_mumbai_heldout_chips.json")


def chip_of(name: str) -> str:
    """Return the SpaceNet chip id embedded in a tile name (e.g. ``chip12``)."""
    m = CHIP_RE.search(name)
    if not m:
        raise ValueError(f"no chip id in {name!r}")
    return m.group(1)


def make_split(chips: list[str], test_frac: float = 0.2, seed: int = 17) -> tuple[list[str], list[str]]:
    """Deterministic chip-level train/test split (sorted then seeded-shuffled)."""
    ordered = sorted(set(chips))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_test = max(1, round(len(ordered) * test_frac))
    test = sorted(ordered[:n_test], key=lambda c: int(c[4:]))
    train = sorted(set(ordered) - set(test), key=lambda c: int(c[4:]))
    return train, test


def load_or_make_heldout(
    chips: list[str], manifest: Path, test_frac: float = 0.2, seed: int = 17
) -> list[str]:
    """Return the frozen held-out TEST chips, creating the manifest on first use."""
    manifest = Path(manifest)
    if manifest.is_file():
        return json.loads(manifest.read_text())["test_chips"]
    _, test = make_split(chips, test_frac=test_frac, seed=seed)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(
        {"dataset": "spacenet5_mumbai_aoi8", "test_frac": test_frac, "seed": seed,
         "n_test_chips": len(test), "test_chips": test}, indent=2))
    return test


def heldout_pairs(corpus: Path, test_chips: list[str]) -> list[tuple[Path, Path]]:
    """(sat, mask) pairs whose chip is in the held-out TEST set."""
    test = set(test_chips)
    out = []
    for sat in sorted(Path(corpus).glob("*_sat.jpg")):
        if chip_of(sat.name) in test:
            mask = sat.with_name(sat.name.replace("_sat.jpg", "_mask.png"))
            if mask.exists():
                out.append((sat, mask))
    return out


def evaluate_checkpoints(
    checkpoints: list[Path],
    corpus: Path = DEFAULT_CORPUS,
    manifest: Path = DEFAULT_MANIFEST,
    threshold: float = 0.44,
    image_size: int = 384,
    device: str = "cpu",
) -> dict:
    """Score each checkpoint on the frozen held-out SpaceNet-Mumbai split (IoU/Dice)."""
    from torch.utils.data import DataLoader

    from src.pipeline.p1_segment.dataset import RoadTileDataset, build_val_transform
    from src.pipeline.p1_segment.model import load_checkpoint
    from src.pipeline.p1_segment.train import evaluate

    all_chips = sorted({chip_of(p.name) for p in Path(corpus).glob("*_sat.jpg")})
    test_chips = load_or_make_heldout(all_chips, manifest)
    pairs = heldout_pairs(corpus, test_chips)
    print(f"held-out SpaceNet-Mumbai TEST: {len(test_chips)} chips / {len(pairs)} tiles "
          f"(of {len(all_chips)} chips)", flush=True)

    loader = DataLoader(RoadTileDataset(pairs, build_val_transform(image_size)),
                        batch_size=4, shuffle=False, num_workers=0)
    results = {}
    for ckpt in checkpoints:
        model, meta = load_checkpoint(ckpt, map_location=device)
        model.to(device).eval()
        m = evaluate(model, loader, device, threshold=threshold)
        results[Path(ckpt).name] = {"iou": m["iou"], "dice": m["dice"]}
        print(f"  {Path(ckpt).name:42s} real-GT IoU {m['iou']:.4f}  Dice {m['dice']:.4f}", flush=True)
        del model
    return {"n_test_chips": len(test_chips), "n_test_tiles": len(pairs),
            "threshold": threshold, "image_size": image_size, "models": results}


def main() -> None:
    p = argparse.ArgumentParser(description="A17: eval on held-out SpaceNet-5 Mumbai (real Indian GT).")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--threshold", type=float, default=0.44)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="data/sample/spacenet_mumbai_eval.json")
    args = p.parse_args()

    report = evaluate_checkpoints([Path(c) for c in args.checkpoints], Path(args.corpus),
                                  Path(args.manifest), args.threshold, args.image_size, args.device)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
