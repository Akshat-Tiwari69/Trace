"""A18 (graph-first) common-unit APLS evaluator.

Honest paired comparison of **A18** (SAM-Road++, one graph per 400px chip)
against the deployed **v3.2** segmentation model, scored on the *same* frozen
raw SN5-Mumbai heldout chips, against the *same* vector GT (GeoJSON), in *one*
common coordinate frame.

Why this exists: v3.2's existing APLS number (`apls_eval.py`) is measured on
derived 512px tiles, while A18 predicts one graph per whole 400px chip. Comparing
those two numbers directly is the **same class of confound** that sank the A41
gate (a delta driven by measurement setup, not model quality). So a promotion
decision needs both models on a common unit vs the same GT -- this module.

Common frame: **400px chip pixels -> metres** at the chip's *anisotropic* GSD
(separate eff_x/eff_y from the GeoTIFF geotransform; EPSG:4326 gives ~5-6% x/y
anisotropy at Mumbai latitude), then rescaled to apls' degree convention. All
three graph sources live in this one frame with one ``(x=col, y=row)`` convention
-- the coordinate contract fixed in A18 (raster row/col, NOT Cartesian y=400-row).

Each model runs at its **own intended input scale** (so neither is handicapped),
then both outputs map into the common 400px frame:
  * v3.2: to_rgb8 percentile-stretch -> resize native 1300 by 0.6 -> 780px (~0.5m,
    its deployed GSD) -> predict_large_prob (A27 512/stride384 Hann blend) ->
    threshold at the checkpoint's meta threshold -> mask 780->400 (INTER_NEAREST).
  * A18: predicted graph loaded from the inferencer's canonical pickle
    ``<a18-pred-dir>/graph/mumbai_{chip}.p`` -- adj-dict ``{(r,c): [(r,c), ...]}``
    in raster (row,col), no flip.

Gate mode (``--strict``, both models present) FAILS unless every scorable GT chip
has BOTH scores -- a missing A18 prediction on a hard chip must count against A18,
not silently drop out of the paired bootstrap (the selection bias that would
inflate a promotion delta). requested/scorable/missing lists are always reported.

    python -m src.pipeline.p1_segment.chip_apls_eval --self-check --n-chips 3
    python -m src.pipeline.p1_segment.chip_apls_eval --strict \
        --v32 models/road_pan.pt --a18-pred-dir .tmp/a43_samroad/infer_out \
        --n-chips 80 --n-samples 600 --out .tmp/a18_vs_v32_chip_apls.json

ASCII-only prints (Windows redirected stdout is cp1252 -- bit A38/A39). Mandatory
__main__ guard (Windows subprocess/DataLoader safety).
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np

from src.pipeline.p1_segment.apls_eval import _DEG_X, _DEG_Y

ROOT = Path(__file__).resolve().parents[3]
SRC_RGB = ROOT / "data/raw/spacenet/SN5_roads_train_AOI_8_Mumbai/PS-RGB"
SRC_GEO = ROOT / "data/raw/spacenet/SN5_roads_train_AOI_8_Mumbai/geojson_roads_speed"
HELDOUT = ROOT / "data/sample/spacenet_mumbai_heldout_chips.json"

IMAGE_SIZE = 400   # 400px common frame -- must match the A18 converter/dataset
NATIVE_PX = 1300   # native SN5 PS-RGB chip size
V32_SCALE = 0.6    # native 1300 -> 780px (~0.5m, v3.2's deployed GSD)
GATE_N_SAMPLES = 600   # apls sample count for a gate-grade score


def _native_coord_to_frame(value: float) -> int:
    """Round a native-chip coordinate into the closed 400px raster frame."""
    return min(IMAGE_SIZE - 1, max(0, int(round(value * IMAGE_SIZE / NATIVE_PX))))


def _write_report(path: Path | str, report: dict) -> None:
    """Write JSON after ensuring a caller-supplied output directory exists."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))


def _chip_eff_xy(src) -> tuple[float, float]:
    """Anisotropic metres-per-pixel in the 400px frame: (eff_x [col], eff_y [row]).

    EPSG:4326 (geographic): lon-degrees compress by cos(lat), so the x pixel is
    smaller in metres than the y pixel -- deriving one axis for both (the old bug)
    biases edge lengths. Projected CRS: transform.a / transform.e are already metric.
    """
    ax, ey = abs(src.transform.a), abs(src.transform.e)
    if src.crs and src.crs.is_geographic:
        lat0 = (src.bounds.top + src.bounds.bottom) / 2.0
        mx = ax * 111_320.0 * math.cos(math.radians(lat0))
        my = ey * 110_540.0
    else:
        mx, my = ax, ey
    return mx * src.width / IMAGE_SIZE, my * src.height / IMAGE_SIZE


def _read_chip(chip: str):
    """Return (bands (3,H,W) uint16/8, geotransform, eff_x, eff_y) for a chip id.

    Bands are returned raw (channel-first) so the v3.2 path can apply its own
    to_rgb8 percentile stretch; GT/A18 need only the transform + eff_x/eff_y.
    """
    import rasterio

    tif = SRC_RGB / f"SN5_roads_train_AOI_8_Mumbai_PS-RGB_{chip}.tif"
    with rasterio.open(tif) as src:
        bands = src.read([1, 2, 3])
        transform = src.transform
        eff_x, eff_y = _chip_eff_xy(src)
    return bands, transform, eff_x, eff_y


def _geojson_adj(chip: str, transform) -> dict:
    """GeoJSON roads -> adj-dict ``{(r,c): [(r,c), ...]}`` in the 400px frame.

    Mirrors the A18 converter's ``geo_to_pixel`` so the eval GT is the identical
    vector-GT contract A18 trains against (no second convention).
    """
    from rasterio.transform import rowcol

    geo = SRC_GEO / f"SN5_roads_train_AOI_8_Mumbai_geojson_roads_speed_{chip}.geojson"
    gj = json.loads(geo.read_text())
    adj: dict = {}

    def add_edge(a, b):
        if a == b:
            return
        adj.setdefault(a, [])
        adj.setdefault(b, [])
        if b not in adj[a]:
            adj[a].append(b)
        if a not in adj[b]:
            adj[b].append(a)

    def to_px(lon, lat):
        r, c = rowcol(transform, lon, lat)
        return (_native_coord_to_frame(r), _native_coord_to_frame(c))

    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        if gtype == "LineString":
            lines = [geom["coordinates"]]
        elif gtype == "MultiLineString":
            lines = geom["coordinates"]
        else:
            continue
        for line in lines:
            pts = [to_px(lon, lat) for lon, lat, *_ in line]
            for a, b in zip(pts[:-1], pts[1:]):
                if (0 <= a[0] < IMAGE_SIZE and 0 <= a[1] < IMAGE_SIZE and
                        0 <= b[0] < IMAGE_SIZE and 0 <= b[1] < IMAGE_SIZE):
                    add_edge(a, b)
    return adj


def adj_to_apls_graph(adj: dict, eff_x: float, eff_y: float):
    """adj-dict ``{(r,c): [(r,c), ...]}`` (400px) -> apls-ready graph.

    Node ``x, y`` use the ``(x=col, y=row)`` contract in metres (anisotropic
    eff_x/eff_y), then rescaled to the degrees apls projects back. Edge ``length_m``
    is the true anisotropic metric length. GT-vector and A18-prediction graphs both
    flow through here, guaranteeing an identical frame to the v3.2 mask graph.
    """
    import networkx as nx

    g = nx.Graph()
    idx: dict = {}

    def nid(rc):
        if rc not in idx:
            i = len(idx)
            idx[rc] = i
            r, c = rc
            g.add_node(i, x=(c * eff_x) / _DEG_X, y=(r * eff_y) / _DEG_Y)
        return idx[rc]

    for a, nbrs in adj.items():
        ia = nid(a)
        for b in nbrs:
            ib = nid(b)
            if ia == ib:
                continue
            length = math.hypot((a[1] - b[1]) * eff_x, (a[0] - b[0]) * eff_y)
            g.add_edge(ia, ib, length_m=max(length, 1e-6))
    return g


def mask_to_apls_graph_aniso(mask01: np.ndarray, eff_x: float, eff_y: float):
    """Skeletonise a binary mask into an apls-ready graph with anisotropic metres.

    Uses ``skeleton_to_graph`` with ``Affine(eff_x,0,0, 0,eff_y,0)`` so node x,y and
    edge length_m are true metres (col*eff_x, row*eff_y), then rescales node metres
    to the same degree convention as :func:`adj_to_apls_graph`.
    """
    from affine import Affine
    from skimage.morphology import skeletonize

    from src.pipeline.p2_graph.skeleton_graph import skeleton_to_graph

    g = skeleton_to_graph(skeletonize(mask01.astype(bool)),
                          transform=Affine(eff_x, 0.0, 0.0, 0.0, eff_y, 0.0))
    for _, d in g.nodes(data=True):
        d["x"] = d["x"] / _DEG_X
        d["y"] = d["y"] / _DEG_Y
    return g


def chip_apls(pred_g, gt_g, n_samples: int = GATE_N_SAMPLES, tol_m: float = 15.0) -> float:
    """APLS of a predicted graph vs the GT graph, with the same guards as
    ``apls_eval.tile_apls``: GT<2 nodes -> NaN (skip); GT has roads but pred has
    none -> 0.0 (worst)."""
    from src.pipeline.p3_analysis.apls import apls

    if gt_g.number_of_nodes() < 2:
        return float("nan")
    if pred_g.number_of_nodes() < 2:
        return 0.0
    return apls(gt_g, pred_g, n_samples=n_samples, tol_m=tol_m)["apls"]


def v32_chip_graph(model, bands: np.ndarray, thr: float, eff_x: float, eff_y: float,
                   device: str = "cpu"):
    """Run v3.2 at its deployed scale and map the mask into the 400px frame.

    to_rgb8 stretch -> resize 1300 by V32_SCALE (~780px, 0.5m) -> predict_large_prob
    (A27 overlapped Hann blend, 512/stride384) -> threshold -> mask 780->400
    (INTER_NEAREST) -> anisotropic skeleton graph. Never single-shot at 400/416
    (out of v3.2's training distribution -- would handicap it).
    """
    import cv2

    from src.pipeline.p1_segment.model import predict_large_prob
    from src.pipeline.p1_segment.raster_io import to_rgb8

    rgb = to_rgb8(bands)
    nh = max(1, int(round(bands.shape[1] * V32_SCALE)))
    nw = max(1, int(round(bands.shape[2] * V32_SCALE)))
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    prob = predict_large_prob(model, rgb, tile_size=512, stride=384, device=device, tta=False)
    mask = (prob >= thr).astype(np.uint8)
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    return mask_to_apls_graph_aniso(mask, eff_x, eff_y)


def a18_pred_path(a18_pred_dir: Path | str, chip: str) -> Path:
    """Canonical inferencer artifact path for a chip: ``<dir>/graph/mumbai_{chip}.p``
    (raster row,col adj-dict, no flip). Single source of the path contract so the
    loader and its test agree."""
    return Path(a18_pred_dir) / "graph" / f"mumbai_{chip}.p"


def strict_exit_code(rep: dict) -> int:
    """Process status for a STRICT promotion gate (call only in ``--strict`` mode).

    A gate's exit code must encode *promotion success*, not merely evaluation
    completion:
      * 0 -- complete coverage AND A18 wins ("b wins": CI excludes 0, delta>0) -> promote
      * 2 -- incomplete coverage (fail-loud; a missing artifact must not pass)
      * 3 -- no promotion: regression ("a wins"), inconclusive, or missing comparison
    Pure function so it is unit-testable against the real BootstrapCI.verdict strings.
    """
    cmp = rep.get("compare")
    if not cmp:
        return 3
    verdict = str(cmp.get("verdict", ""))
    if verdict.startswith("GATE FAIL"):
        return 2
    return 0 if verdict == "b wins" else 3


def _heldout_chips(n_chips: int | None, seed: int) -> list[str]:
    chips = json.loads(HELDOUT.read_text())["test_chips"]
    chips = [c for c in chips
             if (SRC_RGB / f"SN5_roads_train_AOI_8_Mumbai_PS-RGB_{c}.tif").exists()]
    if n_chips and n_chips < len(chips):
        chips = random.Random(seed).sample(chips, n_chips)
    return sorted(chips, key=lambda x: int(x.replace("chip", "")))


def coverage(scorable: list[str], v32_scores: dict, a18_scores: dict,
             want_v32: bool, want_a18: bool) -> dict:
    """Which scorable chips are missing a requested model's score.

    A chip is *scorable* if it has >=2 GT nodes. Gate coverage is complete only when
    every scorable chip has every requested score -- otherwise the paired bootstrap
    would run on a biased subset (a hard chip a model failed on must not vanish).
    """
    missing_v32 = [c for c in scorable if want_v32 and c not in v32_scores]
    missing_a18 = [c for c in scorable if want_a18 and c not in a18_scores]
    return {
        "n_scorable": len(scorable),
        "missing_v32": missing_v32,
        "missing_a18": missing_a18,
        "complete": not missing_v32 and not missing_a18,
    }


def _normalized(scores: dict[str, float], ceilings: dict[str, float]) -> dict[str, float]:
    """Divide each chip score by its GT-vs-self ceiling -> fraction of achievable
    routing, clipped to [0,1]. Chips with a degenerate ceiling (<=1e-6, i.e. GT so
    fragmented nothing routes) are dropped: their raw score is 0/0, not a model
    failure. Keeps the paired bootstrap on chips where the metric can discriminate."""
    out: dict[str, float] = {}
    for chip, s in scores.items():
        c = ceilings.get(chip, 0.0)
        if c > 1e-6:
            out[chip] = min(1.0, s / c)
    return out


def _norm_mean(scores: dict[str, float], ceilings: dict[str, float]) -> float | None:
    norm = _normalized(scores, ceilings)
    return float(np.mean(list(norm.values()))) if norm else None


def _paired_comparison(scorable: list[str], v32_scores: dict[str, float],
                       a18_scores: dict[str, float], coverage_complete: bool) -> dict:
    """Build the paired A18-vs-v3.2 result, including an empty-overlap report."""
    common = [c for c in scorable if c in v32_scores and c in a18_scores]
    if not common:
        return {"n_paired": 0, "coverage_complete": coverage_complete,
                "delta_a18_minus_v32": None, "ci_low": None, "ci_high": None,
                "p_two_sided": None, "excludes_zero": False,
                "verdict": "no paired chips"}

    from src.pipeline.p1_segment.stats import paired_bootstrap_ci
    ci = paired_bootstrap_ci([v32_scores[c] for c in common],
                             [a18_scores[c] for c in common])
    print(f"  A18 vs v3.2 (chip-level, n={len(common)}, "
          f"coverage_complete={coverage_complete}): {ci.summary()}", flush=True)
    return {"n_paired": len(common), "coverage_complete": coverage_complete,
            "delta_a18_minus_v32": ci.delta, "ci_low": ci.ci_low,
            "ci_high": ci.ci_high, "p_two_sided": ci.p_two_sided,
            "excludes_zero": ci.excludes_zero, "verdict": ci.verdict}


def compare_on_chips(v32_ckpt: Path | None, a18_pred_dir: Path | None,
                     n_chips: int | None = None, threshold: float | None = None,
                     device: str = "cpu", seed: int = 7,
                     n_samples: int = GATE_N_SAMPLES, strict: bool = False) -> dict:
    """Score v3.2 and/or A18 vs vector GT on common heldout chips.

    Returns per-chip scores, a coverage report, and -- when both models are present
    and (in ``strict`` mode) coverage is complete -- a paired-bootstrap CI on the
    APLS delta (A18 - v3.2). Promotion is only justified when coverage is complete
    AND the CI excludes zero.
    """
    chips = _heldout_chips(n_chips, seed)

    model = thr = None
    if v32_ckpt is not None:
        from src.pipeline.p1_segment.model import load_checkpoint
        model, meta = load_checkpoint(v32_ckpt, map_location=device)
        model.to(device).eval()
        thr = threshold if threshold is not None else float(meta.get("threshold", 0.44))

    scorable: list[str] = []
    v32_scores: dict[str, float] = {}
    a18_scores: dict[str, float] = {}
    ceilings: dict[str, float] = {}   # per-chip apls(GT,GT): the achievable max on this chip
    for chip in chips:
        bands, transform, eff_x, eff_y = _read_chip(chip)
        gt_g = adj_to_apls_graph(_geojson_adj(chip, transform), eff_x, eff_y)
        if gt_g.number_of_nodes() < 2:
            continue  # no GT roads on this chip -> undefined, skip for both
        scorable.append(chip)
        # apls(GT,GT) < 1.0 when the chip's vector GT is genuinely fragmented
        # (real dead-ends / roads clipped at the 400px boundary; measured gaps are
        # 19-66px, not rounding). Unreachable-in-GT pairs score 0 for BOTH models,
        # so this ceiling divides that shared GT-fragmentation penalty out of the
        # absolute number (the raw paired delta already cancels it; normalized makes
        # the absolute score interpretable as "fraction of achievable routing").
        ceilings[chip] = float(chip_apls(gt_g, gt_g, n_samples))
        if model is not None:
            s = chip_apls(v32_chip_graph(model, bands, thr, eff_x, eff_y, device), gt_g, n_samples)
            if not math.isnan(s):
                v32_scores[chip] = float(s)
        if a18_pred_dir is not None:
            pp = a18_pred_path(a18_pred_dir, chip)
            if pp.exists():   # file MISSING -> coverage gap; file present but empty -> real 0.0
                adj = pickle.loads(pp.read_bytes())
                s = chip_apls(adj_to_apls_graph(adj, eff_x, eff_y), gt_g, n_samples)
                if not math.isnan(s):
                    a18_scores[chip] = float(s)

    cov = coverage(scorable, v32_scores, a18_scores,
                   want_v32=model is not None, want_a18=a18_pred_dir is not None)
    out: dict = {
        "n_requested": len(chips), "coverage": cov, "n_samples": n_samples,
        "gt_ceiling_mean": float(np.mean([ceilings[c] for c in scorable])) if scorable else None,
        "v32": {"checkpoint": v32_ckpt.name if v32_ckpt else None, "n_scored": len(v32_scores),
                "apls_mean": float(np.mean(list(v32_scores.values()))) if v32_scores else None,
                "apls_norm_mean": _norm_mean(v32_scores, ceilings), "threshold": thr},
        "a18": {"pred_dir": str(a18_pred_dir) if a18_pred_dir else None, "n_scored": len(a18_scores),
                "apls_mean": float(np.mean(list(a18_scores.values()))) if a18_scores else None,
                "apls_norm_mean": _norm_mean(a18_scores, ceilings)},
    }
    if model is not None and a18_pred_dir is not None:
        if strict and not cov["complete"]:
            out["compare"] = {"verdict": "GATE FAIL: incomplete coverage",
                              "missing_v32": cov["missing_v32"], "missing_a18": cov["missing_a18"]}
            print(f"  GATE FAIL: {len(cov['missing_v32'])} v32 + "
                  f"{len(cov['missing_a18'])} a18 scorable chips missing", flush=True)
        else:
            # Raw paired delta stays the gate verdict (shared GT fragmentation
            # cancels in the pairing). Normalized paired delta is reported alongside
            # so the promotion signal can also be read on an interpretable [0,1] scale.
            out["compare"] = _paired_comparison(
                scorable, v32_scores, a18_scores, cov["complete"])
            out["compare_normalized"] = _paired_comparison(
                scorable, _normalized(v32_scores, ceilings),
                _normalized(a18_scores, ceilings), cov["complete"])
    return out


def _self_check(n_chips: int, device: str) -> None:
    """Validate the scoring machinery without a trained A18:

    1. GT-vs-itself must recover ~all *routable* GT pairs -- i.e. perfect ~= the GT
       reachable-pair fraction, NOT ~1.0: real SN5 chips have genuine dead-ends and
       roads clipped at the 400px border, so the achievable ceiling is <1.0 (~0.64
       mean here). Asserting perfect>=0.95 was wrong -- it assumed a connected GT.
    2. An empty prediction must score 0.0 (worst) on a chip that has GT roads.
    3. coverage() flags a missing A18 score as incomplete.
    4. If models/road_pan.pt exists, print the real v3.2-vs-GT chip score (no assert).
    """
    import networkx as nx

    from src.pipeline.p3_analysis.apls import _reachable_pair_fraction

    chips = _heldout_chips(n_chips, seed=7)
    print(f"[self-check] chips: {chips}", flush=True)
    checked = 0
    for chip in chips:
        _, transform, eff_x, eff_y = _read_chip(chip)
        gt_g = adj_to_apls_graph(_geojson_adj(chip, transform), eff_x, eff_y)
        if gt_g.number_of_nodes() < 2:
            print(f"  {chip}: no GT roads, skip", flush=True)
            continue
        perfect = chip_apls(gt_g, gt_g)
        empty = chip_apls(nx.Graph(), gt_g)
        ceiling = _reachable_pair_fraction(gt_g)   # achievable max given GT fragmentation
        print(f"  {chip}: nodes={gt_g.number_of_nodes():4d} eff=({eff_x:.3f},{eff_y:.3f}) "
              f"perfect={perfect:.4f} ceiling={ceiling:.4f} empty={empty:.4f}", flush=True)
        # frame is self-consistent iff GT-vs-self recovers a large share of the
        # routable pairs. The bar is half the reachable ceiling (<1.0 for genuinely
        # fragmented chips), loose enough to absorb apls's densification snap-collision
        # loss on dense chips but tight enough to catch gross scoring breakage (~0).
        assert perfect >= 0.5 * ceiling, (
            f"GT-vs-itself {perfect:.4f} below half the routable ceiling {ceiling:.4f} on {chip}")
        assert empty == 0.0, f"empty-vs-GT should be 0.0, got {empty} on {chip}"
        checked += 1
    assert checked > 0, "no heldout chip had scorable GT roads -- check data paths"
    cov = coverage(["a", "b"], {"a": 0.5, "b": 0.4}, {"a": 0.6}, want_v32=True, want_a18=True)
    assert not cov["complete"] and cov["missing_a18"] == ["b"], "coverage() must flag missing a18"

    v32 = ROOT / "models/road_pan.pt"
    if v32.exists():
        rep = compare_on_chips(v32, None, n_chips=n_chips, device=device)
        print(f"[self-check] v3.2 chip-level APLS mean={rep['v32']['apls_mean']} "
              f"(n={rep['v32']['n_scored']}) thr={rep['v32']['threshold']}", flush=True)
    print(f"[self-check] PASS ({checked} chips validated)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="A18 common-unit chip-level APLS (A18 vs v3.2 vs vector GT).")
    p.add_argument("--v32", default=None, help="v3.2 checkpoint (e.g. models/road_pan.pt)")
    p.add_argument("--a18-pred-dir", default=None,
                   help="inferencer output dir; reads <dir>/graph/mumbai_{chip}.p adj-dicts")
    p.add_argument("--n-chips", type=int, default=None, help="random heldout chips to score (None=all)")
    p.add_argument("--n-samples", type=int, default=GATE_N_SAMPLES, help="apls sampled node pairs per chip")
    p.add_argument("--threshold", type=float, default=None, help="shared v3.2 threshold override")
    p.add_argument("--strict", action="store_true",
                   help="gate mode: fail the compare unless every scorable chip has both scores")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=".tmp/a18_vs_v32_chip_apls.json")
    p.add_argument("--self-check", action="store_true",
                   help="validate scoring machinery (GT-vs-self ~1.0, empty ~0.0, coverage) without a trained A18")
    args = p.parse_args()

    if args.self_check:
        _self_check(args.n_chips or 3, args.device)
        return

    if not args.v32 and not args.a18_pred_dir:
        raise SystemExit("provide --v32 and/or --a18-pred-dir (or --self-check)")
    if args.strict and not (args.v32 and args.a18_pred_dir):
        raise SystemExit("--strict is a promotion gate: it requires BOTH --v32 and --a18-pred-dir")
    rep = compare_on_chips(
        Path(args.v32) if args.v32 else None,
        Path(args.a18_pred_dir) if args.a18_pred_dir else None,
        n_chips=args.n_chips, threshold=args.threshold, device=args.device,
        n_samples=args.n_samples, strict=args.strict)
    _write_report(args.out, rep)
    print(f"-> {args.out}", flush=True)
    # Fail-loud for automation: a strict gate encodes promotion success in its exit
    # code (write JSON first). Exploratory (non-strict) runs always exit 0.
    raise SystemExit(strict_exit_code(rep) if args.strict else 0)


if __name__ == "__main__":
    main()
