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

Gate mode (``--strict``) compares the candidate with both v3.2 and the LoRA r=4
incumbent.  It fails unless every scorable GT chip has all three scores, the paired
candidate-minus-incumbent CI is positive, raw mean APLS improves by at least 0.02,
and normalized APLS reaches 0.25.  A missing prediction on a hard chip must count
against the candidate, not silently drop out of the paired bootstrap.

    python -m src.pipeline.p1_segment.chip_apls_eval --self-check --n-chips 3
    python -m src.pipeline.p1_segment.chip_apls_eval \
        --candidates-json .tmp/a46_candidates.json \
        --chip-manifest data/sample/a46_selection_chips.json \
        --chip-key validation --out .tmp/a46_selection.json
    python -m src.pipeline.p1_segment.chip_apls_eval --strict \
        --v32 models/road_pan.pt --incumbent-pred-dir .tmp/a18_lora_reference \
        --a18-pred-dir .tmp/a46_candidate --n-samples 600 \
        --out .tmp/a46_registered_comparison.json

ASCII-only prints (Windows redirected stdout is cp1252 -- bit A38/A39). Mandatory
__main__ guard (Windows subprocess/DataLoader safety).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pickle
import random
import time
from pathlib import Path

import numpy as np

from src.pipeline.p1_segment.apls_eval import _DEG_X, _DEG_Y

ROOT = Path(__file__).resolve().parents[3]
SRC_RGB = ROOT / "data/raw/spacenet/SN5_roads_train_AOI_8_Mumbai/PS-RGB"
SRC_GEO = ROOT / "data/raw/spacenet/SN5_roads_train_AOI_8_Mumbai/geojson_roads_speed"
HELDOUT = ROOT / "data/sample/spacenet_mumbai_heldout_chips.json"
A46_SELECTION = ROOT / "data/sample/a46_selection_chips.json"

IMAGE_SIZE = 400   # 400px common frame -- must match the A18 converter/dataset
NATIVE_PX = 1300   # native SN5 PS-RGB chip size
V32_SCALE = 0.6    # native 1300 -> 780px (~0.5m, v3.2's deployed GSD)
GATE_N_SAMPLES = 600   # apls sample count for a gate-grade score
# Pre-registered A46 candidate-vs-incumbent metric floors.  The incumbent LoRA
# r=4 reaches 0.1808 of its GT-self ceiling; a deployable research candidate must
# make a visible step beyond that result, not merely win a noisy relative test.
MIN_MATERIAL_APLS_DELTA = 0.02
MIN_NORMALIZED_APLS = 0.25


def _native_coord_to_frame(value: float) -> int:
    """Round a native-chip coordinate into the closed 400px raster frame."""
    return min(IMAGE_SIZE - 1, max(0, int(round(value * IMAGE_SIZE / NATIVE_PX))))


def _write_report(path: Path | str, report: dict) -> None:
    """Write JSON after ensuring a caller-supplied output directory exists."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False))


def _load_chip_ids(manifest: Path | str, key: str, n_chips: int | None,
                   seed: int) -> list[str]:
    """Read an explicit frozen chip list without silently dropping inputs.

    Selection manifests use ``validation`` while the historical comparison file
    uses ``test_chips``.  Sampling is deterministic and the returned IDs are
    naturally sorted so report ordering is stable across platforms.
    """
    data = json.loads(Path(manifest).read_text())
    if key not in data or not isinstance(data[key], list):
        raise ValueError(f"chip manifest must contain a list at key {key!r}")
    chips = [str(chip).removeprefix("mumbai_") for chip in data[key]]
    if not chips:
        raise ValueError(f"chip manifest list at key {key!r} must not be empty")
    if len(chips) != len(set(chips)):
        raise ValueError(f"chip manifest list at key {key!r} contains duplicate IDs")
    if n_chips and n_chips < len(chips):
        chips = random.Random(seed).sample(chips, n_chips)
    return sorted(chips, key=lambda x: int(x.replace("chip", "").replace("mumbai_", "")))


def _same_manifest(path: Path | str, registered: Path) -> bool:
    from src.pipeline.p1_segment.provenance import sha256_file
    candidate = Path(path)
    return candidate.is_file() and sha256_file(candidate) == sha256_file(registered)


def _require_registered_selection(manifest: Path | str, key: str,
                                  n_chips: int | None, n_samples: int) -> None:
    if key != "validation" or not _same_manifest(manifest, A46_SELECTION):
        raise ValueError("selection mode requires the registered 102-chip A46 manifest/key")
    if n_chips is not None:
        raise ValueError("selection mode requires all registered 102 chips")
    if n_samples != GATE_N_SAMPLES:
        raise ValueError(f"selection mode requires {GATE_N_SAMPLES} APLS samples per chip")


def _require_registered_comparison(manifest: Path | str, key: str,
                                   n_chips: int | None, n_samples: int,
                                   threshold: float | None) -> None:
    if key != "test_chips" or not _same_manifest(manifest, HELDOUT):
        raise ValueError("strict mode requires the registered 127-chip comparison manifest/key")
    if n_chips is not None:
        raise ValueError("strict mode requires all registered comparison chips")
    if n_samples != GATE_N_SAMPLES:
        raise ValueError(f"strict mode requires {GATE_N_SAMPLES} APLS samples per chip")
    if threshold is not None:
        raise ValueError("strict mode rejects a v3.2 threshold override")


class _AdjacencyUnpickler(pickle.Unpickler):
    """Load only pickle primitives; prediction files are never a public input."""

    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(f"pickle global {module}.{name} is not allowed")


def _load_adjacency(path: Path) -> dict:
    stream = io.BytesIO(path.read_bytes())
    adjacency = _AdjacencyUnpickler(stream).load()
    if stream.read(1):
        raise ValueError(f"prediction {path} has trailing pickle data")
    if not isinstance(adjacency, dict):
        raise ValueError(f"prediction {path} must contain an adjacency dictionary")
    if len(adjacency) > IMAGE_SIZE * IMAGE_SIZE:
        raise ValueError(f"prediction {path} contains too many nodes")

    def validate_node(node, role: str) -> None:
        if (not isinstance(node, tuple) or len(node) != 2 or
                any(isinstance(value, bool) or not isinstance(value, int)
                    for value in node)):
            raise ValueError(f"prediction {path} has invalid {role} {node!r}")
        if any(value < 0 or value >= IMAGE_SIZE for value in node):
            raise ValueError(f"prediction {path} has out-of-frame {role} {node!r}")

    for node, neighbors in adjacency.items():
        validate_node(node, "node")
        if not isinstance(neighbors, (list, tuple, set)):
            raise ValueError(f"prediction {path} neighbors for {node!r} must be a sequence")
        for neighbor in neighbors:
            validate_node(neighbor, "neighbor")
    return adjacency


def _preflight_chip_inputs(chips: list[str], rgb_dir: Path = SRC_RGB,
                           geo_dir: Path = SRC_GEO) -> None:
    missing_rgb = [
        chip for chip in chips
        if not (rgb_dir / f"SN5_roads_train_AOI_8_Mumbai_PS-RGB_{chip}.tif").is_file()
    ]
    missing_gt = [
        chip for chip in chips
        if not (geo_dir / (
            f"SN5_roads_train_AOI_8_Mumbai_geojson_roads_speed_{chip}.geojson"
        )).is_file()
    ]
    if missing_rgb or missing_gt:
        raise FileNotFoundError(
            f"chip protocol inputs incomplete; missing RGB={missing_rgb}; "
            f"missing GT={missing_gt}")


def _graph_diagnostics(graph) -> dict:
    """Exact fragmentation and size diagnostics for one undirected road graph."""
    import networkx as nx

    n = graph.number_of_nodes()
    components = list(nx.connected_components(graph)) if n else []
    component_sizes = [len(nodes) for nodes in components]
    reachable_pairs = sum(size * (size - 1) for size in component_sizes)
    total_pairs = n * (n - 1)
    return {
        "nodes": n,
        "edges": graph.number_of_edges(),
        "components": len(components),
        "isolates": len(list(nx.isolates(graph))) if n else 0,
        "largest_component_node_fraction": max(component_sizes, default=0) / n if n else 0.0,
        "reachable_pair_fraction": reachable_pairs / total_pairs if total_pairs else 0.0,
        "total_edge_length_m": float(sum(
            float(data.get("length_m", 0.0)) for _, _, data in graph.edges(data=True))),
    }


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


def _graph_artifact_digest(pred_dir: Path) -> tuple[str, int]:
    """Digest the exact named prediction artifacts in a directory."""
    from src.pipeline.p1_segment.provenance import sha256_file

    paths = sorted((pred_dir / "graph").glob("*.p"), key=lambda path: path.name)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest().upper(), len(paths)


def _run_manifest_errors(manifest: object) -> list[str]:
    if not isinstance(manifest, dict):
        return ["run manifest must be an object"]
    errors: list[str] = []
    expected_top = {
        "licenses": dict, "code": dict, "data": dict, "model": dict,
        "environment": dict, "inference": dict, "promotion_contract": dict,
    }
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    for key in ("run_id", "purpose"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            errors.append(f"{key} must be a non-empty string")
    if not isinstance(manifest.get("redistribution_allowed"), bool):
        errors.append("redistribution_allowed must be boolean")
    for key, expected in expected_top.items():
        if not isinstance(manifest.get(key), expected):
            errors.append(f"{key} must be an object")
    if errors:
        return errors

    required_strings = {
        "code": ("trace_git_sha", "samroadplus_upstream_git_sha",
                 "samroadplus_local_patch_sha256"),
        "model": ("checkpoint_sha256", "config_sha256",
                  "effective_config_sha256", "sam_checkpoint_sha256"),
        "data": ("selection_manifest_sha256", "selection_key", "coordinate_contract"),
        "inference": ("graphs_sha256",),
        "environment": (
            "python", "platform", "torch", "lightning", "gpu",
            "historical_checkpoint_reproducibility"),
    }
    for section, keys in required_strings.items():
        for key in keys:
            if not isinstance(manifest[section].get(key), str) or not manifest[section][key]:
                errors.append(f"{section}.{key} must be a non-empty string")
    chip_ids = manifest["data"].get("chip_ids")
    if (not isinstance(chip_ids, list) or not chip_ids or
            any(not isinstance(chip, str) or not chip for chip in chip_ids) or
            len(chip_ids) != len(set(chip_ids))):
        errors.append("data.chip_ids must be a non-empty unique string list")
    for section, key in (
            ("data", "chip_count"), ("model", "epoch"),
            ("inference", "graph_count")):
        value = manifest[section].get(key)
        minimum = 0 if (section, key) == ("model", "epoch") else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            errors.append(f"{section}.{key} must be an integer >= {minimum}")
    command = manifest["inference"].get("command")
    if (not isinstance(command, list) or not command or
            any(not isinstance(part, str) for part in command)):
        errors.append("inference.command must be a non-empty string list")
    if not isinstance(manifest["inference"].get("thresholds"), dict):
        errors.append("inference.thresholds must be an object")
    wall_seconds = manifest["inference"].get("wall_seconds_inference")
    if (isinstance(wall_seconds, bool) or not isinstance(wall_seconds, (int, float))
            or not math.isfinite(wall_seconds) or wall_seconds <= 0):
        errors.append("inference.wall_seconds_inference must be a finite number > 0")
    environment = manifest["environment"]
    if not isinstance(environment.get("deterministic_algorithms"), bool):
        errors.append("environment.deterministic_algorithms must be boolean")
    if not isinstance(environment.get("cuda_runtime"), (str, type(None))):
        errors.append("environment.cuda_runtime must be a string or null")
    if (isinstance(environment.get("cudnn"), bool) or
            not isinstance(environment.get("cudnn"), (int, str, type(None)))):
        errors.append("environment.cudnn must be an integer, string or null")
    promotion = manifest["promotion_contract"]
    if not isinstance(promotion.get("selection_split_only"), bool):
        errors.append("promotion_contract.selection_split_only must be boolean")
    for key in ("min_candidate_minus_incumbent_raw_apls",
                "min_candidate_normalized_apls"):
        value = promotion.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0):
            errors.append(f"promotion_contract.{key} must be a finite number >= 0")
    allowed = promotion.get("comparison_split_runs_allowed_after_preregistration")
    if isinstance(allowed, bool) or not isinstance(allowed, int) or allowed < 0:
        errors.append(
            "promotion_contract.comparison_split_runs_allowed_after_preregistration "
            "must be an integer >= 0")
    return errors


def _protocol_manifest_errors(manifest: object, pred_dir: Path) -> list[str]:
    """Match a run to one frozen A46 split and its exact graph filenames."""
    if not isinstance(manifest, dict) or not isinstance(manifest.get("data"), dict):
        return ["run manifest has no valid data protocol"]
    from src.pipeline.p1_segment.provenance import sha256_file

    data = manifest["data"]
    key = data.get("selection_key")
    registered = A46_SELECTION if key == "validation" else HELDOUT if key == "test_chips" else None
    if registered is None:
        return ["data.selection_key is not a registered A46 protocol"]
    registered_data = json.loads(registered.read_text())
    expected_ids = registered_data[key]
    errors: list[str] = []
    if str(data.get("selection_manifest_sha256", "")).upper() != sha256_file(registered).upper():
        errors.append("registered split digest does not match run manifest")
    if data.get("chip_ids") != expected_ids:
        errors.append("chip IDs/order do not match the registered split")
    if data.get("chip_count") != len(expected_ids):
        errors.append("chip count does not match the registered split")
    actual_names = {path.name for path in (pred_dir / "graph").glob("*.p")}
    expected_names = {f"mumbai_{chip}.p" for chip in expected_ids}
    if actual_names != expected_names:
        errors.append("graph filenames do not match the registered split")
    selection_only = (manifest.get("promotion_contract") or {}).get("selection_split_only")
    if selection_only != (key == "validation"):
        errors.append("selection_split_only does not match the registered split")
    return errors


def _prediction_provenance(pred_dir: Path | None) -> dict | None:
    """Verify hashes/metadata emitted by the local-only A46 inference runner."""
    if pred_dir is None:
        return None
    from src.pipeline.p1_segment.provenance import sha256_file

    config = pred_dir / "config.yaml"
    manifest = pred_dir / "run_manifest.json"
    errors: list[str] = []
    manifest_data = None
    if manifest.is_file():
        try:
            manifest_data = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid run manifest: {exc}")
    else:
        errors.append("run_manifest.json is missing")
    errors.extend(_run_manifest_errors(manifest_data))
    errors.extend(_protocol_manifest_errors(manifest_data, pred_dir))

    graph_sha, graph_count = _graph_artifact_digest(pred_dir)
    effective_config_sha = sha256_file(config) if config.is_file() else None
    if effective_config_sha is None:
        errors.append("config.yaml is missing")
    if isinstance(manifest_data, dict):
        inference = manifest_data.get("inference") or {}
        model = manifest_data.get("model") or {}
        if str(inference.get("graphs_sha256", "")).upper() != graph_sha:
            errors.append("graph digest does not match run manifest")
        if inference.get("graph_count") != graph_count:
            errors.append("graph count does not match run manifest")
        if (effective_config_sha and
                str(model.get("effective_config_sha256", "")).upper()
                != effective_config_sha.upper()):
            errors.append("effective config digest does not match run manifest")
        source_config = Path(model.get("config", ""))
        if source_config.is_file():
            source_sha = sha256_file(source_config)
            if str(model.get("config_sha256", "")).upper() != source_sha.upper():
                errors.append("source config digest does not match run manifest")
    out = {
        "directory": str(pred_dir),
        "valid": not errors,
        "errors": errors,
        "effective_config_sha256": effective_config_sha,
        "graphs_sha256": graph_sha,
        "graph_count": graph_count,
        "run_manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
        "run_manifest": manifest_data,
    }
    return out


def _chip_model_result(graph, score: float, ceiling: float,
                       runtime_seconds: float, gt_length_m: float) -> dict:
    diagnostics = _graph_diagnostics(graph)
    diagnostics["edge_length_gt_fraction"] = (
        diagnostics["total_edge_length_m"] / gt_length_m if gt_length_m > 0 else None)
    return {
        "apls": float(score),
        "apls_normalized": min(1.0, float(score) / ceiling) if ceiling > 1e-6 else None,
        "runtime_seconds": float(runtime_seconds),
        "diagnostics": diagnostics,
    }


def _model_summary(scores: dict[str, float], ceilings: dict[str, float],
                   per_chip: dict[str, dict], model_key: str) -> dict:
    rows = [per_chip[chip][model_key] for chip in scores if model_key in per_chip[chip]]
    diag_keys = (
        "components", "isolates", "largest_component_node_fraction",
        "reachable_pair_fraction", "total_edge_length_m", "edge_length_gt_fraction")
    return {
        "n_scored": len(scores),
        "apls_mean": float(np.mean(list(scores.values()))) if scores else None,
        "apls_norm_mean": _norm_mean(scores, ceilings),
        "runtime_seconds": float(sum(row["runtime_seconds"] for row in rows)),
        "diagnostics_mean": {
            key: float(np.mean([
                row["diagnostics"][key] for row in rows
                if row["diagnostics"].get(key) is not None
            ])) if any(row["diagnostics"].get(key) is not None for row in rows) else None
            for key in diag_keys
        },
    }


def _candidate_sort_key(summary: dict) -> tuple:
    """APLS-first deterministic ranking with connectivity-only tiebreakers."""
    diagnostics = summary.get("diagnostics_mean") or {}
    coverage_complete = bool(summary.get("coverage_complete", True))
    provenance_valid = bool(summary.get("provenance_valid", True))
    return (
        0 if coverage_complete and provenance_valid else 1,
        len(summary.get("missing_chips") or []),
        -float(summary.get("apls_mean") or 0.0),
        -float(summary.get("apls_norm_mean") or 0.0),
        -float(diagnostics.get("reachable_pair_fraction") or 0.0),
        -float(diagnostics.get("largest_component_node_fraction") or 0.0),
        float(diagnostics.get("components") or math.inf),
        str(summary.get("label") or ""),
    )


def _select_complete_candidate(ranked: list[dict]) -> str | None:
    for summary in ranked:
        if summary.get("coverage_complete") and summary.get("provenance_valid", True):
            return str(summary["label"])
    return None


def _promotion_gate(rep: dict,
                    min_delta: float = MIN_MATERIAL_APLS_DELTA,
                    min_normalized: float = MIN_NORMALIZED_APLS) -> dict:
    """Evaluate the pre-registered A46 metric gate against the LoRA incumbent.

    This is deliberately stronger than "candidate beats v3.2": it requires a
    statistically supported paired win over the incumbent, a material raw APLS
    step, and a useful absolute fraction of the GT-self ceiling.  Passing this
    metric gate does not waive the separate license, reproducibility, sensor and
    deployment checks recorded in ``docs/Evaluation.md``.
    """
    comparison = rep.get("compare_incumbent") or {}
    normalized = (rep.get("a18") or {}).get("apls_norm_mean")
    delta = comparison.get("delta_a18_minus_incumbent")
    provenance = rep.get("prediction_provenance") or {}
    checks = {
        "complete_coverage": bool((rep.get("coverage") or {}).get("complete")),
        "incumbent_provenance": bool((provenance.get("incumbent") or {}).get("valid")),
        "candidate_provenance": bool((provenance.get("candidate") or {}).get("valid")),
        "paired_ci_win": (comparison.get("verdict") == "b wins" and
                          bool(comparison.get("excludes_zero"))),
        "material_raw_delta": delta is not None and float(delta) >= min_delta,
        "absolute_normalized_apls": (normalized is not None and
                                     float(normalized) >= min_normalized),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "verdict": "metric gate passed" if passed else "metric gate failed",
        "thresholds": {
            "min_candidate_minus_incumbent_apls": min_delta,
            "min_candidate_normalized_apls": min_normalized,
        },
        "checks": checks,
    }


def strict_exit_code(rep: dict) -> int:
    """Process status for a STRICT promotion gate (call only in ``--strict`` mode).

    A gate's exit code must encode *promotion success*, not merely evaluation
    completion:
      * 0 -- the full pre-registered A46 metric gate passes
      * 2 -- incomplete coverage (fail-loud; a missing artifact must not pass)
      * 3 -- no promotion: an effect floor, absolute floor, CI, or comparison fails
    """
    cmp = rep.get("compare")
    if cmp and str(cmp.get("verdict", "")).startswith("GATE FAIL"):
        return 2
    gate = rep.get("promotion_gate") or {}
    return 0 if gate.get("passed") is True else 3


def _heldout_chips(n_chips: int | None, seed: int) -> list[str]:
    return _load_chip_ids(HELDOUT, "test_chips", n_chips, seed)


def coverage(scorable: list[str], v32_scores: dict, a18_scores: dict,
             want_v32: bool, want_a18: bool,
             incumbent_scores: dict | None = None,
             want_incumbent: bool = False) -> dict:
    """Which scorable chips are missing a requested model's score.

    A chip is *scorable* if it has >=2 GT nodes. Gate coverage is complete only when
    every scorable chip has every requested score -- otherwise the paired bootstrap
    would run on a biased subset (a hard chip a model failed on must not vanish).
    """
    missing_v32 = [c for c in scorable if want_v32 and c not in v32_scores]
    missing_a18 = [c for c in scorable if want_a18 and c not in a18_scores]
    incumbent_scores = incumbent_scores or {}
    missing_incumbent = [
        c for c in scorable if want_incumbent and c not in incumbent_scores]
    return {
        "n_scorable": len(scorable),
        "missing_v32": missing_v32,
        "missing_a18": missing_a18,
        "missing_incumbent": missing_incumbent,
        "complete": not missing_v32 and not missing_a18 and not missing_incumbent,
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
                     n_samples: int = GATE_N_SAMPLES, strict: bool = False,
                     chip_manifest: Path | None = None,
                     chip_key: str = "test_chips",
                     incumbent_pred_dir: Path | None = None) -> dict:
    """Score v3.2 and/or A18 vs vector GT on common heldout chips.

    Returns per-chip scores, a coverage report, and -- when both models are present
    and (in ``strict`` mode) coverage is complete -- a paired-bootstrap CI on the
    APLS delta (A18 - v3.2). Promotion is only justified when coverage is complete
    AND the CI excludes zero.
    """
    manifest = chip_manifest or HELDOUT
    chips = _load_chip_ids(manifest, chip_key, n_chips, seed)
    _preflight_chip_inputs(chips)

    model = thr = None
    if v32_ckpt is not None:
        from src.pipeline.p1_segment.model import load_checkpoint
        model, meta = load_checkpoint(v32_ckpt, map_location=device)
        model.to(device).eval()
        thr = threshold if threshold is not None else float(meta.get("threshold", 0.44))

    scorable: list[str] = []
    v32_scores: dict[str, float] = {}
    a18_scores: dict[str, float] = {}
    incumbent_scores: dict[str, float] = {}
    ceilings: dict[str, float] = {}   # per-chip apls(GT,GT): the achievable max on this chip
    per_chip: dict[str, dict] = {}
    for chip in chips:
        gt_start = time.perf_counter()
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
        gt_diagnostics = _graph_diagnostics(gt_g)
        per_chip[chip] = {
            "gt": {
                "apls_self": ceilings[chip],
                "runtime_seconds": float(time.perf_counter() - gt_start),
                "diagnostics": gt_diagnostics,
            }
        }
        gt_length = gt_diagnostics["total_edge_length_m"]
        if model is not None:
            started = time.perf_counter()
            pred_g = v32_chip_graph(model, bands, thr, eff_x, eff_y, device)
            s = chip_apls(pred_g, gt_g, n_samples)
            if not math.isnan(s):
                v32_scores[chip] = float(s)
                per_chip[chip]["v32"] = _chip_model_result(
                    pred_g, s, ceilings[chip], time.perf_counter() - started, gt_length)
        if incumbent_pred_dir is not None:
            started = time.perf_counter()
            pp = a18_pred_path(incumbent_pred_dir, chip)
            if pp.exists():
                incumbent_g = adj_to_apls_graph(_load_adjacency(pp), eff_x, eff_y)
                s = chip_apls(incumbent_g, gt_g, n_samples)
                if not math.isnan(s):
                    incumbent_scores[chip] = float(s)
                    per_chip[chip]["incumbent"] = _chip_model_result(
                        incumbent_g, s, ceilings[chip], time.perf_counter() - started,
                        gt_length)
        if a18_pred_dir is not None:
            started = time.perf_counter()
            pp = a18_pred_path(a18_pred_dir, chip)
            if pp.exists():   # file MISSING -> coverage gap; file present but empty -> real 0.0
                adj = _load_adjacency(pp)
                a18_g = adj_to_apls_graph(adj, eff_x, eff_y)
                s = chip_apls(a18_g, gt_g, n_samples)
                if not math.isnan(s):
                    a18_scores[chip] = float(s)
                    per_chip[chip]["a18"] = _chip_model_result(
                        a18_g, s, ceilings[chip], time.perf_counter() - started, gt_length)

    cov = coverage(scorable, v32_scores, a18_scores,
                   want_v32=model is not None, want_a18=a18_pred_dir is not None,
                   incumbent_scores=incumbent_scores,
                   want_incumbent=incumbent_pred_dir is not None)
    from src.pipeline.p1_segment.provenance import sha256_file

    v32_summary = _model_summary(v32_scores, ceilings, per_chip, "v32")
    v32_summary.update({
        "checkpoint": v32_ckpt.name if v32_ckpt else None,
        "checkpoint_sha256": sha256_file(v32_ckpt) if v32_ckpt else None,
        "threshold": thr,
    })
    incumbent_summary = _model_summary(
        incumbent_scores, ceilings, per_chip, "incumbent")
    incumbent_summary.update({
        "pred_dir": str(incumbent_pred_dir) if incumbent_pred_dir else None,
    })
    a18_summary = _model_summary(a18_scores, ceilings, per_chip, "a18")
    a18_summary.update({"pred_dir": str(a18_pred_dir) if a18_pred_dir else None})
    out: dict = {
        "protocol": {
            "name": "a46-common-unit-chip-apls-v1",
            "chip_manifest": str(manifest),
            "chip_manifest_sha256": sha256_file(manifest),
            "chip_key": chip_key,
            "selection_seed": seed,
            "apls_samples_per_chip": n_samples,
            "coordinate_contract": "x=column,y=row",
        },
        "n_requested": len(chips), "requested_chips": chips,
        "scorable_chips": scorable, "coverage": cov, "n_samples": n_samples,
        "gt_ceiling_mean": float(np.mean([ceilings[c] for c in scorable])) if scorable else None,
        "v32": v32_summary,
        "incumbent": incumbent_summary,
        "a18": a18_summary,
        "prediction_provenance": {
            "incumbent": _prediction_provenance(incumbent_pred_dir),
            "candidate": _prediction_provenance(a18_pred_dir),
        },
        "per_chip": per_chip,
    }
    if model is not None and a18_pred_dir is not None:
        if strict and not cov["complete"]:
            out["compare"] = {"verdict": "GATE FAIL: incomplete coverage",
                              "missing_v32": cov["missing_v32"],
                              "missing_incumbent": cov["missing_incumbent"],
                              "missing_a18": cov["missing_a18"]}
            print(f"  GATE FAIL: {len(cov['missing_v32'])} v32 + "
                  f"{len(cov['missing_incumbent'])} incumbent + "
                  f"{len(cov['missing_a18'])} candidate scorable chips missing",
                  flush=True)
        else:
            # Raw paired delta stays the gate verdict (shared GT fragmentation
            # cancels in the pairing). Normalized paired delta is reported alongside
            # so the promotion signal can also be read on an interpretable [0,1] scale.
            out["compare"] = _paired_comparison(
                scorable, v32_scores, a18_scores, cov["complete"])
            out["compare_normalized"] = _paired_comparison(
                scorable, _normalized(v32_scores, ceilings),
                _normalized(a18_scores, ceilings), cov["complete"])
    if incumbent_pred_dir is not None and a18_pred_dir is not None:
        comparison = _paired_comparison(
            scorable, incumbent_scores, a18_scores, cov["complete"])
        comparison["delta_a18_minus_incumbent"] = comparison.pop(
            "delta_a18_minus_v32")
        out["compare_incumbent"] = comparison
        normalized_comparison = _paired_comparison(
            scorable, _normalized(incumbent_scores, ceilings),
            _normalized(a18_scores, ceilings), cov["complete"])
        normalized_comparison["delta_a18_minus_incumbent"] = normalized_comparison.pop(
            "delta_a18_minus_v32")
        out["compare_incumbent_normalized"] = normalized_comparison
    if strict:
        out["promotion_gate"] = _promotion_gate(out)
    return out


def select_prediction_dirs(candidates: dict[str, Path],
                           chip_manifest: Path,
                           chip_key: str = "validation",
                           n_samples: int = GATE_N_SAMPLES,
                           seed: int = 7) -> dict:
    """Rank existing prediction directories on one frozen selection split.

    GT conversion and GT-self ceilings are computed once per chip, while every
    checkpoint is scored in the same process.  This avoids 27 repeated reference
    passes and makes it difficult to accidentally mix selection and comparison
    manifests.
    """
    from src.pipeline.p1_segment.provenance import sha256_file

    chips = _load_chip_ids(chip_manifest, chip_key, None, seed)
    _preflight_chip_inputs(chips)
    labels = sorted(candidates)
    scores: dict[str, dict[str, float]] = {label: {} for label in labels}
    ceilings: dict[str, float] = {}
    scorable: list[str] = []
    per_chip: dict[str, dict] = {}

    for chip in chips:
        gt_started = time.perf_counter()
        _, transform, eff_x, eff_y = _read_chip(chip)
        gt_g = adj_to_apls_graph(_geojson_adj(chip, transform), eff_x, eff_y)
        if gt_g.number_of_nodes() < 2:
            continue
        scorable.append(chip)
        ceiling = float(chip_apls(gt_g, gt_g, n_samples))
        ceilings[chip] = ceiling
        gt_diagnostics = _graph_diagnostics(gt_g)
        per_chip[chip] = {
            "gt": {
                "apls_self": ceiling,
                "runtime_seconds": float(time.perf_counter() - gt_started),
                "diagnostics": gt_diagnostics,
            },
            "candidates": {},
        }
        gt_length = gt_diagnostics["total_edge_length_m"]
        for label in labels:
            started = time.perf_counter()
            pred_path = a18_pred_path(candidates[label], chip)
            if not pred_path.is_file():
                continue
            graph = adj_to_apls_graph(_load_adjacency(pred_path), eff_x, eff_y)
            score = chip_apls(graph, gt_g, n_samples)
            if math.isnan(score):
                continue
            scores[label][chip] = float(score)
            per_chip[chip]["candidates"][label] = _chip_model_result(
                graph, score, ceiling, time.perf_counter() - started, gt_length)

    summaries: list[dict] = []
    for label in labels:
        flat = {
            chip: {"candidate": row["candidates"][label]}
            for chip, row in per_chip.items()
            if label in row["candidates"]
        }
        summary = _model_summary(scores[label], ceilings, flat, "candidate")
        provenance = _prediction_provenance(candidates[label])
        summary.update({
            "label": label,
            "pred_dir": str(candidates[label]),
            "missing_chips": [chip for chip in scorable if chip not in scores[label]],
            "coverage_complete": all(chip in scores[label] for chip in scorable),
            "provenance_valid": bool(provenance and provenance["valid"]),
            "provenance": provenance,
        })
        summaries.append(summary)

    ranked = sorted(summaries, key=_candidate_sort_key)
    for index, summary in enumerate(ranked, start=1):
        summary["rank"] = index
    return {
        "protocol": {
            "name": "a46-checkpoint-selection-v1",
            "selection_only": True,
            "chip_manifest": str(chip_manifest),
            "chip_manifest_sha256": sha256_file(chip_manifest),
            "chip_key": chip_key,
            "selection_seed": seed,
            "apls_samples_per_chip": n_samples,
            "coordinate_contract": "x=column,y=row",
        },
        "n_requested": len(chips),
        "requested_chips": chips,
        "scorable_chips": scorable,
        "gt_ceiling_mean": (
            float(np.mean([ceilings[chip] for chip in scorable])) if scorable else None),
        "candidates": ranked,
        "selected_label": _select_complete_candidate(ranked),
        "per_chip": per_chip,
    }


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
    p.add_argument("--incumbent-pred-dir", default=None,
                   help="incumbent LoRA prediction dir required by the A46 strict gate")
    p.add_argument("--chip-manifest", default=str(HELDOUT),
                   help="JSON containing the frozen chip IDs")
    p.add_argument("--chip-key", default="test_chips",
                   help="list key inside --chip-manifest (selection uses validation)")
    p.add_argument("--candidates-json", default=None,
                   help="selection mode: JSON object mapping checkpoint labels to prediction dirs")
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

    if args.candidates_json:
        try:
            _require_registered_selection(
                args.chip_manifest, args.chip_key, args.n_chips, args.n_samples)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        candidate_data = json.loads(Path(args.candidates_json).read_text())
        if not isinstance(candidate_data, dict) or not candidate_data:
            raise SystemExit("--candidates-json must contain a non-empty JSON object")
        report = select_prediction_dirs(
            {str(label): Path(path) for label, path in candidate_data.items()},
            chip_manifest=Path(args.chip_manifest), chip_key=args.chip_key,
            n_samples=args.n_samples)
        _write_report(args.out, report)
        print(f"-> {args.out}", flush=True)
        return

    if not args.v32 and not args.a18_pred_dir:
        raise SystemExit("provide --v32 and/or --a18-pred-dir (or --self-check)")
    if args.strict and not (args.v32 and args.incumbent_pred_dir and args.a18_pred_dir):
        raise SystemExit(
            "--strict is the A46 metric gate: it requires --v32, "
            "--incumbent-pred-dir and --a18-pred-dir")
    if args.strict:
        try:
            _require_registered_comparison(
                args.chip_manifest, args.chip_key, args.n_chips,
                args.n_samples, args.threshold)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    rep = compare_on_chips(
        Path(args.v32) if args.v32 else None,
        Path(args.a18_pred_dir) if args.a18_pred_dir else None,
        n_chips=args.n_chips, threshold=args.threshold, device=args.device,
        n_samples=args.n_samples, strict=args.strict,
        chip_manifest=Path(args.chip_manifest), chip_key=args.chip_key,
        incumbent_pred_dir=(
            Path(args.incumbent_pred_dir) if args.incumbent_pred_dir else None))
    _write_report(args.out, rep)
    print(f"-> {args.out}", flush=True)
    # Fail-loud for automation: a strict gate encodes promotion success in its exit
    # code (write JSON first). Exploratory (non-strict) runs always exit 0.
    raise SystemExit(strict_exit_code(rep) if args.strict else 0)


if __name__ == "__main__":
    main()
