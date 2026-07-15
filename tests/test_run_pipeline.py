"""CPU checks for the local P1-P4 orchestration and CLI contracts.

P1 (the real model) is covered by test_model; here we inject a deterministic
synthetic-mask segmenter so the P2→P3→P4 wiring is tested end-to-end without a
190 MB checkpoint, and assert the dashboard (P4) contract holds.
"""

from __future__ import annotations

import builtins
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.pipeline.p1_segment.osm_mask import save_binary_png
from src.pipeline.run_pipeline import DASHBOARD_CRITICALITY_COLUMNS, run


def _fake_segment(image_path, checkpoint, aoi, interim_dir, tile_size, threshold, device, tta=False,
                  blend=True, postprocess=False, min_component_size=50, pp_open_radius=0,
                  pp_close_radius=0, fill_holes=0):
    """Stand-in for P1: write a synthetic road grid mask at the contract path."""
    mask = np.zeros((256, 256), np.uint8)
    for r in (64, 128, 192):
        mask[r - 1 : r + 2, 20:236] = 1
    for c in (64, 128, 192):
        mask[20:236, c - 1 : c + 2] = 1
    out = Path(interim_dir) / f"{aoi}_mask.png"
    save_binary_png(mask, out)
    return out, float(mask.mean())


def test_walking_skeleton_flows_p1_to_p4(tmp_path):
    res = run(
        "unused.jpg", "unused.pt", "a5test",
        interim_dir=tmp_path / "interim", processed_dir=tmp_path / "processed",
        curve_steps=5, segment_fn=_fake_segment,
    )

    # P2 produced a real graph
    assert res["nodes"] > 0 and res["edges"] > 0

    # P3 artifacts exist
    assert (tmp_path / "processed" / "a5test_criticality.csv").exists()
    assert (tmp_path / "processed" / "a5test_resilience.csv").exists()
    assert (tmp_path / "processed" / "a5test_graph.geojson").exists()

    # P4 seam: the criticality CSV matches the dashboard's column contract
    assert res["p4"]["columns_match"] is True
    assert res["p4"]["geojson_exists"] is True

    # analysis summary came through
    assert "targeted_end_ri" in res["analysis"]


def test_dashboard_contract_columns_are_stable():
    # guard against a silent drift of the P4 contract
    assert DASHBOARD_CRITICALITY_COLUMNS == ["node_id", "betweenness", "rank", "is_critical", "x", "y"]


def test_pipeline_signature_invalidates_p1_on_config_change(tmp_path):
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs.get("threshold"))
        return _fake_segment(*args, **kwargs)

    common = dict(
        image_path="unused.jpg", checkpoint="unused.pt", aoi="signature",
        interim_dir=tmp_path / "interim", processed_dir=tmp_path / "processed",
        curve_steps=2, segment_fn=counted,
    )
    run(**common, threshold=0.4)
    changed = run(**common, threshold=0.5)
    unchanged = run(**common, threshold=0.5)

    assert calls == [0.4, 0.5]
    assert changed["stages"][0]["ran"] is True
    assert unchanged["stages"][0]["ran"] is False
    assert unchanged["stages"][0]["reason"] == "signature-match"


def test_cli_config_preserves_unspecified_values(tmp_path, monkeypatch):
    from src.pipeline import run_pipeline

    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps({
        "aoi": "from-config",
        "image": "config.png",
        "checkpoint": "config.pt",
        "resolution_m": 2.5,
        "device": "cuda",
        "tta": True,
        "blend": False,
        "postprocess": True,
        "curve_steps": 7,
    }))
    captured = {}
    monkeypatch.setattr(run_pipeline, "run", lambda *_args, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["run_pipeline", "--config", str(config_path)])

    run_pipeline.main()

    config = captured["config"]
    assert (config.resolution_m, config.device, config.curve_steps) == (2.5, "cuda", 7)
    assert (config.tta, config.blend, config.postprocess) == (True, False, True)


def test_cli_explicit_flags_override_config(tmp_path, monkeypatch):
    from src.pipeline import run_pipeline

    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps({
        "aoi": "from-config",
        "image": "config.png",
        "checkpoint": "config.pt",
        "gap_max_m": 12.0,
    }))
    captured = {}
    monkeypatch.setattr(run_pipeline, "run", lambda *_args, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(sys, "argv", [
        "run_pipeline", "--config", str(config_path),
        "--aoi", "from-cli", "--image", "cli.png", "--checkpoint", "cli.pt",
        "--resolution-m", "3.0", "--tile-size", "256", "--threshold", "0.6",
        "--device", "cpu", "--curve-steps", "9", "--tta", "--no-blend",
        "--postprocess", "--min-component-size", "20", "--pp-open-radius", "1",
        "--pp-close-radius", "2", "--fill-holes", "30",
        "--min-corridor-support", "0.4",
    ])

    run_pipeline.main()

    config = captured["config"]
    assert (config.aoi, config.image, config.checkpoint) == ("from-cli", "cli.png", "cli.pt")
    assert (config.resolution_m, config.tile_size, config.threshold) == (3.0, 256, 0.6)
    assert (config.device, config.curve_steps, config.tta, config.blend) == ("cpu", 9, True, False)
    assert (config.postprocess, config.min_component_size) == (True, 20)
    assert (config.pp_open_radius, config.pp_close_radius, config.fill_holes) == (1, 2, 30)
    assert config.min_corridor_support == 0.4
    assert config.gap_max_m == 12.0


def test_cli_help_is_cp1252_safe_and_current():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"

    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline.run_pipeline", "--help"],
        cwd=Path(__file__).parents[1], env=env, capture_output=True,
    )

    assert result.returncode == 0, result.stderr.decode("cp1252")
    assert "Run road segmentation, graph construction, and resilience analysis" in (
        result.stdout.decode("cp1252")
    )


def test_run_hashes_unchanged_p1_inputs_once(tmp_path, monkeypatch):
    import networkx as nx

    from src.pipeline.p1_segment.provenance import build_provenance, write_provenance
    from src.pipeline import run_pipeline

    image = tmp_path / "image.bin"
    checkpoint = tmp_path / "checkpoint.pt"
    image.write_bytes(b"image" * 100)
    checkpoint.write_bytes(b"checkpoint" * 100)
    expected_checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    targets = {image.resolve(), checkpoint.resolve()}
    scans = Counter()
    path_open = Path.open
    builtin_open = builtins.open

    def count_path_open(path, mode="r", *args, **kwargs):
        if Path(path).resolve() in targets and mode == "rb":
            scans[Path(path).resolve()] += 1
        return path_open(path, mode, *args, **kwargs)

    def count_builtin_open(path, mode="r", *args, **kwargs):
        if Path(path).resolve() in targets and mode == "rb":
            scans[Path(path).resolve()] += 1
        return builtin_open(path, mode, *args, **kwargs)

    def fake_inference(image_path, checkpoint_path, aoi, interim_dir,
                       checkpoint_sha256=None, threshold=None, **_kwargs):
        mask = np.zeros((256, 256), np.uint8)
        mask[64:192, 126:130] = 1
        mask[126:130, 64:192] = 1
        out = Path(interim_dir) / f"{aoi}_mask.png"
        save_binary_png(mask, out)
        provenance_kwargs = ({"checkpoint_sha256": checkpoint_sha256}
                             if checkpoint_sha256 is not None else {})
        record = build_provenance(
            checkpoint_path, {"encoder": "test", "arch": "test"},
            threshold if threshold is not None else 0.5, **provenance_kwargs,
        )
        write_provenance(Path(interim_dir) / aoi / "provenance.json", record)
        return out, float(mask.mean())

    monkeypatch.setattr(Path, "open", count_path_open)
    monkeypatch.setattr(builtins, "open", count_builtin_open)
    monkeypatch.setattr("src.pipeline.p1_segment.predict.run_inference", fake_inference)
    monkeypatch.setattr(run_pipeline, "build_graph", lambda _cfg: None)
    monkeypatch.setattr(run_pipeline, "analyze", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run_pipeline, "_load_graph_for_summary", lambda _cfg: nx.path_graph(2))
    monkeypatch.setattr(run_pipeline, "verify_dashboard_ready", lambda _cfg: {
        "columns_match": True, "geojson_exists": True,
    })

    summary = run_pipeline.run(
        image, checkpoint, "hash-once",
        interim_dir=tmp_path / "interim", processed_dir=tmp_path / "processed",
        curve_steps=2,
    )

    assert scans == Counter({image.resolve(): 1, checkpoint.resolve(): 1})
    assert summary["provenance"]["model_sha256"] == expected_checkpoint_sha
    manifest = json.loads((tmp_path / "interim/hash-once_mask.png.stage.json").read_text())
    assert manifest["inputs"][1]["sha256"] == expected_checkpoint_sha


def test_file_identity_cache_invalidates_on_stat_change(tmp_path):
    from src.pipeline.run_pipeline import _file_identity

    path = tmp_path / "artifact.bin"
    path.write_bytes(b"first")
    hashes = {}
    first = _file_identity(path, hashes)

    path.write_bytes(b"second-value")
    second = _file_identity(path, hashes)
    assert second["sha256"] != first["sha256"]

    previous_mtime = path.stat().st_mtime_ns
    path.write_bytes(b"same-length!")
    os.utime(path, ns=(path.stat().st_atime_ns, previous_mtime + 1_000_000_000))
    third = _file_identity(path, hashes)
    assert third["sha256"] != second["sha256"]
