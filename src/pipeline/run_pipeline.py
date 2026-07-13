"""Run P1-P3 for one AOI and verify the dashboard artifact contract.

Stage signatures make reruns content-aware; the run summary records timings,
resolved configuration, and model provenance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable

from src.pipeline.config import PipelineConfig
from src.pipeline.p2_graph.build_graph import build_graph
from src.pipeline.p2_graph.config import GraphConfig
from src.pipeline.p3_analysis.analyze import analyze

log = logging.getLogger("trace.pipeline")

# Tracker §4 P4 contract: columns the dashboard reads from the criticality CSV.
DASHBOARD_CRITICALITY_COLUMNS = ["node_id", "betweenness", "rank", "is_critical", "x", "y"]

_STAGES = ("p1", "p2", "p3")


def _configure_logging() -> None:
    """Attach a console handler once (idempotent) so stage logs are visible."""
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)


def segment(image_path: str | Path, checkpoint: str | Path, aoi: str, interim_dir: str | Path,
            tile_size: int | None = 512, threshold: float | None = None,
            device: str = "cpu", tta: bool = False, blend: bool = True,
            postprocess: bool = False, min_component_size: int = 50,
            pp_open_radius: int = 0, pp_close_radius: int = 0,
            fill_holes: int = 0, checkpoint_sha256: str | None = None) -> tuple[Path, float]:
    """P1: predict a road mask from imagery → ``data/interim/{aoi}_mask.png``.

    Thin wrapper over the shared :func:`~src.pipeline.p1_segment.predict.run_inference`
    (A36) so this and ``predict.py`` can't drift.
    """
    from src.pipeline.p1_segment.predict import run_inference

    return run_inference(
        image_path, checkpoint, aoi, interim_dir,
        tile_size=tile_size, threshold=threshold, blend=blend, tta=tta, device=device,
        postprocess=postprocess, min_component_size=min_component_size,
        pp_open_radius=pp_open_radius, pp_close_radius=pp_close_radius, fill_holes=fill_holes,
        checkpoint_sha256=checkpoint_sha256,
    )


def verify_dashboard_ready(cfg: GraphConfig) -> dict[str, Any]:
    """P4 seam: confirm the P3 artifacts match what the dashboard consumes."""
    crit = cfg.processed_dir / f"{cfg.aoi}_criticality.csv"
    header = ""
    if crit.exists():
        text = crit.read_text(encoding="utf-8")
        header = text.splitlines()[0] if text.strip() else ""
    # Required columns must be present; extra columns (e.g. S8's is_articulation)
    # are fine — the dashboard reads by name, so the contract only *grows*.
    return {
        "criticality_csv": str(crit),
        "columns_match": set(DASHBOARD_CRITICALITY_COLUMNS).issubset(header.split(",")),
        "geojson": str(cfg.geojson_path),
        "geojson_exists": cfg.geojson_path.exists(),
    }


def _file_identity(
    path: Path,
    hashes: dict[tuple[str, int, int], str] | None = None,
) -> dict[str, Any]:
    """Content identity used by stage manifests (missing files are explicit)."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    hexdigest = hashes.get(key) if hashes is not None else None
    if hexdigest is None:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        hexdigest = digest.hexdigest()
        if hashes is not None:
            hashes[key] = hexdigest
    return {"path": str(path), "exists": True, "size": stat.st_size,
            "sha256": hexdigest}


def _stage_signature(
    name: str,
    inputs: list[Path],
    settings: dict[str, Any],
    hashes: dict[tuple[str, int, int], str] | None = None,
) -> dict:
    return {"version": 1, "stage": name,
            "inputs": [_file_identity(path, hashes) for path in inputs], "settings": settings}


def _stage_manifest_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.stage.json")


def _stage_is_current(output: Path, signature: dict) -> bool:
    manifest = _stage_manifest_path(output)
    if not output.exists() or not manifest.exists():
        return False
    try:
        return json.loads(manifest.read_text(encoding="utf-8")) == signature
    except (OSError, json.JSONDecodeError):
        return False


def _write_stage_manifest(output: Path, signature: dict) -> None:
    manifest = _stage_manifest_path(output)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(signature, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, manifest)


def _stage_enabled(stage: str, from_stage: str | None, force: bool) -> bool:
    """Whether a stage must run because of ``--force`` / ``--from-stage``."""
    if force:
        return True
    if from_stage is not None:
        return _STAGES.index(stage) >= _STAGES.index(from_stage)
    return False


def run(image_path: str | Path, checkpoint: str | Path, aoi: str,
        interim_dir: str | Path = "data/interim", processed_dir: str | Path = "data/processed",
        resolution_m: float = 1.0, tile_size: int | None = None, threshold: float | None = None,
        device: str = "cpu", curve_steps: int = 25, tta: bool = False, blend: bool = True,
        postprocess: bool = False, min_component_size: int = 50, pp_open_radius: int = 0,
        pp_close_radius: int = 0, fill_holes: int = 0,
        segment_fn: Callable[..., tuple[Path, float]] = segment,
        config: PipelineConfig | None = None,
        force: bool = False, from_stage: str | None = None) -> dict[str, Any]:
    """Run the whole pipeline on one tile and return a summary dict.

    ``config`` (a :class:`~src.pipeline.config.PipelineConfig`) is the single
    source of truth; when omitted it is built from the individual kwargs for
    backward compatibility. ``segment_fn`` is injectable so the P2→P4
    orchestration can be tested without a real checkpoint.

    Idempotent: a stage is **skipped when its completed content/config
    signature matches** the current inputs, unless ``force`` or ``from_stage``
    requires it.
    Raises ``RuntimeError`` (fail-loud) on a degenerate graph or a violated P4
    contract — a broken run must not exit 0. Writes ``{aoi}_run.json`` with the
    per-stage timings, resolved config, and model provenance.
    """
    _configure_logging()
    if from_stage is not None and from_stage not in _STAGES:
        raise ValueError(f"--from-stage must be one of {_STAGES}, got {from_stage!r}")

    if config is None:
        config = PipelineConfig(
            aoi=aoi, image=str(image_path), checkpoint=str(checkpoint),
            interim_dir=str(interim_dir), processed_dir=str(processed_dir),
            resolution_m=resolution_m, tile_size=tile_size, threshold=threshold,
            device=device, tta=tta, blend=blend, postprocess=postprocess,
            min_component_size=min_component_size, pp_open_radius=pp_open_radius,
            pp_close_radius=pp_close_radius, fill_holes=fill_holes, curve_steps=curve_steps,
        )
    cfg = config.graph_config()
    stages: list[dict[str, Any]] = []
    hashes: dict[tuple[str, int, int], str] = {}

    def _run_stage(name: str, output: Path, inputs: list[Path],
                   settings: dict[str, Any], fn: Callable[[], Any]) -> Any:
        """Run unless output has an exact input/config content signature."""
        forced = _stage_enabled(name, from_stage, force)
        signature = _stage_signature(name, inputs, settings, hashes)
        if not forced and _stage_is_current(output, signature):
            log.info("[%s] skip '%s' — %s signature matches", name.upper(), config.aoi, output.name)
            stages.append({"stage": name, "ran": False, "reason": "signature-match", "seconds": 0.0})
            return None
        t0 = time.perf_counter()
        result = fn()
        # Recompute after the stage because upstream callbacks may create inputs
        # (notably injected test segmenters); persist only a completed run.
        _write_stage_manifest(output, _stage_signature(name, inputs, settings, hashes))
        dt = round(time.perf_counter() - t0, 3)
        log.info("[%s] done in %ss", name.upper(), dt)
        stages.append({"stage": name, "ran": True, "seconds": dt})
        return result

    log.info("end-to-end pipeline for '%s' (force=%s, from_stage=%s)", config.aoi, force, from_stage)

    # P1 — segment imagery → mask
    mask_path = cfg.mask_path
    p1_inputs = [Path(config.image or image_path), Path(config.checkpoint or checkpoint)]

    def _p1() -> tuple[Path, float]:
        extra = {}
        if segment_fn is segment:
            extra["checkpoint_sha256"] = _file_identity(p1_inputs[1], hashes).get("sha256")
        return segment_fn(config.image or image_path, config.checkpoint or checkpoint, config.aoi,
                          config.interim_dir, tile_size=config.tile_size, threshold=config.threshold,
                          device=config.device, tta=config.tta, blend=config.blend,
                          postprocess=config.postprocess, min_component_size=config.min_component_size,
                          pp_open_radius=config.pp_open_radius, pp_close_radius=config.pp_close_radius,
                          fill_holes=config.fill_holes, **extra)

    p1_settings = {key: getattr(config, key) for key in (
        "tile_size", "threshold", "device", "tta", "blend", "postprocess",
        "min_component_size", "pp_open_radius", "pp_close_radius", "fill_holes")}
    p1_result = _run_stage("p1", mask_path, p1_inputs, p1_settings, _p1)
    coverage = p1_result[1] if p1_result is not None else None

    # P2 — mask → healed routable graph
    def _p2() -> Any:
        return build_graph(cfg)

    p2_inputs = [mask_path, cfg.manifest_path, cfg.prob_path, cfg.provenance_path]
    p2_settings = {key: getattr(config, key) for key in (
        "resolution_m", "gap_max_m", "angle_max_deg", "angle_penalty_factor", "min_edge_len_m",
        "simplify", "min_stub_len_m", "consolidate", "consolidate_tol_m",
        "simplify_geom", "geom_tol_m", "min_corridor_support", "corridor_samples")}
    _run_stage("p2", cfg.graphml_path, p2_inputs, p2_settings, _p2)

    # Copy the P1 provenance next to the processed artifacts (best-effort).
    from src.pipeline.p1_segment.provenance import read_provenance, write_provenance
    provenance = read_provenance(cfg.provenance_path)
    if provenance is not None:
        write_provenance(cfg.processed_provenance_path, provenance)

    # P3 — criticality + resilience
    def _p3() -> Any:
        return analyze(cfg, curve_steps=config.curve_steps)

    analysis = _run_stage("p3", cfg.processed_dir / f"{config.aoi}_criticality.csv",
                          [cfg.graphml_path], {"curve_steps": config.curve_steps}, _p3)

    # P4 — dashboard contract check (fail-loud)
    graph = _load_graph_for_summary(cfg)
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise RuntimeError(
            f"Pipeline aborted: degenerate graph for '{config.aoi}' "
            f"({graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges)")
    p4 = verify_dashboard_ready(cfg)
    log.info("[P4] criticality columns match: %s | geojson present: %s",
             p4["columns_match"], p4["geojson_exists"])
    if not (p4["columns_match"] and p4["geojson_exists"]):
        raise RuntimeError(f"Pipeline finished but dashboard contract violated: {p4}")

    summary = {
        "aoi": config.aoi,
        "status": "ok",
        "mask": str(mask_path),
        "road_fraction": coverage,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "analysis": analysis,
        "p4": p4,
        "stages": stages,
        "config": config.to_dict(),
        "provenance": provenance,
    }
    _write_run_summary(cfg, summary)
    log.info("pipeline done '%s': %d nodes / %d edges -> %s",
             config.aoi, graph.number_of_nodes(), graph.number_of_edges(), config.processed_dir)
    return summary


def _load_graph_for_summary(cfg: GraphConfig):
    """Load the processed graph for the P4 check + node/edge counts."""
    from src.pipeline.p2_graph.graph_io import load_graphml

    return load_graphml(cfg.graphml_path)


def _write_run_summary(cfg: GraphConfig, summary: dict[str, Any]) -> Path:
    """Write ``{aoi}_run.json`` (status + timings + config + provenance)."""
    from src.pipeline.p2_graph.graph_io import atomic_write

    out = cfg.processed_dir / f"{cfg.aoi}_run.json"
    payload = json.dumps(summary, indent=2, default=str)
    atomic_write(out, lambda tmp: tmp.write_text(payload))
    return out


def main() -> None:
    import argparse

    from src.pipeline.p1_segment.model import DEPLOYED_RELEASE

    p = argparse.ArgumentParser(
        description="Run road segmentation, graph construction, and resilience analysis for one AOI."
    )
    p.add_argument("--image", help="RGB satellite tile (jpg/png/3-band tif)")
    p.add_argument("--checkpoint", help=f"trained .pt (deployed: Release {DEPLOYED_RELEASE})")
    p.add_argument("--aoi", help="AOI id for all artifact filenames")
    p.add_argument("--config", help="JSON/YAML PipelineConfig; CLI flags override its fields")
    p.add_argument("--resolution-m", type=float, default=None, help="m/px (pixel-space masks)")
    p.add_argument("--tile-size", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None, help="override; default = checkpoint meta")
    p.add_argument("--device", default=None)
    p.add_argument("--curve-steps", type=int, default=None)
    p.add_argument("--tta", action="store_true", default=None,
                   help="D4 test-time augmentation in P1")
    p.add_argument("--no-blend", dest="blend", action="store_false", default=None,
                   help="disable A27 Hann-blended inference")
    p.add_argument("--postprocess", action="store_true", default=None,
                   help="A10 mask cleanup before P2")
    p.add_argument("--min-component-size", type=int, default=None)
    p.add_argument("--pp-open-radius", type=int, default=None)
    p.add_argument("--pp-close-radius", type=int, default=None)
    p.add_argument("--fill-holes", type=int, default=None,
                   help="A10 postprocess: fill holes up to this area (px); 0 = off")
    p.add_argument("--min-corridor-support", type=float, default=None,
                   help="reject a healing bridge if mean P1 prob-map support along it is below "
                        "this (0 disables; needs prob.png from the blended P1 path, A39)")
    p.add_argument("--force", action="store_true", help="rerun every stage regardless of signatures")
    p.add_argument("--from-stage", choices=_STAGES, default=None,
                   help="force a rerun starting at this stage (earlier stages may match signatures)")
    args = p.parse_args()

    if args.config:
        values = PipelineConfig.from_file(args.config).to_dict()
    else:
        if not (args.image and args.checkpoint and args.aoi):
            p.error("--image, --checkpoint and --aoi are required unless --config is given")
        values = {}

    config_fields = {field.name for field in fields(PipelineConfig)}
    values.update((name, value) for name, value in vars(args).items()
                  if name in config_fields and value is not None)
    config = PipelineConfig(**values)

    run(config.image, config.checkpoint, config.aoi, config=config,
        force=args.force, from_stage=args.from_stage)


if __name__ == "__main__":
    main()
