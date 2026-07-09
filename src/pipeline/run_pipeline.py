"""A5 walking skeleton: one command, P1→P2→P3→P4 on a single tile (task A5).

Chains the released pieces end-to-end with **no manual steps**::

    imagery + checkpoint --[P1 predict]----> binary road mask
                         --[P2 build_graph]-> healed routable graph
                         --[P3 analyze]-----> criticality + resilience
                         --[P4 check]-------> dashboard-ready artifacts

P4 here is the *seam*, not the UI: the dashboard (Saanvi's lane) reads
`{aoi}_graph.geojson` + `{aoi}_criticality.csv`, so the skeleton verifies those
land with the contracted columns rather than editing the app.

Orchestration hardening (A36 / bugs.md §5): structured logging with per-stage
timings, a written ``{aoi}_run.json`` summary (status + timings + resolved
config + provenance), idempotent reruns (``--from-stage`` / ``--force`` /
skip-if-fresh), and a single ``PipelineConfig`` as the source of truth.

Example::

    python -m src.pipeline.run_pipeline --image data/raw/tile.jpg \
        --checkpoint models/road_pan.pt --aoi mytile
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from src.pipeline.config import PipelineConfig
from src.pipeline.p2_graph.build_graph import build_graph
from src.pipeline.p2_graph.config import GraphConfig
from src.pipeline.p3_analysis.analyze import analyze

log = logging.getLogger("trace.pipeline")

# P4 contract: the columns the dashboard reads from the criticality CSV (§4).
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
            fill_holes: int = 0) -> tuple[Path, float]:
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


def _fresh(output: Path, *inputs: Path) -> bool:
    """True when ``output`` exists and is at least as new as every present input."""
    if not output.exists():
        return False
    out_mtime = output.stat().st_mtime
    return all(out_mtime >= i.stat().st_mtime for i in inputs if i.exists())


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

    Idempotent (bugs.md §5A): a stage is **skipped when its output is fresh**
    relative to its input, unless ``force`` or ``from_stage`` requires it.
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

    def _run_stage(name: str, output: Path, inputs: list[Path], fn: Callable[[], Any]) -> Any:
        """Run a stage unless its output is fresh (and not forced); time + log it."""
        forced = _stage_enabled(name, from_stage, force)
        if not forced and _fresh(output, *inputs):
            log.info("[%s] skip '%s' — %s is fresh", name.upper(), config.aoi, output.name)
            stages.append({"stage": name, "ran": False, "reason": "fresh", "seconds": 0.0})
            return None
        t0 = time.perf_counter()
        result = fn()
        dt = round(time.perf_counter() - t0, 3)
        log.info("[%s] done in %ss", name.upper(), dt)
        stages.append({"stage": name, "ran": True, "seconds": dt})
        return result

    log.info("end-to-end pipeline for '%s' (force=%s, from_stage=%s)", config.aoi, force, from_stage)

    # P1 — segment imagery → mask
    mask_path = cfg.mask_path

    def _p1() -> tuple[Path, float]:
        return segment_fn(config.image or image_path, config.checkpoint or checkpoint, config.aoi,
                          config.interim_dir, tile_size=config.tile_size, threshold=config.threshold,
                          device=config.device, tta=config.tta, blend=config.blend,
                          postprocess=config.postprocess, min_component_size=config.min_component_size,
                          pp_open_radius=config.pp_open_radius, pp_close_radius=config.pp_close_radius,
                          fill_holes=config.fill_holes)

    p1_result = _run_stage("p1", mask_path, [], _p1)
    coverage = p1_result[1] if p1_result is not None else None

    # P2 — mask → healed routable graph
    def _p2() -> Any:
        return build_graph(cfg)

    _run_stage("p2", cfg.graphml_path, [mask_path], _p2)

    # Copy the P1 provenance next to the processed artifacts (best-effort).
    from src.pipeline.p1_segment.provenance import read_provenance, write_provenance
    provenance = read_provenance(cfg.provenance_path)
    if provenance is not None:
        write_provenance(cfg.processed_provenance_path, provenance)

    # P3 — criticality + resilience
    def _p3() -> Any:
        return analyze(cfg, curve_steps=config.curve_steps)

    analysis = _run_stage("p3", cfg.processed_dir / f"{config.aoi}_criticality.csv",
                          [cfg.graphml_path], _p3)

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
    log.info("A5 done '%s': %d nodes / %d edges -> %s",
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

    p = argparse.ArgumentParser(description="A5 walking skeleton: P1→P2→P3→P4 on one tile.")
    p.add_argument("--image", help="RGB satellite tile (jpg/png/3-band tif)")
    p.add_argument("--checkpoint", help=f"trained .pt (deployed: Release {DEPLOYED_RELEASE})")
    p.add_argument("--aoi", help="AOI id for all artifact filenames")
    p.add_argument("--config", help="JSON/YAML PipelineConfig; CLI flags override its fields")
    p.add_argument("--resolution-m", type=float, default=1.0, help="m/px (pixel-space masks)")
    p.add_argument("--tile-size", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None, help="override; default = checkpoint meta")
    p.add_argument("--device", default="cpu")
    p.add_argument("--curve-steps", type=int, default=25)
    p.add_argument("--tta", action="store_true", help="D4 test-time augmentation in P1")
    p.add_argument("--no-blend", action="store_true", help="disable A27 Hann-blended inference")
    p.add_argument("--postprocess", action="store_true", help="A10 mask cleanup before P2")
    p.add_argument("--min-component-size", type=int, default=50)
    p.add_argument("--pp-open-radius", type=int, default=0)
    p.add_argument("--pp-close-radius", type=int, default=0)
    p.add_argument("--fill-holes", type=int, default=0,
                   help="A10 postprocess: fill holes up to this area (px); 0 = off")
    p.add_argument("--min-corridor-support", type=float, default=0.3,
                   help="reject a healing bridge if mean P1 prob-map support along it is below "
                        "this (0 disables; needs prob.png from the blended P1 path, bugs.md §4)")
    p.add_argument("--force", action="store_true", help="rerun every stage even if outputs are fresh")
    p.add_argument("--from-stage", choices=_STAGES, default=None,
                   help="force a rerun starting at this stage (earlier stages skip if fresh)")
    args = p.parse_args()

    if args.config:
        config = PipelineConfig.from_file(args.config)
        # Let explicit CLI values override the file (only when the user set them).
        if args.image:
            config.image = args.image
        if args.checkpoint:
            config.checkpoint = args.checkpoint
        if args.aoi:
            config.aoi = args.aoi
    else:
        if not (args.image and args.checkpoint and args.aoi):
            p.error("--image, --checkpoint and --aoi are required unless --config is given")
        config = PipelineConfig(
            aoi=args.aoi, image=args.image, checkpoint=args.checkpoint,
            resolution_m=args.resolution_m, tile_size=args.tile_size, threshold=args.threshold,
            device=args.device, tta=args.tta, blend=not args.no_blend, postprocess=args.postprocess,
            min_component_size=args.min_component_size, pp_open_radius=args.pp_open_radius,
            pp_close_radius=args.pp_close_radius, fill_holes=args.fill_holes, curve_steps=args.curve_steps,
            min_corridor_support=args.min_corridor_support,
        )

    run(config.image, config.checkpoint, config.aoi, config=config,
        force=args.force, from_stage=args.from_stage)


if __name__ == "__main__":
    main()
