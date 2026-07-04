"""One config object for an end-to-end pipeline run (bugs.md §5B).

The tunables for a run were duplicated across ``run()``'s long parameter list,
``GraphConfig``, and three separate argparse blocks — with independent defaults,
so "works via build_graph, wrong via run_pipeline" drift was a live risk and no
single reproducible record of a run's settings existed.

``PipelineConfig`` is that single record: P1 (segmentation) + P2 (graph/heal) +
P3 (analysis) parameters in one dataclass, loadable from a JSON/YAML file, and
serialisable back into the run summary so a result is always reproducible from
its recorded config. The per-stage CLIs still exist for standalone use, but the
orchestrator resolves everything through this one object.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from src.pipeline.p2_graph.config import GraphConfig, sanitize_aoi


@dataclasses.dataclass
class PipelineConfig:
    """All parameters for a single AOI's P1→P3 run, in one place."""

    aoi: str
    image: str | None = None
    checkpoint: str | None = None

    # --- IO roots -------------------------------------------------------------
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"

    # --- P1: segmentation -----------------------------------------------------
    tile_size: int | None = None       # None → checkpoint meta image_size
    threshold: float | None = None     # None → checkpoint meta threshold
    device: str = "cpu"
    tta: bool = False
    blend: bool = True                 # A27 Hann-blended inference (default on)
    postprocess: bool = False
    min_component_size: int = 50
    pp_open_radius: int = 0
    pp_close_radius: int = 0
    fill_holes: int = 0

    # --- P2: graph build + healing (mirrors GraphConfig) ----------------------
    resolution_m: float = 1.0
    gap_max_m: float = 40.0
    angle_max_deg: float = 60.0
    angle_penalty_factor: float = 2.0
    min_edge_len_m: float = 1.0
    simplify: bool = True
    min_stub_len_m: float = 15.0
    consolidate: bool = True
    consolidate_tol_m: float = 10.0
    simplify_geom: bool = True
    geom_tol_m: float = 1.5

    # --- P3: analysis ---------------------------------------------------------
    curve_steps: int = 25

    def __post_init__(self) -> None:
        sanitize_aoi(self.aoi)

    # ------------------------------------------------------------------ #
    def graph_config(self) -> GraphConfig:
        """Build the P2/P3 ``GraphConfig`` this run should use (one source)."""
        return GraphConfig(
            aoi=self.aoi,
            interim_dir=Path(self.interim_dir),
            processed_dir=Path(self.processed_dir),
            resolution_m=self.resolution_m,
            gap_max_m=self.gap_max_m,
            angle_max_deg=self.angle_max_deg,
            angle_penalty_factor=self.angle_penalty_factor,
            min_edge_len_m=self.min_edge_len_m,
            simplify=self.simplify,
            min_stub_len_m=self.min_stub_len_m,
            consolidate=self.consolidate,
            consolidate_tol_m=self.consolidate_tol_m,
            simplify_geom=self.simplify_geom,
            geom_tol_m=self.geom_tol_m,
        )

    def to_dict(self) -> dict:
        """Plain-dict form for the run summary (JSON-serialisable)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_file(cls, path: str | Path) -> "PipelineConfig":
        """Load a config from JSON or YAML (``.yaml``/``.yml``). Unknown keys error."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml  # optional dep; only needed for YAML configs

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)
