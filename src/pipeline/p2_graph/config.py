"""Configuration for Phase II — graph build + healing (Shaivi's lane).

All tunables live here, not scattered through the code (``docs/Rules.md`` →
"No magic numbers"). The same ``GraphConfig`` drives the S1 spike (graph built
from an OSM-derived mask) and S2 (graph built from a real predicted mask) — the
P2 code path is identical, only the input mask differs.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class GraphConfig:
    """Parameters for one AOI's graph build + healing pass.

    Distances are in **metres** (the grid is metric — see ``osm_mask.build_grid``),
    so ``gap_max_m`` and friends are resolution-independent.
    """

    aoi: str

    # --- healing: which gaps to bridge, and how strictly ----------------------
    gap_max_m: float = 40.0            # never bridge endpoints farther apart than this
    angle_max_deg: float = 60.0        # never bridge if the road would turn more than this
    angle_penalty_factor: float = 2.0  # how hard a turn is penalised vs. a straight run
    min_edge_len_m: float = 1.0        # drop degenerate sub-pixel edges below this

    # --- simplification (S3): lighter graph, same connectivity ----------------
    simplify: bool = True              # prune short stubs + collapse degree-2 chains
    min_stub_len_m: float = 15.0       # trim degree-1 spurs shorter than this

    # --- consolidation (S4): merge near-duplicate junctions -------------------
    consolidate: bool = True           # merge node clusters joined by sub-tol edges
    consolidate_tol_m: float = 10.0    # junctions joined by an edge shorter than this

    # --- grid fallback (used only when no alignment manifest is present) -------
    resolution_m: float = 1.0          # m/px; overridden by the manifest when available

    # --- IO paths (per ``docs/Tracker.md`` §4 contract) -----------------------
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")

    @property
    def mask_path(self) -> Path:
        """Input contract: the binary road mask from P1 (or the OSM spike)."""
        return self.interim_dir / f"{self.aoi}_mask.png"

    @property
    def manifest_path(self) -> Path:
        """Optional grid alignment (CRS + transform) written alongside the mask."""
        return self.interim_dir / self.aoi / "manifest.json"

    @property
    def graphml_path(self) -> Path:
        """Output contract: the healed routable graph (GraphML)."""
        return self.processed_dir / f"{self.aoi}_graph.graphml"

    @property
    def geojson_path(self) -> Path:
        """Output contract: the healed graph as GeoJSON (for P4 / inspection)."""
        return self.processed_dir / f"{self.aoi}_graph.geojson"
