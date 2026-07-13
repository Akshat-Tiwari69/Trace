"""Configuration for Phase II — graph build + healing (Shaivi's lane).

All tunables live here, not scattered through the code (``docs/Rules.md`` →
"No magic numbers"). The same ``GraphConfig`` drives the S1 spike (graph built
from an OSM-derived mask) and S2 (graph built from a real predicted mask) — the
P2 code path is identical, only the input mask differs.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

# AOI ids are interpolated into artifact filenames (mask/graph/CSV paths), so
# they must never carry path separators or traversal sequences.
_AOI_PATTERN = re.compile(r"^[a-z0-9_-]{1,64}$")


def sanitize_aoi(aoi: str) -> str:
    """Validate an AOI id for safe use in artifact paths; return it unchanged.

    Accepts only ``^[a-z0-9_-]{1,64}$`` (lowercase letters, digits, underscore,
    hyphen). Anything else — path separators, ``..``, spaces, uppercase, empty —
    raises ``ValueError`` so a hostile or mistyped id can never escape the data
    directories via path interpolation.
    """
    if not isinstance(aoi, str) or not _AOI_PATTERN.fullmatch(aoi):
        raise ValueError(
            f"invalid AOI id {aoi!r}: must be 1-64 chars of lowercase letters, "
            "digits, '_' or '-' (no path separators, dots, or spaces)"
        )
    return aoi


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

    # --- healing: probability-map corridor check -------------------------------
    # Distance/angle alone can't distinguish a real gap from a frontage road
    # running parallel to a highway broken by the same occlusion. When P1's
    # prob.png is present, a candidate bridge is also rejected if the mean P1
    # probability along its curve is too low — the model's own belief that
    # *something* road-like is there, even sub-threshold. 0 disables the check
    # even if prob.png exists (mask-only behaviour).
    min_corridor_support: float = 0.3
    corridor_samples: int = 16         # points sampled along a bridge's curve

    # --- simplification (S3): lighter graph, same connectivity ----------------
    simplify: bool = True              # prune short stubs + collapse degree-2 chains
    min_stub_len_m: float = 15.0       # trim degree-1 spurs shorter than this

    # --- consolidation (S4): merge near-duplicate junctions -------------------
    consolidate: bool = True           # merge node clusters joined by sub-tol edges
    consolidate_tol_m: float = 10.0    # junctions joined by an edge shorter than this

    # --- polyline simplification (S5): lighter geometry, shape preserved ------
    simplify_geom: bool = True         # Douglas-Peucker each edge's geometry
    geom_tol_m: float = 1.5            # drop vertices within this of the line

    # --- grid fallback (used only when no alignment manifest is present) -------
    resolution_m: float = 1.0          # m/px; overridden by the manifest when available

    # --- IO paths (per ``docs/Tracker.md`` §4 contract) -----------------------
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")

    def __post_init__(self) -> None:
        """Guard every path-interpolating consumer: reject unsafe AOI ids."""
        sanitize_aoi(self.aoi)
        if not 0.0 <= self.min_corridor_support <= 1.0:
            raise ValueError(
                f"min_corridor_support must be in [0, 1], got {self.min_corridor_support!r}"
            )
        if self.corridor_samples < 1:
            raise ValueError(f"corridor_samples must be >= 1, got {self.corridor_samples!r}")

    @property
    def mask_path(self) -> Path:
        """Input contract: the binary road mask from P1 (or the OSM spike)."""
        return self.interim_dir / f"{self.aoi}_mask.png"

    @property
    def manifest_path(self) -> Path:
        """Optional grid alignment (CRS + transform) written alongside the mask."""
        return self.interim_dir / self.aoi / "manifest.json"

    @property
    def prob_path(self) -> Path:
        """Optional P1 probability map, co-located with the manifest.

        Present only when P1 ran the blended (Hann-window) inference path; a
        mask-only input (upload, OSM spike, old artifact) has no such file, and
        healing falls back to distance/angle/crossing only.
        """
        return self.interim_dir / self.aoi / "prob.png"

    @property
    def provenance_path(self) -> Path:
        """P1 provenance record (checkpoint/threshold/commit) written by inference."""
        return self.interim_dir / self.aoi / "provenance.json"

    @property
    def processed_provenance_path(self) -> Path:
        """Provenance copied next to the processed graph/criticality artifacts."""
        return self.processed_dir / f"{self.aoi}_provenance.json"

    @property
    def graphml_path(self) -> Path:
        """Output contract: the healed routable graph (GraphML)."""
        return self.processed_dir / f"{self.aoi}_graph.graphml"

    @property
    def geojson_path(self) -> Path:
        """Output contract: the healed graph as GeoJSON (for P4 / inspection)."""
        return self.processed_dir / f"{self.aoi}_graph.geojson"
