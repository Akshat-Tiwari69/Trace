"""P1 · Segmentation lane (Akshat).

Phase 0 ships the OSM→mask label generator (``osm_mask``); Phase I adds the
fine-tuned segmentation model that predicts masks from imagery.
"""

from src.pipeline.p1_segment.osm_mask import (
    MaskConfig,
    build_grid,
    fetch_osm_roads,
    rasterize_roads,
    tile_array,
)

__all__ = [
    "MaskConfig",
    "build_grid",
    "fetch_osm_roads",
    "rasterize_roads",
    "tile_array",
]
