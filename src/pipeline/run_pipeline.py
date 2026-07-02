"""A5 walking skeleton: one command, P1→P2→P3→P4 on a single tile (task A5).

Chains the released pieces end-to-end with **no manual steps**::

    imagery + checkpoint --[P1 predict]----> binary road mask
                         --[P2 build_graph]-> healed routable graph
                         --[P3 analyze]-----> criticality + resilience
                         --[P4 check]-------> dashboard-ready artifacts

P4 here is the *seam*, not the UI: the dashboard (Saanvi's lane) reads
`{aoi}_graph.geojson` + `{aoi}_criticality.csv`, so the skeleton verifies those
land with the contracted columns rather than editing the app. The single CLI is
the integration glue Akshat owns; each stage is the teammates' code, unchanged.

Example::

    python -m src.pipeline.run_pipeline --image data/raw/tile.jpg \
        --checkpoint models/deepglobe_mit_b3_scse_512px_best.pt --aoi mytile
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.pipeline.p2_graph.build_graph import build_graph
from src.pipeline.p2_graph.config import GraphConfig
from src.pipeline.p3_analysis.analyze import analyze

# P4 contract: the columns the dashboard reads from the criticality CSV (§4).
DASHBOARD_CRITICALITY_COLUMNS = ["node_id", "betweenness", "rank", "is_critical", "x", "y"]


def segment(image_path: str | Path, checkpoint: str | Path, aoi: str, interim_dir: str | Path,
            tile_size: int = 512, threshold: float | None = None,
            device: str = "cpu", tta: bool = False,
            postprocess: bool = False, min_component_size: int = 50,
            pp_close_radius: int = 0, fill_holes: int = 0) -> tuple[Path, float]:
    """P1: predict a road mask from imagery → ``data/interim/{aoi}_mask.png``.

    Threshold defaults to the checkpoint's deploy threshold (its ``meta``).
    ``tta`` applies D4 test-time augmentation. ``postprocess`` runs the A10 mask
    cleanup (drop tiny false components, optional close/fill) before writing.
    """
    from src.pipeline.p1_segment.model import load_checkpoint, predict_large
    from src.pipeline.p1_segment.osm_mask import save_binary_png
    from src.pipeline.p1_segment.raster_io import read_image_any, write_manifest

    image, transform, crs = read_image_any(image_path)  # A26: GeoTIFF-aware (keeps CRS/PAN)
    model, meta = load_checkpoint(checkpoint, map_location=device)
    thr = threshold if threshold is not None else float(meta.get("threshold", 0.5))
    mask = predict_large(model, image, tile_size=tile_size, device=device, threshold=thr, tta=tta)
    if postprocess:
        from src.pipeline.p1_segment.postprocess import postprocess_mask
        mask = postprocess_mask(mask, min_size=min_component_size,
                                close_radius=pp_close_radius, fill_holes=fill_holes)
    out = Path(interim_dir) / f"{aoi}_mask.png"
    save_binary_png(mask, out)
    write_manifest(aoi, interim_dir, transform, crs)  # A26: georeference the graph if available
    return out, float(mask.mean())


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


def run(image_path: str | Path, checkpoint: str | Path, aoi: str,
        interim_dir: str | Path = "data/interim", processed_dir: str | Path = "data/processed",
        resolution_m: float = 1.0, tile_size: int = 512, threshold: float | None = None,
        device: str = "cpu", curve_steps: int = 25, tta: bool = False,
        postprocess: bool = False, min_component_size: int = 50, pp_close_radius: int = 0,
        fill_holes: int = 0,
        segment_fn: Callable[..., tuple[Path, float]] = segment) -> dict[str, Any]:
    """Run the whole pipeline on one tile and return a summary dict.

    ``segment_fn`` is injectable so the P2→P4 orchestration can be tested without
    a real checkpoint; production uses the default :func:`segment` (P1).
    """
    cfg = GraphConfig(aoi=aoi, interim_dir=Path(interim_dir),
                      processed_dir=Path(processed_dir), resolution_m=resolution_m)

    print(f"[A5] end-to-end pipeline for '{aoi}'")
    print("[P1] segment imagery → road mask")
    mask_path, coverage = segment_fn(image_path, checkpoint, aoi, interim_dir, tile_size, threshold, device, tta,
                                     postprocess, min_component_size, pp_close_radius, fill_holes)
    print(f"     → {mask_path}  ({coverage:.2%} road px)")

    print("[P2] mask → healed routable graph")
    graph, report = build_graph(cfg)

    print("[P3] criticality + global-efficiency resilience")
    analysis = analyze(cfg, curve_steps=curve_steps)

    print("[P4] dashboard-ready check (artifact contract, not the UI)")
    p4 = verify_dashboard_ready(cfg)
    print(f"     criticality columns match: {p4['columns_match']} | geojson present: {p4['geojson_exists']}")

    print(f"\nA5 ✓ one tile flowed P1→P2→P3→P4: "
          f"{graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges → {processed_dir}")
    return {
        "aoi": aoi, "mask": str(mask_path), "road_fraction": coverage,
        "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
        "bridges_added": report.bridges_added, "analysis": analysis, "p4": p4,
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="A5 walking skeleton: P1→P2→P3→P4 on one tile.")
    p.add_argument("--image", required=True, help="RGB satellite tile (jpg/png/3-band tif)")
    p.add_argument("--checkpoint", required=True, help="trained .pt (Release a4-roadseg-v1)")
    p.add_argument("--aoi", required=True, help="AOI id for all artifact filenames")
    p.add_argument("--resolution-m", type=float, default=1.0, help="m/px (pixel-space masks)")
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--threshold", type=float, default=None, help="override; default = checkpoint meta")
    p.add_argument("--device", default="cpu")
    p.add_argument("--curve-steps", type=int, default=25)
    p.add_argument("--tta", action="store_true", help="D4 test-time augmentation in P1")
    p.add_argument("--postprocess", action="store_true", help="A10 mask cleanup before P2")
    p.add_argument("--min-component-size", type=int, default=50)
    p.add_argument("--pp-close-radius", type=int, default=0)
    p.add_argument("--fill-holes", type=int, default=0,
                   help="A10 postprocess: fill holes up to this area (px); 0 = off")
    args = p.parse_args()
    run(args.image, args.checkpoint, args.aoi, resolution_m=args.resolution_m,
        tile_size=args.tile_size, threshold=args.threshold, device=args.device,
        curve_steps=args.curve_steps, tta=args.tta, postprocess=args.postprocess,
        min_component_size=args.min_component_size, pp_close_radius=args.pp_close_radius,
        fill_holes=args.fill_holes)


if __name__ == "__main__":
    main()
