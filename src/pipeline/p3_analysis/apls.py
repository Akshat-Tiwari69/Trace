"""S7 — APLS topology validation against OSM ground truth.

Pixel overlap (IoU) can look fine while the *graph* is unroutable — a mask at
F1=0.72 can score APLS=0.25 (``docs/Evaluation.md``, ``Research.md``). **APLS**
(Average Path Length Similarity, the SpaceNet-3 metric) measures whether our
healed graph routes the *same way* the real roads do: it samples node pairs,
compares the shortest-path length between them in our graph vs. in the ground
truth, and penalises mismatches.

This is a self-contained, node-based port (no heavy CosmiQ dependency):

    contribution(a, b) = max(0, 1 − |L_gt(a,b) − L_prop(a',b')| / L_gt(a,b))

where ``a', b'`` are the proposal nodes spatially nearest to ``a, b`` (within a
snap tolerance); a missing correspondence or missing path scores 0. We average
over sampled pairs in both directions (gt→prop and prop→gt) and take their
**harmonic mean** — the standard symmetric APLS in ``[0, 1]`` (1 = identical
routing).

Ground truth is the OSM road graph for the same AOI (``build_osm_truth`` fetches
it once via osmnx and commits a small GeoJSON so the CLI is re-runnable offline).
Pure CPU, classical Python (Shaivi's lane).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from bisect import bisect_left
from pathlib import Path

import numpy as np

from src.pipeline.p2_graph.graph_io import load_geojson_graph, save_geojson
from src.pipeline.p2_graph.skeleton_graph import _annotate_degree_and_type

DEFAULT_BBOX = (73.823, 15.488, 73.842, 15.501)  # panaji_demo (matches the spike)


# --------------------------------------------------------------------------- #
# Local metric projection (small AOI → equirectangular metres)
# --------------------------------------------------------------------------- #
def _to_metres(lonlat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    """Project lon/lat to local metres about (lon0, lat0). Fine over a city AOI."""
    k = math.cos(math.radians(lat0))
    x = (lonlat[:, 0] - lon0) * 111_320.0 * k
    y = (lonlat[:, 1] - lat0) * 110_540.0
    return np.column_stack([x, y])


def _densify(graph, interval_m: float):
    """Inject evenly-spaced points along edges so node density is comparable.

    APLS compares *locations*, not nodes — but our healed graph has many mid-edge
    nodes while OSM has only junctions. Subdividing every edge into ≈``interval_m``
    segments (in both graphs) gives each point a correspondent, so the score
    reflects routing fidelity, not how finely each graph happens to be noded. Path
    lengths are preserved (we only split edges).
    """
    import networkx as nx

    dense = nx.Graph()
    nid = (max(graph.nodes) + 1) if graph.number_of_nodes() else 0
    for n, d in graph.nodes(data=True):
        dense.add_node(n, x=d["x"], y=d["y"])
    for u, v, d in graph.edges(data=True):
        length = float(d.get("length_m", 0.0))
        k = max(1, round(length / interval_m)) if interval_m > 0 else 1
        if k <= 1:
            # dense is a plain Graph, so a short parallel edge (u, v now a
            # MultiGraph-sourced input, A37) would silently overwrite its
            # sibling here; keep the shorter one — the one Dijkstra would
            # actually route over.
            length = min(length, dense.edges[u, v]["length_m"]) if dense.has_edge(u, v) else length
            dense.add_edge(u, v, length_m=max(length, 1e-6))
            continue
        ux, uy = graph.nodes[u]["x"], graph.nodes[u]["y"]
        vx, vy = graph.nodes[v]["x"], graph.nodes[v]["y"]
        geometry = [list(map(float, point)) for point in
                    (d.get("geometry") or [[ux, uy], [vx, vy]])]
        if ((geometry[0][0] - vx) ** 2 + (geometry[0][1] - vy) ** 2 <
                (geometry[0][0] - ux) ** 2 + (geometry[0][1] - uy) ** 2):
            geometry.reverse()
        cumulative = [0.0]
        for p, q in zip(geometry, geometry[1:]):
            cumulative.append(cumulative[-1] + math.hypot(q[0] - p[0], q[1] - p[1]))

        def point_at_fraction(fraction: float) -> tuple[float, float]:
            target = fraction * cumulative[-1]
            j = min(len(cumulative) - 1, max(1, bisect_left(cumulative, target)))
            if j == 0:
                return geometry[0][0], geometry[0][1]
            span = cumulative[j] - cumulative[j - 1]
            local = 0.0 if span == 0 else (target - cumulative[j - 1]) / span
            p, q = geometry[j - 1], geometry[j]
            return p[0] + (q[0] - p[0]) * local, p[1] + (q[1] - p[1]) * local

        seg, prev = length / k, u
        for i in range(1, k):
            t = i / k
            x, y = point_at_fraction(t)
            dense.add_node(nid, x=x, y=y)
            dense.add_edge(prev, nid, length_m=seg)
            prev, nid = nid, nid + 1
        dense.add_edge(prev, v, length_m=seg)
    return dense


def _metric_xy(
    xy: np.ndarray,
    coordinate_system: str,
    lon0: float,
    lat0: float,
) -> np.ndarray:
    if coordinate_system == "geographic":
        return _to_metres(xy, lon0, lat0)
    if coordinate_system == "projected":
        return xy
    raise ValueError("coordinate_system must be 'geographic' or 'projected'")


def _snap_map(
    src,
    dst,
    lon0: float,
    lat0: float,
    tol_m: float,
    coordinate_system: str,
) -> dict:
    """Map each ``src`` node to the nearest ``dst`` node within ``tol_m`` (or None)."""
    from scipy.spatial import cKDTree

    dst_nodes = list(dst.nodes)
    if not dst_nodes:
        return {n: None for n in src.nodes}
    dst_xy = np.array([[dst.nodes[n]["x"], dst.nodes[n]["y"]] for n in dst_nodes])
    tree = cKDTree(_metric_xy(dst_xy, coordinate_system, lon0, lat0))

    src_nodes = list(src.nodes)
    src_xy = np.array([[src.nodes[n]["x"], src.nodes[n]["y"]] for n in src_nodes])
    dists, idxs = tree.query(
        _metric_xy(src_xy, coordinate_system, lon0, lat0),
        distance_upper_bound=tol_m,
    )

    out: dict = {}
    for n, dist, idx in zip(src_nodes, dists, idxs):
        out[n] = dst_nodes[idx] if math.isfinite(dist) and idx < len(dst_nodes) else None
    return out


def _apls_oneway(src, dst, snap: dict, n_samples: int, weight: str, seed: int) -> float:
    """Mean path-length-similarity for sampled ``src`` pairs routed through ``dst``."""
    import networkx as nx

    nodes = list(src.nodes)
    if len(nodes) < 2:
        # A <2-node graph can't support any routing claim — that's a failure of
        # the graph, not a perfect score.
        print("[apls] WARNING: <2 nodes in one direction — scoring 0.0")
        return 0.0
    rng = random.Random(seed)
    pairs = [rng.sample(nodes, 2) for _ in range(n_samples)]
    src_groups: dict = {}
    for index, (a, b) in enumerate(pairs):
        src_groups.setdefault(a, []).append((index, b))

    contribs = [0.0] * n_samples
    dst_groups: dict = {}
    skipped = 0
    for a, targets in src_groups.items():
        lengths = nx.single_source_dijkstra_path_length(src, a, weight=weight)
        for index, b in targets:
            length_src = lengths.get(b)
            if length_src is None or length_src <= 0:
                skipped += 1
                continue
            a2, b2 = snap.get(a), snap.get(b)
            if a2 is not None and b2 is not None and a2 != b2:
                dst_groups.setdefault(a2, []).append((index, b2, length_src))

    for a2, targets in dst_groups.items():
        lengths = nx.single_source_dijkstra_path_length(dst, a2, weight=weight)
        for index, b2, length_src in targets:
            length_dst = lengths.get(b2)
            if length_dst is not None:
                contribs[index] = max(0.0, 1.0 - abs(length_src - length_dst) / length_src)
    if skipped > n_samples // 2:
        print(f"[apls] WARNING: {skipped}/{n_samples} sampled pairs unreachable — "
              "effective sample is small; treat this score with caution")
    return sum(contribs) / n_samples


def _reachable_pair_fraction(graph) -> float:
    """Fraction of ordered node pairs connected by any path."""
    import networkx as nx

    n = graph.number_of_nodes()
    if n < 2:
        return 0.0
    reachable = sum(len(component) * (len(component) - 1)
                    for component in nx.connected_components(graph))
    return reachable / (n * (n - 1))


def apls(
    gt,
    prop,
    *,
    coordinate_system: str,
    n_samples: int = 600,
    tol_m: float = 15.0,
    interval_m: float = 10.0,
    weight: str = "length_m",
    seed: int = 42,
) -> dict:
    """Symmetric APLS between ground-truth ``gt`` and proposal ``prop`` graphs.

    ``coordinate_system`` is mandatory: ``geographic`` projects declared lon/lat
    coordinates to local metres, while ``projected`` uses x/y as metres directly.
    No coordinate-magnitude guessing is permitted. Both graphs are densified to
    ≈``interval_m`` spacing first. Returns the two one-way scores and harmonic mean.
    """
    if coordinate_system not in {"geographic", "projected"}:
        raise ValueError("coordinate_system must be 'geographic' or 'projected'")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not math.isfinite(tol_m) or tol_m <= 0:
        raise ValueError("tol_m must be a finite positive number")

    common = {
        "n_samples": n_samples,
        "snap_tol_m": tol_m,
        "coordinate_system": coordinate_system,
    }
    if gt.number_of_nodes() == 0:
        return {
            "apls": 0.0,
            "apls_gt_to_prop": 0.0,
            "apls_prop_to_gt": 0.0,
            **common,
            "reachable_pair_fraction_gt": 0.0,
            "reachable_pair_fraction_prop": _reachable_pair_fraction(prop),
        }
    if gt.number_of_nodes() >= 2 and prop.number_of_nodes() < 2:
        return {
            "apls": 0.0,
            "apls_gt_to_prop": 0.0,
            "apls_prop_to_gt": 0.0,
            **common,
            "reachable_pair_fraction_gt": round(_reachable_pair_fraction(gt), 4),
            "reachable_pair_fraction_prop": 0.0,
        }

    gt = _densify(gt, interval_m)
    prop = _densify(prop, interval_m)
    lat0 = float(np.mean([gt.nodes[n]["y"] for n in gt.nodes]))
    lon0 = float(np.mean([gt.nodes[n]["x"] for n in gt.nodes]))

    gt_to_prop = _snap_map(gt, prop, lon0, lat0, tol_m, coordinate_system)
    prop_to_gt = _snap_map(prop, gt, lon0, lat0, tol_m, coordinate_system)
    a = _apls_oneway(gt, prop, gt_to_prop, n_samples, weight, seed)
    b = _apls_oneway(prop, gt, prop_to_gt, n_samples, weight, seed)
    harmonic = 0.0 if (a + b) == 0 else 2 * a * b / (a + b)
    return {
        "apls": round(harmonic, 4),
        "apls_gt_to_prop": round(a, 4),
        "apls_prop_to_gt": round(b, 4),
        **common,
        "reachable_pair_fraction_gt": round(_reachable_pair_fraction(gt), 4),
        "reachable_pair_fraction_prop": round(_reachable_pair_fraction(prop), 4),
    }


# --------------------------------------------------------------------------- #
# OSM ground truth
# --------------------------------------------------------------------------- #
def build_osm_truth(bbox: tuple[float, float, float, float], path: Path):
    """Fetch the OSM drive network for ``bbox`` and save it as a schema GeoJSON.

    Run once to produce the committed ``{aoi}_osm_truth.geojson``; afterwards the
    CLI reads that file and needs no network.
    """
    import networkx as nx
    import osmnx as ox

    west, south, east, north = bbox
    try:
        osm = ox.graph_from_bbox(
            north=north, south=south, east=east, west=west,
            network_type="drive", simplify=True, retain_all=True, truncate_by_edge=True,
        )
    except Exception as exc:  # Overpass down / rate-limited / network error
        raise SystemExit(
            f"OSM ground-truth fetch failed ({type(exc).__name__}: {exc}).\n"
            "  Overpass may be unreachable or rate-limited — retry in a few minutes,\n"
            f"  or reuse a previously cached truth file at {path} if one exists."
        ) from exc
    relabel = {osmid: i for i, osmid in enumerate(osm.nodes)}
    truth = nx.Graph()
    for osmid, i in relabel.items():
        d = osm.nodes[osmid]
        truth.add_node(i, x=float(d["x"]), y=float(d["y"]))
    for u, v, d in osm.edges(data=True):
        a, b = relabel[u], relabel[v]
        if a == b:
            continue
        length = float(d.get("length", 0.0)) or 1e-6
        if truth.has_edge(a, b) and length >= truth.edges[a, b]["length_m"]:
            continue
        truth.add_edge(
            a, b, length_m=length, is_bridged=False, edge_betweenness=0.0,
            geometry=[[truth.nodes[a]["x"], truth.nodes[a]["y"]],
                      [truth.nodes[b]["x"], truth.nodes[b]["y"]]],
        )
    _annotate_degree_and_type(truth)
    save_geojson(truth, path)
    return truth


def _load_or_build_truth(bbox, path: Path):
    if path.exists():
        return load_geojson_graph(path)
    print(f"  fetching OSM ground truth -> {path}")
    return build_osm_truth(bbox, path)


def validate(
    aoi: str,
    sample_dir: Path = Path("data/sample"),
    bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
    n_samples: int = 600,
    tol_m: float = 15.0,
) -> dict:
    """Compute APLS of the AOI's healed graph vs OSM truth; write a report."""
    prop = load_geojson_graph(sample_dir / f"{aoi}_graph.geojson")
    truth = _load_or_build_truth(bbox, sample_dir / f"{aoi}_osm_truth.geojson")

    result = apls(
        truth, prop, coordinate_system="geographic",
        n_samples=n_samples, tol_m=tol_m,
    )
    result["aoi"] = aoi
    result["graph_nodes"] = prop.number_of_nodes()
    result["osm_truth_nodes"] = truth.number_of_nodes()

    out = sample_dir / f"{aoi}_apls.json"
    out.write_text(json.dumps(result, indent=2))
    print(
        f"\n=== APLS — {aoi} ===\n"
        f"healed graph: {prop.number_of_nodes()} nodes | OSM truth: {truth.number_of_nodes()} nodes\n"
        f"APLS = {result['apls']:.3f}  (gt->prop {result['apls_gt_to_prop']:.3f}, "
        f"prop->gt {result['apls_prop_to_gt']:.3f}; {n_samples} pairs, snap {tol_m} m)\n"
        f"  -> {out}"
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="APLS topology validation vs OSM ground truth.")
    p.add_argument("--aoi", default="panaji_demo", help="AOI id (default panaji_demo)")
    p.add_argument("--sample-dir", default="data/sample", help="dir with {aoi}_graph.geojson")
    p.add_argument("--bbox", help="west,south,east,north (for fetching OSM truth)")
    p.add_argument("--n-samples", type=int, default=600, help="node pairs to sample")
    p.add_argument("--tol-m", type=float, default=15.0, help="node snap tolerance (m)")
    args = p.parse_args()

    bbox = DEFAULT_BBOX
    if args.bbox:
        bbox = tuple(float(v) for v in args.bbox.split(","))  # type: ignore[assignment]
    validate(args.aoi, Path(args.sample_dir), bbox, args.n_samples, args.tol_m)


if __name__ == "__main__":
    main()
