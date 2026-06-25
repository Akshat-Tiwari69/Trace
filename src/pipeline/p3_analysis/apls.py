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
            dense.add_edge(u, v, length_m=max(length, 1e-6))
            continue
        ux, uy = graph.nodes[u]["x"], graph.nodes[u]["y"]
        vx, vy = graph.nodes[v]["x"], graph.nodes[v]["y"]
        seg, prev = length / k, u
        for i in range(1, k):
            t = i / k
            dense.add_node(nid, x=ux + (vx - ux) * t, y=uy + (vy - uy) * t)
            dense.add_edge(prev, nid, length_m=seg)
            prev, nid = nid, nid + 1
        dense.add_edge(prev, v, length_m=seg)
    return dense


def _snap_map(src, dst, lon0: float, lat0: float, tol_m: float) -> dict:
    """Map each ``src`` node to the nearest ``dst`` node within ``tol_m`` (or None)."""
    from scipy.spatial import cKDTree

    dst_nodes = list(dst.nodes)
    if not dst_nodes:
        return {n: None for n in src.nodes}
    dst_xy = np.array([[dst.nodes[n]["x"], dst.nodes[n]["y"]] for n in dst_nodes])
    tree = cKDTree(_to_metres(dst_xy, lon0, lat0))

    src_nodes = list(src.nodes)
    src_xy = np.array([[src.nodes[n]["x"], src.nodes[n]["y"]] for n in src_nodes])
    dists, idxs = tree.query(_to_metres(src_xy, lon0, lat0), distance_upper_bound=tol_m)

    out: dict = {}
    for n, dist, idx in zip(src_nodes, dists, idxs):
        out[n] = dst_nodes[idx] if math.isfinite(dist) and idx < len(dst_nodes) else None
    return out


def _apls_oneway(src, dst, snap: dict, n_samples: int, weight: str, seed: int) -> float:
    """Mean path-length-similarity for sampled ``src`` pairs routed through ``dst``."""
    import networkx as nx

    nodes = list(src.nodes)
    if len(nodes) < 2:
        return 1.0
    rng = random.Random(seed)
    contribs: list[float] = []
    for _ in range(n_samples):
        a, b = rng.sample(nodes, 2)
        try:
            length_src = nx.shortest_path_length(src, a, b, weight=weight)
        except nx.NetworkXNoPath:
            continue  # unreachable in src → not a routing claim, skip
        if length_src <= 0:
            continue
        a2, b2 = snap.get(a), snap.get(b)
        if a2 is None or b2 is None or a2 == b2:
            contribs.append(0.0)  # no correspondence → worst score
            continue
        try:
            length_dst = nx.shortest_path_length(dst, a2, b2, weight=weight)
        except nx.NetworkXNoPath:
            contribs.append(0.0)
            continue
        contribs.append(max(0.0, 1.0 - abs(length_src - length_dst) / length_src))
    return sum(contribs) / len(contribs) if contribs else 1.0


def apls(
    gt,
    prop,
    n_samples: int = 600,
    tol_m: float = 15.0,
    interval_m: float = 10.0,
    weight: str = "length_m",
    seed: int = 42,
) -> dict:
    """Symmetric APLS between ground-truth ``gt`` and proposal ``prop`` graphs.

    Both graphs are densified to ≈``interval_m`` spacing first (so the score is
    independent of how finely each is noded). Returns a dict with the two one-way
    scores and their harmonic mean ``apls``.
    """
    if gt.number_of_nodes() == 0:
        return {"apls": 0.0, "apls_gt_to_prop": 0.0, "apls_prop_to_gt": 0.0,
                "n_samples": n_samples, "snap_tol_m": tol_m}

    gt = _densify(gt, interval_m)
    prop = _densify(prop, interval_m)
    lat0 = float(np.mean([gt.nodes[n]["y"] for n in gt.nodes]))
    lon0 = float(np.mean([gt.nodes[n]["x"] for n in gt.nodes]))

    gt_to_prop = _snap_map(gt, prop, lon0, lat0, tol_m)
    prop_to_gt = _snap_map(prop, gt, lon0, lat0, tol_m)
    a = _apls_oneway(gt, prop, gt_to_prop, n_samples, weight, seed)
    b = _apls_oneway(prop, gt, prop_to_gt, n_samples, weight, seed)
    harmonic = 0.0 if (a + b) == 0 else 2 * a * b / (a + b)
    return {
        "apls": round(harmonic, 4),
        "apls_gt_to_prop": round(a, 4),
        "apls_prop_to_gt": round(b, 4),
        "n_samples": n_samples,
        "snap_tol_m": tol_m,
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
    osm = ox.graph_from_bbox(
        north=north, south=south, east=east, west=west,
        network_type="drive", simplify=True, retain_all=True, truncate_by_edge=True,
    )
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

    result = apls(truth, prop, n_samples=n_samples, tol_m=tol_m)
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
