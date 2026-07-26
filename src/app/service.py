"""Framework-neutral application service for the web and API surfaces."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
from shapely.geometry import shape

from src.pipeline.p3_analysis.resilience import resilience_index


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "data" / "sample"
SAMPLE_AOI = "panaji_demo"
DEFAULT_EFFICIENCY_SEED = 42


def _finite(value: object, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _unit_interval(value: object, field: str) -> float:
    number = _finite(value, field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return number


def _truthy(value: object) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def _json_file(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _graph_from_geojson(payload: dict) -> nx.MultiGraph:
    if payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
        raise ValueError("sample graph must be a GeoJSON FeatureCollection")

    graph = nx.MultiGraph()
    edge_rows: list[tuple[dict, object]] = []
    for feature in payload["features"]:
        properties = feature.get("properties") or {}
        geometry_payload = feature.get("geometry")
        if not isinstance(properties, dict) or not isinstance(geometry_payload, dict):
            raise ValueError("sample graph contains an invalid feature")
        geometry = shape(geometry_payload)
        if properties.get("feature_type") == "node":
            node_id = int(properties["node_id"])
            graph.add_node(
                node_id,
                x=_finite(geometry.x, "node.x"),
                y=_finite(geometry.y, "node.y"),
                betweenness=_finite(properties.get("betweenness", 0.0), "betweenness"),
                is_critical=_truthy(properties.get("is_critical")),
                is_articulation=_truthy(properties.get("is_articulation")),
            )
        elif properties.get("feature_type") == "edge":
            edge_rows.append((properties, geometry))

    for properties, geometry in edge_rows:
        u, v = int(properties["u"]), int(properties["v"])
        if u not in graph or v not in graph:
            raise ValueError(f"edge references missing node: {u}-{v}")
        length = _finite(properties["length_m"], "length_m")
        if length <= 0:
            raise ValueError("edge length_m must be positive")
        graph.add_edge(
            u,
            v,
            key=int(properties.get("edge_key", graph.number_of_edges(u, v))),
            length_m=length,
            geometry=geometry,
            is_bridged=_truthy(properties.get("is_bridged")),
            edge_betweenness=_finite(
                properties.get("edge_betweenness", 0.0), "edge_betweenness"
            ),
        )
    if graph.number_of_nodes() < 2 or graph.number_of_edges() < 1:
        raise ValueError("sample graph is empty")
    return graph


def graph_from_features(features) -> nx.MultiGraph:
    """Build a routable MultiGraph from the legacy GeoDataFrame artifact view."""
    nodes = features[features["feature_type"] == "node"]
    edges = features[features["feature_type"] == "edge"]
    if nodes.empty or edges.empty:
        raise ValueError("GeoJSON must contain node and edge features")
    graph = nx.MultiGraph()
    for _, node in nodes.iterrows():
        graph.add_node(
            int(node["node_id"]),
            x=_finite(node.geometry.x, "node.x"),
            y=_finite(node.geometry.y, "node.y"),
            betweenness=_finite(node.get("betweenness", 0.0), "betweenness"),
            is_critical=_truthy(node.get("is_critical")),
            is_articulation=_truthy(node.get("is_articulation")),
        )
    for _, edge in edges.iterrows():
        length = _finite(edge["length_m"], "length_m")
        if length <= 0:
            raise ValueError("edge length_m must be positive")
        graph.add_edge(
            int(edge["u"]),
            int(edge["v"]),
            key=int(edge.get("edge_key", graph.number_of_edges(int(edge["u"]), int(edge["v"])))),
            length_m=length,
            geometry=edge.geometry,
            is_bridged=_truthy(edge.get("is_bridged")),
        )
    return graph


def _min_edge_data(graph: nx.MultiGraph, u: int, v: int) -> dict:
    return min(graph[u][v].values(), key=lambda data: float(data["length_m"]))


def path_length(graph: nx.MultiGraph, path: tuple[int, ...] | list[int]) -> float:
    """Return a MultiGraph path length using Dijkstra's shortest parallel edge."""
    return sum(
        float(_min_edge_data(graph, start, end)["length_m"])
        for start, end in zip(path, path[1:])
    )


def representative_reroute(graph: nx.MultiGraph, disabled_node: int) -> dict | None:
    """Return the worst neighbour-pair detour, preferring disconnection."""
    if disabled_node not in graph:
        return None
    neighbours = list(graph.neighbors(disabled_node))
    if len(neighbours) < 2:
        return None
    perturbed = graph.copy()
    perturbed.remove_node(disabled_node)
    finite: list[dict] = []
    disconnected: list[dict] = []

    for origin, destination in combinations(neighbours, 2):
        baseline_path = tuple(nx.shortest_path(graph, origin, destination, weight="length_m"))
        if disabled_node not in baseline_path:
            continue
        baseline_length = path_length(graph, baseline_path)
        if baseline_length <= 0:
            continue
        try:
            rerouted_path = tuple(
                nx.shortest_path(perturbed, origin, destination, weight="length_m")
            )
        except nx.NetworkXNoPath:
            disconnected.append({
                "origin": int(origin),
                "destination": int(destination),
                "baseline_path": list(baseline_path),
                "rerouted_path": None,
                "baseline_length_m": baseline_length,
                "rerouted_length_m": None,
                "travel_time_delta_pct": None,
                "delay_segments": [],
                "disconnected": True,
            })
            continue

        rerouted_length = path_length(perturbed, rerouted_path)
        baseline_edges = {
            tuple(sorted((start, end))) for start, end in zip(baseline_path, baseline_path[1:])
        }
        delay_segments = []
        for start, end in zip(rerouted_path, rerouted_path[1:]):
            if tuple(sorted((start, end))) not in baseline_edges:
                contribution = (
                    100.0
                    * float(_min_edge_data(graph, start, end)["length_m"])
                    / baseline_length
                )
                delay_segments.append({
                    "edge": [int(start), int(end)],
                    "contribution_pct": contribution,
                })
        finite.append({
            "origin": int(origin),
            "destination": int(destination),
            "baseline_path": list(baseline_path),
            "rerouted_path": list(rerouted_path),
            "baseline_length_m": baseline_length,
            "rerouted_length_m": rerouted_length,
            "travel_time_delta_pct": 100.0 * (rerouted_length / baseline_length - 1.0),
            "delay_segments": sorted(
                delay_segments,
                key=lambda item: item["contribution_pct"],
                reverse=True,
            )[:5],
            "disconnected": False,
            "disconnected_pairs": 0,
        })
    if disconnected:
        return {**disconnected[0], "disconnected_pairs": len(disconnected)}
    return max(finite, key=lambda route: route["travel_time_delta_pct"]) if finite else None


def _criticality(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "node_id": int(row["node_id"]),
                "betweenness": _finite(row["betweenness"], "betweenness"),
                "rank": int(row["rank"]),
                "is_critical": _truthy(row["is_critical"]),
                "is_articulation": _truthy(row.get("is_articulation")),
                "x": _finite(row["x"], "x"),
                "y": _finite(row["y"], "y"),
            })
    return sorted(rows, key=lambda row: row["rank"])


def _resilience_curve(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {"n_removed": int(row["n_removed"])}
            for key, value in row.items():
                if key == "n_removed":
                    continue
                if key == "efficiency_method":
                    parsed[key] = value.strip().lower()
                elif key in {"efficiency_k", "efficiency_seed", "random_seed"}:
                    parsed[key] = int(value) if value.strip() else None
                elif key.endswith(("resilience_index", "largest_cc_fraction")):
                    parsed[key] = _unit_interval(value, key)
                else:
                    parsed[key] = _finite(value, key)
            rows.append(parsed)
    return rows


@dataclass(frozen=True)
class AoiDataset:
    graph: nx.MultiGraph
    graph_bytes: bytes
    graph_etag: str
    critical_nodes: list[dict]
    resilience_curve: list[dict]
    bounds: tuple[float, float, float, float]
    baseline_efficiency: float
    simulation_baseline_efficiency: float
    simulation_sample_size: int | None
    simulation_seed: int | None
    evidence: dict


@lru_cache(maxsize=1)
def sample_dataset() -> AoiDataset:
    graph_path = SAMPLE_DIR / "panaji_demo_graph.geojson"
    graph_bytes = graph_path.read_bytes()
    graph = _graph_from_geojson(json.loads(graph_bytes))
    xs = [float(data["x"]) for _, data in graph.nodes(data=True)]
    ys = [float(data["y"]) for _, data in graph.nodes(data=True)]
    resilience_curve = _resilience_curve(SAMPLE_DIR / "panaji_demo_resilience.csv")
    if not resilience_curve or resilience_curve[0]["n_removed"] != 0:
        raise ValueError("sample resilience curve must begin at zero removals")
    baseline = _finite(resilience_curve[0]["targeted_efficiency"], "targeted_efficiency")
    if baseline <= 0 or not math.isfinite(baseline):
        raise ValueError("sample graph baseline efficiency is invalid")
    method = resilience_curve[0].get("efficiency_method", "exact")
    if method not in {"exact", "sampled"}:
        raise ValueError("sample resilience efficiency_method is invalid")
    sample_size = resilience_curve[0].get("efficiency_k") if method == "sampled" else None
    seed = resilience_curve[0].get("efficiency_seed") if sample_size else None
    if sample_size is not None and (sample_size <= 0 or seed is None):
        raise ValueError("sample resilience sampling metadata is incomplete")
    return AoiDataset(
        graph=graph,
        graph_bytes=graph_bytes,
        graph_etag=f'"{hashlib.sha256(graph_bytes).hexdigest()}"',
        critical_nodes=_criticality(SAMPLE_DIR / "panaji_demo_criticality.csv"),
        resilience_curve=resilience_curve,
        bounds=(min(xs), min(ys), max(xs), max(ys)),
        baseline_efficiency=baseline,
        simulation_baseline_efficiency=baseline,
        simulation_sample_size=sample_size,
        simulation_seed=seed,
        evidence=_json_file(SAMPLE_DIR / "panaji_demo_evidence_manifest.json"),
    )


def aoi_summary(aoi: str) -> dict:
    if aoi != SAMPLE_AOI:
        raise KeyError(aoi)
    dataset = sample_dataset()
    return {
        "aoi": SAMPLE_AOI,
        "label": "Panaji, Goa",
        "coordinate_system": "EPSG:4326",
        "bounds": list(dataset.bounds),
        "node_count": dataset.graph.number_of_nodes(),
        "edge_count": dataset.graph.number_of_edges(),
        "critical_count": sum(row["is_critical"] for row in dataset.critical_nodes),
        "baseline_efficiency": dataset.baseline_efficiency,
        "graph_url": f"/api/v1/aois/{SAMPLE_AOI}/graph",
        "critical_nodes": dataset.critical_nodes,
        "resilience_curve": dataset.resilience_curve,
        "evidence": dataset.evidence,
    }


@lru_cache(maxsize=256)
def _simulate_cached(aoi: str, removed_node_ids: tuple[int, ...]) -> dict:
    if aoi != SAMPLE_AOI:
        raise KeyError(aoi)
    dataset = sample_dataset()
    removed = list(removed_node_ids)
    unknown = [node for node in removed if node not in dataset.graph]
    if unknown:
        raise ValueError("Unknown node IDs: " + ", ".join(map(str, unknown)))
    kwargs = {
        "baseline_efficiency": dataset.simulation_baseline_efficiency,
        "k": dataset.simulation_sample_size,
    }
    if "seed" in inspect.signature(resilience_index).parameters:
        kwargs["seed"] = dataset.simulation_seed or DEFAULT_EFFICIENCY_SEED
    metrics = resilience_index(dataset.graph, removed, **kwargs)
    ri = _unit_interval(metrics["resilience_index"], "resilience_index")
    active_nodes = [node for node in dataset.graph if node not in removed]
    active = dataset.graph.subgraph(active_nodes)
    active_lcc = (
        max(map(len, nx.connected_components(active))) / len(active_nodes)
        if active_nodes
        else 0.0
    )
    return {
        "aoi": SAMPLE_AOI,
        "removed_node_ids": removed,
        "baseline_node_count": dataset.graph.number_of_nodes(),
        "baseline_efficiency": _finite(metrics["baseline_efficiency"], "baseline_efficiency"),
        "perturbed_efficiency": _finite(metrics["perturbed_efficiency"], "perturbed_efficiency"),
        "resilience_index": ri,
        "efficiency_loss": 1.0 - ri,
        "largest_cc_fraction": _unit_interval(
            metrics["largest_cc_fraction"], "largest_cc_fraction"
        ),
        "active_largest_cc_fraction": active_lcc,
        "efficiency_method": "sampled" if dataset.simulation_sample_size else "exact",
        "efficiency_sample_size": dataset.simulation_sample_size,
        "efficiency_seed": dataset.simulation_seed,
        "representative_route": (
            representative_reroute(dataset.graph, removed[0]) if len(removed) == 1 else None
        ),
    }


def simulate(aoi: str, removed_node_ids: list[int]) -> dict:
    """Return a cached, deterministic failure simulation."""
    key = tuple(sorted(set(removed_node_ids)))
    return dict(_simulate_cached(aoi, key))


simulate.cache_clear = _simulate_cached.cache_clear  # type: ignore[attr-defined]
simulate.cache_info = _simulate_cached.cache_info  # type: ignore[attr-defined]


def _json_value(value: Any) -> Any:
    """Convert pandas/numpy scalars without importing either at API startup."""
    return value.item() if hasattr(value, "item") else value


def analysis_payload(result: Any, job_id: str) -> dict:
    """Serialize the stable, public portion of a queued analysis result."""
    base = f"/api/v1/analyses/{job_id}"
    criticality = [
        {key: _json_value(value) for key, value in row.items()}
        for row in result.criticality.to_dict(orient="records")
    ]
    return {
        "id": job_id,
        "job_id": job_id,
        "status": "done",
        "n_nodes": int(result.n_nodes),
        "n_edges": int(result.n_edges),
        "resolution_m": _finite(result.resolution_m, "resolution_m"),
        "resilience_index": _unit_interval(result.resilience_index, "resilience_index"),
        "top_node": None if result.top_node is None else int(result.top_node),
        "summary": {key: _json_value(value) for key, value in result.summary.items()},
        "criticality": criticality,
        "graph_url": f"{base}/graph",
        "exports": {
            "json": f"{base}/result",
            "geojson": f"{base}/graph",
        },
    }


def _edge_coordinates(graph: nx.MultiGraph, u: int, v: int, geometry: Any) -> list[list[float]]:
    if hasattr(geometry, "coords"):
        coordinates = list(geometry.coords)
    elif isinstance(geometry, (list, tuple)):
        coordinates = geometry
    else:
        coordinates = [
            (graph.nodes[u]["x"], graph.nodes[u]["y"]),
            (graph.nodes[v]["x"], graph.nodes[v]["y"]),
        ]
    return [[_finite(x, "edge.x"), _finite(y, "edge.y")] for x, y, *_ in coordinates]


def analysis_graph_geojson(result: Any) -> dict:
    """Return an uploaded graph in its honest image-space metre coordinates."""
    graph = result.graph
    features: list[dict] = []
    for node_id, data in graph.nodes(data=True):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [_finite(data["x"], "node.x"), _finite(data["y"], "node.y")],
            },
            "properties": {
                "feature_type": "node",
                "node_id": int(node_id),
                "betweenness": _finite(data.get("betweenness", 0.0), "betweenness"),
                "is_critical": bool(data.get("is_critical", False)),
                "is_articulation": bool(data.get("is_articulation", False)),
            },
        })
    for u, v, key, data in graph.edges(keys=True, data=True):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": _edge_coordinates(graph, u, v, data.get("geometry")),
            },
            "properties": {
                "feature_type": "edge",
                "u": int(u),
                "v": int(v),
                "edge_key": int(key),
                "length_m": _finite(data["length_m"], "length_m"),
                "is_bridged": bool(data.get("is_bridged", False)),
                "edge_betweenness": _finite(
                    data.get("edge_betweenness", 0.0), "edge_betweenness"
                ),
            },
        })
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "coordinate_system": "image-space metres",
            "resolution_m": _finite(result.resolution_m, "resolution_m"),
        },
    }
