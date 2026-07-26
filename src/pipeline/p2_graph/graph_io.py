"""Read/write the healed graph in the Tracker §4 contract formats.

The graph is the central hand-off artifact (``docs/Tracker.md`` §4): P3 and P4
both consume it. We persist it two ways:

* **GraphML** (``{aoi}_graph.graphml``) — the canonical, loss-less form P3 reads.
  GraphML only stores scalar attributes, so the edge polyline ``geometry`` is
  serialised to a JSON string and restored on load.
* **GeoJSON** (``{aoi}_graph.geojson``) — a map-ready ``FeatureCollection`` of
  node Points and edge LineStrings (WGS84 lon/lat) that the Folium dashboard and
  the committed ``data/sample/`` set use directly.

Coordinates are expected in WGS84 lon/lat by the time this runs (call
``reproject_graph_to_wgs84`` after healing).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import networkx as nx


def atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    """Write to ``<path>.tmp`` via ``writer``, then atomically replace ``path``.

    A crash mid-write leaves the previous artifact intact (the dashboard reads
    these files live), instead of a truncated GraphML/GeoJSON/CSV. The temp file
    lives in the same directory so ``os.replace`` stays atomic (same filesystem).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # no-op after a successful replace


def _finite(value: object, label: str, source: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} in {source} must be a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} in {source} must be finite, got {value!r}")
    return number


def validate_graph_contract(graph: "nx.Graph", source: object = "in-memory graph") -> None:
    """Validate the P2 graph seam before writes and after loads.

    Nodes need finite ``x/y`` coordinates; edges need finite positive route
    lengths and geometry attached to their endpoint nodes (either orientation).
    The resulting map payload must also be strict JSON so browsers never receive
    Python's non-standard ``NaN``/``Infinity`` tokens.
    """
    for node, data in graph.nodes(data=True):
        _finite(data.get("x"), f"node {node!r} x", source)
        _finite(data.get("y"), f"node {node!r} y", source)

    multi = graph.is_multigraph()
    edges = graph.edges(data=True, keys=True) if multi else (
        (u, v, 0, data) for u, v, data in graph.edges(data=True)
    )
    for u, v, key, data in edges:
        length = _finite(data.get("length_m"), f"edge ({u}, {v}, {key}) length_m", source)
        if length <= 0.0:
            raise ValueError(
                f"edge ({u}, {v}, {key}) in {source} has length_m={length}; "
                "the graph contract requires a positive route weight"
            )
        geometry = data.get("geometry")
        if geometry is None:
            geometry = [
                [graph.nodes[u]["x"], graph.nodes[u]["y"]],
                [graph.nodes[v]["x"], graph.nodes[v]["y"]],
            ]
        if not isinstance(geometry, (list, tuple)) or len(geometry) < 2:
            raise ValueError(f"edge ({u}, {v}, {key}) in {source} needs at least two geometry points")
        points: list[tuple[float, float]] = []
        for index, point in enumerate(geometry):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(
                    f"edge ({u}, {v}, {key}) geometry point {index} in {source} must be [x, y]"
                )
            points.append((
                _finite(point[0], f"edge ({u}, {v}, {key}) geometry x", source),
                _finite(point[1], f"edge ({u}, {v}, {key}) geometry y", source),
            ))
        u_xy = (float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"]))
        v_xy = (float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"]))

        def close(left: tuple[float, float], right: tuple[float, float]) -> bool:
            return (math.isclose(left[0], right[0], rel_tol=0.0, abs_tol=1e-6)
                    and math.isclose(left[1], right[1], rel_tol=0.0, abs_tol=1e-6))

        aligned = ((close(points[0], u_xy) and close(points[-1], v_xy))
                   or (close(points[0], v_xy) and close(points[-1], u_xy)))
        if not aligned:
            raise ValueError(
                f"edge ({u}, {v}, {key}) geometry endpoints in {source} do not match its nodes"
            )

    try:
        json.dumps(graph_to_geojson(graph), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"graph in {source} is not browser-safe JSON: {exc}") from exc


def graph_contract_fingerprint(graph: "nx.Graph") -> str:
    """Hash every field shared by the GraphML and GeoJSON contracts."""
    validate_graph_contract(graph)
    nodes = sorted(
        (
            int(node), round(float(data["x"]), 6), round(float(data["y"]), 6),
            int(data.get("degree", 0)), str(data.get("type", "intersection")),
            float(data.get("betweenness", 0.0)), bool(data.get("is_critical", False)),
            bool(data.get("is_articulation", False)),
        )
        for node, data in graph.nodes(data=True)
    )
    edge_rows = graph.edges(data=True, keys=True) if graph.is_multigraph() else (
        (u, v, 0, data) for u, v, data in graph.edges(data=True)
    )
    edges = []
    for u, v, key, data in edge_rows:
        geometry = data.get("geometry") or [
            [graph.nodes[u]["x"], graph.nodes[u]["y"]],
            [graph.nodes[v]["x"], graph.nodes[v]["y"]],
        ]
        points = [
            (round(float(point[0]), 6), round(float(point[1]), 6))
            for point in geometry
        ]
        first_node = u if int(u) <= int(v) else v
        first_xy = (
            round(float(graph.nodes[first_node]["x"]), 6),
            round(float(graph.nodes[first_node]["y"]), 6),
        )
        if points and points[-1] == first_xy and points[0] != first_xy:
            points.reverse()
        edges.append((
            min(int(u), int(v)), max(int(u), int(v)), int(key),
            round(float(data["length_m"]), 3), bool(data.get("is_bridged", False)),
            bool(data.get("is_bridge", False)), float(data.get("edge_betweenness", 0.0)),
            (round(float(data["width_m"]), 3) if data.get("width_m") is not None else None),
            (round(float(data["confidence"]), 3) if data.get("confidence") is not None else None),
            tuple(points),
        ))
    edges.sort()
    payload = {
        "meta": {
            key: graph.graph[key]
            for key in ("heal", "simplify", "consolidate", "polyline", "provenance", "coordinate_frame")
            if key in graph.graph
        },
        "nodes": nodes,
        "edges": edges,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def graph_artifacts_match(graphml_path: Path, geojson_path: Path) -> bool:
    """Whether both artifacts validate and contain the same analyzed graph."""
    if not Path(graphml_path).exists() or not Path(geojson_path).exists():
        return False
    try:
        return graph_contract_fingerprint(load_graphml(graphml_path)) == graph_contract_fingerprint(
            load_geojson_graph(geojson_path)
        )
    except Exception:
        return False


def save_graphml(graph: "nx.Graph", path: Path) -> None:
    """Write ``graph`` to GraphML, JSON-encoding edge geometry to a string."""
    import networkx as nx

    validate_graph_contract(graph, Path(path))
    out = graph.copy()
    for _, _, data in out.edges(data=True):
        if isinstance(data.get("geometry"), list):
            data["geometry"] = json.dumps(data["geometry"])
    for key in ("heal", "simplify", "consolidate", "polyline", "provenance", "coordinate_frame"):  # graph-level metadata → JSON string
        if isinstance(out.graph.get(key), dict):
            out.graph[key] = json.dumps(out.graph[key])
    atomic_write(Path(path), lambda tmp: nx.write_graphml(out, str(tmp)))


def load_graphml(path: Path) -> "nx.Graph":
    """Read a GraphML graph back, decoding edge geometry from its JSON string.

    ``force_multigraph=True`` so the loaded type is always a MultiGraph, even
    when this particular file has no parallel edges — otherwise nx.read_graphml
    infers plain Graph/MultiGraph from the file's own ``<graph edgedefault>``
    tag, and callers branching on ``is_multigraph()`` would silently see a
    different type per file, violating the A37 MultiGraph contract.
    """
    import networkx as nx

    graph = nx.read_graphml(str(path), node_type=int, force_multigraph=True)
    for _, _, data in graph.edges(data=True):
        geom = data.get("geometry")
        if isinstance(geom, str):
            data["geometry"] = json.loads(geom)
    for key in ("heal", "simplify", "consolidate", "polyline", "provenance", "coordinate_frame"):  # decode graph-level metadata
        if isinstance(graph.graph.get(key), str):
            graph.graph[key] = json.loads(graph.graph[key])
    validate_graph_contract(graph, Path(path))
    return graph


def graph_to_geojson(graph: "nx.Graph") -> dict:
    """Build a mixed Point+LineString ``FeatureCollection`` from the graph.

    Each feature carries ``feature_type`` ("node"/"edge") so the dashboard can
    style junctions and roads separately; node features expose ``betweenness`` /
    ``is_critical`` for the criticality heatmap, edges expose ``is_bridged`` so
    healed roads can be drawn distinctly, preserving the design honesty rule.

    On a MultiGraph each keyed edge becomes its own LineString feature carrying
    an ``edge_key`` property, so parallel branches (loops, dual carriageways)
    survive the GeoJSON round-trip instead of being collapsed.
    """
    multi = graph.is_multigraph()
    features: list[dict] = []

    def rounded(coord: list) -> list:
        """6-dp lon/lat (~0.1 m) — enough for mapping, keeps the sample small."""
        return [round(float(coord[0]), 6), round(float(coord[1]), 6)]

    for node_id, data in graph.nodes(data=True):
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": rounded([data["x"], data["y"]])},
                "properties": {
                    "feature_type": "node",
                    "node_id": int(node_id),
                    "degree": int(data.get("degree", 0)),
                    "type": data.get("type", "intersection"),
                    "betweenness": float(data.get("betweenness", 0.0)),
                    "is_critical": bool(data.get("is_critical", False)),
                    "is_articulation": bool(data.get("is_articulation", False)),
                },
            }
        )

    # keys=True only on a MultiGraph; the key is recorded as ``edge_key`` so a
    # parallel edge is distinguishable on reload (load_geojson_graph rebuilds a
    # MultiGraph and re-keys deterministically by insertion order).
    edges = graph.edges(data=True, keys=True) if multi else (
        (u, v, 0, d) for u, v, d in graph.edges(data=True)
    )
    for u, v, key, data in edges:
        coords = data.get("geometry") or [
            [graph.nodes[u]["x"], graph.nodes[u]["y"]],
            [graph.nodes[v]["x"], graph.nodes[v]["y"]],
        ]
        props = {
            "feature_type": "edge",
            "u": int(u),
            "v": int(v),
            "edge_key": int(key),
            "length_m": round(float(data.get("length_m", 0.0)), 3),
            "is_bridged": bool(data.get("is_bridged", False)),
            "is_bridge": bool(data.get("is_bridge", False)),
            "edge_betweenness": float(data.get("edge_betweenness", 0.0)),
        }
        if data.get("width_m") is not None:  # optional road width
            props["width_m"] = round(float(data["width_m"]), 3)
        if data.get("confidence") is not None:  # optional mean probability
            props["confidence"] = round(float(data["confidence"]), 3)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [rounded(c) for c in coords]},
                "properties": props,
            }
        )

    fc = {"type": "FeatureCollection", "features": features}
    # Carry build-time graph metadata (authoritative heal/simplify stats) so the
    # evaluator reports true numbers instead of re-deriving them from the graph.
    meta = {k: graph.graph[k] for k in ("heal", "simplify", "consolidate", "polyline", "provenance", "coordinate_frame") if k in graph.graph}
    if meta:
        fc["meta"] = meta
    return fc


def save_geojson(graph: "nx.Graph", path: Path) -> None:
    """Write the graph as a GeoJSON FeatureCollection (atomically)."""
    validate_graph_contract(graph, Path(path))
    payload = json.dumps(graph_to_geojson(graph), allow_nan=False)
    atomic_write(Path(path), lambda tmp: tmp.write_text(payload, encoding="utf-8"))


def save_graph_pair(graph: "nx.Graph", graphml_path: Path, geojson_path: Path) -> None:
    """Pre-serialize both artifacts and restore the last good pair on failure."""
    graphml_path, geojson_path = Path(graphml_path), Path(geojson_path)
    validate_graph_contract(graph)
    graphml_stage = graphml_path.with_name(graphml_path.name + ".pair-stage")
    geojson_stage = geojson_path.with_name(geojson_path.name + ".pair-stage")
    graphml_backup = graphml_path.with_name(graphml_path.name + ".pair-backup")
    geojson_backup = geojson_path.with_name(geojson_path.name + ".pair-backup")
    originals = ((graphml_path, graphml_backup), (geojson_path, geojson_backup))
    replaced: list[Path] = []
    restore_failed = False
    try:
        save_graphml(graph, graphml_stage)
        save_geojson(graph, geojson_stage)
        if not graph_artifacts_match(graphml_stage, geojson_stage):
            raise RuntimeError("staged GraphML and GeoJSON do not describe the same routing graph")
        for path, backup in originals:
            if path.exists():
                shutil.copy2(path, backup)
        os.replace(graphml_stage, graphml_path)
        replaced.append(graphml_path)
        os.replace(geojson_stage, geojson_path)
        replaced.append(geojson_path)
    except BaseException as original_error:
        for path, backup in reversed(originals):
            if path in replaced:
                try:
                    if backup.exists():
                        os.replace(backup, path)
                    else:
                        path.unlink(missing_ok=True)
                except OSError:
                    restore_failed = True
        if restore_failed:
            raise RuntimeError(
                "graph pair replacement and rollback failed; .pair-backup files were retained"
            ) from original_error
        raise
    finally:
        graphml_stage.unlink(missing_ok=True)
        geojson_stage.unlink(missing_ok=True)
        if not restore_failed:
            graphml_backup.unlink(missing_ok=True)
            geojson_backup.unlink(missing_ok=True)


def load_geojson_graph(path: Path) -> "nx.MultiGraph":
    """Rebuild a NetworkX MultiGraph from a :func:`graph_to_geojson` FeatureCollection.

    The inverse of :func:`graph_to_geojson`: node Point features restore
    ``x, y, degree, type, betweenness, is_critical``; edge LineString features
    restore ``length_m, is_bridged, edge_betweenness`` (and ``width_m`` /
    ``confidence`` when present), including the LineString geometry. Returns a
    **MultiGraph** so parallel branches keep their persisted ``edge_key`` values.
    Lets the committed
    ``data/sample/`` GeoJSON be re-analysed without the (gitignored) GraphML.
    """
    import networkx as nx

    fc = json.loads(Path(path).read_text())
    graph = nx.MultiGraph()
    for key, value in fc.get("meta", {}).items():  # restore build-time metadata
        graph.graph[key] = value
    for feat in fc["features"]:
        props = feat["properties"]
        if props.get("feature_type") == "node":
            lon, lat = feat["geometry"]["coordinates"]
            graph.add_node(
                int(props["node_id"]),
                x=float(lon),
                y=float(lat),
                degree=int(props.get("degree", 0)),
                type=props.get("type", "intersection"),
                betweenness=float(props.get("betweenness", 0.0)),
                is_critical=bool(props.get("is_critical", False)),
                is_articulation=bool(props.get("is_articulation", False)),
            )
    for feat in fc["features"]:
        props = feat["properties"]
        if props.get("feature_type") == "edge":
            attrs = dict(
                length_m=float(props.get("length_m", 0.0)),
                geometry=feat["geometry"]["coordinates"],
                is_bridged=bool(props.get("is_bridged", False)),
                is_bridge=bool(props.get("is_bridge", False)),
                edge_betweenness=float(props.get("edge_betweenness", 0.0)),
            )
            if props.get("width_m") is not None:  # optional road width
                attrs["width_m"] = float(props["width_m"])
            if props.get("confidence") is not None:  # optional mean prob (A37)
                attrs["confidence"] = float(props["confidence"])
            edge_key = props.get("edge_key")
            if edge_key is None:
                graph.add_edge(int(props["u"]), int(props["v"]), **attrs)
            else:
                graph.add_edge(int(props["u"]), int(props["v"]),
                               key=int(edge_key), **attrs)
    validate_graph_contract(graph, Path(path))
    return graph
