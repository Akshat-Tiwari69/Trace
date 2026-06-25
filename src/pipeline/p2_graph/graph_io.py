"""Read/write the healed graph in the §4 contract formats.

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

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx


def save_graphml(graph: "nx.Graph", path: Path) -> None:
    """Write ``graph`` to GraphML, JSON-encoding edge geometry to a string."""
    import networkx as nx

    out = graph.copy()
    for _, _, data in out.edges(data=True):
        if isinstance(data.get("geometry"), list):
            data["geometry"] = json.dumps(data["geometry"])
    for key in ("heal", "simplify"):  # graph-level metadata → JSON string
        if isinstance(out.graph.get(key), dict):
            out.graph[key] = json.dumps(out.graph[key])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(out, str(path))


def load_graphml(path: Path) -> "nx.Graph":
    """Read a GraphML graph back, decoding edge geometry from its JSON string."""
    import networkx as nx

    graph = nx.read_graphml(str(path), node_type=int)
    for _, _, data in graph.edges(data=True):
        geom = data.get("geometry")
        if isinstance(geom, str):
            data["geometry"] = json.loads(geom)
    for key in ("heal", "simplify"):  # decode graph-level metadata
        if isinstance(graph.graph.get(key), str):
            graph.graph[key] = json.loads(graph.graph[key])
    return graph


def graph_to_geojson(graph: "nx.Graph") -> dict:
    """Build a mixed Point+LineString ``FeatureCollection`` from the graph.

    Each feature carries ``feature_type`` ("node"/"edge") so the dashboard can
    style junctions and roads separately; node features expose ``betweenness`` /
    ``is_critical`` for the criticality heatmap, edges expose ``is_bridged`` so
    healed roads can be drawn distinctly (``docs/Design.md`` §1, honesty).
    """
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
                },
            }
        )

    for u, v, data in graph.edges(data=True):
        coords = data.get("geometry") or [
            [graph.nodes[u]["x"], graph.nodes[u]["y"]],
            [graph.nodes[v]["x"], graph.nodes[v]["y"]],
        ]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [rounded(c) for c in coords]},
                "properties": {
                    "feature_type": "edge",
                    "u": int(u),
                    "v": int(v),
                    "length_m": round(float(data.get("length_m", 0.0)), 3),
                    "is_bridged": bool(data.get("is_bridged", False)),
                    "edge_betweenness": float(data.get("edge_betweenness", 0.0)),
                },
            }
        )

    fc = {"type": "FeatureCollection", "features": features}
    # Carry build-time graph metadata (authoritative heal/simplify stats) so the
    # evaluator reports true numbers instead of re-deriving them from the graph.
    meta = {k: graph.graph[k] for k in ("heal", "simplify") if k in graph.graph}
    if meta:
        fc["meta"] = meta
    return fc


def save_geojson(graph: "nx.Graph", path: Path) -> None:
    """Write the graph as a GeoJSON FeatureCollection."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(graph_to_geojson(graph)))


def load_geojson_graph(path: Path) -> "nx.Graph":
    """Rebuild a NetworkX graph from a :func:`graph_to_geojson` FeatureCollection.

    The inverse of :func:`graph_to_geojson`: node Point features restore
    ``x, y, degree, type, betweenness, is_critical``; edge LineString features
    restore ``length_m, is_bridged, edge_betweenness``. Lets the committed
    ``data/sample/`` GeoJSON be re-analysed without the (gitignored) GraphML.
    """
    import networkx as nx

    fc = json.loads(Path(path).read_text())
    graph = nx.Graph()
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
            )
    for feat in fc["features"]:
        props = feat["properties"]
        if props.get("feature_type") == "edge":
            graph.add_edge(
                int(props["u"]),
                int(props["v"]),
                length_m=float(props.get("length_m", 0.0)),
                is_bridged=bool(props.get("is_bridged", False)),
                edge_betweenness=float(props.get("edge_betweenness", 0.0)),
            )
    return graph
