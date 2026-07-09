"""Contract tests on the committed ``data/sample`` artifacts (bugs.md §4/§5F).

The deployed dashboard's ONLY data source is the committed sample. Runtime
validation in ``app.py`` surfaces a broken sample to *users*; these tests catch
it in CI instead — a regenerated sample with a renamed column or a degenerate
edge fails here before it ships.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.pipeline.p2_graph.graph_io import graph_to_geojson, load_geojson_graph, save_geojson

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_GEOJSON = REPO_ROOT / "data" / "sample" / "panaji_demo_graph.geojson"
SAMPLE_CRITICALITY = REPO_ROOT / "data" / "sample" / "panaji_demo_criticality.csv"

# The §4 dashboard contract (mirrors app.py's runtime checks).
REQUIRED_CRITICALITY_COLUMNS = {"node_id", "betweenness", "rank", "is_critical", "x", "y"}


def test_sample_graph_loads_and_meets_contract():
    graph = load_geojson_graph(SAMPLE_GEOJSON)

    assert graph.number_of_nodes() > 0 and graph.number_of_edges() > 0

    for node, data in graph.nodes(data=True):
        assert "x" in data and "y" in data, f"node {node} lacks coordinates"

    for u, v, data in graph.edges(data=True):
        # length_m > 0 is also enforced by the loader itself; assert explicitly
        # so a loosened loader still fails this contract test.
        assert float(data["length_m"]) > 0, f"edge ({u}, {v}) has non-positive length_m"
        assert "is_bridged" in data, f"edge ({u}, {v}) lacks is_bridged"


def test_sample_criticality_csv_meets_contract():
    with SAMPLE_CRITICALITY.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows, "criticality CSV is empty"
    assert REQUIRED_CRITICALITY_COLUMNS <= set(rows[0].keys())

    for row in rows:
        b = float(row["betweenness"])
        assert 0.0 <= b <= 1.0, f"betweenness {b} out of [0, 1] for node {row['node_id']}"
        int(row["node_id"])  # ids parse as ints
        int(row["rank"])

    ranks = [int(r["rank"]) for r in rows]
    assert sorted(ranks) == list(range(1, len(rows) + 1)), "ranks are not 1..N"


def test_sample_graph_roundtrips_through_geojson(tmp_path):
    graph = load_geojson_graph(SAMPLE_GEOJSON)
    out = tmp_path / "roundtrip.geojson"
    save_geojson(graph, out)
    again = load_geojson_graph(out)

    assert again.number_of_nodes() == graph.number_of_nodes()
    assert again.number_of_edges() == graph.number_of_edges()

    # Attribute survival on a spot-checked edge and the feature count.
    # MultiGraph edges are keyed: graph.edges[u, v] is a {key: attrs} dict, so
    # pull the single (u, v, key) triple and address the reloaded edge by key.
    u, v, key, data = next(iter(graph.edges(data=True, keys=True)))
    reloaded = again.edges[u, v, key] if again.is_multigraph() else again.edges[u, v]
    assert "length_m" in reloaded and "is_bridged" in reloaded
    n_features = len(graph_to_geojson(graph)["features"])
    assert n_features == graph.number_of_nodes() + graph.number_of_edges()
