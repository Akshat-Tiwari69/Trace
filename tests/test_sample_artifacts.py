"""Contract tests on the committed ``data/sample`` artifacts (Tracker §4).

The deployed dashboard's ONLY data source is the committed sample. Runtime
validation in ``app.py`` surfaces a broken sample to *users*; these tests catch
it in CI instead — a regenerated sample with a renamed column or a degenerate
edge fails here before it ships.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from src.pipeline.p2_graph.graph_io import graph_to_geojson, load_geojson_graph, save_geojson

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample"
SAMPLE_GEOJSON = SAMPLE_DIR / "panaji_demo_graph.geojson"
SAMPLE_CRITICALITY = SAMPLE_DIR / "panaji_demo_criticality.csv"
EVIDENCE_MANIFEST = SAMPLE_DIR / "panaji_demo_evidence_manifest.json"

# The Tracker §4 dashboard contract (mirrors app.py's runtime checks).
REQUIRED_CRITICALITY_COLUMNS = {"node_id", "betweenness", "rank", "is_critical", "x", "y"}


def _artifact_sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix in {".csv", ".geojson", ".json"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def test_sample_evidence_manifest_matches_committed_artifacts():
    manifest = json.loads(EVIDENCE_MANIFEST.read_text(encoding="utf-8"))
    source = manifest["input"]
    artifacts = manifest["artifacts"]

    assert source == {
        "path": SAMPLE_GEOJSON.name,
        "sha256": _artifact_sha256(SAMPLE_GEOJSON),
    }
    assert set(artifacts) == {
        "panaji_demo_flood_curve.png",
        "panaji_demo_graph_eval.json",
        "panaji_demo_percolation.json",
        "panaji_demo_resilience.csv",
        "panaji_demo_resilience_curve.png",
    }
    for name, record in artifacts.items():
        expected_sha256 = _artifact_sha256(SAMPLE_DIR / name)
        assert record["sha256"] == expected_sha256
        assert record["command"].startswith("python -m src.pipeline.")


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
