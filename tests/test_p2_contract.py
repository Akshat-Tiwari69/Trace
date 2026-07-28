"""P2 trust-boundary and paired-artifact contract regressions."""

from __future__ import annotations

import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from PIL import Image


def _valid_graph() -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.graph["coordinate_frame"] = {
        "coordinates": "wgs84",
        "crs": "EPSG:4326",
        "axis_order": "x,y",
        "length_unit": "metre",
    }
    graph.add_node(0, x=72.8, y=19.1)
    graph.add_node(1, x=72.801, y=19.1)
    graph.add_edge(
        0,
        1,
        key=7,
        length_m=105.0,
        geometry=[[72.8, 19.1], [72.8005, 19.1002], [72.801, 19.1]],
        is_bridged=False,
    )
    return graph


def test_graph_contract_rejects_non_finite_coordinates_and_browser_json():
    from src.pipeline.p2_graph.graph_io import validate_graph_contract

    graph = _valid_graph()
    graph.nodes[0]["x"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_graph_contract(graph)

    graph = _valid_graph()
    graph.graph["coordinate_frame"]["resolution_m"] = float("inf")
    with pytest.raises(ValueError, match="browser-safe JSON"):
        validate_graph_contract(graph)


def test_graph_contract_rejects_non_positive_length_and_detached_geometry():
    from src.pipeline.p2_graph.graph_io import validate_graph_contract

    graph = _valid_graph()
    graph.edges[0, 1, 7]["length_m"] = 0.0
    with pytest.raises(ValueError, match="length_m"):
        validate_graph_contract(graph)

    graph = _valid_graph()
    graph.edges[0, 1, 7]["geometry"][0] = [0.0, 0.0]
    with pytest.raises(ValueError, match="geometry endpoints"):
        validate_graph_contract(graph)


def test_graph_contract_accepts_reversed_edge_geometry():
    from src.pipeline.p2_graph.graph_io import validate_graph_contract

    graph = _valid_graph()
    graph.edges[0, 1, 7]["geometry"].reverse()
    validate_graph_contract(graph)


def test_coordinate_frame_roundtrips_and_paired_artifacts_must_match(tmp_path):
    from src.pipeline.p2_graph.graph_io import (
        graph_artifacts_match,
        load_geojson_graph,
        load_graphml,
        save_geojson,
        save_graphml,
    )

    graph = _valid_graph()
    graphml = tmp_path / "graph.graphml"
    geojson = tmp_path / "graph.geojson"
    save_graphml(graph, graphml)
    save_geojson(graph, geojson)

    assert load_graphml(graphml).graph["coordinate_frame"] == graph.graph["coordinate_frame"]
    assert load_geojson_graph(geojson).graph["coordinate_frame"] == graph.graph["coordinate_frame"]
    assert graph_artifacts_match(graphml, geojson)

    payload = json.loads(geojson.read_text())
    node = next(f for f in payload["features"] if f["properties"].get("node_id") == 0)
    edge = next(f for f in payload["features"] if f["properties"].get("feature_type") == "edge")
    node["geometry"]["coordinates"][0] += 0.01
    edge["geometry"]["coordinates"][0][0] += 0.01
    geojson.write_text(json.dumps(payload))
    assert not graph_artifacts_match(graphml, geojson)

    save_geojson(graph, geojson)
    payload = json.loads(geojson.read_text())
    edge = next(f for f in payload["features"] if f["properties"].get("feature_type") == "edge")
    edge["geometry"]["coordinates"][1][1] += 0.001
    geojson.write_text(json.dumps(payload))
    assert not graph_artifacts_match(graphml, geojson)

    save_geojson(graph, geojson)
    payload = json.loads(geojson.read_text())
    edge = next(f for f in payload["features"] if f["properties"].get("feature_type") == "edge")
    edge["properties"]["length_m"] += 1.0
    geojson.write_text(json.dumps(payload))
    assert not graph_artifacts_match(graphml, geojson)

    save_geojson(graph, geojson)
    payload = json.loads(geojson.read_text())
    edge = next(f for f in payload["features"] if f["properties"].get("feature_type") == "edge")
    edge["properties"]["is_bridged"] = True
    geojson.write_text(json.dumps(payload))
    assert not graph_artifacts_match(graphml, geojson)

    save_geojson(graph, geojson)
    payload = json.loads(geojson.read_text())
    node = next(f for f in payload["features"] if f["properties"].get("feature_type") == "node")
    node["properties"]["betweenness"] = 0.5
    geojson.write_text(json.dumps(payload))
    assert not graph_artifacts_match(graphml, geojson)


def test_graph_pair_validation_preserves_last_good_artifacts(tmp_path):
    from src.pipeline.p2_graph.graph_io import save_graph_pair

    graphml = tmp_path / "graph.graphml"
    geojson = tmp_path / "graph.geojson"
    save_graph_pair(_valid_graph(), graphml, geojson)
    before = graphml.read_bytes(), geojson.read_bytes()

    invalid = _valid_graph()
    invalid.edges[0, 1, 7]["length_m"] = 0.0
    with pytest.raises(ValueError, match="positive route weight"):
        save_graph_pair(invalid, graphml, geojson)
    assert (graphml.read_bytes(), geojson.read_bytes()) == before


def test_graph_pair_rolls_back_when_second_replace_fails(tmp_path, monkeypatch):
    from src.pipeline.p2_graph import graph_io

    graphml = tmp_path / "graph.graphml"
    geojson = tmp_path / "graph.geojson"
    graph_io.save_graph_pair(_valid_graph(), graphml, geojson)
    before = graphml.read_bytes(), geojson.read_bytes()
    updated = _valid_graph()
    updated.edges[0, 1, 7]["length_m"] = 106.0
    real_replace = os.replace
    failed = False

    def fail_geojson_once(source, destination):
        nonlocal failed
        if Path(destination) == geojson and not failed:
            failed = True
            raise OSError("simulated second replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(graph_io.os, "replace", fail_geojson_once)
    with pytest.raises(OSError, match="second replace"):
        graph_io.save_graph_pair(updated, graphml, geojson)
    assert (graphml.read_bytes(), geojson.read_bytes()) == before
    assert not list(tmp_path.glob("*.pair-*"))


def test_mask_loader_accepts_binary_encodings_and_rejects_grayscale(tmp_path):
    from src.pipeline.p2_graph.build_graph import _load_mask

    valid = tmp_path / "valid.png"
    Image.fromarray(np.array([[0, 255], [1, 0]], dtype=np.uint8)).save(valid)
    assert set(np.unique(_load_mask(valid))) == {0, 1}

    invalid = tmp_path / "invalid.png"
    Image.fromarray(np.array([[0, 128], [255, 0]], dtype=np.uint8)).save(invalid)
    with pytest.raises(ValueError, match="binary"):
        _load_mask(invalid)


def test_alignment_manifest_requires_complete_matching_grid(tmp_path):
    from src.pipeline.p2_graph.build_graph import _load_alignment

    path = tmp_path / "manifest.json"
    valid = {
        "crs": "EPSG:32643",
        "transform": [1.0, 0.0, 100.0, 0.0, -1.0, 200.0],
        "resolution_m": 1.0,
        "width": 3,
        "height": 2,
    }
    path.write_text(json.dumps(valid))
    transform, crs = _load_alignment(path, (2, 3))
    assert crs == "EPSG:32643" and tuple(transform)[:6] == tuple(valid["transform"])

    missing = dict(valid)
    missing.pop("crs")
    path.write_text(json.dumps(missing))
    with pytest.raises(ValueError, match="missing"):
        _load_alignment(path, (2, 3))

    path.write_text(json.dumps(valid | {"width": 4}))
    with pytest.raises(ValueError, match="dimensions"):
        _load_alignment(path, (2, 3))

    path.write_text(json.dumps(valid | {"transform": [True, 0, 100, 0, -1, 200]}))
    with pytest.raises(ValueError, match="finite transform"):
        _load_alignment(path, (2, 3))

    path.write_text(json.dumps(valid | {"transform": [0, 0, 0, 0, 0, 0]}))
    with pytest.raises(ValueError, match="invertible"):
        _load_alignment(path, (2, 3))

    path.write_text(json.dumps(valid | {"resolution_m": True}))
    with pytest.raises(ValueError, match="resolution_m"):
        _load_alignment(path, (2, 3))


def test_probability_map_must_match_mask_before_graph_write(tmp_path):
    from src.pipeline.p2_graph.build_graph import _load_prob, build_graph
    from src.pipeline.p2_graph.config import GraphConfig

    prob = tmp_path / "prob.png"
    Image.fromarray(np.zeros((3, 4), dtype=np.uint8)).save(prob)
    with pytest.raises(ValueError, match="dimensions"):
        _load_prob(prob, (4, 4))

    interim = tmp_path / "interim"
    (interim / "shape").mkdir(parents=True)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(interim / "shape_mask.png")
    Image.fromarray(np.zeros((3, 4), dtype=np.uint8)).save(interim / "shape" / "prob.png")
    cfg = GraphConfig(aoi="shape", interim_dir=interim, processed_dir=tmp_path / "processed")
    with pytest.raises(ValueError, match="dimensions"):
        build_graph(cfg)
    assert not cfg.graphml_path.exists() and not cfg.geojson_path.exists()


def test_destructive_consolidation_is_opt_in_in_both_configs():
    from src.pipeline.config import PipelineConfig
    from src.pipeline.p2_graph.config import GraphConfig

    assert GraphConfig(aoi="defaults").consolidate is False
    assert PipelineConfig(aoi="defaults").consolidate is False
    assert PipelineConfig(aoi="opt-in", consolidate=True).graph_config().consolidate is True


def test_graph_config_rejects_non_finite_or_impossible_values():
    from src.pipeline.p2_graph.config import GraphConfig

    invalid = {
        "gap_max_m": float("nan"),
        "angle_max_deg": 181.0,
        "angle_penalty_factor": -1.0,
        "min_edge_len_m": 0.0,
        "min_corridor_support": float("inf"),
        "corridor_samples": 1,
        "min_stub_len_m": -1.0,
        "consolidate_tol_m": -1.0,
        "geom_tol_m": -1.0,
        "resolution_m": 0.0,
        "simplify": 1,
        "consolidate": 0,
        "simplify_geom": "yes",
    }
    for field, value in invalid.items():
        with pytest.raises((TypeError, ValueError), match=field):
            GraphConfig(aoi=f"bad-{field.replace('_', '-')}", **{field: value})
