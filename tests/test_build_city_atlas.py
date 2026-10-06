import hashlib
import json
import math
import shutil
from pathlib import Path

from src.pipeline.build_city_atlas import AREAS, area_bbox, square_bbox, write_record

ROOT = Path(__file__).resolve().parents[1]


def _side_m(bbox):
    west, south, east, north = bbox
    return ((north - south) * 111_320.0,
            (east - west) * 111_320.0 * math.cos(math.radians((south + north) / 2)))


def test_square_bbox_is_a_centred_square():
    bbox = square_bbox((77.20, 28.62, 77.24, 28.65), 2000.0)
    height, width = _side_m(bbox)
    assert abs(height - 2000) < 1 and abs(width - 2000) < 5
    assert abs((bbox[0] + bbox[2]) / 2 - 77.22) < 1e-6


def test_area_square_shrinks_to_fit_a_smaller_cached_extract(tmp_path):
    extract = [77.214, 28.629, 77.226, 28.641]
    (tmp_path / "delhi_cp_osm_roads.meta.json").write_text(json.dumps({"bbox": extract, "network_type": "drive"}))
    bbox = area_bbox("delhi_cp", raw_dir=tmp_path)
    assert extract[0] <= bbox[0] and extract[1] <= bbox[1] and extract[2] >= bbox[2] and extract[3] >= bbox[3]
    assert max(_side_m(bbox)) < 2000


def test_record_counts_match_the_artifacts(tmp_path):
    for suffix in ("_graph.geojson", "_criticality.csv", "_resilience.csv"):
        shutil.copyfile(ROOT / "data/sample" / f"panaji_demo{suffix}", tmp_path / f"panaji_demo{suffix}")
    record = json.loads(write_record("panaji_demo", tmp_path, {"source": "osm"}).read_text())
    assert (record["node_count"], record["edge_count"], record["critical_count"]) == (573, 828, 57)
    assert 0 < record["worst_single_loss"] < 1


def test_committed_atlases_match_their_records():
    records = sorted((ROOT / "data/atlas").glob("*_atlas.json"))
    assert len(records) == len(AREAS)
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        graph = path.with_name(f"{record['aoi']}_graph.geojson")
        assert record["source"] == "osm" and record["model"] is None, path.name
        assert hashlib.sha256(graph.read_bytes()).hexdigest() == record["graph_sha256"], path.name
        assert record["label"] and record["region"].startswith("IN-")
