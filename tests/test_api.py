"""Public web API contracts for the F9 presentation replacement."""

from __future__ import annotations

import io
import math
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.app import api
from src.app.api import create_app
from src.app.service import representative_reroute


JOB_ID = "a" * 32


def _image_bytes(*, size: tuple[int, int] = (64, 64), format: str = "PNG") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, "white").save(output, format=format)
    return output.getvalue()


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(create_app(start_worker=False)) as test_client:
        yield test_client


def test_health_and_security_headers(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in response.headers["content-security-policy"]


def test_sample_manifest_is_contract_shaped(client: TestClient) -> None:
    response = client.get("/api/v1/aois/panaji_demo")
    assert response.status_code == 200
    payload = response.json()
    assert payload["aoi"] == "panaji_demo"
    assert payload["label"] == "Panaji, Goa"
    assert payload["node_count"] == len(payload["critical_nodes"])
    assert payload["edge_count"] > 0
    assert payload["graph_url"] == "/api/v1/aois/panaji_demo/graph"
    assert payload["critical_nodes"][0]["rank"] == 1
    assert payload["resilience_curve"][0]["targeted_resilience_index"] == 1.0
    assert len(payload["bounds"]) == 4
    assert response.headers["cache-control"] == "public, max-age=60, stale-while-revalidate=300"


def test_unknown_aoi_is_not_interpolated_into_paths(client: TestClient) -> None:
    response = client.get("/api/v1/aois/not-a-real-aoi")
    assert response.status_code == 404
    assert response.json() == {"detail": "Area not found"}


def test_graph_artifact_has_etag_and_conditional_response(client: TestClient) -> None:
    response = client.get("/api/v1/aois/panaji_demo/graph")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")
    assert response.json()["type"] == "FeatureCollection"
    etag = response.headers["etag"]
    assert len(etag) == 66 and etag.startswith('"')

    cached = client.get(
        "/api/v1/aois/panaji_demo/graph",
        headers={"if-none-match": etag},
    )
    assert cached.status_code == 304
    assert cached.content == b""


def test_simulation_preserves_finite_baseline_universe(client: TestClient) -> None:
    response = client.post(
        "/api/v1/simulations",
        json={"aoi": "panaji_demo", "removed_node_ids": [278]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["removed_node_ids"] == [278]
    summary = client.get("/api/v1/aois/panaji_demo").json()
    assert payload["baseline_node_count"] == summary["node_count"]
    assert 0.0 <= payload["resilience_index"] <= 1.0
    assert payload["efficiency_loss"] >= 0.0
    assert math.isfinite(payload["baseline_efficiency"])
    assert math.isfinite(payload["perturbed_efficiency"])
    assert 0.0 <= payload["largest_cc_fraction"] <= 1.0
    curve_zero = client.get("/api/v1/aois/panaji_demo").json()["resilience_curve"][0]
    expected_method = curve_zero.get("efficiency_method", "exact")
    assert payload["efficiency_method"] == expected_method
    assert payload["efficiency_sample_size"] == curve_zero.get("efficiency_k")
    assert payload["efficiency_seed"] == curve_zero.get("efficiency_seed")
    assert "representative_route" in payload


def test_simulation_rejects_unknown_or_unbounded_node_sets(client: TestClient) -> None:
    unknown = client.post(
        "/api/v1/simulations",
        json={"aoi": "panaji_demo", "removed_node_ids": [9999]},
    )
    assert unknown.status_code == 422
    assert unknown.json() == {"detail": "Unknown node IDs: 9999"}

    too_many = client.post(
        "/api/v1/simulations",
        json={"aoi": "panaji_demo", "removed_node_ids": list(range(51))},
    )
    assert too_many.status_code == 422


def test_simulation_cache_normalizes_equivalent_node_sets(client: TestClient) -> None:
    api.simulate.cache_clear()
    client.post(
        "/api/v1/simulations",
        json={"aoi": "panaji_demo", "removed_node_ids": [278, 12, 278]},
    )
    before = api.simulate.cache_info()
    client.post(
        "/api/v1/simulations",
        json={"aoi": "panaji_demo", "removed_node_ids": [12, 278]},
    )
    after = api.simulate.cache_info()
    assert after.hits == before.hits + 1


def test_upload_validates_then_queues_modal_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _image_bytes()
    mask = io.BytesIO()
    Image.new("L", (64, 64), 255).save(mask, format="PNG")
    queued: dict[str, object] = {}

    monkeypatch.setattr(api.modal_client, "MODAL_SEG_URL", "https://modal.invalid/segment")
    monkeypatch.setenv("MODAL_SEG_KEY", "test-key")
    monkeypatch.setattr(api.modal_client, "call", lambda value: (mask.getvalue(), 0.52))
    monkeypatch.setattr(api.job_queue, "cleanup", lambda: 0)
    monkeypatch.setattr(api.job_queue, "pending_count", lambda: 0)
    monkeypatch.setattr(api.job_queue, "ensure_worker", lambda: None)

    def submit(value: np.ndarray, resolution_m: float) -> str:
        queued.update(mask=value, resolution_m=resolution_m)
        return JOB_ID

    monkeypatch.setattr(api.job_queue, "submit", submit)
    monkeypatch.setattr(api.job_queue, "position", lambda _job_id: 2)
    with TestClient(create_app()) as upload_client:
        response = upload_client.post(
            "/api/v1/analyses",
            files={"image": ("roads.png", source, "image/png")},
            data={"resolution_m": "0.75", "confirm_external_processing": "true"},
        )

    assert response.status_code == 202
    assert response.headers["location"] == f"/api/v1/analyses/{JOB_ID}"
    assert response.json() == {
        "id": JOB_ID,
        "job_id": JOB_ID,
        "status": "queued",
        "position": 2,
        "status_url": f"/api/v1/analyses/{JOB_ID}",
        "result_url": f"/api/v1/analyses/{JOB_ID}/result",
        "threshold": 0.52,
    }
    assert queued["resolution_m"] == 0.75
    assert isinstance(queued["mask"], np.ndarray)
    assert queued["mask"].shape == (64, 64)
    assert set(np.unique(queued["mask"])) == {1}


@pytest.mark.parametrize(
    ("filename", "content_type", "content", "consent"),
    [
        ("roads.png", "image/png", b"not an image", "true"),
        ("roads.jpg", "image/png", _image_bytes(format="JPEG"), "true"),
        ("roads.png", "image/png", _image_bytes(), "false"),
    ],
)
def test_upload_rejects_untrusted_input_before_modal(
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    content_type: str,
    content: bytes,
    consent: str,
) -> None:
    called = False

    def modal_call(_value: bytes):
        nonlocal called
        called = True
        raise AssertionError("Modal must not receive invalid input")

    monkeypatch.setattr(api.modal_client, "MODAL_SEG_URL", "https://modal.invalid/segment")
    monkeypatch.setattr(api.modal_client, "call", modal_call)
    monkeypatch.setattr(api.job_queue, "ensure_worker", lambda: None)
    with TestClient(create_app()) as upload_client:
        response = upload_client.post(
            "/api/v1/analyses",
            files={"image": (filename, content, content_type)},
            data={"resolution_m": "0.5", "confirm_external_processing": consent},
        )

    assert response.status_code == 422
    assert called is False


def test_upload_size_cap_is_checked_before_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4)
    monkeypatch.setattr(api.job_queue, "ensure_worker", lambda: None)
    with TestClient(create_app()) as upload_client:
        response = upload_client.post(
            "/api/v1/analyses",
            files={"image": ("roads.png", b"12345", "image/png")},
            data={"resolution_m": "0.5", "confirm_external_processing": "true"},
        )
    assert response.status_code == 413


def test_upload_rejects_invalid_modal_metadata_before_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _image_bytes()
    mask = io.BytesIO()
    Image.new("L", (64, 64), 255).save(mask, format="PNG")
    monkeypatch.setattr(api.modal_client, "MODAL_SEG_URL", "https://modal.invalid/segment")
    monkeypatch.setenv("MODAL_SEG_KEY", "test-key")
    monkeypatch.setattr(api.modal_client, "call", lambda _value: (mask.getvalue(), "bad"))
    monkeypatch.setattr(api.job_queue, "pending_count", lambda: 0)
    monkeypatch.setattr(api.job_queue, "ensure_worker", lambda: None)
    monkeypatch.setattr(
        api.job_queue,
        "submit",
        lambda *_args, **_kwargs: pytest.fail("invalid metadata must not be queued"),
    )
    with TestClient(create_app()) as upload_client:
        response = upload_client.post(
            "/api/v1/analyses",
            files={"image": ("roads.png", source, "image/png")},
            data={"resolution_m": "0.5", "confirm_external_processing": "true"},
        )
    assert response.status_code == 502


def test_analysis_status_and_result_are_public_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = nx.MultiGraph()
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=1.0, y=0.0)
    graph.add_edge(1, 2, key=0, length_m=1.0, geometry=[(0.0, 0.0), (1.0, 0.0)])
    result = SimpleNamespace(
        graph=graph,
        criticality=pd.DataFrame([{
            "node_id": 1,
            "betweenness": 1.0,
            "rank": 1,
            "is_critical": True,
            "is_articulation": False,
            "x": 0.0,
            "y": 0.0,
        }]),
        resilience_index=0.75,
        top_node=1,
        n_nodes=2,
        n_edges=1,
        resolution_m=0.5,
        summary={"critical_junctions": 1},
    )
    state = {
        "job_id": JOB_ID,
        "status": "done",
        "created_utc": "2026-07-22T00:00:00+00:00",
        "started_utc": "2026-07-22T00:00:01+00:00",
        "finished_utc": "2026-07-22T00:00:02+00:00",
        "error": None,
    }
    monkeypatch.setattr(api.job_queue, "ensure_worker", lambda: None)
    monkeypatch.setattr(api.job_queue, "status", lambda _job_id: state)
    monkeypatch.setattr(api.job_queue, "position", lambda _job_id: 0)
    monkeypatch.setattr(api.job_queue, "result", lambda _job_id: result)
    with TestClient(create_app()) as analysis_client:
        status_response = analysis_client.get(f"/api/v1/analyses/{JOB_ID}")
        result_response = analysis_client.get(f"/api/v1/analyses/{JOB_ID}/result")
        graph_response = analysis_client.get(f"/api/v1/analyses/{JOB_ID}/graph")

    assert status_response.status_code == 200
    assert status_response.json()["status"] == "done"
    payload = result_response.json()
    assert payload["resilience_index"] == 0.75
    assert payload["criticality"][0]["node_id"] == 1
    assert payload["exports"]["geojson"].endswith(f"/{JOB_ID}/graph")
    assert graph_response.status_code == 200
    assert graph_response.headers["content-type"].startswith("application/geo+json")
    assert graph_response.json()["type"] == "FeatureCollection"


def test_analysis_job_ids_and_failures_do_not_leak_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api.job_queue, "ensure_worker", lambda: None)
    monkeypatch.setattr(api.job_queue, "status", lambda _job_id: {
        "job_id": JOB_ID,
        "status": "failed",
        "created_utc": None,
        "started_utc": None,
        "finished_utc": None,
        "error": "C:/secret/path/model.py exploded",
    })
    with TestClient(create_app()) as analysis_client:
        unsafe = analysis_client.get("/api/v1/analyses/not-a-job")
        failed = analysis_client.get(f"/api/v1/analyses/{JOB_ID}")
        failed_result = analysis_client.get(f"/api/v1/analyses/{JOB_ID}/result")

    assert unsafe.status_code == 404
    assert failed.json()["error"] == "Analysis failed. Try another image."
    assert "secret" not in failed.text
    assert failed_result.status_code == 422
    assert "secret" not in failed_result.text


def test_pending_result_is_a_retryable_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.job_queue, "ensure_worker", lambda: None)
    monkeypatch.setattr(api.job_queue, "status", lambda _job_id: {
        "job_id": JOB_ID,
        "status": "running",
        "created_utc": None,
        "started_utc": None,
        "finished_utc": None,
        "error": None,
    })
    with TestClient(create_app()) as analysis_client:
        response = analysis_client.get(f"/api/v1/analyses/{JOB_ID}/result")
    assert response.status_code == 409
    assert response.headers["retry-after"] == "2"


def test_static_export_is_served_after_api_routes(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<h1>Field atlas</h1>", encoding="utf-8")
    with TestClient(create_app(web_out=tmp_path, start_worker=False)) as static_client:
        homepage = static_client.get("/")
        health = static_client.get("/healthz")
    assert homepage.status_code == 200
    assert "Field atlas" in homepage.text
    assert health.json() == {"status": "ok"}


def test_expensive_public_work_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api,
        "simulate",
        lambda aoi, removed: {"aoi": aoi, "removed_node_ids": removed},
    )
    with TestClient(create_app(start_worker=False)) as guarded_client:
        responses = [
            guarded_client.post(
                "/api/v1/simulations",
                json={"aoi": "panaji_demo", "removed_node_ids": [index]},
            )
            for index in range(13)
        ]
    assert [response.status_code for response in responses[:12]] == [200] * 12
    assert responses[12].status_code == 429


def test_full_upload_queue_fails_before_modal(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def modal_call(_value: bytes):
        nonlocal called
        called = True

    monkeypatch.setattr(api.modal_client, "MODAL_SEG_URL", "https://modal.invalid/segment")
    monkeypatch.setenv("MODAL_SEG_KEY", "test-key")
    monkeypatch.setattr(api.modal_client, "call", modal_call)
    monkeypatch.setattr(api.job_queue, "cleanup", lambda: 0)
    monkeypatch.setattr(api.job_queue, "pending_count", lambda: 8)
    with TestClient(create_app(start_worker=False)) as guarded_client:
        response = guarded_client.post(
            "/api/v1/analyses",
            files={"image": ("roads.png", _image_bytes(), "image/png")},
            data={"resolution_m": "0.5", "confirm_external_processing": "true"},
        )
    assert response.status_code == 503
    assert called is False


def test_representative_reroute_discloses_disconnection_and_finite_detour() -> None:
    graph = nx.MultiGraph()
    graph.add_edge(0, 1, length_m=1.0)
    graph.add_edge(1, 2, length_m=1.0)
    disconnected = representative_reroute(graph, 1)
    assert disconnected is not None
    assert disconnected["disconnected"] is True
    assert disconnected["disconnected_pairs"] == 1
    assert disconnected["rerouted_path"] is None

    graph.add_edge(0, 3, length_m=2.0)
    graph.add_edge(3, 2, length_m=2.0)
    finite = representative_reroute(graph, 1)
    assert finite is not None
    assert finite["disconnected"] is False
    assert finite["baseline_path"] == [0, 1, 2]
    assert finite["rerouted_path"] == [0, 3, 2]
    assert finite["travel_time_delta_pct"] == 100.0
