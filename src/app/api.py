"""FastAPI entry point for the Route Resilience web experience."""

from __future__ import annotations

from contextlib import asynccontextmanager
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
from collections import defaultdict, deque

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles

from src.app import job_queue, modal_client
from src.app.service import (
    SAMPLE_AOI,
    analysis_graph_geojson,
    analysis_payload,
    aoi_summary,
    sample_dataset,
    simulate,
)


SUMMARY_CACHE = "public, max-age=60, stale-while-revalidate=300"
ARTIFACT_CACHE = "public, max-age=31536000, immutable"
MAX_UPLOAD_BYTES = 11 * 1024 * 1024
MAX_IMAGE_PIXELS = 4096 * 4096
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
IMAGE_FORMATS = {
    "image/png": ("PNG", {".png"}),
    "image/jpeg": ("JPEG", {".jpg", ".jpeg"}),
}
log = logging.getLogger("trace.api")


class SimulationRequest(BaseModel):
    aoi: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    removed_node_ids: list[int] = Field(default_factory=list, max_length=50)

    @field_validator("removed_node_ids")
    @classmethod
    def normalize_nodes(cls, values: list[int]) -> list[int]:
        return sorted(set(values))


class _RateLimiter:
    def __init__(
        self,
        limit: int = 180,
        window_seconds: int = 60,
        max_keys: int = 10_000,
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float) -> bool:
        cutoff = now - self.window_seconds
        with self._lock:
            if key not in self._hits and len(self._hits) >= self.max_keys:
                self._hits.pop(next(iter(self._hits)))
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True


def _security_headers() -> dict[str, str]:
    return {
        "Content-Security-Policy": (
            "default-src 'self'; base-uri 'self'; frame-ancestors 'none'; "
            "object-src 'none'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: https:; "
            "font-src 'self'; worker-src 'self' blob:; "
            "connect-src 'self' https://tiles.openfreemap.org https://*.openfreemap.org"
        ),
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    }


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _validate_image(upload: UploadFile) -> tuple[bytes, tuple[int, int]]:
    content_type = (upload.content_type or "").lower()
    expected = IMAGE_FORMATS.get(content_type)
    suffix = Path(upload.filename or "").suffix.lower()
    if expected is None or suffix not in expected[1]:
        raise HTTPException(status_code=422, detail="Upload a PNG or JPEG image")

    content = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 11 MiB limit")
    if not content:
        raise HTTPException(status_code=422, detail="Image is empty")

    try:
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
            if image.format != expected[0]:
                raise HTTPException(
                    status_code=422,
                    detail="Image content does not match its file type",
                )
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=422,
                    detail="Image exceeds the 4096×4096 pixel limit",
                )
            image.verify()
    except HTTPException:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Upload a valid PNG or JPEG image") from exc
    return content, (width, height)


def _decode_mask(mask_png: bytes, expected_size: tuple[int, int]) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(mask_png)) as image:
            if image.format != "PNG" or image.size != expected_size:
                raise ValueError("unexpected mask contract")
            image.load()
            mask = np.asarray(image.convert("L"))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Segmentation service returned an invalid mask",
        ) from exc
    return (mask > 0).astype(np.uint8)


def _read_job(job_id: str) -> dict:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="Analysis not found")
    try:
        return job_queue.status(job_id)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail="Analysis not found") from exc


def _status_payload(job_id: str, state: dict) -> dict:
    status = state.get("status")
    return {
        "id": job_id,
        "job_id": job_id,
        "status": status,
        "position": job_queue.position(job_id) if status == "queued" else 0,
        "created_utc": state.get("created_utc"),
        "started_utc": state.get("started_utc"),
        "finished_utc": state.get("finished_utc"),
        "error": "Analysis failed. Try another image." if status == "failed" else None,
        "result_url": f"/api/v1/analyses/{job_id}/result",
    }


def _finished_result(job_id: str):
    state = _read_job(job_id)
    if state.get("status") in {"queued", "running"}:
        raise HTTPException(
            status_code=409,
            detail="Analysis is still running",
            headers={"Retry-After": "2"},
        )
    if state.get("status") == "failed":
        log.warning("analysis failed job_id=%s", job_id)
        raise HTTPException(status_code=422, detail="Analysis failed. Try another image.")
    if state.get("status") != "done":
        raise HTTPException(status_code=503, detail="Analysis state is unavailable")
    try:
        return job_queue.result(job_id)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Analysis result is not available yet",
            headers={"Retry-After": "2"},
        ) from exc


def create_app(
    *,
    web_out: Path | None = None,
    start_worker: bool = True,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if start_worker:
            job_queue.ensure_worker()
        yield

    app = FastAPI(
        title="Route Resilience API",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/v1/openapi.json",
        lifespan=lifespan,
    )
    allowed_hosts = [
        host.strip()
        for host in os.getenv(
            "TRACE_ALLOWED_HOSTS",
            "trace.tiwaribabu.in,localhost,127.0.0.1,testserver",
        ).split(",")
        if host.strip()
    ]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    development_origins = [
        origin.strip()
        for origin in os.getenv("TRACE_DEV_ORIGINS", "").split(",")
        if origin.strip()
    ]
    if development_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=development_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["content-type", "if-none-match"],
        )

    limiter = _RateLimiter()
    upload_limiter = _RateLimiter(limit=6, window_seconds=60)
    simulation_limiter = _RateLimiter(limit=12, window_seconds=60)

    @app.middleware("http")
    async def secure_response(request: Request, call_next):
        if not limiter.allow(_client_key(request), time.monotonic()):
            return JSONResponse(
                {"detail": "Too many requests"},
                status_code=429,
                headers={"Retry-After": "60", **_security_headers()},
            )
        response = await call_next(request)
        response.headers.update(_security_headers())
        if request.url.path.startswith("/api/v1/analyses"):
            response.headers["Cache-Control"] = "no-store"
        elif request.url.path.startswith("/_next/static/"):
            response.headers["Cache-Control"] = ARTIFACT_CACHE
        return response

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/aois/{aoi}")
    def get_aoi(aoi: str, response: Response) -> dict:
        try:
            payload = aoi_summary(aoi)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Area not found") from exc
        response.headers["Cache-Control"] = SUMMARY_CACHE
        return payload

    @app.get("/api/v1/aois/{aoi}/graph")
    def get_graph(aoi: str, request: Request) -> Response:
        if aoi != SAMPLE_AOI:
            raise HTTPException(status_code=404, detail="Area not found")
        dataset = sample_dataset()
        headers = {"ETag": dataset.graph_etag, "Cache-Control": ARTIFACT_CACHE}
        if request.headers.get("if-none-match") == dataset.graph_etag:
            return Response(status_code=304, headers=headers)
        return Response(
            content=dataset.graph_bytes,
            media_type="application/geo+json",
            headers=headers,
        )

    @app.post("/api/v1/simulations")
    def run_simulation(request: Request, body: SimulationRequest) -> dict:
        if not simulation_limiter.allow(_client_key(request), time.monotonic()):
            raise HTTPException(
                status_code=429,
                detail="Simulation rate limit exceeded",
                headers={"Retry-After": "60"},
            )
        try:
            return simulate(body.aoi, body.removed_node_ids)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Area not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/analyses", status_code=202)
    def create_analysis(
        request: Request,
        image: UploadFile = File(...),
        resolution_m: float = Form(0.5, ge=0.1, le=2.0, allow_inf_nan=False),
        confirm_external_processing: bool = Form(...),
    ) -> JSONResponse:
        if not confirm_external_processing:
            raise HTTPException(
                status_code=422,
                detail="Consent to external GPU processing is required",
            )
        if not upload_limiter.allow(_client_key(request), time.monotonic()):
            raise HTTPException(
                status_code=429,
                detail="Upload rate limit exceeded",
                headers={"Retry-After": "60"},
            )
        source, size = _validate_image(image)
        if not modal_client.MODAL_SEG_URL or not os.environ.get("MODAL_SEG_KEY"):
            raise HTTPException(status_code=503, detail="Image analysis is not configured")
        job_queue.cleanup()
        if job_queue.pending_count() >= 8:
            raise HTTPException(
                status_code=503,
                detail="Image analysis queue is full. Try again later.",
                headers={"Retry-After": "60"},
            )
        try:
            mask_png, threshold = modal_client.call(source)
        except modal_client.EndpointBusyError as exc:
            raise HTTPException(
                status_code=503,
                detail="Image analysis is busy. Try again shortly.",
                headers={"Retry-After": "10"},
            ) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Image analysis timed out") from exc
        except urllib.error.URLError as exc:
            raise HTTPException(
                status_code=503,
                detail="Image analysis is temporarily unavailable",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - upstream details stay server-side
            log.exception("Modal segmentation failed")
            raise HTTPException(status_code=502, detail="Image analysis failed") from exc

        mask = _decode_mask(mask_png, size)
        if threshold is not None:
            try:
                threshold = float(threshold)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=502,
                    detail="Segmentation service returned invalid metadata",
                ) from exc
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise HTTPException(
                    status_code=502,
                    detail="Segmentation service returned invalid metadata",
                )
        job_id = job_queue.submit(mask, resolution_m=resolution_m)
        position = job_queue.position(job_id)
        base = f"/api/v1/analyses/{job_id}"
        payload = {
            "id": job_id,
            "job_id": job_id,
            "status": "queued",
            "position": position,
            "status_url": base,
            "result_url": f"{base}/result",
            "threshold": threshold,
        }
        return JSONResponse(
            payload,
            status_code=202,
            headers={"Location": base, "Retry-After": "2"},
        )

    @app.get("/api/v1/analyses/{job_id}")
    def get_analysis(job_id: str) -> dict:
        return _status_payload(job_id, _read_job(job_id))

    @app.get("/api/v1/analyses/{job_id}/result")
    def get_analysis_result(job_id: str) -> dict:
        return analysis_payload(_finished_result(job_id), job_id)

    @app.get("/api/v1/analyses/{job_id}/graph")
    def get_analysis_graph(job_id: str) -> Response:
        payload = analysis_graph_geojson(_finished_result(job_id))
        return Response(
            json.dumps(payload, separators=(",", ":")),
            media_type="application/geo+json",
            headers={
                "Content-Disposition": f'attachment; filename="analysis-{job_id}.geojson"'
            },
        )

    static_dir = Path(web_out or os.getenv("TRACE_WEB_OUT", "web/out"))
    if (static_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="web")
    return app


app = create_app()
