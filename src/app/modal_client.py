"""Bounded, retrying client for the authenticated Modal segmentation endpoint."""

from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


MODAL_SEG_URL = os.environ.get("MODAL_SEG_URL")
MAX_MODAL_RESPONSE_BYTES = 24 * 1024 * 1024
_MAX_INFLIGHT = 2
_semaphore = threading.BoundedSemaphore(_MAX_INFLIGHT)


class EndpointBusyError(RuntimeError):
    """Raised when all in-flight GPU slots are occupied."""


def post_once(body: bytes) -> dict:
    """Send one authenticated request and parse its JSON response."""
    parsed = urllib.parse.urlsplit(MODAL_SEG_URL or "")
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise RuntimeError("MODAL_SEG_URL must be an HTTPS endpoint")
    key = os.environ.get("MODAL_SEG_KEY", "")
    if not key:
        raise RuntimeError("MODAL_SEG_KEY is not configured")
    req = urllib.request.Request(
        MODAL_SEG_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": key,
        },
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        status = getattr(response, "status", 200)
        if status != 200:
            raise RuntimeError(f"endpoint returned HTTP {status}")
        raw = response.read(MAX_MODAL_RESPONSE_BYTES + 1)
        if len(raw) > MAX_MODAL_RESPONSE_BYTES:
            raise RuntimeError("endpoint response exceeded the size limit")
    try:
        output = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("endpoint returned a non-JSON response") from error
    if not isinstance(output, dict):
        raise RuntimeError("endpoint returned an invalid JSON response")
    return output


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code >= 500
    return isinstance(error, urllib.error.URLError)


def call(image_bytes: bytes, retries: int = 2) -> tuple[bytes, float | None]:
    """Return decoded mask PNG bytes and the deployed threshold."""
    body = json.dumps({"image_b64": base64.b64encode(image_bytes).decode()}).encode()
    if not _semaphore.acquire(blocking=False):
        raise EndpointBusyError(
            "The GPU endpoint is busy with other requests — please retry in a moment."
        )
    try:
        for attempt in range(retries + 1):
            try:
                output = post_once(body)
                break
            except Exception as error:
                if attempt < retries and _is_retryable(error):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
    finally:
        _semaphore.release()

    if "mask_png_b64" not in output:
        raise RuntimeError(output.get("error", "unexpected response from endpoint"))
    try:
        mask_png = base64.b64decode(output["mask_png_b64"], validate=True)
    except ValueError as error:
        raise RuntimeError("endpoint returned an undecodable mask") from error
    if not mask_png:
        raise RuntimeError("endpoint returned an empty mask")
    return mask_png, output.get("threshold")
