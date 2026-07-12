"""Serverless-GPU road-segmentation endpoint (Modal) for the Route Resilience app.

Runs the deployed SegFormer MiT-B3 + SCSE checkpoint (`road_pan.pt`, release
a4-roadseg-v3.2) on a T4 that scales to zero. The dashboard POSTs a base64 image
with the shared key in an HTTP header and gets back a base64 binary road-mask
PNG. Only `model.py` is
vendored (torch/smp/numpy only), so none of the repo's heavy geo `__init__` chain
is pulled in. The auth key lives in a Modal Secret (`roadseg-key`, env
`ROADSEG_KEY`) — never in the repo.

Deploy (from repo root, Modal authed):  modal deploy deploy/modal_app.py
"""
from __future__ import annotations

import modal
from fastapi import Request

MODEL_URL = (
    "https://github.com/Akshat-Tiwari69/Trace/releases/download/"
    "a4-roadseg-v3.2/road_pan.pt"
)
# SHA-256 of the release asset above (== local models/road_pan.pt). The image
# build fails closed on mismatch so a tampered/corrupted download can never be
# baked in. Recompute (Get-FileHash / sha256sum) when a new release is deployed.
MODEL_SHA256 = "0ebedf973c0ffe0b1148d3de38382d17ac88398c6a55f4550d661a6b1e9eed1d"
MODEL_PATH = "/model/road_pan.pt"

# Payload guardrails for the public endpoint. The UI enforces the same original-
# byte limit; derive the base64 ceiling so transport expansion cannot disagree.
MAX_SOURCE_BYTES = 11 * 1024 * 1024
MAX_B64_LEN = 4 * ((MAX_SOURCE_BYTES + 2) // 3)
MAX_IMAGE_PIXELS = 4096 * 4096  # PIL decompression-bomb ceiling


def _bake_checkpoint() -> None:
    import hashlib
    import os
    import urllib.error
    import urllib.request

    os.makedirs("/model", exist_ok=True)
    for attempt in (1, 2):  # one retry on transient network failure
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            break
        except urllib.error.URLError:
            if attempt == 2:
                raise

    sha = hashlib.sha256()
    with open(MODEL_PATH, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    digest = sha.hexdigest()
    if digest != MODEL_SHA256:
        raise RuntimeError(
            f"checkpoint checksum mismatch: expected {MODEL_SHA256}, got {digest}. "
            "Refusing to bake an unverified model into the image. If this is a new "
            "release, update MODEL_SHA256 from the local models/road_pan.pt."
        )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "segmentation-models-pytorch==0.3.4",
        "numpy==1.26.4",
        "pillow==10.4.0",
        "fastapi[standard]==0.115.14",
    )
    .add_local_file("src/pipeline/p1_segment/model.py", "/root/model.py", copy=True)
    .run_function(_bake_checkpoint)
)

app = modal.App("roadresilience-seg", image=image)


# Cost controls: at most 2 warm T4 containers, kill any request after 120 s,
# scale to zero 120 s after the last request. (Kwarg names per the current
# Modal SDK — `max_containers`/`scaledown_window` replaced the older
# `concurrency_limit`/`container_idle_timeout`; verify on redeploy if the
# pinned modal version predates the rename.)
@app.cls(
    gpu="T4",
    secrets=[modal.Secret.from_name("roadseg-key")],
    max_containers=2,
    timeout=120,
    scaledown_window=120,
)
class Segmenter:
    @modal.enter()
    def load(self) -> None:
        import sys

        import PIL.Image

        # Decompression-bomb ceiling for untrusted uploads (PIL raises
        # DecompressionBombError beyond 2x this).
        PIL.Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

        sys.path.insert(0, "/root")
        import model as M  # vendored standalone module

        net, meta = M.load_checkpoint(MODEL_PATH, map_location="cuda")
        self.M = M
        self.net = net.to("cuda").eval()
        self.threshold = float(meta.get("threshold", 0.5))
        self.tile = int(meta.get("image_size", 512))

    def _infer_png(self, image_bytes: bytes) -> bytes:
        import io

        import numpy as np
        from PIL import Image

        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.load()  # force full decode here so bombs/corruption fail here
            img = img.convert("RGB")
        except Exception as exc:
            # UnidentifiedImageError, DecompressionBombError, truncated files…
            # normalized to ValueError so the endpoint returns a generic 400.
            raise ValueError("invalid image") from exc
        arr = np.asarray(img)  # HxWx3 uint8
        prob = self.M.predict_large_prob(
            self.net, arr, tile_size=self.tile, device="cuda"
        )
        mask = ((prob >= self.threshold).astype("uint8")) * 255
        out = io.BytesIO()
        Image.fromarray(mask, mode="L").save(out, format="PNG")
        return out.getvalue()

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def segment(self, request: Request):
        """POST {"image_b64"} with X-API-Key -> mask response.

        Real HTTP errors: 401 bad key, 400 missing/invalid image, 413 oversized.
        """
        import base64
        import binascii
        import hmac
        import os

        from fastapi import HTTPException

        # Auth FIRST (constant-time), before any decode/parse work on the body.
        expected_key = os.environ.get("ROADSEG_KEY", "")
        if not expected_key:
            raise HTTPException(status_code=503, detail="endpoint auth is not configured")
        if not hmac.compare_digest(request.headers.get("x-api-key", ""), expected_key):
            raise HTTPException(status_code=401, detail="unauthorized")

        try:
            item = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid JSON") from exc
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="invalid JSON object")

        img_b64 = item.get("image_b64")
        if not img_b64 or not isinstance(img_b64, str):
            raise HTTPException(status_code=400, detail="missing image_b64")
        if len(img_b64) > MAX_B64_LEN:
            raise HTTPException(status_code=413, detail="image too large")
        try:
            image_bytes = base64.b64decode(img_b64, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail="invalid image")
        if len(image_bytes) > MAX_SOURCE_BYTES:
            raise HTTPException(status_code=413, detail="image too large")
        try:
            png = self._infer_png(image_bytes)
        except ValueError:
            # Generic message on purpose — never a traceback body.
            raise HTTPException(status_code=400, detail="invalid image")
        return {
            "mask_png_b64": base64.b64encode(png).decode(),
            "threshold": self.threshold,
        }
