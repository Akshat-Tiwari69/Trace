"""Serverless-GPU road-segmentation endpoint (Modal) for the Route Resilience app.

Runs the deployed SegFormer MiT-B3 + SCSE checkpoint (`road_pan.pt`, release
a4-roadseg-v3.2) on a T4 that scales to zero. The dashboard POSTs a base64 image
and gets back a base64 binary road-mask PNG. Only `model.py` is vendored (it
imports torch/smp/numpy only), so none of the repo's heavy geo `__init__` chain
is pulled in.

Deploy (from repo root, Modal authed):  modal deploy deploy/modal_app.py
"""
from __future__ import annotations

import modal

MODEL_URL = (
    "https://github.com/Akshat-Tiwari69/Trace/releases/download/"
    "a4-roadseg-v3.2/road_pan.pt"
)
MODEL_PATH = "/model/road_pan.pt"


def _bake_checkpoint() -> None:
    import os
    import urllib.request

    os.makedirs("/model", exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "segmentation-models-pytorch==0.3.4",
        "numpy==1.26.4",
        "pillow==10.4.0",
        "fastapi[standard]",
    )
    .add_local_file("src/pipeline/p1_segment/model.py", "/root/model.py", copy=True)
    .run_function(_bake_checkpoint)
)

app = modal.App("roadresilience-seg", image=image)


@app.cls(gpu="T4")
class Segmenter:
    @modal.enter()
    def load(self) -> None:
        import sys

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

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.asarray(img)  # HxWx3 uint8
        prob = self.M.predict_large_prob(
            self.net, arr, tile_size=self.tile, device="cuda"
        )
        mask = ((prob >= self.threshold).astype("uint8")) * 255
        out = io.BytesIO()
        Image.fromarray(mask, mode="L").save(out, format="PNG")
        return out.getvalue()

    @modal.fastapi_endpoint(method="POST", docs=True)
    def segment(self, item: dict):
        """POST {"image_b64": "<base64 image>"} -> {"mask_png_b64", "threshold"}."""
        import base64

        img_b64 = item.get("image_b64")
        if not img_b64:
            return {"error": "missing image_b64"}
        png = self._infer_png(base64.b64decode(img_b64))
        return {
            "mask_png_b64": base64.b64encode(png).decode(),
            "threshold": self.threshold,
        }
