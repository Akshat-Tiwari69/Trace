# SETUP.md — Environment Setup

> Get your machine ready to work on Route Resilience. **You do this yourself** — no one remote-accesses anyone's machine. Pick the path that matches your role; it takes ~15–20 minutes. If anything here is out of date, fix it and commit (this file is shared — note it in `Tracker.md` §10).

## Which path is mine?

| You are | What you run locally | Your path |
|---|---|---|
| **Saanvi** (dashboard) | Streamlit dashboard on CPU, off `data/sample/` | **Path A (CPU)** — done in 2 steps |
| **Shaivi** (graph/resilience) | graph + analysis on CPU | **Path A (CPU)**. Optional **Path C** if you want to train locally on your GPU |
| **Akshat** (ML) | data pipeline on CPU; training on cloud or local GPU | **Path A** + **Path B (cloud)**; optional local GPU |
| **Anyone training** | the segmentation model | **Path B (Colab/Kaggle)** — works on any laptop |

---

## 0 · Prerequisites (everyone)

- **Python 3.10+** and **git**.
- Clone the repo:
  ```bash
  git clone <repo-url> && cd <repo>
  ```
- Create an isolated environment (conda recommended because of GDAL — see the I-1 note):
  ```bash
  conda create -n routeres python=3.10 -y && conda activate routeres
  # or: python -m venv .venv && source .venv/bin/activate   (Windows: .venv\Scripts\activate)
  ```

## Path A · CPU (Saanvi, Shaivi, and the base for everyone)

This runs the dashboard and the whole graph/resilience pipeline — **no GPU needed.**

**If you have conda** (recommended on Linux/macOS, or if pip's GDAL/rasterio wheels fail):
```bash
# GDAL/rasterio are the fussy ones — install via conda-forge FIRST (avoids the classic build errors)
conda install -c conda-forge gdal rasterio geopandas -y
# then the rest
pip install -r requirements.txt
```

**If you don't have conda** (e.g. plain Windows + venv): prebuilt wheels for `rasterio`/`fiona`/`geopandas` are available on PyPI for common platforms, so plain pip usually works:
```bash
pip install -r requirements.txt
```
If pip fails to build `rasterio`/`fiona`/`GDAL` from source on your platform, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and fall back to the conda-forge path above.

```bash
# CPU-only PyTorch (smaller, no CUDA): only needed if you run inference locally
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
Verify:
```bash
python -c "import streamlit, folium, networkx, skimage, sknw, rasterio, osmnx; print('CPU env OK')"
```
**Saanvi:** that's it — run the dashboard with `streamlit run src/app/app.py` (it reads `data/sample/`).
**Shaivi:** that's it — your graph/resilience code (`src/pipeline/p2_graph/`, `p3_analysis/`) runs on this.

## Path B · Cloud training (Colab / Kaggle — works on ANY laptop)

The **primary, hardware-agnostic** way to train. Same notebook for everyone.
1. Open `notebooks/train_segmentation.ipynb` in **Google Colab** or **Kaggle**.
2. Enable GPU: Colab → *Runtime → Change runtime type → GPU (T4)*; Kaggle → *Settings → Accelerator → **GPU T4×2***. **Avoid Kaggle P100** — our torch build silently fails on it (see `Tracker.md` §10); the API push also defaults to P100, so launch T4 from the Kaggle UI.
3. Run all cells. The notebook installs its own deps, pulls the dataset, fine-tunes, and saves the checkpoint.
4. **Save the checkpoint off-device** (Google Drive / Kaggle Dataset) — cloud sessions are wiped when they end.
   - Kaggle free GPU: ~30 hrs/week, ≤9 hr/session. Colab free: a T4 with variable limits.

## Path C · Local NVIDIA GPU (optional — Akshat's 3070 Ti, Shaivi's 5070)

Only if you want faster local training. **Not required** — Path B covers training for everyone.

**Check your GPU architecture matters here:**
- **Ampere / Ada (e.g. RTX 30-series, 40-series):** standard install —
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```
- **Blackwell / RTX 50-series (e.g. RTX 5070, compute capability sm_120):** you **must** use PyTorch ≥2.7 with CUDA 12.8, or the GPU is silently ignored —
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  ```

Verify the GPU is actually seen:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '-')"
# RTX 50-series should print: True ... (12, 0)
```
If it prints `False` or a `sm_120 not compatible` warning on a 50-series card, you're on the wrong wheel — reinstall with the **cu128** line above.

**Laptop GPU tips:** use a cooling pad / performance power mode; keep tiles small (256²); checkpoint often so a thermal shutdown loses ≤1 epoch.

---

## Datasets

Download pointers (sources, licenses, roles) are in `docs/Research.md` → *Dataset Analysis*. The data-pipeline + OSM→mask script (task **A3**) automates label generation. Do **not** commit raw imagery or checkpoints (they're `.gitignore`d); commit only the small `data/sample/` set.

## Run P1 inference (imagery → road mask)

Once you have a trained checkpoint (grab the best one, `road_pan.pt`, from the [`a4-roadseg-v3.2` release](https://github.com/Akshat-Tiwari69/Trace/releases/tag/a4-roadseg-v3.2), or train your own), turn an image into the road-mask artifact P2 consumes — runs on **CPU**, no GPU needed:

```bash
python -m src.pipeline.p1_segment.predict \
    --image data/raw/<tile>.tif --checkpoint models/road_pan.pt --aoi <id> \
    [--blend] [--postprocess]
# writes data/interim/<id>_mask.png  (binary {0,1}); large images are tiled + stitched.
# threshold/tile-size default to the checkpoint meta; --blend = seamless Hann-overlap
# inference; --postprocess = A10 cleanup. GeoTIFFs (incl. 1-band Cartosat PAN) are read
# with their CRS/transform and a georef manifest is written for P2 automatically.
```

For the full CLI (eval on the truthful Indian benchmark, the v3 fine-tune, whole-pipeline) see the AUTO-GENERATED reference in [`README.md`](README.md).

## Troubleshooting (I-1)

- **GDAL/rasterio build errors on pip:** prebuilt wheels usually cover Windows/macOS/Linux + common Python versions, so plain `pip install -r requirements.txt` works on most machines. If pip tries to build from source and fails, install Miniconda and use the conda-forge path in Path A instead.
- **`torch.cuda.is_available()` is False on a 50-series card:** wrong wheel → use **cu128** (Path C).
- **Dashboard shows nothing:** confirm `data/sample/` has the graph + criticality files (Shaivi's S1 output).

> Found a fix that isn't here? Add it and log it in `Tracker.md` §10 — this file is the shared setup memory.