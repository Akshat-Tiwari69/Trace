# SETUP.md — Development Environments

> Use an isolated environment. CI targets Python 3.11; the Oracle application uses its own Python 3.12 venv. Do not mix this project with unrelated TensorFlow/Google/agent packages and then treat `pip check` conflicts as a project baseline.

## Prerequisites

- Git
- Python **3.11** for development and CI parity
- Optional NVIDIA GPU for training/heavy evaluation
- No remote access to a teammate’s machine; each contributor uses these reproducible paths

```bash
git clone https://github.com/Akshat-Tiwari69/Trace.git
cd Trace
python -m venv .venv
```

Activate it:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Upgrade the installer consistently with CI:

```bash
python -m pip install --upgrade pip==24.2
```

Choose the smallest dependency role that matches the work:

- `requirements-core.txt` — shared CPU geospatial, graph and image-processing runtime.
- `deploy/requirements-app.txt` — core plus the hosted dashboard only; no Torch or training stack.
- `requirements-train.txt` — core plus raster/model training and evaluation packages; install Torch separately first.
- `requirements-dev.txt` — additive test/notebook tooling; install it alongside a runtime role, or use the aggregate.
- `requirements.txt` — aggregate app + training + development environment used by CI and full-repository work.

## Path A — Sample dashboard and CPU upload analysis

Use this when you need the app, sample graph or P2/P3 upload-analysis path but not P1 inference/training:

```bash
python -m pip install -r deploy/requirements-app.txt
python -c "import streamlit, folium, networkx, geopandas, skimage, sknw; print('app/graph env OK')"
streamlit run src/app/app.py
```

The committed Panaji sample works without a checkpoint, GPU, Modal URL or secret.

The **Your imagery** tab is enabled only when both `MODAL_SEG_URL` and `MODAL_SEG_KEY` are present in the environment. Local sample exploration remains available when they are absent.

## Path B — Full development on CPU

Install the intended CPU Torch wheels **before** root requirements so `segmentation-models-pytorch` does not resolve an unintended default Torch build:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

This mirrors CI. Verify:

```bash
python -c "import torch, streamlit, rasterio, geopandas, osmnx, sknw; print('full CPU env OK', torch.__version__)"
python -m pytest tests/ -q
```

PyPI wheels cover the supported rasterio/geopandas/pyogrio path on common Windows/macOS/Linux Python versions. If a platform tries to compile GDAL-family packages and fails, use a clean conda-forge Python 3.11 environment for the geospatial packages, then install the remaining requirements. Direct system GDAL installation is not the normal project path.

## Path C — Local NVIDIA GPU

Create a fresh Python 3.11 venv and use the current official [PyTorch installation selector](https://pytorch.org/get-started/locally/) or [official version matrix](https://pytorch.org/get-started/previous-versions/). Do not copy an old CUDA wheel URL from a log.

As of July 2026, the project’s working local research environment used PyTorch `2.12.1+cu126`; official 2.12.1 wheels also include newer CUDA variants. For Blackwell-class GPUs, use the current official CUDA 13.x recommendation rather than the deprecated 12.8 path. Re-check the selector whenever PyTorch changes.

After installing Torch/Torchvision from the chosen official index:

```bash
python -m pip install -r requirements-train.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
```

Do not proceed with a GPU experiment unless `torch.cuda.is_available()` is true and the device capability is supported by that wheel. Record Python, Torch, CUDA, device, config and Git commit with the experiment.

## Path D — Colab or Kaggle

Cloud GPU is the hardware-agnostic training path. Accelerator names, quotas and default images change, so verify them in the provider UI rather than relying on a fixed promise here.

1. Start a GPU notebook/session and clone the reviewed branch/commit.
2. Install the Torch build compatible with that runtime, then `requirements-train.txt` (plus `requirements-dev.txt` when tests or notebook tooling are needed).
3. Run the module/CLI recipe from `src/pipeline/p1_segment/`; notebooks should remain thin launchers over those modules.
4. Save checkpoints, configs and result JSON outside ephemeral session storage.
5. Record the exact runtime/package versions and source commit.

`notebooks/train_segmentation.ipynb` is a thin launcher over the maintained combined trainer and records which old notebook-only knobs are archival rather than silently reimplemented; a fresh run does not reproduce the exact A4 artifact. Current v3.x adaptation uses `src.pipeline.p1_segment.finetune`; A46 graph-first experiments have their own tracked protocol in `Research.md`/`Evaluation.md`.

## Data and checkpoints

- Raw/restricted imagery belongs under ignored `data/raw/` paths.
- Training/intermediate corpora remain ignored.
- Checkpoints belong under ignored `models/` and in an approved external artifact store/release.
- Only small, license-safe contract fixtures/evidence belong in `data/sample/`.
- Dataset roles, limitations and licenses are in `docs/Research.md`.

Download the intended deployed mask checkpoint from [`a4-roadseg-v3.2`](https://github.com/Akshat-Tiwari69/Trace/releases/tag/a4-roadseg-v3.2) as `models/road_pan.pt`. Verify release/checksum guidance before production use.

## Run the sample dashboard

```bash
streamlit run src/app/app.py
```

Open the displayed local URL. Start in Briefing/Analysis; upload remains optional.

## Run P1 locally

Blended Hann inference is the default. Use `--no-blend` only for a deliberate legacy comparison.

```bash
python -m src.pipeline.p1_segment.predict \
  --image data/raw/<tile>.tif \
  --checkpoint models/road_pan.pt \
  --aoi <safe-id> \
  --postprocess
```

Outputs include `data/interim/<id>_mask.png`, provenance, and georeference/probability sidecars when applicable. Threshold and tile size default to checkpoint metadata.

## Run the full batch pipeline

```bash
python -m src.pipeline.run_pipeline \
  --image data/raw/<tile>.tif \
  --checkpoint models/road_pan.pt \
  --aoi <safe-id> \
  --postprocess
```

The runner performs P1→P2→P3, verifies the dashboard artifact seam and writes `data/processed/<id>_run.json` only after success. Use `--force` or `--from-stage` deliberately; normal reuse is based on input/config signatures.

## Evaluation commands

```bash
# Historical/current mask-model development benchmark
python -m src.pipeline.p1_segment.eval_spacenet --help

# Legacy mask→skeleton tile APLS
python -m src.pipeline.p1_segment.apls_eval --help

# Strict common-unit v3.2 vs graph-first chip gate
python -m src.pipeline.p1_segment.chip_apls_eval --help

# Current sample graph/APLS evidence
python -m src.pipeline.p3_analysis.evaluate --aoi panaji_demo --sample-dir data/sample
python -m src.pipeline.p3_analysis.apls --aoi panaji_demo --sample-dir data/sample
```

Read `docs/Evaluation.md` before interpreting or comparing outputs; the chip and tile protocols are not interchangeable.

## Troubleshooting

- **Imports missing under the default shell Python:** activate the project venv; do not install into a global mixed environment.
- **Torch pulled the wrong build:** recreate the venv and install the intended CPU/GPU Torch wheels before `requirements.txt`.
- **GPU unavailable/unsupported capability:** select a compatible current official PyTorch/CUDA wheel; do not proceed on silent CPU fallback.
- **Geospatial wheel build failure:** use a clean Python 3.11 environment and common supported platform; fall back to conda-forge rather than hand-building GDAL.
- **Dashboard sample missing:** verify `data/sample/panaji_demo_graph.geojson`, `_criticality.csv` and `_resilience.csv` exist.
- **Upload tab disabled:** set both Modal environment variables or use sample mode.
- **Local environment has unrelated dependency conflicts:** recreate it by role; clean CI and the commands above are the reference.

Production setup is intentionally separate: see `deploy/README.md`.
