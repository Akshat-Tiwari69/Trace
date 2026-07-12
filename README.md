# Route Resilience

**Find the road junctions a city can least afford to lose.**

Route Resilience converts satellite imagery into a routable road graph, flags inferred/healed links, ranks structural chokepoints and shows how junction or area failures change routing and global efficiency.

[Open the public dashboard](https://trace.tiwaribabu.in) · [Setup](SETUP.md) · [Evaluation](docs/Evaluation.md) · [Current work](docs/Tracker.md)

> **Status (2026-07-13):** the P1→P4 product and public sample dashboard are working; repository CI is green. Mask model v3.2 is the intended production checkpoint. A18-LoRA validates a graph-first research direction but is not deploy-ready. The exact live Oracle/Modal ref/checksum still requires the O1 operator audit.

## What it does

```mermaid
flowchart LR
    A[Imagery] --> B["P1 · road evidence"]
    B --> C["P2 · MultiGraph + explicit healing"]
    C --> D["P3 · criticality + resilience"]
    D --> E["P4 · Streamlit/Folium dashboard"]
```

- **P1:** pretrained PyTorch road segmentation with tiled/Hann inference, GeoTIFF/PAN reading, optional probability maps and provenance.
- **P2:** skeletonization, parallel-edge-preserving MultiGraph extraction, simplification and probability/geometry-aware gap healing.
- **P3:** betweenness, articulation points/bridges and global-efficiency failure analysis.
- **P4:** Briefing, Analysis, uploaded imagery and Methodology views with junction/area scenarios, rerouting and exports.

The system is designed to recover useful continuity under occlusion and fragmentation, but it does not claim literal knowledge of every hidden road. Inferred links are marked and model evidence is reported with its limitations.

## Current evidence

### Mask-model development evidence

SpaceNet-5 Mumbai has been repeatedly consulted, so these are **development-benchmark**, not untouched-test, results. RGB→gray is a PAN proxy, not real Cartosat PAN.

| Checkpoint | Development-selected threshold | RGB IoU | Gray-proxy IoU | Legacy tile-mask APLS |
|---|---:|---:|---:|---:|
| **v3.2** `road_pan.pt` | 0.52 | **0.4594** | **0.4177** | **0.4987** |
| v3/v3.1 weights | 0.50 | 0.4493 | 0.4046 | 0.4374 |
| v1 weights | 0.50 sweep result | 0.3993 | 0.3447 | 0.4198 |

The released v1 deploy threshold is `0.44`; its row above is a development sweep at `0.50`, not the v1 release protocol. Exact fixed-threshold tables and source JSON are in [Evaluation.md](docs/Evaluation.md).

### Graph-first research evidence

On the strict common-unit chip/vector-GT gate (127/127 chips), A18-LoRA raw APLS was `0.1051` vs v3.2 `0.0121`; the paired delta was `+0.0930`, CI `[+0.0782,+0.1095]`. Normalized by each chip’s GT-self ceiling, A18-LoRA captured about **18%** of achievable routing—roughly twice the frozen A18 result, but still far below deploy quality.

Do not compare those absolute chip scores to the legacy tile-mask APLS column above; they use different units and ground truth.

### Current committed Panaji sample

- 364 nodes / 500 edges
- build-time healing: 8 → 3 components; five bridges added, four surviving final inferred edges
- 79 articulation nodes / 92 structural bridges
- APLS vs cached OSM truth: `0.5369`

Multi-step resilience-curve absolutes are temporarily withheld from headline use because A45 must fix a shrinking-node denominator in `ablation_curve()` and regenerate the curves. The single-scenario dashboard RI already preserves the baseline node universe.

## Quickstart — sample dashboard

Python 3.11:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip==24.2
python -m pip install -r deploy/requirements-app.txt
streamlit run src/app/app.py
```

The committed sample needs no model, GPU or Modal secret. The **Your imagery** tab is enabled only when `MODAL_SEG_URL` and `MODAL_SEG_KEY` are configured.

## Run your own image through P1→P3

Install the full environment from [SETUP.md](SETUP.md) and download `road_pan.pt` from the [`a4-roadseg-v3.2` model release](https://github.com/Akshat-Tiwari69/Trace/releases/tag/a4-roadseg-v3.2) into ignored `models/`.

```bash
python -m src.pipeline.run_pipeline \
  --image data/raw/your_tile.tif \
  --checkpoint models/road_pan.pt \
  --aoi your_area \
  --resolution-m 1.0 \
  --postprocess
```

Blended Hann inference is the default. GeoTIFF CRS/transform is carried into metric graph construction. For PNG/JPEG or another non-georeferenced source, `--resolution-m` is the ground-sample-distance assumption and currently defaults to `1.0` m/px; set it explicitly when real metric lengths matter.

Stage CLIs:

```bash
python -m src.pipeline.p1_segment.predict --help
python -m src.pipeline.p2_graph.build_graph --help
python -m src.pipeline.p3_analysis.analyze --help
```

## Evaluation entry points

```bash
# Mumbai mask-model development benchmark
python -m src.pipeline.p1_segment.eval_spacenet --help

# Legacy tile mask→graph APLS
python -m src.pipeline.p1_segment.apls_eval --help

# Strict same-chip v3.2 vs graph-first promotion gate
python -m src.pipeline.p1_segment.chip_apls_eval --help

# Committed Panaji sample reports
python -m src.pipeline.p3_analysis.evaluate --aoi panaji_demo --sample-dir data/sample
python -m src.pipeline.p3_analysis.apls --aoi panaji_demo --sample-dir data/sample
```

Read [Evaluation.md](docs/Evaluation.md) before comparing outputs.

## Repository map

```text
src/pipeline/p1_segment/   model, inference, data, training and evaluation
src/pipeline/p2_graph/     skeleton/MultiGraph, simplification, healing and IO
src/pipeline/p3_analysis/  criticality, resilience, APLS and scenarios
src/pipeline/run_pipeline.py
src/app/                   Streamlit/Folium app, Modal client and upload queue/analysis
data/sample/               committed demo contracts and evidence
deploy/                    Oracle/Modal/Caddy/systemd configuration
docs/                      product, architecture, evidence, research and coordination
tests/                     CPU/unit/contract/application tests
```

## Non-negotiable design rules

- Streamlit + Folium, pure Python; no database, user-login product or JavaScript SPA in this release.
- PyTorch and pretrained fine-tuning only.
- Resilience is based on global efficiency and must preserve the baseline node universe.
- P2/P3/dashboard remain CPU-capable; training/hosted P1 may use GPU.
- Raw/provider imagery, restricted data, secrets and checkpoints are not committed.
- Negative results and benchmark limits are recorded rather than hidden.

## Tests

```bash
python -m pytest tests/ -q
```

A44 baseline: **284 passed** locally; remote `dev` CI also runs dashboard import and a clean local mask-to-resilience contract smoke under production dependencies.

## Current roadmap

1. **A44:** reconcile documentation and evidence.
2. **A45:** fix resilience/inference protocol correctness, simplify code and measure performance.
3. **A46:** improve graph-first absolute routing and resolve license/reproducibility/new-geography gates.
4. **F9:** UI/UX overhaul after architecture/model outputs stabilize.
5. **O1/X1:** immutable live rollout verification and final demo capture.

See [Tracker.md](docs/Tracker.md) for status and [bugs.md](bugs.md) for the current issue ledger.

## Licensing and data

OSM, DeepGlobe, SpaceNet, Esri/provider imagery, Cartosat and any upstream model code each have separate terms. Raw/provider imagery is not redistributed; dataset use and limitations are recorded in [Research.md](docs/Research.md).

This repository currently has **no top-level code license**, so do not assume permission to reuse or redistribute the project code. Selecting a code license is tracked as `LEGAL-1` in [bugs.md](bugs.md).
