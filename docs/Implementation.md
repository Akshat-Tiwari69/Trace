# Implementation.md

> **Purpose.** This document is the execution roadmap for **Route Resilience**: the build phases, the order they happen in, who owns what, the milestones that mark progress, and the deliverables that come out. It turns the *what* (`PRD.md`) and the *how* (`TRD.md`, `Schema.md`) into a concrete plan of action. It is a living plan — update it as reality changes, and track day-to-day status in `Tracker.md`.

---

## Team Roles

Three parallel workstreams, one lead each. Leads own their workstream's code and its file outputs; integration is shared.

| Role | Owns | Lead |
|---|---|---|
| **ML / Segmentation Lead** (also integration + overall coordination) | Phase 0 data pipeline, Phase I segmentation, stitching the phases together | Akshat |
| **Graph & Analysis Lead** | Phase II graph build + healing, Phase III criticality + resilience | Shaivi |
| **Frontend / Design Lead** | Phase IV dashboard + `Design.md` | Saanvi |

The pipeline is built so workstreams **don't block each other**: the graph/analysis half can be developed against an OpenStreetMap-derived graph before any segmentation output exists, and the dashboard can be built against mock/precomputed artifacts.

## Project Phases

The build mirrors the pipeline, plus a foundations phase first and an integration phase last.

| Phase | Goal | Key tasks | Depends on | Output artifact |
|---|---|---|---|---|
| **Phase 0 — Foundations** | A working environment + data ready | Env setup (incl. Blackwell CUDA), pinned deps, repo scaffolding, dataset download, OSM→mask script, tiling | — | runnable env, cached datasets, label masks |
| **Phase I — Segmentation** | Occlusion-robust road masks | Fine-tune pretrained SegFormer/U-Net, occlusion augmentation, Dice+clDice loss, inference | Phase 0 | binary road masks |
| **Phase II — Graph build + healing** | A routable, connected graph | Skeletonize → sknw graph → MST/Union-Find healing | masks (or OSM graph for early dev) | healed `graph.graphml` |
| **Phase III — Criticality + resilience** | Critical-node + resilience analysis | Betweenness, node ablation, global-efficiency Resilience Index | Phase II | criticality CSV + resilience curve |
| **Phase IV — Dashboard** | Interactive visualization | Map + criticality heatmap + click-to-disable simulation | precomputed artifacts | Streamlit app |
| **Phase 5 — Integration + evaluation** | End-to-end demo + numbers | Wire phases via file handoffs, run evaluation, polish | all | working demo + metrics report |

## Sprint Planning

The plan front-loads the slow, painful work (setup + data) so the intensive build is spent on the interesting parts.

- **Sprint 0 — Pre-build (do now, ahead of the intensive window).** Environment on both machines, dependency shake-out (especially GDAL/rasterio and the RTX 50-series CUDA 12.8 install), dataset download/caching, the OSM→mask script, and a "walking skeleton" run of the whole pipeline on **one** sample tile. The graph/analysis workstream starts here on an OSM graph; the segmentation model is pretrained here so the intensive window only fine-tunes.
- **Sprint 1 — Core build (the intensive window).** Three workstreams run in parallel to working state: segmentation producing real masks; healing + criticality + resilience on those masks; dashboard wired to real artifacts.
- **Sprint 2 — Integration + polish.** Connect everything end-to-end, run the evaluation suite (`Evaluation.md`), record a backup demo capture, finalize docs.

A "walking skeleton" (a rough run through all four phases on one tile) should exist **by the end of Sprint 0** — it is the single best way to surface integration problems early.

## Milestones

| ID | Milestone | Definition of done |
|---|---|---|
| **M0** | Environment ready | Both machines run PyTorch on GPU (5070 verified on CUDA 12.8); all core libs import; deps pinned |
| **M1** | Data pipeline ready | Datasets cached; OSM→mask script produces aligned masks for an AOI; tiling works |
| **M2** | Segmentation working | Fine-tuned model outputs road masks; IoU/Occlusion-Recall measured |
| **M3** | Routable graph | Mask → skeleton → graph → MST/Union-Find healing; Connectivity Ratio measured |
| **M4** | Resilience analysis | Betweenness + node ablation + global-efficiency Resilience Index produce a degradation curve |
| **M5** | Dashboard interactive | Map renders; clicking a node disables it and shows rerouting + travel-time impact |
| **M6** | Integrated demo | Full pipeline runs end-to-end on a demo tile; evaluation report produced; backup capture recorded |

## Deliverables

- Trained model checkpoint (`models/`, git-ignored).
- Road masks for the demo AOI.
- Healed routable graph (`graph.graphml` / GeoJSON).
- Criticality scores (CSV) + resilience-degradation curve.
- Streamlit + Folium dashboard (`app.py`).
- Evaluation report (metrics per `Evaluation.md`).
- The documentation set (this `docs/` folder).
- A short screen-capture of a working end-to-end run (demo insurance).

## Dependencies

- **Internal:** Phase II needs masks (Phase I) **or** an OSM graph for early dev; Phase III needs a healed graph; Phase IV needs precomputed artifacts; Phase 5 needs all. The OSM-graph path is what lets the graph/analysis workstream start immediately.
- **External:** datasets (DeepGlobe/SpaceNet/OpenSatMap/Sentinel-2/LISS-IV — see `Research.md`); Cartosat-3 access for the high-res demo tile; open-source libraries; the RTX 50-series CUDA 12.8 toolchain; free Colab/Kaggle for overflow training.

## Release Strategy

- **Versioning:** `v0.1` = walking skeleton (end-to-end on one tile, rough); `v0.x` = each phase reaching working state; `v1.0` = integrated demo with evaluation numbers.
- **What "released" means:** a repo that runs locally with `streamlit run app.py` against committed sample artifacts, plus pinned dependencies (and optionally a `Dockerfile`) so anyone can reproduce it. An optional public demo can be hosted on a free CPU tier (see `TRD.md` → Deployment), since the dashboard serves precomputed outputs.
- **Reproducibility:** fixed seeds, documented configs, checkpoints saved off-device.

## Risk Mitigation

The plan bakes in the key mitigations (full detail in `RiskRegister.md`):
- **Walking skeleton early** so integration risk surfaces in Sprint 0, not at the end.
- **Graph/analysis on OSM first** so that workstream never waits on segmentation.
- **Pretrained models + fine-tuning only** (no training from scratch) to fit the compute budget.
- **Free-cloud overflow** (Colab/Kaggle) for any run that won't fit 8 GB.
- **Frequent checkpointing** so a laptop thermal shutdown loses ≤ 1 epoch.
- **Fallback MVP:** if segmentation underperforms, demonstrate Phases II–IV directly on an OSM-derived graph — the graph-theoretic resilience analysis (the most novel part) still shines.