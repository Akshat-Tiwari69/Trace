# Tracker.md

> **Purpose.** This is the **living progress tracker** for Route Resilience — the single place to see what's done, what's in flight, what's next, and what's blocking. Update it as you go (it's meant to be edited constantly, unlike the other docs). Status reflects the project as of the last update date below.

**Last updated:** 2026-06-22

**Status legend:** ✅ done · 🔄 in progress · ⏳ pending · ⛔ blocked

---

## Completed Tasks

| Item | Owner | Notes |
|---|---|---|
| ✅ Repo + docs framework scaffolding | ML Lead | `docs/` structure, `Index.md`, CODEOWNERS |
| ✅ `Research.md` (Phase 1) | ML Lead | Literature, datasets, hardware feasibility verdict |
| ✅ `PRD.md` (Phase 1) | ML Lead | Vision, requirements, success metrics |
| ✅ `TRD.md` (Phase 2) | ML Lead | Architecture, stacks, file-based design |
| ✅ `UserJourney.md` (Phase 2) | ML Lead | Flows, error/edge cases |
| ✅ `Schema.md` (Phase 2) | Graph Lead | Data architecture (file-based artifacts) |
| ✅ `Implementation.md` / `Rules.md` / `Tracker.md` / `Evaluation.md` / `RiskRegister.md` | ML Lead | Phases 3–4 docs |

## Active Tasks

| Item | Owner | Notes |
|---|---|---|
| 🔄 `Design.md` (neutral redo) | Frontend Lead | Replacing the earlier over-scoped draft with a lean, neutral version |
| 🔄 Environment setup on both machines | ML Lead | **5070 needs PyTorch ≥2.7 + CUDA 12.8 (cu128)** — verify `get_device_capability()==(12,0)` |
| 🔄 Dependency shake-out | ML Lead | Confirm GDAL/rasterio, smp, networkx, sknw, skimage, albumentations, osmnx import cleanly; pin `requirements.txt` |
| 🔄 Dataset download + caching | ML Lead | DeepGlobe, SpaceNet sample, Sentinel-2/LISS-IV tiles |
| 🔄 OSM→mask auto-labelling script | ML Lead | osmnx → rasterio; buffer 3–5 px; QC samples |
| 🔄 Graph/analysis on OSM graph | Graph Lead | Build skeleton→graph→MST/Union-Find, betweenness, ablation, global-efficiency RI on an OSMnx graph (no segmentation needed yet) |

## Pending Tasks

| Item | Owner |
|---|---|
| ⏳ Fine-tune segmentation model (SegFormer/U-Net) | ML Lead |
| ⏳ Run healing on real predicted masks | Graph Lead |
| ⏳ Build Streamlit + Folium dashboard | Frontend Lead |
| ⏳ Wire phases end-to-end (walking skeleton → integration) | ML Lead |
| ⏳ Run evaluation suite + ablations (`Evaluation.md`) | All |
| ⏳ Record backup demo capture | All |

## Bugs

| ID | Description | Status |
|---|---|---|
| — | none logged yet | — |

## Issues

| ID | Description | Status |
|---|---|---|
| I-1 | GDAL/rasterio installs can differ across machines — validate before relying on them | 🔄 watching |

## Blockers

| ID | Description | Status |
|---|---|---|
| — | none currently | — |

## Team Notes

- **Workstream split:** ML/Segmentation (Akshat, also integration), Graph & Analysis (Shaivi), Frontend/Design (Saanvi). Workstreams are decoupled via file handoffs so nobody blocks anyone.
- **Key decisions on record:** Resilience Index uses **global efficiency** (not raw average path length); frontend is **Streamlit + Folium** (pure Python, no JS SPA); repo kept **neutral/generic**; segmentation is **fine-tune pretrained only** (no training from scratch).
- **Hardware note:** the strongest GPU (RTX 50-series) is on the Graph Lead's machine; ML Lead sets up its training environment remotely, then it hosts the long training runs.
- **Design.md:** being rebuilt neutral and lean; other docs reference it only generically, so the redo doesn't affect them.

## Daily Logs

> Template — copy the block for each working day.

**2026-06-22**
- Completed Phase 1–2 docs and Phase 3–4 docs (Implementation, Rules, Tracker, Evaluation, RiskRegister).
- Agreed parallel-workstream plan; ML + Graph leads starting environment setup, data pipeline, and the OSM-graph analysis spike.
- Frontend lead restarting `Design.md` as a neutral version.

```
**YYYY-MM-DD**
- Done:
- In progress:
- Blockers:
- Next:
```