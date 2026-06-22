# RiskRegister.md

> **Purpose.** This document lists the things that could go wrong on **Route Resilience**, how likely and how damaging each is, and the plan to prevent or absorb it. Identifying risks early is what turns "we got unlucky" into "we planned for that." Probability (P) and Impact (I) are rated **Low / Medium / High**. Update as risks open, close, or change.

---

## How to read this

- **P** = how likely it is to happen. **I** = how bad it is if it does.
- **Owner** = the role responsible for watching and mitigating it (roles defined in `Implementation.md`).
- The highest-priority risks (High×High / High×Medium) are summarized at the bottom.

## Technical Risks

| ID | Risk | P | I | Mitigation | Owner |
|---|---|---|---|---|---|
| T-1 | Segmentation underperforms on heavy occlusion | M | H | Occlusion augmentation + clDice loss + MST healing; fall back to a simpler U-Net baseline; **fallback MVP: run Phases II–IV on an OSM-derived graph** so the resilience story still works | ML Lead |
| T-2 | Resilience Index breaks (÷ by infinity) when the graph disconnects | L | H | **Already mitigated by design** — use global efficiency, which stays finite. Keep it; never revert to raw average-path-length ratio | Graph Lead |
| T-3 | Betweenness centrality too slow on large city graphs | M | M | NetworkX **k-sample** approximate betweenness; precompute once; analyze a sub-region if needed | Graph Lead |
| T-4 | Phase-to-phase integration fails late | M | H | Define file-handoff contracts up front; build the **walking skeleton in Sprint 0**; integration buffer in Sprint 2 | ML Lead |
| T-5 | "Betweenness = resilience" critique | L | M | Pair betweenness with global-efficiency degradation under ablation; frame betweenness as a chokepoint *indicator*, not a guaranteed resilience measure (see `Research.md`) | Graph Lead |

## Dataset Risks

| ID | Risk | P | I | Mitigation | Owner |
|---|---|---|---|---|---|
| D-1 | OSM auto-labels are incomplete/misaligned (weak labels) | H | M | Buffer masks (3–5 px), visual QC, use relaxed/buffered IoU; treat metrics as indicative | ML Lead |
| D-2 | Cartosat-3 access/format uncertain until provided | M | M | Build the pipeline format-agnostic (rasterio/GDAL); pretrain entirely on open data so we only adapt at the end | ML Lead |
| D-3 | Domain gap — benchmarks are mostly non-Indian cities | M | M | Fine-tune on LISS-IV + OSM-labelled Indian AOIs; hold out an Indian city for the generalisation metric | ML Lead |
| D-4 | License non-compliance (OpenSatMap non-commercial, OSM ODbL) | L | M | Record source + license per dataset; attribute imagery; **don't commit raw/restricted data**; keep usage non-commercial | All |

## Infrastructure Risks

| ID | Risk | P | I | Mitigation | Owner |
|---|---|---|---|---|---|
| I-1 | RTX 50-series (Blackwell, sm_120) not picked up by PyTorch | M | H | Install **PyTorch ≥2.7 + CUDA 12.8 (cu128)**; verify `get_device_capability()==(12,0)` **early**; ML Lead sets it up remotely on the Graph Lead's machine | ML Lead |
| I-2 | 8 GB VRAM out-of-memory during training | M | M | AMP/FP16, small tiles (256/512), gradient accumulation + checkpointing; **free Colab/Kaggle (16 GB) overflow** | ML Lead |
| I-3 | Laptop GPU thermal throttling on long runs | M | L | Cooling/airflow, performance power mode, smaller tiles, **frequent checkpoints** (lose ≤1 epoch on shutdown) | ML Lead |
| I-4 | GDAL/rasterio install differs across machines | M | M | Validate installs in Sprint 0; pin versions / prefer conda; document the working setup | ML Lead |
| I-5 | Free-cloud session limits + ephemeral storage | M | M | Save checkpoints off-device (Drive/Kaggle Datasets); plan runs within session caps | ML Lead |

## Team Risks

| ID | Risk | P | I | Mitigation | Owner |
|---|---|---|---|---|---|
| TM-1 | Strongest GPU sits with the less-experienced coder | M | M | ML Lead configures that machine remotely; Graph Lead owns the more approachable **classical-Python** graph half + hosts training runs | ML Lead |
| TM-2 | Single points of knowledge (only one person understands a part) | M | M | Docs-first; shared repo; brief walkthroughs at handoffs | All |
| TM-3 | Coordination friction across three machines | L | M | Clean file-handoff contracts; git discipline; `Tracker.md` kept current | All |

## Timeline Risks

| ID | Risk | P | I | Mitigation | Owner |
|---|---|---|---|---|---|
| TL-1 | Integration underestimated, eats the end of the build | M | H | Walking skeleton early; reserve Sprint 2 for integration; protect the core interaction over polish | ML Lead |
| TL-2 | Scope creep from the ambitious design vision | M | M | v1 scope is fixed (map + criticality + disable-node sim); aspirational features parked (see `PRD.md`/`Design.md`) | All |
| TL-3 | Pre-build (setup/data) not finished before the intensive window | M | H | **Front-load Sprint 0 now** — env, deps, data, OSM masks, walking skeleton done ahead of time | ML Lead |

## Financial Risks

| ID | Risk | P | I | Mitigation | Owner |
|---|---|---|---|---|---|
| F-1 | Project cost | L | L | Effectively zero — all tools free/open, free-cloud GPUs, commodity hardware. Optional small cloud-GPU buffer only if needed | ML Lead |

## Top Risks to Watch

The few that most deserve attention (high impact and non-trivial probability):

1. **I-1 — Blackwell/CUDA setup** (M×H): verify the RTX 50-series PyTorch install *first*, before anything depends on it.
2. **T-4 / TL-1 — integration left too late** (M×H): the walking skeleton in Sprint 0 is the single best defence.
3. **TL-3 — pre-build not done in time** (M×H): front-load setup, data, and the OSM-graph spike now.
4. **T-1 — segmentation underperforms on occlusion** (M×H): mitigated by the augmentation+loss+healing stack, with the OSM-graph fallback as insurance.
5. **D-1 — weak OSM labels** (H×M): expected and manageable with buffering, QC, and relaxed metrics.