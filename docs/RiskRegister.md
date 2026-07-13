# RiskRegister.md — Current Risks

> Probability (P) and impact (I) are Low, Medium or High. Closed historical setup risks are not kept as active warnings; their evidence remains in `Tracker.md`, `Research.md` and Git history.

## Active risks

| ID | Risk | P | I | Current mitigation / next gate | Owner |
|---|---|---|---|---|---|
| **M-1** | Mumbai has been repeatedly consulted, so model decisions overfit a single-city development benchmark | H | H | Treat Mumbai as development only; reserve a new geography/sensor before final claims | Akshat |
| **M-2** | A18-LoRA wins relatively but still captures only about 18% of achievable chip routing | H | H | A46 targets encoder adaptation, training coverage and context; require absolute and paired gains | Akshat |
| **M-3** | Real Cartosat PAN behavior may differ from the RGB-to-gray proxy | M | H | Keep proxy claims explicit; run real-PAN QC/evaluation as soon as data is available | Akshat |
| **M-4** | The inspected SAM-Road++ snapshot has no clear license or reproducible dependency lock | H | H | Research locally only; do not redistribute upstream code/derived weights until licensing is clarified | Akshat |
| **G-1** | Graph simplification/healing can create plausible but false connections | M | H | Preserve probability support, crossing/angle checks, inferred-edge flags and APLS/fragmentation evaluation | Shaivi |
| **G-2** | Exact centrality/efficiency remains expensive on city-scale graphs | M | M | Use fixed-source sampling and caches; benchmark A45 changes on large synthetic graphs | Shaivi |
| **G-3** | Multi-step `ablation_curve` currently shrinks the node universe, so RI can exceed 1 and old curve evidence is invalid | H | H | A45 must isolate failed nodes/preserve the baseline denominator, update tests, then regenerate sample/evaluation curves | Shaivi |
| **P-1** | Repository deploy hardening may be ahead of the live Oracle/Modal configuration | M | H | Complete O1 operator checklist and verify the upload flow from the public URL | Akshat |
| **P-2** | Public endpoint has no verified rate limiting | M | M | Decide/install Caddy rate-limit module or document an alternative control; retain Modal concurrency/cost caps | Akshat |
| **P-3** | Filesystem queue is appropriate for one host but not horizontal scale | L | M | Keep single-host scope explicit; preserve atomic claim/lease/JSON recovery tests | Saanvi/Akshat |
| **C-1** | Large modules and accumulated experiment paths increase change risk | H | M | A45 characterization tests, responsibility splits and dead-path quarantine; measure complexity before/after | Coordinator |
| **C-2** | Local GPU environment contains dependency conflicts despite passing tests | M | M | Use documented isolated environments; CI is the clean reference; avoid mixing TensorFlow/Google stacks | Akshat |
| **U-1** | UI overhaul can regress mature analysis and recovery behavior | M | H | F9 follows A45/A46; retain browser flows, job recovery and production smoke as gates | Saanvi |
| **D-1** | Documentation can drift faster than code and misroute agents/operators | M | H | A44 reduces duplication; Tracker holds current state, topic docs hold details, CI/docs checks added where practical | Coordinator |

## Accepted constraints

- Training needs GPU access; Colab/Kaggle and optional local NVIDIA GPUs are supported. P2/P3/dashboard remain CPU-capable.
- The public demo is a single-host research prototype, not a multi-tenant production platform.
- No database, user accounts or JavaScript SPA are planned for this release.
- SpaceNet/DeepGlobe/OSM/Cartosat licensing constrains data and model redistribution.

## Closed or materially reduced risks

- Disconnected-graph arithmetic: mitigated by baseline-normalized global efficiency and tests.
- Phase integration: the P1–P4 seam and end-to-end runner exist and fail loudly.
- Stale stage reuse: content/config signatures replaced mtime-only checks.
- Deployment rollback/health transaction: implemented. Immutable-ref **enforcement** is still open because `update.sh` accepts branch names; A45-C5/O1 must close it before claiming moving-branch prevention.
- Missing production dependencies: CI installs `deploy/requirements-app.txt` and exercises mask-to-resilience.
- Queue duplicate/recovery hazards: atomic claims, owner leases, persistent ordering and JSON results are tested for the single-host design.

## Highest-priority watch list

1. New-geography/sensor evidence before any final model claim.
2. A18 licensing before redistribution or deployment.
3. O1 live-service reconciliation and rate limiting.
4. A45 complexity reduction without contract or performance regressions.
5. F9 browser/accessibility regression coverage.
