# bugs.md — Current Open Issues

> Historical review findings and their original section numbers are preserved in [`docs/ProductionReadinessAudit-2026-07.md`](docs/ProductionReadinessAudit-2026-07.md). Current code uses durable task/contract references rather than depending on those archived section numbers.

**Last reconciled:** 2026-07-14 · **Active program:** A46 → web replacement/O1

## Correctness and evidence

- [x] **A45-C1 — Preserve the baseline node universe in every resilience path.** Multi-step and uploaded-image failure paths now isolate failed nodes/use the shared metric contract; fixed-source regression tests keep RI comparable and bounded.
- [x] **A45-C2 — Correct graph-evaluator bridge labeling.** Reports now distinguish five build-time healing additions from four final surviving `is_bridged` edges.
- [x] **A45-C3 — Unify model-selection inference.** Selection, clean/gray/forget gates and occlusion evaluation use labeled `hann_blended_probability_v1`; incompatible resume histories fail closed.
- [x] **A45-C4 — Regenerate current derived sample evidence.** Resilience, flood, percolation, graph evaluation and plots were regenerated; the manifest records the source graph, artifact hashes and commands.
- [x] **A45-C5 — Enforce immutable deployment refs.** `update.sh` accepts only an exact full SHA or application release tag and rejects branches, abbreviations and model tags.
- [x] **A45-C6 — Correct public Methodology evidence.** DeepGlobe evidence is labeled historical, and the corrected current graph/resilience report is exposed.
- [x] **A45-C7 — Make CLI help Windows-console safe.** Current ASCII help succeeds under cp1252.
- [x] **A45-C8 — Make `run_pipeline --config` behavior match help.** Explicit CLI fields override config; unspecified fields retain file/default values.

## Model and data

- [ ] **A46-M1 — Raise A18 absolute routing.** LoRA r=4 wins the relative common-unit gate but captures only about 18% of achievable chip routing.
- [ ] **A46-M2 — Resolve SAM-Road++ licensing/reproducibility.** The inspected upstream snapshot has no clear license/config/dependency lock; do not redistribute code or weights until resolved.
- [ ] **A46-M3 — Add untouched geography/sensor evidence.** Mumbai is development-only; freeze a new set before inspecting results.
- [ ] **A46-M4 — Validate real Cartosat PAN.** Current grayscale numbers are only an RGB-to-gray proxy.

## Production operator checklist (O1)

- [ ] Create/approve an immutable application release ref that contains the July hardening work.
- [ ] Make `roadresilience-update.service` require its environment file instead of treating it as optional.
- [ ] Install/update the user systemd units, dependency environment, Caddy body cap and journald cap on the Oracle host.
- [ ] Redeploy the Modal endpoint from the approved code/checkpoint checksum and rotate/verify the shared key.
- [ ] Verify port 8501 is closed in both Oracle networking and host firewall; external probing currently cannot prove both internal controls.
- [ ] Decide and apply Caddy/edge rate limiting.
- [ ] Run the public sample and upload flows, including cold start, queue, restart/recovery and failure cases.
- [ ] Record the deployed ref, Modal function/checkpoint checksum and smoke timestamp in `Tracker.md`.

## Repository and product hygiene

- [x] **A45-R1 — Separate clean dependency roles.** Core, hosted-app, training and additive dev roles now compose into the CI/full-development aggregate.
- [x] **A45-R2 — Make the notebook a thin launcher.** The notebook calls maintained training code; a structural test prevents local trainer definitions from returning.
- [x] **A45-R3 — Replace archived `bugs.md §…` code comments.** Current code/tests use durable task/contract references or self-contained rationale.
- [ ] **O1-R1 — Reconcile the public/default branch and application release.** GitHub `main` remains far behind `dev`; decide the human stage-gate update and create an approved application ref before claiming the public repository/release is current.
- [ ] **A46-R1 — Track A18/A46 configs and result artifacts** in a license-safe reproducible form.
- [ ] **LEGAL-1 — Choose a repository code license.** No top-level `LICENSE` currently grants reuse rights; dataset/provider licenses are separate and must not be treated as the code license.
- [ ] **F9 — Replace Streamlit/Folium with the authorized researched web experience** after A45/A46, with visual, browser, responsive, accessibility and performance-budget gates from `Design.md`.

## Recently closed

- [x] First production-readiness batch: metric/geospatial corrections, content/config stage signatures, queue claims/leases/JSON results, auth-before-decode, dependency smoke, rollback tooling and architecture/docs corrections (PR #125).
- [x] Graph-label coordinate invariant and upstream-license warning recorded for A18 (PR #126).
- [x] Current Panaji graph/APLS evidence was refreshed during A44; A45 then regenerated resilience, flood and percolation artifacts and added the evidence manifest.
