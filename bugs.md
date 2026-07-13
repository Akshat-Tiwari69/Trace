# bugs.md — Current Open Issues

> Historical review findings and their original section numbers are preserved in [`docs/ProductionReadinessAudit-2026-07.md`](docs/ProductionReadinessAudit-2026-07.md). Older code comments that say `bugs.md §…` refer to that archive until A45 replaces them with durable rationale/task references.

**Last reconciled:** 2026-07-13 · **Active program:** A44 → A45/A46 → F9/O1

## Correctness and evidence

- [ ] **A45-C1 — Preserve the baseline node universe in multi-step ablation.** `ablation_curve()` removes nodes and recomputes efficiency with a smaller denominator, so RI can exceed 1 and historical resilience/flood curve absolutes are invalid. Isolate failed nodes or otherwise keep the baseline universe, add regression tests, then regenerate sample curves/plots/evaluation prose.
- [ ] **A45-C2 — Correct graph-evaluator bridge labeling.** The current Panaji build added five healed bridges but four survive in final GeoJSON. `graph_eval.json` reports the build-time count as final `bridged_edges`; report both concepts explicitly.
- [ ] **A45-C3 — Unify model-selection inference.** Threshold selection uses Hann-blended probabilities while forget/gray/per-tile scoring still uses the legacy hard-tiled path. Select thresholds and report/gate metrics from one shared probability protocol, then transparently re-baseline affected evidence.
- [ ] **A45-C4 — Regenerate current derived sample evidence.** A44 refreshed graph/APLS JSON, but percolation and resilience/flood artifacts still require regeneration after C1; ensure every committed report names the graph/input fingerprint.
- [ ] **A45-C5 — Enforce immutable deployment refs.** `update.sh` currently accepts `DEPLOY_REF=dev` because it resolves `origin/$DEPLOY_REF`; reject moving branches and accept only an approved tag or full commit SHA.
- [ ] **A45-C6 — Correct public Methodology evidence.** It currently labels the historical DeepGlobe `segmentation_eval.json` as SpaceNet-Mumbai held-out evidence and exposes the provisional graph-eval resilience subsection. Point it to correctly labeled development evidence and suppress/flag curve RI until C1 regeneration.
- [ ] **A45-C7 — Make CLI help Windows-console safe.** `python -m src.pipeline.run_pipeline --help` currently raises a cp1252 `UnicodeEncodeError` on the Unicode arrow in argparse text; use ASCII help text or an explicit UTF-8-safe console path.
- [ ] **A45-C8 — Make `run_pipeline --config` behavior match help.** The CLI claims flags override config fields, but with `--config` only image/checkpoint/AOI are applied; resolution/tile/threshold/device and other flags are ignored. Define explicit precedence and test it; also replace the stale “A5 walking skeleton” description.

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

- [ ] **A45-R1 — Separate clean dependency roles.** The mixed local GPU environment has package conflicts even though tests pass; create/test app/graph, training and dev/test environments while retaining a compatible aggregate path.
- [ ] **A45-R2 — Make the notebook a thin launcher.** Consolidate duplicated training/evaluation logic in modules before further model experiments.
- [ ] **A45-R3 — Replace archived `bugs.md §…` code comments** with task IDs or explanatory comments that survive document cleanup.
- [ ] **O1-R1 — Reconcile the public/default branch and application release.** GitHub `main` remains far behind `dev`; decide the human stage-gate update and create an approved application ref before claiming the public repository/release is current.
- [ ] **A46-R1 — Track A18/A46 configs and result artifacts** in a license-safe reproducible form.
- [ ] **LEGAL-1 — Choose a repository code license.** No top-level `LICENSE` currently grants reuse rights; dataset/provider licenses are separate and must not be treated as the code license.
- [ ] **F9 — UI/UX overhaul** after A45/A46, with browser, responsive and accessibility gates from `Design.md`.

## Recently closed

- [x] First production-readiness batch: metric/geospatial corrections, content/config stage signatures, queue claims/leases/JSON results, auth-before-decode, dependency smoke, rollback tooling and architecture/docs corrections (PR #125).
- [x] Graph-label coordinate invariant and upstream-license warning recorded for A18 (PR #126).
- [x] Current Panaji graph evaluation and APLS JSON regenerated during A44; remaining curve/percolation artifacts are explicitly queued above.
