# bugs.md — Current Open Issues

> Historical review findings and their original section numbers are preserved in [`docs/ProductionReadinessAudit-2026-07.md`](docs/ProductionReadinessAudit-2026-07.md). Current code uses durable task/contract references rather than depending on those archived section numbers.

**Last reconciled:** 2026-09-29 · **Active program:** X1 demo capture

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

- [x] Closed 2026-07-28: immutable release ref, required updater environment file, host units/caps, Modal redeploy and key rotation, closed legacy ports, application throttles, public sample/upload smoke and deployment record. Evidence: `docs/Tracker.md` §10 (2026-07-28).

## 2026-09-29 audit follow-ups

- [x] **F12-C1 — Empty `failed=` URL pre-failed junction 0.** `Number("")` is `0`, so every fresh visit silently added J-0 to the scenario (J-278 alone showed 2.2% loss instead of 1.5%). Fixed with `parseNodeIds()` and a unit regression.
- [x] **F12-D1 — `next` critical/high advisories.** Updated to 16.3.6 with non-breaking transitive fixes (npm audit 11 → 3).
- [x] **O2 — Deploy F12.** Live at `828fd531e2213068c9bdb4cb1b9498be5b4ccbfb` since 2026-09-28 22:27:56 UTC; public J-278 scenario verified.
- [ ] **D2 — `maplibre-gl` critical advisory (DOM.sanitize XSS).** Low exposure (no popups/`setHTML`; attribution comes from the OpenFreeMap style), but the fix needs the 6.x major upgrade with map regression checks.
- [ ] **D3 — `vitest` moderate advisory.** Test-only; needs the 5.x major upgrade.
- [ ] **D4 — Python dependencies are not vulnerability-scanned** and GitHub Dependabot alerts are disabled for the repository.
- [ ] **OPS-1 — No external uptime monitoring.** A transient network-path outage (~21:40–21:45 UTC 2026-09-28) was noticed only by manual probing; the host itself stayed up.
- [ ] **F12-W1 — Methodology prefetch 404 (cosmetic).** The static export writes `methodology/__next.methodology/__PAGE__.txt`, but the client prefetches `__next.methodology.__PAGE__.txt`; still present on `next` 16.3.6. Navigation works; only a console error appears. `prefetch={false}` on the Method link would silence it.
- [x] **OPS-2 — `rpcbind` listened on the host.** Disabled during the 2026-09-29 maintenance (no NFS mounts or dependants).
- [x] **OPS-3 — HTML was served without `Cache-Control`.** Browsers could heuristically reuse a stale page for days after a deploy (observed during O2). The API middleware now defaults unhashed responses to `no-cache` (ETag revalidation returns `304`); hashed `/_next/static/` assets stay immutable and routes with their own policy keep it. Live at `ae4c6ff` since 2026-09-29 14:36 UTC.

## Repository and product hygiene

- [x] **A45-R1 — Separate clean dependency roles.** Core, hosted-app, training and additive dev roles now compose into the CI/full-development aggregate.
- [x] **A45-R2 — Make the notebook a thin launcher.** The notebook calls maintained training code; a structural test prevents local trainer definitions from returning.
- [x] **A45-R3 — Replace archived `bugs.md §…` code comments.** Current code/tests use durable task/contract references or self-contained rationale.
- [ ] **O1-R1 — Reconcile the public/default branch and application release.** GitHub `main` remains far behind `dev`; decide the human stage-gate update and create an approved application ref before claiming the public repository/release is current.
- [ ] **A46-R1 — Track A18/A46 configs and result artifacts** in a license-safe reproducible form.
- [ ] **LEGAL-1 — Choose a repository code license.** No top-level `LICENSE` currently grants reuse rights; dataset/provider licenses are separate and must not be treated as the code license.
- [x] **F9 — Replace Streamlit/Folium with the authorized researched web experience** (merged in PR #132 and live since O1).

## Recently closed

- [x] First production-readiness batch: metric/geospatial corrections, content/config stage signatures, queue claims/leases/JSON results, auth-before-decode, dependency smoke, rollback tooling and architecture/docs corrections (PR #125).
- [x] Graph-label coordinate invariant and upstream-license warning recorded for A18 (PR #126).
- [x] Current Panaji graph/APLS evidence was refreshed during A44; A45 then regenerated resilience, flood and percolation artifacts and added the evidence manifest.
