# TRACE product requirements

## Product statement

TRACE is a map-led route-resilience demonstrator that converts satellite imagery into a routable road graph, ranks vulnerable junctions, and shows how targeted failures change network efficiency. It is designed to make an ML/graph pipeline inspectable and memorable in a recruiter or technical-review setting.

## Goals

- Demonstrate a real end-to-end P1→P2→P3 system, not a static design mock.
- Present committed sample evidence immediately without a checkpoint or GPU.
- Let users stress, compare, recover, and export a scenario.
- Let authorized users submit their own imagery through the authenticated Modal P1 boundary and CPU P2/P3 queue.
- Be responsive, keyboard-operable, accessible, performant, and deployable on the existing small Oracle ARM host.
- State model, metric, coordinate, sampling, and generalization limits honestly.

## Non-goals

- Turn-by-turn navigation, traffic prediction, or real-time road closures.
- A claim that criticality-ordered recovery is an optimal capital plan.
- A database, account system, team workspace, or saved cloud projects.
- Invented longitude/latitude for uploaded imagery without georeference.
- Deployment of an unlicensed research model, even if its evaluation score is higher.

## Functional requirements

| ID | Requirement | Acceptance |
|---|---|---|
| FR1 | Load a committed sample AOI | Panaji metadata, graph, criticality, curve, and manifest load without GPU/checkpoint |
| FR2 | Explore critical junctions | Map and semantic ranking select the same junction and expose rank/role |
| FR3 | Run node-failure scenarios | API returns bounded RI, efficiency loss, component fractions, sampling disclosure, and route/disconnection evidence |
| FR4 | Compare baseline/scenario | Same graph supports legible comparison without a duplicate payload |
| FR5 | Recover failures | Removed junctions can be restored in criticality order with deterministic recalculation |
| FR6 | Preserve/share state | Mode, selected junction, and failed set survive reload through the URL |
| FR7 | Export current evidence | GeoJSON and criticality CSV reflect the current scenario |
| FR8 | Analyze uploaded imagery | Validated PNG/JPEG → Modal mask → queued CPU P2/P3 → status/result/image-space graph |
| FR9 | Recover queued work | Filesystem queue survives a normal process restart and cleans stale artifacts |
| FR10 | Explain methodology | Metric, sample/model provenance, coordinate policy, sampling, and limitations are public |
| FR11 | Fail honestly | Tile, API, upload, configuration, and analysis failures show actionable safe states; no fabricated output |

## Quality requirements

| ID | Requirement | Acceptance |
|---|---|---|
| QR1 | Correctness | Baseline-normalized global efficiency preserves the baseline node universe and is finite in `[0,1]` |
| QR2 | Graph fidelity | Parallel edges, closed rings, positive lengths, coordinate metadata, and GraphML/GeoJSON agreement are tested |
| QR3 | Accessibility | WCAG 2.2 AA; zero serious/critical axe findings; keyboard completion; 44 px primary targets; reduced motion |
| QR4 | Responsiveness | No horizontal overflow and usable flows at 375, 768, 1024, and 1440 px; 200% zoom remains usable |
| QR5 | Performance | Static bundle/payload budgets pass; live p75 target LCP ≤2.5 s, INP ≤200 ms, CLS ≤0.1 |
| QR6 | Security | Upload type/signature/size/dimension validation, explicit processing consent, rate limits, trusted hosts, safe errors, loopback app port |
| QR7 | Reproducibility | Seeds, sample size/method, versions, immutable release ref, artifact hashes, and evidence are recorded |
| QR8 | Operability | One-worker Uvicorn, Caddy TLS/compression/body limit, systemd hardening, health checks, automatic rollback |

## Evidence and model policy

- Production P1 remains the licensed PyTorch `a4-roadseg-v3.2` checkpoint at threshold `0.52` until a replacement passes the frozen promotion protocol and license review.
- The SAM-Road++ graph-first candidate won the registered routing comparison but cannot be shipped because no usable upstream license is published.
- SpaceNet-5 Mumbai is a repeatedly consulted development benchmark, not an untouched final test set.
- The product does not claim demonstrated generalization to every city, sensor, weather condition, or road class.
- Sample and live metrics come from Python domain code; the React client only formats and visualizes them.

## Release definition

A release is complete only when:

1. Python unit/integration tests and production-dependency smoke pass.
2. TypeScript typecheck, lint, unit tests, static build, and bundle budgets pass.
3. Browser stress/upload/export/responsive/axe journeys pass.
4. Independent code, Python, security, and UI reviews have no unresolved blocking findings.
5. The PR is approved and merged into `dev`, never `main`.
6. An immutable full SHA/tag is deployed; live home, health, simulation, upload validation, authenticated upload, firewall, and rollback checks pass.
7. The deployed SHA, Modal checkpoint checksum/ref, time, and evidence are recorded in `docs/Tracker.md`.
