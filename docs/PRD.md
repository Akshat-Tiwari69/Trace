# PRD.md — Product Requirements

## Product statement

Route Resilience turns satellite imagery or a prepared road graph into an explainable resilience analysis. It extracts road evidence, repairs selected gaps with explicit uncertainty, identifies structural chokepoints and lets a planner explore how junction or area failures change routing and global efficiency.

The product is a research prototype for decision support—not a certified road map, emergency-routing authority or proof that every inferred connection exists on the ground.

## Problem

Satellite-derived road masks fragment under shadows, vegetation, buildings, vehicles and sensor/domain change. Pixel accuracy alone does not show whether the result is useful for routing. Even a good road graph still needs a finite, interpretable way to describe which failures matter.

Route Resilience addresses both layers:

1. produce road masks/graphs while preserving confidence, geometry and provenance;
2. measure topology and network degradation rather than relying only on pixels;
3. make the evidence explorable by a non-technical user.

## Users

- **Urban and transport planners:** identify junctions and corridors that deserve redundancy or maintenance attention.
- **Disaster-management teams:** explore localized and compound failures before an incident.
- **Geospatial/remote-sensing reviewers:** inspect extraction, topology, provenance and limitations.
- **Developers/researchers:** reproduce pipeline stages and evaluate model or graph changes.

## Current product capabilities

| ID | Requirement | Current state |
|---|---|---|
| **FR1** | Run an end-to-end imagery→mask→graph→analysis pipeline | Shipped via `src.pipeline.run_pipeline` |
| **FR2** | Accept local raster imagery, including georeferenced RGB/multiband/one-band GeoTIFF/PAN | Shipped for batch/local inference |
| **FR3** | Accept bounded PNG/JPEG imagery in the public dashboard | Shipped through authenticated Modal P1 |
| **FR4** | Fine-tune pretrained PyTorch road models; never train a backbone from scratch | Shipped; v3.2 remains deployed |
| **FR5** | Preserve mask alignment, optional probabilities and model provenance | Shipped |
| **FR6** | Convert masks to a routable MultiGraph and improve fragmentation without forcing false full connectivity | Shipped |
| **FR7** | Preserve parallel edges, geometry, inferred-edge flags, confidence when available and positive metric lengths | Shipped |
| **FR8** | Rank critical nodes and expose articulation points/bridges | Shipped |
| **FR9** | Simulate single-junction and area/compound failures | Shipped, including drawn flood polygons and keyboard alternatives |
| **FR10** | Report a finite, bounded Resilience Index based on baseline-normalized global efficiency | Single-scenario path shipped; multi-step curve denominator fix is queued in A45 |
| **FR11** | Show rerouting, criticality, resilience curves and rankings in a Streamlit/Folium dashboard | Shipped |
| **FR12** | Export the current graph and a concise visual summary | Shipped as GeoJSON and PNG |
| **FR13** | Evaluate routing/topology with a common-unit protocol and paired uncertainty | Shipped for v3.2 vs A18 research comparisons |

## Quality requirements

- **Honesty:** distinguish observed, predicted and healed roads; state benchmark and sensor limitations next to claims.
- **Reproducibility:** record checkpoint checksum, architecture, threshold, Git revision, config and stage signatures.
- **Correctness:** metric coordinates when georeference exists; positive edge lengths; graph/file round trips; bounded RI.
- **Performance:** P2/P3/dashboard remain CPU-capable; hosted P1 uses a scale-to-zero GPU; large operations use measured sampling/caching.
- **Accessibility:** important map interactions have labeled, keyboard-accessible alternatives; color is never the only signal.
- **Reliability:** queued uploads survive normal app reruns/restarts on the single host; production deploys use immutable refs, health checks and rollback.
- **Privacy/security:** no user accounts or permanent upload library; source bytes are processed in memory, while derived queue artifacts are age-cleaned; service authentication and size validation fail closed.

## Evidence and success criteria

| Area | Promotion/success criterion |
|---|---|
| Mask model | Report IoU/Dice on a named development/evaluation unit at the model’s deploy protocol; pass anti-forgetting and runtime/checkpoint gates |
| Routing model | Improve strict common-unit chip APLS with complete coverage and a paired interval excluding zero; also improve the absolute share of achievable routing |
| Graph healing | Improve connectivity/routing evidence without unacceptable false bridges; keep inferred edges inspectable |
| Resilience | Targeted failures degrade baseline-normalized global efficiency faster than matched random failures on the evaluated graph; RI remains in `[0,1]` |
| Generalization | Final claims require an untouched geography/sensor; Mumbai alone is development evidence |
| Product | A non-technical user can understand the baseline, run a scenario or upload, interpret limitations and export a result without developer assistance |
| Operations | The approved immutable ref is live and the public sample/upload flows pass an operator smoke test |

## Current evidence

- v3.2 is the best deployed **mask** model on the repeatedly consulted SpaceNet-5 Mumbai development benchmark.
- Heavy TTA, stronger occlusion fine-tuning, clDice-first fine-tuning, Massachusetts mixing, OSM mean-teacher self-training, foreground-biased crops and SDT-BCE did not produce a safe routing promotion.
- A18 SAM-Road++ and its LoRA variant beat v3.2 on the same chip/vector-GT routing frame, validating graph-first research. Absolute routing remains too low for deployment.
- Real Cartosat PAN and new-geography generalization remain unproven.

`Evaluation.md` owns exact numbers and protocols; `Research.md` owns experiment rationale and negative results.

## Current scope

In scope:

- Single-AOI research analysis from committed sample artifacts or one uploaded/local image.
- Pretrained PyTorch model fine-tuning and graph-first research.
- Classical CPU graph construction, healing, criticality and global-efficiency scenarios.
- Public Streamlit/Folium demonstration on a single Oracle host with Modal P1.
- Reproducible files, evaluation and exports.

Out of scope:

- Certified navigation, emergency dispatch or claims of complete road-map accuracy.
- Accounts, collaborative projects, permanent storage, database or multi-tenant platform.
- Live traffic/GPS feeds and lane-level travel-time modeling.
- National-scale serving or horizontally distributed queue workers.
- JavaScript SPA/mobile rewrite in this release.
- Final sensor/geographic claims before new held-out evidence exists.

## Next product gates

1. A44 documentation and evidence coherence.
2. A45 code simplification/performance with preserved contracts.
3. A46 graph-first absolute-routing improvement plus licensing/reproducibility resolution.
4. F9 UI/UX overhaul on the stable architecture.
5. O1 live deployment reconciliation and X1 final capture.

The ordered execution plan lives in `Implementation.md`; live status lives only in `Tracker.md`.
