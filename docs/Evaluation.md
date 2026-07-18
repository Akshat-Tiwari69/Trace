# Evaluation.md — Evidence, Protocols and Verdicts

> Exact results live here. `Research.md` explains hypotheses; `Tracker.md` records current decisions. Numbers from different spatial units or ground-truth constructions are not compared as if they were the same metric.

## Evidence boundaries

- **SpaceNet-5 Mumbai is a development benchmark.** Its chips have guided repeated model, threshold and architecture decisions; it is not an untouched final test set.
- **Grayscale is a PAN proxy.** It is RGB imagery converted to gray, not real Cartosat-3 PAN with its native spectral response, noise and MTF.
- **DeepGlobe is historical in-domain evidence.** It does not prove Indian or sensor generalization.
- **Panaji sample graph metrics** describe one committed demonstration graph and OSM comparison.
- **A18 chip APLS** and the older tile-mask APLS use different prediction units/ground-truth constructions. Compare candidates only within the same protocol.
- A final claim requires a new geography/sensor that has not guided development.

## Metrics

### Mask metrics

- **IoU:** road-mask intersection over union.
- **Dice:** overlap harmonic score.
- **Synthetic occlusion recall:** recall on programmatically hidden pixels; useful for controlled comparison but not proof of real canopy/shadow recovery.

### Routing/topology metrics

- **APLS:** similarity of shortest-path lengths after spatial snapping.
- **Reachable-pair fraction/fragmentation:** reported alongside APLS so a tiny reachable subset cannot appear healthy.
- **Common-unit chip gate:** both models are evaluated on the same frozen SpaceNet chips, vector ground truth and coordinate frame, with complete coverage and paired uncertainty.

### Graph/resilience metrics

- **Global efficiency:** mean inverse shortest-path distance, finite under disconnection.
- **Resilience Index:** post-failure efficiency divided by intact efficiency while preserving the baseline node universe.
- **Connectivity/healing:** component count plus routing evidence before/after healing; a lower component count alone does not prove bridges are correct.

The single-scenario and multi-step paths now use the same node-universe contract. `ablation_curve()` isolates failed nodes by removing their incident edges, keeps every baseline node in the denominator and reuses one fixed sampled-source set. RI is validated as finite and bounded in `[0, 1]`. Curve artifacts committed before A45 remain superseded; the current sample set was regenerated from the corrected implementation and is fingerprinted by `panaji_demo_evidence_manifest.json`.

## Segmentation evidence

### Historical DeepGlobe v1

Source: `data/sample/segmentation_eval.json`.

| Model/protocol | IoU | Synthetic occlusion recall | Note |
|---|---:|---:|---|
| v1, flip + multi-scale validation | 0.6699 | — | Best historical DeepGlobe validation view |
| v1, deploy threshold 0.44 | 0.6617 | 0.7927 | Threshold selected to trade ≤0.01 clean IoU for recall |

This establishes a reproducible historical baseline, not deployment-domain generalization.

### Mumbai development benchmark — fixed threshold 0.44

Sources: `spacenet_mumbai_eval.json` and `_gray.json`, 127 chips / 449 tiles, 512 px.

| Checkpoint | RGB IoU | Gray-proxy IoU | RGB Dice | Gray Dice |
|---|---:|---:|---:|---:|
| v3.2 `road_pan.pt` | 0.4017 | 0.3418 | 0.5732 | 0.5094 |
| v3/v3.1 weights `road_spacenet.pt` | **0.4311** | **0.3752** | **0.6025** | **0.5457** |
| v1 `deepglobe_…pt` | 0.3752 | 0.3183 | 0.5456 | 0.4829 |

This table isolates checkpoint behavior at one threshold. It is not each release at its deployed threshold.

### Mumbai development benchmark — development-set threshold sweep

Sources: `spacenet_mumbai_threshold_sweep*.json`. Threshold selection and reporting use the same repeatedly consulted development data, so these are best-case development numbers.

| Checkpoint | Best threshold | RGB IoU | Gray-proxy IoU | Release interpretation |
|---|---:|---:|---:|---|
| v3.2 `road_pan.pt` | 0.52 | **0.4594** | **0.4177** | 0.52 is the deployed v3.2 threshold |
| v3/v3.1 weights `road_spacenet.pt` | 0.50 | 0.4493 | 0.4046 | 0.50 is the v3.1 deploy threshold |
| v1 `deepglobe_…pt` | 0.50 | 0.3993 | 0.3447 | **Not** the v1 release threshold; v1 deploys at 0.44 |

Verdict: v3.2 is the best deployed mask model for this development benchmark, especially under the gray proxy. The result cannot establish new-city or real-PAN generalization.

### Legacy tile-mask routing comparison

Source: `spacenet_mumbai_apls.json`, 80 tiles.

| Checkpoint | Tile-mask APLS |
|---|---:|
| v3.2 | **0.4987** |
| v3/v3.1 weights | 0.4374 |
| v1 | 0.4198 |

This is useful for comparing the segmentation releases under the legacy mask→skeleton unit. It is **not comparable in absolute value** to A18 chip/vector-GT APLS.

## Graph-first evidence

### A41 SDT-BCE candidate — rejected

The anchored rerun preserved DeepGlobe (`0.6299` vs `0.6314`, delta `−0.0015`) and improved Mumbai pixels, but routing regressed:

- n=80 paired tile-mask APLS: v3.2 `0.4942` vs A41b `0.4525`
- delta `−0.0417`, 95% CI `[−0.0735, −0.0113]`, p=`0.006`

Verdict: no v3.3; pixel improvement without routing improvement is insufficient.

### A18 frozen and LoRA — direction validated, not deploy-ready

Protocol: strict common-unit chip APLS on 127/127 frozen SpaceNet-Mumbai chips, same vector ground truth/frame and paired comparison against v3.2.

| Candidate | Raw APLS | v3.2 raw | Raw delta | Normalized by GT-self ceiling | v3.2 normalized | Verdict |
|---|---:|---:|---:|---:|---:|---|
| A18 frozen encoder | 0.0502 | 0.0121 | +0.0381, CI `[+0.0281,+0.0492]` | 0.0862 | 0.0185 | wins relative gate |
| A18 LoRA r=4 | **0.1051** | 0.0121 | **+0.0930**, CI `[+0.0782,+0.1095]` | **0.1808** | 0.0185 | wins decisively |

The normalized diagnostic estimates the fraction of achievable routing under fragmented chip GT. LoRA roughly doubled the frozen result, showing encoder adaptation was a bottleneck. But `0.1808` is still only about 18% of the achievable ceiling: A18 is a research direction, not a deployment candidate.

Fragmentation diagnosis on those 127 chips: LoRA averages 11.43 components vs 3.46 for GT, largest-component node fraction 0.376 vs 0.719, 4.26 isolated nodes, six empty graphs and roughly 48% of GT edge length. It is more fragmented than GT on 114/127 chips. Missing nodes/edges/connectivity—not background false positives—is the immediate target.

Reproducibility limitation: the inspected upstream snapshot lacks a clear license/dependency lock, and the exact run workspace/config/result JSON currently live outside the tracked release artifacts. A46 must resolve this before redistribution or promotion.

## Current Panaji sample evidence

The sample graph/evidence was regenerated during A45 after the resilience-normalization correction.

Sources: `panaji_demo_graph.geojson`, `panaji_demo_graph_eval.json`, `panaji_demo_apls.json`, `panaji_demo_resilience.csv`, `panaji_demo_percolation.json`, the two current curve plots and `panaji_demo_evidence_manifest.json`. The manifest records the source-graph SHA-256, artifact hashes and exact regeneration commands.

| Property | Current value |
|---|---:|
| Nodes / edges | 364 / 500 |
| Components during build healing | 8 → 3 |
| Connectivity-ratio change | +8.32% |
| Bridges added during build | 5 |
| Surviving final `is_bridged` edges | 4 |
| Final articulation nodes / structural bridges | 79 / 92 |
| Top critical node | 278 (`betweenness=0.3307`) |
| APLS vs cached OSM truth | **0.5369** |
| APLS directions | GT→proposal `0.4709`; proposal→GT `0.6242` |
| Reachable pair fraction | GT `0.9769`; proposal `0.9774` |
| 40-step targeted RI mean / end | `0.6800` / `0.4477` |
| 40-step random RI mean / end | `0.8105` / `0.6135` |
| 25-removal targeted / random RI | `0.576689` / `0.770555` |

`graph_eval.json` now separates the five bridges added during build-time healing from the four final edges that still carry `is_bridged`; it does not conflate construction history with final graph state. Targeted failures degrade this sample faster than the seeded random baseline. These are demonstration-graph results, not a citywide generalization claim.

## A45 correctness and performance evidence

All timings below are local focused benchmarks, not production-SLA claims. Each optimized path retained characterization tests and exact outputs where stated.

| Path | Before | After | Result |
|---|---:|---:|---|
| APLS one-way, 20×20 synthetic grid / 1,000 sampled pairs | `1.0144 s` median | `0.3544 s` median | **2.86× faster**, same score |
| Curved-edge densification locator | `45.25 ms` | `3.51 ms` | **12.9× faster**, same geometry |
| 4,096² blended-inference Python allocation proxy | `378.1 MiB` | `137.0 MiB` | **63.8% lower**; output SHA-256 unchanged |
| 181.4 MiB checkpoint identity, three repeat lookups in one run | three full scans | one scan + `0.0018 s` cached lookups | checkpoint scans **3→1** |
| Synthetic training evaluator | `0.0410 s` median | `0.0218 s` median | **46.8% faster**, exact metrics |

The blended-inference change streams batches instead of retaining every window and probability. Its measured CPU time was effectively flat across 512²–4,096² inputs (worst observed change `+6.8%`), so the supported claim is lower memory, not higher throughput.

Fine-tune threshold selection, clean/gray checks, forget gates and synthetic-occlusion evaluation now share the labeled `hann_blended_probability_v1` probability protocol. Resume rejects histories without that label, preventing a mixed-protocol run. Historical A6–A41 training histories predate the label and remain historical evidence; they are not silently reinterpreted. A46 must establish its own baseline and candidate under the labeled protocol.

## Experiment ledger

| Work | Result | Decision |
|---|---|---|
| A7 D4 TTA | ~flat/slightly worse, 8× compute | opt-in only |
| A8 heavier occlusion fine-tune | occlusion recall flat, clean IoU down | rejected |
| A9 clDice-first fine-tune | IoU and hard clDice down; confounded by Lovász removal | rejected; do not repeat as the same fine-tune |
| A11 Massachusetts mixing | Mumbai zero-shot worse | rejected |
| A12 OSM mean-teacher self-training | held-out development score worse | rejected; A20/A22 retired |
| A23/A24 real SpaceNet supervision + gray/radiometric aug | best deployed Mumbai-development mask/routing result | v3.2 adopted |
| A38/A38b foreground-biased crops | gray pixels improved, APLS significantly worse | rejected |
| A41/A41b SDT-BCE | pixels improved, APLS regressed | rejected |
| A18 frozen → LoRA | common-unit routing improved strongly, absolute routing still low | graph-first direction adopted for A46 |
| A46 epoch-22 + topology 0.60 | 127-chip raw APLS `0.181165`, `+0.076053` vs epoch-18, CI `[+0.059871, +0.092975]`; normalized `0.301519` | metric gate passed; not deployable because upstream has no license |

## Promotion protocol for A46

1. Freeze chip IDs, vector GT, coordinate transform, ground-sample distance and v3.2 reference outputs before training.
2. Assert graph-label alignment (`x=column`, `y=row`) on every converted sample.
3. Use the 102-chip validation split for checkpoint/threshold/hyperparameter selection. Keep the 127-chip comparison closed until one configuration is pre-registered; it remains development evidence, not a final test.
4. Require complete candidate/reference coverage; missing outputs fail the gate.
5. Report raw APLS, GT-self ceiling, normalized diagnostic, component/isolated-node/edge-length/reachability diagnostics and runtime.
6. Use paired chip resampling/randomization against the LoRA r=4 incumbent. The interval must exclude zero in the candidate's favor and the paired raw-APLS mean gain must be at least **0.02**. This floor is frozen before checkpoint selection and before reopening the 127-chip comparison.
7. Require candidate normalized APLS of at least **0.25**, a material absolute step beyond the incumbent's `0.1808` GT-self-ceiling fraction. Passing these metric floors identifies a research candidate; it does not waive step 9.
8. Validate a later candidate on an untouched geography/sensor before final generalization language.
9. Before deployment: license, dependency lock, deterministic run manifest, checkpoint provenance/checksum, runtime/memory, Modal/local compatibility, rollback and live smoke.

## A46 evidence boundary

- The frozen 102-chip selection IDs live in `data/sample/a46_selection_chips.json`; `data/sample/a46_run_manifest.schema.json` defines the local-run evidence contract. The 102 and 127 sets are disjoint.
- Track exact A18/A46 configs/results in a license-safe reproducible artifact set. Upstream SAM-Road++ currently publishes no license, so its source, local patch and derived checkpoints remain ignored/local and cannot be redistributed or deployed.
- A genuinely untouched geography/sensor evaluation set remains required before any generalization claim.

## A46 checkpoint selection and registered comparison (2026-07-18)

All 27 historical LoRA checkpoints were inferred on the registered 102-chip selection split with complete graph coverage and verified config, graph, checkpoint and split hashes. Epoch 22 initially led at raw APLS `0.141189`, but missed the `0.25` normalized floor. The pre-registered stage-1 topology sweep selected threshold `0.60` at raw APLS `0.165779` and normalized APLS `0.282758`; it beat the epoch-18 incumbent by `+0.069943`, with 95% CI `[+0.051983, +0.089659]`. Every floor passed, so stage 2 was not run.

`data/sample/a46_comparison_preregistration.json` froze that exact candidate, incumbent, v3.2 checkpoint, 127-chip manifest and strict 600-sample protocol before the comparison split was opened. The one permitted comparison completed 127/127 coverage. The candidate scored raw APLS `0.181165` and normalized APLS `0.301519`; the incumbent scored `0.105112` and `0.180762`; deployable v3.2 scored `0.012082` and `0.018539`. Candidate-minus-incumbent raw APLS was `+0.076053`, with 95% CI `[+0.059871, +0.092975]` and paired randomization `p=0.000100`. Fragmentation also improved: mean components fell from `11.433` to `7.937`, largest-component node fraction rose from `0.3763` to `0.5413`, and reachable-pair fraction rose from `0.2200` to `0.3859`.

The metric gate passed, but **production promotion remains false**. SAM-Road++ publishes no license, so its source, patch, derived weights and predictions remain local research artifacts and cannot be redistributed or deployed. `data/sample/a46_comparison_result.json` records the license-safe result summary and hashes; licensed v3.2 remains the production checkpoint while MIT-licensed SAM-Road is evaluated next under the same protocol.
