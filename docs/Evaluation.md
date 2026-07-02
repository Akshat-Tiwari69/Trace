# Evaluation.md

> **Purpose.** This document defines how **Route Resilience** is measured — the metrics for each phase, the datasets and procedures used to validate them, the baselines we compare against, the targets we aim for, and the experiments that prove our design choices actually help. For a project like this, a clear evaluation plan is what separates "looks like it works" from "demonstrably works." Numeric results are filled in as runs complete (this is a living document).

---

## Evaluation Metrics

Each metric is tied to the phase it judges. Plain-English meaning first, then the technical note.

### Segmentation (Phase I)

| Metric | Plain English | Note |
|---|---|---|
| **IoU** (Intersection over Union) | How much predicted road overlaps true road, ignoring the easy "not-road" ocean | `intersection / union` of road pixels; the primary pixel metric |
| **Dice / F1** | Similar overlap measure, more forgiving of thin shapes | `2·TP / (2·TP + FP + FN)` |
| **Occlusion-Recall** | Of the roads *hidden* under trees/shadows, how many did we recover? | Recall computed on occluded regions — **the headline metric for this project** |
| **Relaxed / Buffered IoU** | IoU with a 3–5 px tolerance so being slightly off-centre isn't punished | Standard for thin structures; prevents penalising minor alignment shifts |

> Don't use plain pixel accuracy — roads are a tiny fraction of pixels (~4.5% in DeepGlobe), so a model predicting "no road" everywhere scores ~95% while finding nothing. See `Research.md`.

### Topology (Phase II)

| Metric | Plain English | Note |
|---|---|---|
| **Connectivity Ratio** | How much bigger the largest connected road network gets after MST healing | `% increase in largest connected component` — the direct scoreboard for healing |
| **Topological Accuracy / APLS** | Is the graph actually *routable*? Compare shortest-path lengths to OSM | Average Path Length Similarity; penalises missing/broken edges heavily (a mask with F1=0.72 can score APLS=0.25 — see `Research.md`) |

### Resilience (Phase III)

| Metric | Plain English | Note |
|---|---|---|
| **Resilience Index (global efficiency)** | How much the network's overall efficiency drops when critical nodes fail | Ratio of global efficiency after vs. before perturbation; **stays finite even when the graph splits** |
| **Largest-CC curve under ablation** | How fast the network fragments as nodes are removed | The classic percolation curve |
| **Targeted vs. random degradation** | Does removing *critical* nodes hurt more than removing random ones? | Sanity check that betweenness is finding real chokepoints |

## Benchmark Datasets

| Dataset | Used for |
|---|---|
| **DeepGlobe Roads** | Primary pretraining + IoU benchmark |
| **SpaceNet Roads** | Pretraining + APLS/topological benchmark (ships the APLS metric) |
| **OpenSatMap** | Domain-diverse, occlusion-rich fine-tuning/eval (non-commercial license) |
| **OSM-labelled Indian AOIs** | Generalisation to local terrain; auto-generated masks |
| **Cartosat-3 / LISS-IV demo tile** | Final high-res demonstration + qualitative eval |

## Validation Procedures

- **Strict train/val/test split** — never evaluate on data the model trained on.
- **Held-out Indian city** for the generalisation metric (train elsewhere, test there).
- **Spot-check OSM auto-labels** before trusting them — they're weak labels (missing/misaligned roads); use buffered/relaxed metrics.
- **APLS on random point-pairs:** sample many origin–destination pairs, route on our graph vs. OSM, compare path lengths.
- **Resilience sanity:** confirm targeted (high-betweenness) removal degrades global efficiency **faster** than random removal — if it doesn't, something's wrong.

## Baseline Results

Reference points from the literature to contextualize our numbers (full citations in `Research.md`):

| Reference | Result |
|---|---|
| D-LinkNet (DeepGlobe winner) | strong IoU baseline on DeepGlobe |
| SpaceNet 3 baseline (U-Net + skeletonize + sknw) | APLS ≈ 0.49 |
| SpaceNet 3 winner ("albu") | APLS ≈ 0.666 |
| Cautionary point | a mask at F1 = 0.72 can score APLS = 0.25 — pixels ≠ topology |
| **Our baselines** | seg: A4 IoU **0.670**, Occlusion-Recall **0.793** @thr 0.44 (DeepGlobe val) · graph: see below |

### Segmentation-lane numbers (A4 — DeepGlobe held-out validation)

Reproduce with `python -m src.pipeline.p1_segment.evaluate` (reads the trained
checkpoint's embedded validation metrics → `data/sample/segmentation_eval.json`).
Add `--image <tile>` for a live qualitative inference demo + red overlay.

| Metric | Value | Notes |
|---|---|---|
| Model | SegFormer MiT-B3 + SCSE U-Net (EMA), 47.5M params, 30 epochs | full-res sliding-window (Hann) validation |
| **IoU** | **0.6699** (flip + multi-scale TTA) · 0.6638 best single-view | held-out DeepGlobe val |
| **Occlusion-Recall** | **0.793** @ deploy thr 0.44 | the headline metric — recall on hidden roads |
| Deploy threshold | 0.44 → clean IoU 0.6617 | occlusion-aware: max recall within 0.01 IoU of peak |
| Checkpoint | GitHub Release `a4-roadseg-v1` | meta-driven; `load_checkpoint` rebuilds + deploys it |

*Real-world spot-check (full P1→S2 dry-run on a live tile, not DeepGlobe): on an
ESRI World-Imagery tile of tree-canopy-heavy Panaji (Altinho), the model recovered
roads at 5.13% pixel coverage tracing the visible street network; the predicted-mask
graph healed **38→23 components (+15 bridges, +50% connectivity)** and held the
resilience sanity check (**targeted RI 0.503 < random 0.780**). Off-domain + heavy
occlusion ⇒ a deliberately hard, sparse case — reported honestly per the
error-analysis policy below; on open road grids the mask is far denser.*

### Real Indian GT — held-out SpaceNet-5 Mumbai (A17 benchmark, A23 model)

The **truthful** Indian metric. DeepGlobe/OSM-agreement numbers above are in-domain
or weak-label proxies; A12 proved OSM-agreement is *misleading* (it rewarded models
that lose on held-out). A17 froze a chip-level held-out SpaceNet-5 Mumbai split
(real human vector GT, `data/sample/spacenet_mumbai_heldout_chips.json`, 127/637
chips) and re-baselined every model on it. Reproduce with
`python -m src.pipeline.p1_segment.eval_spacenet --checkpoints <ckpts> --device cuda [--grayscale] [--sweep]`
and `… apls_eval …`.

<!-- AUTO-GENERATED: from data/sample/spacenet_mumbai_{eval,eval_gray,threshold_sweep,apls}.json (IoU @0.44, 512px, global aggregation) -->
| Model | RGB IoU | Grayscale (Cartosat-PAN proxy) | APLS (routing, n=50) | best-threshold IoU |
|---|---|---:|---:|---:|
| **v3** `road_spacenet` (SpaceNet-Mumbai fine-tune) | **0.4311** | **0.3752** | **0.4147** | 0.4493 @0.50 |
| v2 `road_v2` (Indian OSM fine-tune) | 0.3727 | — | — | — |
| v1 `deepglobe_…_best` (DeepGlobe baseline) | 0.3752 | 0.3183 | 0.3844 | 0.3993 @0.50 |
<!-- END AUTO-GENERATED -->

**Findings:** (1) **v3 beats v1 on every axis** — RGB IoU +0.056 (+15%), APLS +0.030 (+8%), and grayscale. First genuine gain on real Indian GT after A8/A9/A11/A12 all failed — the lever was *real in-domain labels + a truthful metric*, not architecture. (2) **v2's apparent edge was a metric artifact** — on real GT it ties v1 (0.373 vs 0.375); its OSM-agreement gain didn't transfer. (3) **Cartosat readiness:** grayscale (PAN proxy) drop shrank from v1's −15% to v3's −11% (grayscale aug, A24); v3-grayscale 0.375 ≈ v1-RGB. (4) **Threshold sweep:** optimum is 0.50, not the deployed 0.44 (~+0.02 IoU). v3 released as [`a4-roadseg-v3`](https://github.com/Akshat-Tiwari69/Trace/releases/tag/a4-roadseg-v3).

### Graph-lane first numbers (S1 sample — `panaji_demo`, OSM-derived w/ simulated occlusion)

Reproduce with `python -m src.pipeline.p3_analysis.evaluate` (reads the committed
`data/sample/` graph; writes `panaji_demo_graph_eval.json` + `_resilience_curve.png`).

| Metric | Value | Notes |
|---|---|---|
| Graph size | 400 nodes, 474 edges | after S3 simplification + S4 consolidation; 19 healed/bridged edges |
| **Simplification (S3)** | **−22%** nodes (630 → 489) | 48 short stubs pruned + 93 degree-2 chain nodes collapsed; lossless for routing |
| **Consolidation (S4)** | **489 → 400** nodes | 51 near-duplicate junctions merged (tol 10 m); overpass-guarded (only merges along sub-tolerance edges) |
| **Total node reduction** | **630 → 400 (−37%)** | **all 10 components preserved** throughout |
| **Polyline simplification (S5)** | vertices **15.4k → 2.2k (−86%)** | Douglas-Peucker (1.5 m); GeoJSON 675 KB → 256 KB; shape preserved (Hausdorff ≤ tol), routing weights untouched |
| **Cut structure (S8)** | **136** articulation points, **191** bridge edges | true single-points-of-failure — structural criticality distinct from betweenness |
| **Connectivity Ratio** | **+15.1%** | largest connected component grew after MST/Union-Find healing (components 29 → 10); build-time figure |
| Top "Gatekeeper" node | betweenness **0.488** | node 45; top-5 cluster near the centre |
| Baseline global efficiency | 0.0013 | metric units (1/m); only ratios are interpretable |
| **Resilience: targeted vs random** | mean RI **0.601 vs 0.731** over 40 removals | targeted (high-betweenness-first) ablation degrades the network **far faster** than random ⇒ betweenness finds genuine chokepoints ✓ |

*Numbers are on the OSM stand-in (S1); the same `evaluate` runs unchanged on a real predicted-mask graph (S2). The graph is now S3-simplified + S4-consolidated — 37% lighter, identical connectivity.*

### Topology metric suite (E2)

The connectivity/topology scoreboard, consolidated — does the extracted graph
*route* like the real network, not just *look* like it pixel-wise?

| Metric | Owner | Value | Source |
|---|---|---|---|
| **Connectivity Ratio** | graph | **+15.1%** (largest CC after healing) | `evaluate` · S3/E1 |
| **APLS** (vs OSM) | graph | **0.40** symmetric (densified, snap 15 m) | `apls` · S7 |
| **Relaxed / buffered IoU** | seg | *Akshat — released model, A7/A10* | `p1_segment` |

Graph-side topology metrics (connectivity ratio + APLS) are **reported and
reproducible** (commands above); the buffered-IoU row is the segmentation lane's
contribution. Together they catch the "good pixels, bad topology" failure mode
(F1 0.72 → APLS 0.25) the project exists to avoid.

### Topology validation — APLS vs OSM (S7)

Reproduce with `python -m src.pipeline.p3_analysis.apls` (compares the healed graph
to a committed OSM ground-truth graph for the AOI; writes `panaji_demo_apls.json`).
Self-contained, densified, symmetric node-based APLS (no heavy CosmiQ dependency).

| Metric | Value | Notes |
|---|---|---|
| **APLS (healed graph vs OSM)** | **0.40** | symmetric harmonic mean (gt→prop 0.30, prop→gt 0.62); 600 sampled pairs, 15 m snap, 10 m densification |
| Reference (SpaceNet-3) | baseline ≈ 0.49, winner ≈ 0.67 | *not* directly comparable — those are on clean SpaceNet imagery |

This is a **deliberately hard** case: the S1 sample is built from OSM but then
**simulated-occluded** (80 patches) and healed, so the score measures how well the
heal recovers OSM routing *after* damage — honest, not inflated. On a clean
(non-occluded) or S3/S4-simplified build the score rises. APLS guards against the
"good pixels, bad topology" trap (a mask at F1 0.72 can score APLS 0.25).

### Resilience ablations (E4)

The design choices are only justified if turning them on/off *moves the number*.

**Healing on vs off** — does MST/Union-Find healing actually make a more resilient
network? (Reproduce: `python -m src.pipeline.p3_analysis.evaluate`; "off" = drop the
`is_bridged` edges from the sample.)

| | components | global efficiency |
|---|---|---|
| **Healing OFF** | 26 | 0.001008 |
| **Healing ON** | **10** | **0.001105** |
| Δ | −16 components | **+9.6% efficiency** · **+15.1% connectivity ratio** (build-time) |

**Failure mode — targeted vs random vs flood** (same node count; reproduce:
`python -m src.pipeline.p3_analysis.flood`):

| Failure | end Resilience Index | reading |
|---|---|---|
| **Targeted** (highest-betweenness first) | **0.357** | most damaging — losing the real chokepoints |
| **Random** (scattered) | 0.428 | distributed loss hurts more than a localized one |
| **Flood** (spatial cluster around the top chokepoint) | 0.785 | least damaging — the network reroutes around a *localized* hole |

**Reading:** healing measurably improves resilience; and the network is **robust to
localized floods but vulnerable to targeted chokepoint failure** — exactly the
distinction a disaster planner needs. (A flood of a *redundant* inland area, `--central`,
is even more survivable, RI > 1.) The naive expectation "flood ≫ random" does **not**
hold here because a contiguous flood mostly drowns low-importance local streets;
reported honestly rather than fitted to the hypothesis.

## Target Scores

Stated as intents, not invented precise numbers (actuals filled in as we run):

- **IoU:** competitive with U-Net/D-LinkNet-class baselines; **prioritise Occlusion-Recall** over raw IoU.
- **Occlusion-Recall:** a clear improvement *with* occlusion augmentation vs. without (the delta is the point).
- **Connectivity Ratio:** a large positive jump after MST healing.
- **APLS:** as high as feasible; minimise wrong/missing edges.
- **Resilience curve:** smooth, interpretable degradation where **targeted ≫ random**.

## Experimental Design

The ablation studies are the **evidence that our design choices work** — they matter as much as headline scores.

| Experiment | Question it answers |
|---|---|
| Occlusion augmentation **on vs. off** | Does simulating occlusion improve Occlusion-Recall? |
| clDice connectivity loss **on vs. off** | Does the topology-aware loss improve connectivity/APLS? |
| MST/Union-Find healing **on vs. off** | How much does healing raise Connectivity Ratio and APLS? |
| Targeted **vs.** random node removal | Does betweenness identify genuinely critical nodes? |
| (Optional) SegFormer **vs.** U-Net baseline | Does the transformer help under occlusion? |

Each ablation changes **one** thing and reports the metric delta. That's what makes the results credible.

## Error Analysis Strategy

- **Where models fail:** heavy tree canopy, complex multi-road junctions, rural/unseen terrain, and OSM-misaligned tiles.
- **How to inspect:** overlay predictions on imagery; pull the worst-scoring tiles and look at them; specifically examine recall inside occluded regions (not just overall).
- **Feed back:** use failure patterns to drive the next iteration (more occlusion augmentation, threshold tuning, targeted fine-tuning), and document recurring failure modes here.
- **Honesty:** report failure cases openly — a credible evaluation shows where it breaks, not just where it shines.