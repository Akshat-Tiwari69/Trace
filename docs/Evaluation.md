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

### Graph-lane first numbers (S1 sample — `panaji_demo`, OSM-derived w/ simulated occlusion)

Reproduce with `python -m src.pipeline.p3_analysis.evaluate` (reads the committed
`data/sample/` graph; writes `panaji_demo_graph_eval.json` + `_resilience_curve.png`).

| Metric | Value | Notes |
|---|---|---|
| Graph size | 630 nodes, 749 edges | 19 healed/bridged edges (2.5%) |
| **Connectivity Ratio** | **+15.1%** | largest connected component 524 → 603 nodes after MST/Union-Find healing (components 29 → 10) |
| Top "Gatekeeper" node | betweenness **0.511** | node 45; top-5 all ≈ 0.44–0.51 |
| Baseline global efficiency | 0.0012 | metric units (1/m); only ratios are interpretable |
| **Resilience: targeted vs random** | mean RI **0.674 vs 0.860** over 40 removals | targeted (high-betweenness-first) ablation degrades the network **far faster** than random ⇒ betweenness finds genuine chokepoints ✓ |

*Numbers are on the OSM stand-in (S1); the same `evaluate` runs unchanged on a real predicted-mask graph (S2) once a tile is available.*

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