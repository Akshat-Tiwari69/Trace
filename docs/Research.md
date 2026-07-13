# Research.md — Literature, Experiment History and Next Hypotheses

> `Evaluation.md` owns exact numbers and promotion protocols. This document explains the research context, what was tried, why it worked or failed, and what remains worth testing.

## Research framing

Road extraction methods fall into three broad families:

1. **Pixel segmentation:** predict road/background, then vectorize the mask.
2. **Connectivity-aware segmentation:** add topology-oriented architectures or losses, but still produce a mask.
3. **Direct graph prediction:** predict nodes/edges/connectivity without making a binary mask the product boundary.

Route Resilience began with a segmentation→classical-graph pipeline because it was reproducible on modest compute and easy to inspect. The project’s own evidence now shows the limit of that choice: several changes improved pixels without improving routing, while A18 direct graph prediction improved the same-unit routing comparison. The next research phase is graph-first, with v3.2 retained as the deployable fallback.

## Reference methods

| Method | Core idea | Relevance / project verdict |
|---|---|---|
| SegFormer (Xie et al., 2021) | Efficient hierarchical transformer encoder for segmentation | MiT-B3 is the deployed v3.2 encoder |
| D-LinkNet (Zhou et al., 2018) | Pretrained LinkNet plus dilated context | Strong historical road-segmentation baseline; not yet needed after graph-first pivot |
| CoANet (Mei et al., 2021) | Strip convolutions and connectivity attention | Interesting mask-topology alternative; licensing/code constraints and lower priority after A18 |
| clDice (Shit et al., 2021) | Skeleton-overlap topology loss | Included in the original recipe; clDice-first fine-tune A9 regressed and was confounded, so it is not a default improvement claim |
| RoadTracer (Bastani et al., 2018) | Iterative graph construction | Foundational direct-graph alternative; sequential error accumulation is a concern |
| Sat2Graph (He et al., 2020) | Predict a graph encoding from imagery | Supports the graph-first direction |
| CRESI/CRESIv2 (Van Etten) | City-scale mask→graph→routing and APLS | Relevant evaluation/engineering reference |
| SAM-Road / SAM-Road++ | Frozen/adapted SAM encoder plus topology-aware graph heads | A18 research path; current upstream snapshot has licensing/reproducibility gaps |
| Global efficiency (Latora & Marchiori, 2001) | Mean inverse path length, finite under disconnection | Locked resilience basis |
| Boeing & Ha (2024) | Large-scale street-network disruption simulation | Supports targeted-vs-random failure analysis, not a claim that betweenness alone is resilience |

## References

- Xie, E. et al. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.* NeurIPS. arXiv:2105.15203.
- Zhou, L., Zhang, C., Wu, M. (2018). *D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction.* CVPR Workshops.
- Mei, J. et al. (2021). *CoANet: Connectivity Attention Network for Road Extraction From Satellite Imagery.* IEEE Transactions on Image Processing.
- Shit, S. et al. (2021). *clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation.* CVPR. arXiv:2003.07311.
- Demir, I. et al. (2018). *DeepGlobe 2018: A Challenge to Parse the Earth through Satellite Images.* CVPR Workshops.
- Van Etten, A., Lindenbaum, D., Bacastow, T. (2018). *SpaceNet: A Remote Sensing Dataset and Challenge Series.* arXiv:1807.01232.
- Bastani, F. et al. (2018). *RoadTracer: Automatic Extraction of Road Networks from Aerial Images.* CVPR. arXiv:1802.03680.
- He, S. et al. (2020). *Sat2Graph: Road Graph Extraction through Graph-Tensor Encoding.* ECCV. arXiv:2007.09547.
- Zhao, H. et al. (2024). *OpenSatMap: A Fine-grained High-resolution Satellite Dataset for Large-scale Map Construction.* NeurIPS Datasets & Benchmarks. arXiv:2410.23278.
- Boeing, G., Ha, J. (2024). *Resilient by Design: Simulating Street Network Disruptions across Every Urban Area in the World.* Transportation Research Part A 182, 104016.
- Latora, V., Marchiori, M. (2001). *Efficient Behavior of Small-World Networks.* Physical Review Letters 87(19).
- Buslaev, A. et al. (2018). *Fully Convolutional Network for Automatic Road Extraction.* arXiv:1806.05182.

## Datasets: used versus considered

### Used in tracked experiments

| Data | Role | Evidence limitation |
|---|---|---|
| DeepGlobe Roads | v1 training and anti-forgetting anchor | Non-Indian in-domain benchmark |
| SpaceNet-5 Mumbai | Real vector/mask supervision for v3/v3.2 and A18 | Repeatedly consulted single-city development benchmark |
| OSM + Esri imagery Indian corpus | Weak-label/domain experiments and graph sample generation | OSM agreement is noisy and was misleading for A12 |
| Massachusetts Roads | A11 combined retrain | Domain mismatch; rejected |
| Panaji OSM/sample artifacts | Graph, resilience and dashboard demonstration | One sample AOI, not model generalization evidence |

### Supported or considered, not validated as claimed deployment evidence

| Data/source | Status |
|---|---|
| Cartosat-3 PAN | Reader/georeference path exists; no real labeled PAN evaluation yet |
| LISS-IV / Sentinel-2 | Mentioned as potential Indian/wide-area sources; no current validated headline experiment |
| OpenSatMap | Considered research data; non-commercial/provider terms require careful review; not part of the deployed model evidence |
| New Indian geography/sensor | Required future holdout; not yet assembled |

Raw/provider imagery is not redistributed. Dataset and imagery-provider terms must be checked for every new experiment.

## What the project learned

### Adopted results

- **Real in-domain supervision mattered most.** SpaceNet-5 Mumbai supervision produced v3, and gray/radiometric augmentation produced v3.2, the best deployable mask model on the development benchmark.
- **Topology must gate promotion.** A38 and A41 improved pixel evidence while degrading APLS; both were rejected.
- **Graph-first has measurable headroom.** A18 frozen beat v3.2 on the common chip/vector-GT frame; LoRA r=4 roughly doubled A18’s own raw/normalized APLS.
- **File/provenance/evaluation correctness is model work.** Threshold, frame, unit, data split and coordinate mistakes can reverse a verdict.

### Negative-result ledger

| Task | Hypothesis | Outcome | Do not repeat without new evidence |
|---|---|---|---|
| A7 | D4 TTA will improve v1 | ~flat/slightly worse for ~8× compute | Keep opt-in only |
| A8 | Heavier occlusion fine-tune improves hidden-road recall | Recall flat; clean IoU materially worse | Do not increase the same augmentation recipe |
| A9 | clDice-first fine-tune improves topology | IoU and hard clDice worse; Lovász removal confounded it | Do not cite as a clean loss ablation; only revisit in a controlled from-start design |
| A11 | Massachusetts volume improves Indian performance | Indian zero-shot worse | Domain match beats raw volume |
| A12 | OSM weak-label mean teacher adds signal | Development result worse; OSM validation was misleading | Retire A20/A22 optimizations of this path |
| A38/A38b | Foreground-biased crops improve road continuity | Gray pixels improved, APLS significantly worse | Pixel recall alone is not enough |
| A41/A41b | SDT-BCE improves topology without forgetting | Pixels improved, APLS significantly regressed | Loss-only mask lever closed for now |

The focused A6–A11 history is retained in `Retrospective-A6-A11.md` with a dated-archive banner.

## A18 graph-first contract and result

### Coordinate invariant

The inspected SAM-Road++ loader assumed a Cartesian `y`-up transform (`y = 400 - row`) that did not match this repository’s SpaceNet raster/vector convention. That mirrored supervision. The local adapter must assert:

- `x = column`
- `y = row`
- sampled graph nodes land on the rasterized road mask before training

After correction, the sampled validation alignment check reached zero-pixel median/max error. The assertion is mandatory for future data conversion.

### Current result

- Frozen A18 beat v3.2 on 127/127 common chips but captured only about 8.6% of the achievable routing ceiling.
- LoRA r=4 adapted only a small encoder/head parameter set and raised the normalized ceiling fraction to about 18%, roughly 2.1× the frozen result.
- Increasing the raster frame from 400 to 800/1300 px without changing chip context did not raise the GT-self ceiling; it was a rescale, not additional context.
- The remaining error is dominated by under-connection: on the 127 development chips LoRA averages 11.43 components vs 3.46 for vector GT, largest-component node fraction 0.376 vs 0.719, 4.26 isolated nodes, six empty predictions and about 48% of GT edge length. It has more components than GT on 114/127 chips.

Exact scores and paired intervals are in `Evaluation.md`.

### Reproducibility and license limitation

The inspected upstream snapshot (`44c6e7b` in the local research workspace) had no clear license, standard configs or dependency lock. Therefore:

- do not redistribute upstream code or derived weights until license permission is clear;
- track the exact snapshot/config/checkpoint/result hashes for A46;
- keep research integration behind an optional boundary rather than importing it into the production package;
- do not call A18 reproducible or deployable while the run artifacts remain local/untracked.

Licensed fallback context: [SAM-Road](https://github.com/htcr/sam_road) is MIT-licensed and [Segment Anything](https://github.com/facebookresearch/segment-anything) is Apache-2.0. SAM-Road++ itself remains blocked for redistribution until its authors clarify licensing.

## Ranked A46 experiments

These are hypotheses, not promised gains. Selection uses the 102-chip validation split; the repeatedly consulted 127 chips receive one pre-registered development evaluation only after a configuration is frozen.

1. **Reproducibility gate before another run:** freeze the local patch, upstream/SAM hashes, dependency lock, 412/102/127 split, deterministic seed/cudnn settings, trainable-parameter manifest and result schema. Clarify licensing or keep the work local-only.
2. **Exploit the 27 existing LoRA checkpoints:** epoch 18 was selected by `val_topo_loss`, not APLS. Score all checkpoints on the 102 validation chips and choose by raw APLS plus fragmentation/reachability.
3. **Calibrate inference for this checkpoint:** the current keypoint/road/topology thresholds (`.195/.341/.705`) came from an upstream SpaceNet config. Sweep those plus NMS/neighborhood radius on validation APLS; cache shared inference where possible. This is the highest-ROI no-training experiment. The [SAM-Road++ paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_Towards_Satellite_Image_Road_Graph_Extraction_A_Global-Scale_Dataset_and_CVPR_2025_paper.pdf) also selects thresholds on validation and identifies extended-line processing as an important APLS lever.
4. **Small deterministic LoRA capacity/placement study:** compare r=4/8/16 or r=8 plus a controlled last-block unfreeze across three fixed seeds, selected by validation APLS rather than topology loss alone.
5. **Refit once after choices are frozen:** all 514 road-bearing non-development chips are already used (412 train/102 validation). Only after hyperparameters are frozen, refit on 514 for a predeclared epoch budget. The 375 empty-label chips are lower priority because the dominant error is missing connectivity, not false-positive roads.
6. **Add real context, not resized pixels:** use 2×2/strip multi-chip neighborhoods plus graph stitching/dedup. First define a separate mosaic GT-self ceiling/gate; do not compare its absolute score with chip APLS.
7. **New geography/sensor evidence:** freeze a genuinely untouched set before inspecting results.
8. **Licensed fallback if required:** evaluate MIT-licensed SAM-Road or another clearly licensed graph-first implementation under the same gate. DeH4R is scientifically interesting but its current repository also lacks a license; GLD-Road’s licensed repository does not provide a reproducible training implementation.

Measured planning evidence: the r=4 run took about 32.4 minutes for 27 epochs and the frozen run 44.3 minutes for 40 epochs on the local research GPU; 127-chip LoRA inference took about 191 seconds. A validation checkpoint/calibration pass is therefore hours, not weeks. Storage is the larger avoidable cost: save-every-epoch produced about 8.94 GiB (LoRA) and 13.17 GiB (frozen). A46 should keep top-k plus last, not every full checkpoint.

## A46 stop/go gates

- Alignment assertions pass on every sample.
- Train/validation/test chip identities and spatial grouping are recorded and disjoint.
- Full 127/127 reference/candidate coverage on the current development gate.
- Complete 102/102 coverage for checkpoint/config selection; do not tune on the 127-chip development comparison.
- Paired raw-APLS interval excludes zero and absolute ceiling fraction improves materially over 18%.
- Runtime, memory, trainable parameter count and checkpoint provenance are recorded.
- No deployment or redistribution until license/dependency/checkpoint compatibility is resolved.
- No final generalization claim until a new held-out geography/sensor passes.

## Compute guidance

The supported policy is simple:

- training and heavy evaluation may use an optional local NVIDIA GPU, Colab or Kaggle;
- recipes must fit a practical single-GPU environment or document why they do not;
- P2/P3/dashboard remain CPU-capable;
- exact current installation instructions belong in `SETUP.md`, not in a timeless research claim about specific cloud quotas or GPU generations.

## Resilience research note

Betweenness is a useful chokepoint heuristic, not a resilience metric by itself. Product evidence comes from the global-efficiency response to failures. A45 aligned both resilience paths on one contract: failed nodes remain in the baseline universe as isolates, the sampled-source set stays fixed across a curve, and RI remains bounded in `[0, 1]`. The corrected Panaji curve/flood evidence and its source fingerprint are recorded in `Evaluation.md` and `data/sample/panaji_demo_evidence_manifest.json`.
