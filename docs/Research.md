# Research.md

> **Purpose.** This document is the research foundation for **Route Resilience** — an end-to-end system for *occlusion-robust road extraction and graph-theoretic criticality analysis for urban mobility.* It surveys the state of the art in road extraction, graph-theoretic network resilience, the datasets and benchmarks the project uses, and establishes a definitive, numbers-backed verdict on whether commodity 8 GB GPUs are sufficient to train the models the plan requires. It is written to be approachable while remaining technically precise.

## Literature Review

Road extraction from satellite imagery has matured through three overlapping generations of methods. **(1) Segmentation-based** approaches treat road extraction as per-pixel binary classification (road vs. not-road), then post-process the mask into a graph. **(2) Topology/connectivity-aware** methods add losses or modules that explicitly preserve road *connectedness*, because a single missing pixel can break a route. **(3) Graph-based** methods predict the road graph (nodes and edges) directly, skipping the pixel mask.

The deep-learning era began with the **Massachusetts Roads Dataset** (Mnih, 2013), the first large dataset enabling CNN-based road extraction. The field accelerated sharply in 2018 with two landmark competitions: **DeepGlobe** (Demir et al., 2018) and **SpaceNet** (Van Etten et al., 2018), which released large labelled corpora and attracted global researchers. The DeepGlobe winner, **D-LinkNet** (Zhou et al., 2018), used a LinkNet encoder-decoder with a pretrained ResNet34 encoder and dilated (atrous) convolution in the center block to enlarge the receptive field — it remains a widely adopted baseline.

The central, recurring problem in all this literature is exactly the one this project targets: **occlusion**. Trees, buildings, shadows, and vehicles hide roads, and pixel-based methods consequently produce *fragmented* roads with broken topology. This motivated connectivity-aware architectures (CoANet), topology-preserving losses (clDice), and graph-based trackers (RoadTracer, Sat2Graph).

Plain-English glossary:
- **Segmentation** = labelling every pixel (here: road / not-road).
- **Mask** = the black-and-white image of those labels.
- **Topology** = how things are connected, regardless of exact shape — the property we care most about for routing.
- **Occlusion** = something blocking the view of the road.
- **Receptive field** = how much of the image one output pixel "sees"; bigger is better for long thin roads.

## Existing Solutions

| System | Year / Venue | Idea | Why it matters to us |
|---|---|---|---|
| **D-LinkNet** (Zhou et al.) | CVPR 2018 Workshops | LinkNet + pretrained ResNet34 encoder + dilated convolution center | DeepGlobe champion; strong, lightweight segmentation baseline that fits 8 GB GPUs |
| **CoANet** (Mei et al.) | IEEE TIP 2021 | Strip Convolution Module (4 directional strip convolutions) + Connectivity Attention module; ResNet-101 encoder | Directly targets occlusion and topological correctness; SOTA on SpaceNet + DeepGlobe |
| **clDice / soft-clDice** (Shit et al.) | CVPR 2021 | Topology-preserving loss computed on the intersection of mask and its skeleton | A *loss function* we can bolt onto any model to improve connectivity — a planned value-add |
| **RoadTracer** (Bastani et al.) | CVPR 2018 | Iterative graph construction guided by a CNN decision function | Graph-based alternative; avoids intermediate mask but error accumulates at complex junctions |
| **Sat2Graph** (He et al.) | ECCV 2020 | Graph-tensor encoding (18/19-D) to predict graph directly | Direct graph prediction; not end-to-end trainable, isomorphic-encoding limitation |
| **CRESI / CRESIv2** (Van Etten) | 2019 | City-scale road + travel-time extraction; APLS evaluation | Shows full city-scale segmentation→graph→routing pipeline, similar to ours |
| **SAM-Road** | 2024 | Adapts Segment Anything Model + lightweight transformer GNN for road graphs | Recent transformer-era direction |

Our chosen architecture is **SegFormer** (a Transformer-based segmentation network) via HuggingFace / `segmentation_models_pytorch`, plus a classical, explainable graph-healing pipeline (skeletonize → sknw → NetworkX → MST/Union-Find). This blends a modern occlusion-robust segmentation backbone with a transparent, debuggable graph stage — a good fit for a small team working with limited compute.

## Research Papers (real citations)

- Xie, E. et al. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.* NeurIPS. arXiv:2105.15203. Per the paper verbatim, "SegFormer-B0 achieves a mean Intersection over Union (mIoU) of 37.4% using only 3.8 million parameters and 8.4 billion FLOPs"; SegFormer-B1 = 13.7M params (paper comparison tables).
- Zhou, L., Zhang, C., Wu, M. (2018). *D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction.* CVPR Workshops. (Code: github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge)
- Mei, J., Li, R-J., Gao, W., Cheng, M-M. (2021). *CoANet: Connectivity Attention Network for Road Extraction From Satellite Imagery.* IEEE TIP. (Code: github.com/mj129/CoANet)
- Shit, S. et al. (2021). *clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation.* CVPR, pp. 16555–16564. arXiv:2003.07311.
- Demir, I. et al. (2018). *DeepGlobe 2018: A Challenge to Parse the Earth through Satellite Images.* CVPR Workshops.
- Van Etten, A., Lindenbaum, D., Bacastow, T. (2018). *SpaceNet: A Remote Sensing Dataset and Challenge Series.* arXiv:1807.01232. (Defines the APLS metric.)
- Bastani, F. et al. (2018). *RoadTracer: Automatic Extraction of Road Networks from Aerial Images.* CVPR. arXiv:1802.03680.
- He, S. et al. (2020). *Sat2Graph: Road Graph Extraction through Graph-Tensor Encoding.* ECCV. arXiv:2007.09547.
- Zhao, H. et al. (2024). *OpenSatMap: A Fine-grained High-resolution Satellite Dataset for Large-scale Map Construction.* NeurIPS Datasets & Benchmarks. arXiv:2410.23278.
- Boeing, G., Ha, J. (2024). *Resilient by Design: Simulating Street Network Disruptions across Every Urban Area in the World.* Transportation Research Part A, vol. 182, art. 104016. arXiv:2403.10636.
- Latora, V., Marchiori, M. (2001). *Efficient Behavior of Small-World Networks.* Physical Review Letters 87(19). (Defines global efficiency.)
- Buslaev, A. et al. (2018). *Fully Convolutional Network for Automatic Road Extraction (ResNet34-UNet, DeepGlobe).* arXiv:1806.05182.

## Benchmark Studies

- **DeepGlobe Road Extraction Challenge (2018):** 6,226 train / 1,243 val / 1,101 test images, all 1024×1024 RGB at ~50 cm/pixel; scored by IoU. D-LinkNet won.
- **SpaceNet Roads (Challenge 3):** 2,780 image chips of 1300×1300 px at 30 cm ground sample distance from WorldView-3, over Las Vegas, Paris, Shanghai and Khartoum (Van Etten et al. 2018, arXiv:1807.01232; CRESI v2 confirms "we use the 2780 images/labels in the SpaceNet 3 training dataset"). It introduced the **APLS** metric (Average Path Length Similarity), a graph-theoretic measure that compares shortest-path lengths between ground-truth and predicted graphs rather than pixels — directly relevant to our Topological Accuracy metric.
- **City-Scale dataset** (He et al. 2020): 180 tiles of 2048×2048 at 1 m/pixel over 20 US cities; vector ground-truth graphs supplied; scored with TOPO and APLS.
- **APLS vs pixel-F1:** Van Etten et al. (2018) illustrate the gap directly — a proposal mask scoring **F1 = 0.72 yields an APLS of only 0.25** because "missing intersections and road segments are heavily penalized." The SpaceNet 3 baseline (U-Net + skeletonization + sknw) scored **APLS = 0.49**, while the winner "albu" scored **0.6663 total** (Vegas 0.798, Paris 0.604, Shanghai 0.654, Khartoum 0.609). A single wrong edge can crater APLS — exactly the failure mode our MST healing addresses.

## Alternative Approaches & Differentiation

Common alternative approaches and how this project differs:

1. **Segmentation-only pipelines** ("plain U-Net / D-LinkNet", naive thinning). The most common approach trains a segmentation model, stops at the mask, and does simple thinning. These score poorly on *connectivity* and *topological accuracy* under occlusion. **Our edge:** explicit MST/Union-Find graph healing + clDice connectivity loss.
2. **Heavy graph-based pipelines (RoadTracer / Sat2Graph).** Powerful but hard to train, GPU-hungry, and error-prone at junctions; difficult to bring to a working end-to-end state under tight time and compute budgets.
3. **Brief-literal resilience metrics.** Implementations that take a Resilience Index as a ratio of average shortest path lengths hit a divide-by-infinity wall the moment node removal disconnects the graph. **Our edge:** the **global-efficiency** reformulation (below), which stays finite and is exactly what the Boeing & Ha (2024) study uses.
4. **Static-output tools.** Many pipelines deliver static plots. **Our edge:** an interactive Streamlit + Folium dashboard with a click-to-disable-node simulation that reroutes live.

## Dataset Analysis

| Dataset | Resolution | Size / Coverage | License / Access | Role in our pipeline |
|---|---|---|---|---|
| **Sentinel-2** (Copernicus) | 10 m (visible/NIR) | Global, ~5-day revisit | Free, open; via Copernicus Data Space and mirrored on ISRO **Bhoonidhi** | Wide-area, multi-season context; illumination/seasonal robustness training |
| **Resourcesat-2/2A LISS-IV** | 5.8 m | India-wide | Free/open via ISRO **Bhoonidhi**; LISS-III/AWiFS also via USGS EarthExplorer | Indian-terrain fine-tuning; generalisation to local road styles |
| **Cartosat-3** | Very high-res (sub-metre, panchromatic) | Indian sub-continent | Restricted / on request (ISRO / NRSC Bhoonidhi) | High-res inference / demonstration tiles |
| **SpaceNet Roads** | 30 cm pan-sharpened (WorldView-3) | 4 cities, 2,780 chips | Open (AWS) | Pretraining + APLS-style topological benchmarking |
| **DeepGlobe Roads** | 50 cm | 6,226/1,243/1,101 images, 1024² | Free for research (sign-in / Kaggle mirror) | Primary pretraining/fine-tuning corpus |
| **OpenSatMap** | level-20 (0.15 m) & level-19 (0.3 m) | ~3,787 images, 60 cities, 19 countries | CC BY-NC-SA 4.0 (Google Maps ToS apply) | Recent, fine-grained occlusion-rich supplement |
| **OpenStreetMap (OSM)** | vector | Global | ODbL (open) | **Zero-manual-labelling**: auto-rasterize OSM road vectors into masks via `osmnx` + `rasterio` to create ground truth for any AOI |

The OSM auto-labelling pipeline (`osmnx` to pull road vectors → `rasterio` to burn them into raster masks aligned with Sentinel-2/LISS-IV tiles) lets us generate training labels for Indian cities at zero manual cost. Caveat: OSM/satellite registration offsets and labelling-density differences can degrade APLS (documented in CRESI experiments), so we will buffer masks (3–5 px) and visually QC samples.

## State-of-the-Art Techniques

- **Transformer segmentation (SegFormer).** Hierarchical Mix-Transformer (MiT) encoder + lightweight all-MLP decoder; no positional encodings, robust to resolution change, strong efficiency/accuracy trade-off. SegFormer-B0 = ~3.7–3.8M params; B1 = ~13.7M — both small enough for 8 GB GPUs.
- **Connectivity attention (CoANet).** Strip convolutions aligned to long, thin road shapes; connectivity attention module predicts pixel-to-neighbour links to preserve topology under occlusion.
- **Topology-aware losses (clDice).** Computed on the skeleton intersection; provably preserves connectedness up to homotopy; improves graph similarity and Betti numbers. We will use a Dice + soft-clDice + BCE combination.
- **Occlusion-simulation augmentation.** `CoarseDropout` / Random Erasing (via **Albumentations**) deliberately masks patches during training so the network learns to "see through" trees/vehicles — a direct, cheap intervention for the occlusion brief.
- **Morphological skeletonization + graph extraction.** `skimage.morphology.skeletonize` → **sknw** to build a NetworkX graph → MST/Union-Find healing scored by Euclidean distance AND angular alignment.
- **Graph-theoretic resilience.** Betweenness centrality to find chokepoints; node ablation; **global efficiency** as a finite resilience metric (see Open Problems and Infrastructure sections).

## Open Problems & Innovation Opportunities

1. **Occlusion is unsolved at scale.** Even SOTA models fragment under heavy tree canopy and dense urban clutter. Opportunity: combine SegFormer + clDice + occlusion augmentation + post-hoc MST healing.
2. **Mask→graph healing is usually naive.** Most pipelines thin the mask and connect nearest endpoints. Our **angle-aware MST/Union-Find** bridging (penalising bridges that turn unnaturally) is a genuine, implementable innovation.
3. **The Resilience Index breaks on disconnection.** A commonly proposed Resilience Index is the ratio of average shortest-path length (baseline ÷ perturbed). When removing a high-centrality node *disconnects* the graph, average shortest path length becomes **infinite**, so the ratio is undefined/zero. The fix — the **Latora–Marchiori (2001) global efficiency**, E = 1/[N(N−1)] · Σ (d_ij^euclidean / d_ij), which "allows to overcome the subtleties due to infinite characteristic path lengths… even when the network is not connected" — averages the *inverse* shortest-path lengths over all node pairs, with disconnected pairs contributing 0. This stays finite and degrades smoothly from 1 (fully connected) toward 0. It is precisely the measure Boeing & Ha (2024) use when they "simulate over 2.4 billion trips across more than 8,000 urban areas in 178 countries" and find that "disrupting high-centrality nodes severely impacts network function." Adopting global efficiency is our key analytical differentiator. (Caveat from the literature — e.g., *Findings*, "Betweenness Centrality is not a Network Resilience Metric": betweenness centrality is a *heuristic* for criticality, not a theoretically guaranteed resilience measure — we will present it as a chokepoint indicator, not ground truth.)
4. **Indian-terrain generalisation.** Most benchmarks are US/Europe/China cities. Fine-tuning on LISS-IV + OSM-labelled Indian AOIs is an under-explored, high-impact niche.

## Infrastructure & Hardware Feasibility — The Definitive Verdict

**Verdict: commodity 8 GB GPUs are sufficient for the realistic plan** — fine-tuning a pretrained SegFormer-B0/B1 or ResNet34/50-encoder U-Net/DeepLabV3+ on 256–512 px tiles — provided mixed precision, small batch sizes with gradient accumulation, and gradient checkpointing (where needed) are used. Free Colab/Kaggle GPUs (16 GB T4/P100) serve as overflow for any heavier run. Phases II–IV are CPU-bound and not affected by GPU limits.

### (1)–(2) Is 8 GB enough, and at what settings?

Direct 8 GB-class evidence points: a **U-Net at 512×512 on an 8 GB GPU fits batch size 2 in FP32** (batch ≥4 OOMs); **DeepLabV3+ with a heavy encoder at 512×1024 fits only batch 1 even on a 12 GB GPU.** SegFormer-B0 (3.7M params) and ResNet34-UNet are far lighter than the B5/ResNet101 configurations whose large VRAM figures (13–48 GB) appear in papers — those are at 1024² on big GPUs and are only upper bounds. Crucially, the **ResNet34-UNet DeepGlobe winner (Buslaev et al. 2018) was explicitly designed to fit a single 8 GB GPU.**

Recommended training configuration for 8 GB:

| Setting | 256×256 tiles | 512×512 tiles |
|---|---|---|
| Model | SegFormer-B0/B1 or ResNet34-UNet | SegFormer-B0 or ResNet34-UNet |
| Precision | AMP / FP16 (or BF16 on newer GPUs) | AMP / FP16 (mandatory) |
| Batch size (physical) | 8–16 | 2–4 |
| Gradient accumulation | ×1–2 | ×4–8 (to reach effective batch 16–32) |
| Gradient checkpointing | optional | recommended for B1/ResNet50 (~22% VRAM cut, ~20% slower) |
| Encoder | pretrained (ImageNet) — fine-tune, do NOT train from scratch | same |

AMP roughly halves activation memory; gradient checkpointing cuts peak VRAM a further ~22–60% for ~20–33% extra time (a HuggingFace-measured example dropped peak VRAM from 8,681 MB → 6,775 MB, ~22%, while throughput fell ~20%); gradient accumulation gives a large *effective* batch without extra memory.

### (3) Realistic training time

Zhou et al. (D-LinkNet, CVPR 2018 Workshops) state plainly: **"It took about 40 hours for us to train one model"** — D-LinkNet trained on all 6,226 DeepGlobe images at batch size 4 over ~160 epochs. Independent DeepGlobe road studies report **~1 hour per epoch** on a single 24 GB RTX 3090 at 1024² batch 4. On an 8 GB GPU with 256–512 px tiles and AMP, expect per-epoch times in the same order to somewhat longer; a **practical fine-tune of a few (3–10) epochs on a subset is achievable overnight.** SegFormer-B0 is lighter than ResNet34-UNet and should be comparable or faster per image.

### (4) Recent-GPU (Blackwell / RTX 50-series) compatibility note

General caveat for anyone running on a current-generation Blackwell GPU (compute capability **sm_120**): stable PyTorch wheels did not support sm_120 until **PyTorch 2.7.0**, which per SaladCloud's documentation **"was the first stable release to add native sm_120 support — shipping pre-built CUDA 12.8 wheels with updated cuDNN, NCCL, and Triton 3.3."** Install via the CUDA 12.8 index (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`) and verify `torch.cuda.get_device_capability()`. Older Ampere/Ada GPUs need no special handling. Avoid hard dependence on libraries that lagged on Blackwell (e.g. xFormers, some Triton kernels).

### (5) Sustained-training / thermal note (laptop & small-form-factor GPUs)

Laptop and SFF GPUs run at far lower sustained power than desktop equivalents and will thermally throttle and clock down under long training. Mitigations: good airflow / a cooling pad, the device's performance power mode, smaller (256²) tiles to limit per-step heat, and frequent checkpointing so a thermal shutdown never loses more than one epoch.

### (6) When to offload to free cloud

Use **Kaggle** and **Google Colab** for: (a) any 512²/1024² full-resolution run, (b) larger batch sizes that won't fit 8 GB, (c) parallel experiments running different configs simultaneously, (d) long unattended training (Kaggle background execution). Per Kaggle's official GPU docs: **"You can use up to 30 hours per week of GPU, and individual sessions can run up to 9 hours"** on NVIDIA Tesla P100 (16 GB) GPUs (2×T4 16 GB is also selectable). Colab's free tier offers a T4 (16 GB) with variable, unpublished session limits. Keep local GPUs for prototyping, debugging, and inference/demo. Save checkpoints to Drive/Kaggle Datasets since cloud storage is ephemeral.

### (7) Phases II–IV are CPU-bound — confirmed

Skeletonization (scikit-image), graph build (sknw/NetworkX), MST/Union-Find healing, betweenness centrality, node ablation, global-efficiency computation, and the Streamlit/Folium dashboard all run on CPU and are **not GPU-bottlenecked.** Betweenness centrality on large graphs is the heaviest CPU step (time- and resource-intensive for city-scale networks); we will mitigate with NetworkX's approximate (k-sample) betweenness if needed.

---

## v1 → v2 Improvement Roadmap (techniques & sources)

> The **task rows** for everything below live in `Tracker.md` §6 (Akshat A7–A15, Shaivi S3–S11, Saanvi F3–F8, shared E2–E5). This section is the *researched technique detail + citations* behind them — kept here so the Tracker stays a task board, not a literature review. All proposals respect the §2 locked constraints (PyTorch, fine-tune pretrained only, global-efficiency only, Streamlit+Folium, CPU graph/dashboard, free/Colab GPU).

### Current state & biggest opportunities

v1 is a fully-closed end-to-end pipeline (A1–A5 ✅, E1 ✅): SegFormer MiT-B3 + SCSE U-Net (IoU 0.670 / Occlusion-Recall 0.793) → skimage/sknw + MST/Union-Find healing → betweenness + global-efficiency resilience → Streamlit+Folium dashboard. The model has **genuinely plateaued on DeepGlobe** — it sits right around the original D-LinkNet DeepGlobe winner's **0.6466 validation IoU** (Zhou, Zhang & Wu, CVPR 2018 Workshops). The biggest gains now come from a topology-aware loss (clDice), TTA/ensembling, and domain fine-tuning on Indian imagery — **not** from more DeepGlobe epochs.

Where the new tasks aim:
1. **Segmentation plateaued at ~0.670 IoU**, and pixel IoU is the *wrong* primary metric for a connectivity project — what matters is whether roads stay topologically connected. Pivot toward **connectivity/topology metrics** (clDice, APLS, relaxed/buffered IoU). Topology-aware SOTA is higher: TopoRF-Net (Fu et al., *Sensors* 2025, DOI:10.3390/sensors25247428) reports IoU 69.76% / F1 82.18% on DeepGlobe-Road — real headroom.
2. **Inference-time gains are unbanked** — TTA/ensembling are nearly-free accuracy not wired into `predict.py`.
3. **Occlusion robustness — the whole thesis — is under-tested** (one threshold + CoarseDropout, no systematic occlusion benchmark/ablation).
4. **Domain gap:** DeepGlobe (Indonesia/Thailand/India @0.5m) → Indian basemap tiles. A6 is the right instinct; expand with self-training/pseudo-labeling + held-out Indian eval.
5. **Graph quality un-scored** — no APLS/topology validation vs OSM, no degree-2 simplification or stub pruning.
6. **Resilience is shallow** — plain betweenness + targeted/random only; no articulation points/bridges, demand weighting, or flood/elevation scenarios.
7. **Dashboard** lacks large-graph perf hardening, multi-node (flood) failure, export/report.
8. **No experiment tracking/config files** — results live in Tracker prose, which won't scale across many experiments.

### §A — Segmentation (Akshat: A7–A15)
- **Topology loss — soft-clDice.** Official PyTorch `jocpae/clDice` (Shit & Paetzold et al., CVPR 2021, arXiv:2003.07311); pip `monai.losses.SoftDiceclDiceLoss` (iter_=3, alpha=0.5). Combine `alpha*dice + (1-alpha)*soft_cldice`, α≈0.3–0.5 (paper's road experiments used α=0.4). On the Roads dataset, soft-clDice (α=0.4) achieved **β1 (Betti) Error 0.755 and Opt-Junction F1 0.916 vs soft-Dice's 1.408 and 0.766** — large topology gains. On crack segmentation, clDice added **+2.8–4.9pp Dice / +1.7–3.9pp clDice** over BCE (MDPI 2673-7590/5/4/177). *Caveat:* `soft_skel` loop is memory-hungry → smaller batch/patch on 16 GB GPUs.
- **TTA — ttach (qubvel).** `SegmentationTTAWrapper(model, aliases.d4_transform())` = 8 dihedral augs (orientation-free aerial); multi-scale via `tta.Scale`. A U-Net tutorial measured **+1.33% F1 / +1.89% IoU** from TTA with no retraining (Idiot Developer). Expect ~+0.5–2 IoU.
- **Occlusion augmentation — Albumentations.** `CoarseDropout` (Cutout/RandomErasing evolution), `GridDropout`/GridMask, `XYMasking`, `RandomShadow`, brightness/contrast/hue, elastic/grid distortion — Albumentations calls dropout-style transforms "among the highest-impact" because they "simulate real-world partial occlusion" (the tree/building/shadow mode this project targets).
- **Architectures (smp).** Current MiT-B3+Unet is solid. Options: MiT-B4/B5, or ensemble with a complementary CNN backbone (SE-ResNeXt50, EfficientNet-b4/b5, ResNet101) under Unet/UnetPlusPlus/DeepLabV3+. **Note:** MiT encoders don't support Linknet/UnetPlusPlus in smp; CNN encoders do. D-LinkNet (2018 winner, dilated centre) is the canonical road arch.
- **Connectivity-oriented refs:** CoANet (strip conv + connectivity attention; *non-commercial* license), CAFormer, D3FNet, TopoRF-Net (IoU 69.76 DeepGlobe). They show topology metrics (APLS/clDice) move even when pixel IoU barely does.
- **Datasets past DeepGlobe:** SpaceNet Roads (0.3m, ~8000 km centerlines, 4 cities), Massachusetts Roads (1m, 1108 train), RoadTracer. Domain adaptation: **RoadDA** (arXiv:2108.12611, IEEE TGRS) stagewise GAN + adversarial self-training → **74.92% IoU / 85.81% F1, +15.52pp IoU over ADVENT**; **Topology-aware UDA** (arXiv:2309.15625, ISPRS J.) adds a skeleton head + connectivity pseudo-label refinement → **+7.5pp IoU** over the source model, beating competitors on SpaceNet→DeepGlobe by ≥6.6/6.7/9.8 in IoU/F1/APLS. A Sentinel-2+OSM dataset covers 7 Indian regions (Data in Brief 2025) but **10m only resolves major roads** → prefer sub-meter basemap tiles for narrow roads (why A6 uses Esri, not Sentinel-2), or teacher-student super-resolution (arXiv:2310.11622).
- **Foundation models (optional, heavier):** Road-SAM (frequency adapters), SAM-Road (APLS SOTA on SpaceNet/City-Scale).
- **Post-processing:** morphological open/close, connected-component area filtering, Otsu/threshold tuning, skeleton pruning, optional CRF.

### §B — Graph build + healing (Shaivi: S3–S7)
- **Skeletonization:** skimage `skeletonize` (Zhang/Lee) vs `medial_axis`; tune sknw extraction; simplify polylines with Douglas–Peucker (shapely `simplify`).
- **Graph cleaning (OSMnx-style):** `consolidate_intersections(tolerance, rebuild_graph=True)` merges node clusters into true intersections; `simplify_graph` removes interstitial degree-2 nodes keeping geometry; prune short stubs; snap near-duplicate nodes. Boeing 2025 ("Topological Graph Simplification…," *Transactions in GIS*); `rebuild_graph=True` avoids merging spatially-close-but-topologically-remote nodes (overpasses).
- **Smarter healing:** tune Euclidean+angular gap scoring; Bezier/spline interpolation for natural curves; junction/overpass handling.
- **Graph scoring:** **APLS** (Van Etten/CosmiQ SpaceNet metric; open-source `apls`) + TOPO; connectivity ratio; topology accuracy vs OSM.

### §C — Criticality + resilience (Shaivi: S8–S11)
- **Beyond betweenness:** edge betweenness, current-flow/random-walk betweenness, **approximate k-sample** betweenness (`betweenness_centrality(k=...)`, Brandes/Riondato) for speed, **articulation points & bridges** (`nx.articulation_points`, `nx.bridges`), k-core, **percolation_centrality** (demand/state-weighted), population/demand-weighted centrality.
- **Richer resilience:** targeted-vs-random curves (have these; add degree/weighted-degree per the Brazilian Federal Road Network study, arXiv:2412.15865), percolation analysis, multi-failure scenarios, recovery sequencing, **flood/elevation-based realistic failure** (remove low-elevation nodes), compare global-efficiency degradation across scenarios.
- **Performance:** k-sample betweenness, caching, sparse representations.

### §D — Dashboard (Saanvi: F3–F8)
- **Performance:** `st.cache_data` for artifact loading, `st.cache_resource` for the graph object; simplify geometry before render; FastMarkerCluster for many markers; limit `st_folium` return keys.
- **Interactivity:** draw-a-flood-polygon (Folium Draw plugin) → disable enclosed nodes → multi-node failure; before/after comparison; layer toggles; choropleth criticality heatmap.
- **Visualization:** colorblind-safe ramp (Viridis already good); legend/tooltips; side charts; export GeoJSON/PDF report.
- **Polish:** loading states, error handling, sample-data mode (present).

### §E — Evaluation & rigor (shared: E2–E5)
- **Metrics:** IoU, Dice, Occlusion-Recall, relaxed/buffered IoU (CCQ correctness/completeness/quality), APLS, connectivity ratio; held-out test set; vs published baselines (D-LinkNet 0.6466 DeepGlobe validation IoU).
- **Ablations:** with/without clDice, with/without occlusion aug, with/without healing, targeted vs random.
- **Tracking:** Weights & Biases free tier or simple CSV logging; YAML config files instead of hardcoded params; reproducibility (fixed seeds).

### Recommended order
1. **This week (free, high ROI):** A7 (TTA), A8 (occlusion aug), A9 (clDice weight), S3/S4 (graph cleaning), F3 (dashboard caching) — no new data, measurable gains.
2. **Next:** finish A6, then A11→A12 (datasets + self-training) for the domain-gap win; S7/E2 to start scoring on APLS/connectivity (the metrics that actually matter here).
3. **Then depth:** S8–S11 resilience + F4–F7 dashboard make the demo compelling; A13/A14/A15 are optional accuracy pushes.
4. **Throughout:** stand up E5 (configs + tracking) *before* running many experiments; use E3/E4 ablations to prove each change helps.

**Thresholds that change the plan:**
- If clDice/occlusion ablations (E3) show no connectivity gain, drop them and pour effort into datasets (A11/A12).
- If TTA (A7) gives < 0.5 IoU, keep it only if its runtime cost is acceptable.
- If the A6 domain fine-tune underperforms v1 on an Indian val set, **don't release v2** — invest in self-training (A12) instead.
- If a larger encoder (A14) or connectivity decoder (A15) doesn't beat MiT-B3 on *topology* metrics (not just IoU), don't adopt it.

**Caveats:** quantitative gains cited above (clDice Betti/Opt-J F1, TTA +1.89% IoU, RoadDA +15.5pp, topology-aware UDA +7.5pp) come from *other datasets/papers* — indicative ranges to validate via E3/E4, not guarantees. clDice's soft-skeleton is memory-hungry (reduce batch/patch on free 16 GB T4/P100). CoANet code is non-commercial research license; respect dataset licenses (OSM ODbL, SpaceNet, OpenSatMap non-commercial) per §2.
