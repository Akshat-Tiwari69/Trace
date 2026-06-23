# PRD.md

> **Purpose.** This Product Requirements Document defines **Route Resilience** — an end-to-end system that extracts road networks from satellite imagery even where roads are hidden, heals the network into a routable graph, identifies critical chokepoints, stress-tests the network against simulated disasters, and presents it all in an interactive dashboard. It specifies the vision, users, requirements, constraints (including the compute limits established in Research.md), and success metrics.

## Product Vision

A decision-support tool that turns raw satellite imagery of an Indian city into a **resilience map**: it shows planners and disaster responders not just *where the roads are*, but *which roads are critical* and *what happens to mobility when they fail*. Where existing tools stop at a road mask, our system "sees through" trees, shadows, and vehicles, repairs the broken network, and lets a user click any junction to simulate its loss and instantly see rerouting and added travel time.

## Problem Statement

Urban road networks are vulnerable to floods, accidents, construction, and disasters. Two technical gaps block good planning: **(1)** satellite-derived road maps are *fragmented* because trees, buildings, shadows, and vehicles **occlude** roads, breaking the network so it can't be routed on; and **(2)** even with a good map, planners lack a principled, finite way to quantify how *critical* each road is and how *resilient* the network is to failure. This project delivers an occlusion-robust extraction pipeline plus graph-theoretic criticality analysis for urban mobility. We address both gaps, and we fix a mathematical flaw in the commonly proposed resilience metric.

## Goals & Objectives

1. **G1 — Occlusion-robust segmentation.** Produce road masks from Sentinel-2 / LISS-IV / Cartosat-3 imagery that infer continuity under occlusion, across seasons and illumination. (Transformer-based: SegFormer + clDice + occlusion augmentation.)
2. **G2 — Routable healed graph.** Convert fragmented masks into a single connected, weighted vector graph via skeletonization → sknw/NetworkX → MST/Union-Find bridging scored by distance + angular alignment.
3. **G3 — Criticality & resilience analysis.** Compute betweenness centrality to find "Gatekeeper Nodes"; run node-ablation stress tests; report a **finite Resilience Index based on global efficiency**.
4. **G4 — Interactive dashboard.** Streamlit + Folium/Leaflet app with a criticality heatmap and a click-to-disable-node simulation that reroutes live and reports increased travel time.
5. **G5 — Hardware-agnostic & runnable by everyone.** Training runs on free cloud (Colab/Kaggle) so it works identically on any machine; the graph and dashboard run on CPU; committed sample artifacts let any team member run their part without a GPU. Local 8 GB GPUs are an optional faster path.

## Target Users

- **Urban / transport planners** (municipal corporations, city development authorities) — design and retrofit resilient networks.
- **Disaster-response agencies (NDMA, SDMAs)** — pre-plan evacuation routes and identify chokepoints before floods/earthquakes.
- **Municipal authorities** — prioritise maintenance/redundancy investment on the most critical roads.
- **ISRO / NRSC stakeholders** — demonstrate downstream value of Indian EO data (LISS-IV, Cartosat-3) for civic applications.

## User Personas

**Persona 1 — Meera Nair, Urban Mobility Planner, a metropolitan development authority.** Needs to know which intersections are single points of failure before approving a flyover budget. Not a coder; wants a map she can click. Success = "I can show my committee what happens to commute times if Junction X floods."

**Persona 2 — Capt. Rohan Desai, District Disaster-Management Officer (NDMA-linked).** During monsoon, needs to pre-identify roads whose loss isolates neighbourhoods, using up-to-date satellite maps where official maps lag. Success = "I get a ranked list of chokepoints and an evacuation-route impact estimate within minutes."

**Persona 3 — Dr. Aishwarya Rao, Scientist, NRSC.** Evaluates whether the pipeline genuinely exploits LISS-IV/Cartosat-3 and whether the methodology (metrics, healing, resilience) is sound. Success = "The project uses a finite, defensible resilience metric and reports APLS/IoU honestly."

## Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | Ingest GeoTIFF satellite tiles (Sentinel-2, LISS-IV, Cartosat-3) via rasterio/GDAL; tile to 256²/512². |
| FR2 | Auto-generate training masks from OSM vectors (osmnx → rasterio), zero manual labelling. |
| FR3 | Train/fine-tune a pretrained SegFormer (or ResNet34/50-encoder U-Net/DeepLabV3+) with Dice + soft-clDice + BCE loss and occlusion augmentation (Albumentations CoarseDropout). |
| FR4 | Output binary road masks that bridge occluded gaps. |
| FR5 | Skeletonize masks (skimage) to 1-px centerlines; convert to NetworkX graph via sknw. |
| FR6 | Heal gaps with MST + Union-Find, scoring candidate bridges by Euclidean distance AND angular alignment; output a single weighted routable graph. |
| FR7 | Compute betweenness centrality; rank and highlight Gatekeeper Nodes. |
| FR8 | Node-ablation simulation: remove top-centrality nodes (flood/accident/closure scenarios) and recompute. |
| FR9 | Compute Resilience Index using **global efficiency** (finite under disconnection). |
| FR10 | Dashboard: criticality heatmap overlay (Folium/Leaflet) + click-a-node-to-disable toggle that reroutes and shows added travel time. |
| FR11 | Export results (GeoJSON graph, metric report). |

## Non-Functional Requirements

- **NFR1 — Hardware/compute constraint.** Must train within an 8 GB VRAM budget (commodity consumer/laptop-class GPUs). Mandatory: fine-tune pretrained encoders only (no training from scratch); AMP/FP16; batch 2–4 at 512² (8–16 at 256²) with gradient accumulation to effective batch 16–32; gradient checkpointing for heavier encoders. Free Colab/Kaggle (T4/P100 16 GB, "up to 30 hours per week … sessions up to 9 hours") for any heavier run.
- **NFR2 — Recent-GPU compatibility.** On current-generation Blackwell GPUs (compute capability sm_120, e.g. RTX 50-series), use PyTorch ≥2.7.0 with CUDA 12.8 (`cu128`) wheels — the first stable PyTorch release to add native sm_120 support; avoid hard dependence on libraries that lagged on Blackwell (e.g. xFormers). Older Ampere/Ada GPUs need no special handling.
- **NFR3 — Resilience metric must stay finite** under node removal (global efficiency, not raw average-path-length ratio).
- **NFR4 — Runtime targets.** Inference on a city tile + graph build + analysis within minutes on a laptop CPU/GPU; dashboard node-toggle reroute should feel near-instant (sub-second to a few seconds on a city-scale graph).
- **NFR5 — Reproducibility.** Fixed seeds, documented configs, checkpoints saved off-device (cloud storage is ephemeral).
- **NFR6 — Thermal resilience.** Frequent checkpointing so a laptop thermal shutdown loses ≤1 epoch.
- **NFR7 — Usability.** Non-technical users (planners) operate the dashboard with clicks only.

## Success Metrics (tied to the evaluation criteria)

| Metric | What it measures | Target/intent |
|---|---|---|
| **IoU & Dice (Occlusion-Recall focus)** | Mask overlap with ground truth, weighted to recovering occluded road pixels | Competitive IoU; prioritise recall under occlusion |
| **Generalisation across terrains** | Performance on unseen Indian cities/terrains | Stable IoU across held-out AOIs |
| **Connectivity Ratio** | % increase in the largest connected component after MST healing | Large positive jump vs raw skeleton |
| **Topological Accuracy (APLS)** | Average-path-length error vs OSM benchmark graphs | High APLS; few wrong/missing edges (note: F1=0.72 can mean APLS=0.25, so we optimise topology, not just pixels) |
| **Length-Complete/Relaxed IoU** | IoU with 3–5 px tolerance buffer (centerline-aware) | High relaxed IoU |
| **Resilience Index (global efficiency)** | Finite drop in global efficiency under node ablation | Smooth, interpretable degradation curve |

## Scope Definition

**In scope (initial release):**
- The four-phase pipeline (segmentation → healing → analysis → dashboard) as a working prototype.
- Fine-tuning a pretrained model on DeepGlobe/SpaceNet + OSM-labelled Indian AOIs.
- Global-efficiency Resilience Index; betweenness centrality; node-ablation demo.
- Streamlit/Folium dashboard with the interactive node-disable simulation.
- Demonstration on a high-resolution (Cartosat-3 / LISS-IV) tile of an Indian city.

**Out of scope (this phase):**
- Training large graph-based models (RoadTracer/Sat2Graph) from scratch.
- Real-time traffic feeds / live GPS data.
- Multi-city national-scale deployment.
- Mobile app; production hosting/scaling.
- Sub-metre lane-level (per-lane) extraction.

The scope is deliberately bounded so the core pipeline is demonstrable as a working prototype within a short, focused build window, with model fine-tuning pre-done on local and free-cloud GPUs beforehand. Phases are loosely coupled (clean file handoffs between segmentation, graph, analysis, and dashboard) so they can be developed in parallel.

## Future Enhancements

1. Replace classical MST healing with a learned graph-completion GNN (PyTorch Geometric).
2. Add travel-time-weighted edges (speed estimation, à la CRESIv2) for realistic routing.
3. Multi-temporal change detection (Sentinel-2 revisit) to auto-update maps after disasters.
4. Incorporate elevation (flood-prone low-lying node removal, as in Boeing & Ha 2024).
5. Edge-betweenness and articulation-point analysis for finer criticality.
6. Deploy as a hosted web service for municipal use; integrate with NDMA workflows.