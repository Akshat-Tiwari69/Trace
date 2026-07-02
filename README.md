# Route Resilience

**Find the road junctions a city can't afford to lose.**

Route Resilience extracts roads from satellite imagery — *even where trees, buildings, and shadows hide them* — heals the gaps into a routable network, then scores which junctions are critical and how gracefully the network degrades when they fail. It ends in an interactive map you can click to simulate a closure and watch the city reroute.

> Status: **road-seg v3 released** — full pipeline end-to-end (segmentation → graph → resilience → dashboard); **166 CPU tests** green. **v3 is the first model to beat the DeepGlobe baseline on real Indian ground truth** (held-out SpaceNet-5 Mumbai), and is Cartosat-3 PAN aware.

---

## Why it matters

When a junction floods, collapses, or is blocked, *which* failures actually break the city's connectivity — and which barely matter? Route Resilience answers that quantitatively, using **global efficiency** (a resilience measure that stays meaningful even when the network splits into pieces), and shows it on a map a planner can actually use.

## The pipeline

```mermaid
flowchart LR
    A[Satellite imagery] -->|P1 · SegFormer| B[Road mask]
    B -->|P2 · skeletonize + heal| C[Routable graph]
    C -->|P3 · criticality + resilience| D[Metrics]
    D -->|P4 · Streamlit + Folium| E[Interactive dashboard]
```

| Phase | What it does | Tech |
|---|---|---|
| **P1 · Segment** | RGB tile → binary road mask, robust to occlusion | SegFormer **MiT-B3 + SCSE U-Net** (PyTorch, fine-tuned) |
| **P2 · Heal** | mask → skeleton → graph, then bridge canopy-broken gaps | `skimage` + `sknw` + **MST / Union-Find** healing |
| **P3 · Analyze** | rank junctions by betweenness; measure resilience under failure | `networkx` · **global-efficiency** Resilience Index |
| **P4 · Dashboard** | click a junction → simulate closure → reroute + impact | **Streamlit + Folium** (CPU, no GPU) |

## Results

<!-- AUTO-GENERATED: segmentation results — from data/sample/spacenet_mumbai_*.json; see docs/Evaluation.md -->
**Segmentation — held-out SpaceNet-5 Mumbai (real Indian ground truth, IoU @0.44, 512px):**

| Model | RGB IoU | Grayscale (Cartosat-PAN proxy) | APLS (routing) | Release |
|---|---|---|---|---|
| **v3** — SpaceNet-Mumbai fine-tuned | **0.431** | **0.375** | **0.415** | [`a4-roadseg-v3`](https://github.com/Akshat-Tiwari69/Trace/releases/tag/a4-roadseg-v3) |
| v2 — Indian OSM fine-tuned | 0.373 | — | — | [`a4-roadseg-v2`](https://github.com/Akshat-Tiwari69/Trace/releases/tag/a4-roadseg-v2) |
| v1 — DeepGlobe baseline | 0.375 | 0.318 | 0.384 | [`a4-roadseg-v1`](https://github.com/Akshat-Tiwari69/Trace/releases/tag/a4-roadseg-v1) |

**v3 beats the baseline on every axis** — pixels (+15%), routing (APLS +8%), and grayscale/PAN robustness — the first genuine gain on *real* Indian GT (earlier OSM-agreement gains didn't transfer to held-out; the truthful benchmark is what exposed it). A threshold sweep puts the optimum at **0.50**, not the 0.44 v3 shipped with — the deployed model is **[`a4-roadseg-v3.1`](https://github.com/Akshat-Tiwari69/Trace/releases/tag/a4-roadseg-v3.1)** (same weights, threshold 0.50 → IoU **0.449**). On DeepGlobe (in-domain) v1 still scores IoU **0.670** / Occlusion-Recall **0.793** @0.44.
<!-- END AUTO-GENERATED -->

Model releases: **[`a4-roadseg-v3.1`](https://github.com/Akshat-Tiwari69/Trace/releases/tag/a4-roadseg-v3.1)** (deployed — v3 weights @ threshold 0.50) · `v3` · `v2` · `v1`.

**Resilience (the core thesis holds):** removing high-betweenness junctions collapses global efficiency *far faster* than removing random ones — i.e. the criticality scoring finds genuine chokepoints. On the OSM sample, targeted ablation mean RI **0.674 vs 0.860** random; on a live, tree-occluded Panaji satellite tile run end-to-end, **0.503 vs 0.780**.

**Dashboard:** disabling the top junction drops the Resilience Index **1.000 → 0.925** and draws the rerouted path live on the map.

---

## Quickstart

Runs on **CPU, no GPU, no prior pipeline run** — the dashboard ships with committed sample data.

```bash
pip install -r requirements.txt
streamlit run src/app/app.py
```

Then open the local URL, pick a critical junction, and hit **Simulate closure**.

### Run the whole pipeline on your own tile

Grab the best model from the [`a4-roadseg-v3.1` release](https://github.com/Akshat-Tiwari69/Trace/releases/tag/a4-roadseg-v3.1) (`road_spacenet.pt`) into `models/`, then one command takes imagery all the way to dashboard-ready artifacts:

<!-- AUTO-GENERATED: CLI reference — from src/pipeline/**/__main__ entrypoints -->
```bash
# whole pipeline: imagery → mask → graph → resilience (threshold/tile-size come from the checkpoint meta)
python -m src.pipeline.run_pipeline --image data/raw/your_tile.jpg \
    --checkpoint models/road_spacenet.pt --aoi your_area --postprocess

# stages individually
python -m src.pipeline.p1_segment.predict --image <tile> --checkpoint <pt> --aoi <id> \
    [--blend] [--postprocess]                 # → data/interim/{id}_mask.png (+ georef manifest for GeoTIFFs)
python -m src.pipeline.p2_graph.build_graph  --aoi <id>   # → healed routable graph
python -m src.pipeline.p3_analysis.analyze   --aoi <id>   # → criticality + resilience

# evaluate on the truthful Indian benchmark (held-out SpaceNet-5 Mumbai, real GT)
python -m src.pipeline.p1_segment.eval_spacenet --checkpoints models/road_spacenet.pt <v1.pt> \
    --device cuda [--grayscale] [--sweep]     # IoU/Dice (RGB or Cartosat-PAN proxy; --sweep = best threshold)
python -m src.pipeline.p1_segment.apls_eval    --checkpoints models/road_spacenet.pt <v1.pt> --device cuda  # routing APLS

# reproduce the v3 fine-tune (real SpaceNet labels + Cartosat grayscale aug)
python -m src.pipeline.p1_segment.finetune --init <v1.pt> --spacenet-corpus data/raw/spacenet/dg_format \
    --deepglobe-dir data/raw/deepglobe/train --grayscale-p 0.5 --cldice-weight 0 --oversample 1 \
    --epochs 10 --out models/road_spacenet.pt --device cuda
```

Cartosat-3 GeoTIFFs (RGB **or** 1-band PAN) are read via rasterio and carry their CRS/transform through as a metric graph automatically.
<!-- END AUTO-GENERATED -->

Full environment setup (CPU / cloud-GPU training / local-GPU paths) is in [`SETUP.md`](SETUP.md).

---

## Repository layout

```
src/pipeline/
  p1_segment/   model, predict (+blend/postprocess), finetune, eval_spacenet + apls_eval
                (real-GT SpaceNet benchmark), raster_io (GeoTIFF/PAN), OSM→mask data
  p2_graph/     skeletonize + sknw + MST/Union-Find healing, run_real_mask
  p3_analysis/  betweenness criticality + global-efficiency resilience, evaluate
  run_pipeline.py   A5 walking skeleton — P1→P2→P3→P4 in one command
src/app/        Streamlit + Folium dashboard
notebooks/      Colab/Kaggle training notebook
data/sample/    committed demo artifacts (so the app runs with no GPU)
docs/           Tracker (source of truth) + PRD, TRD, Design, Evaluation, Research, …
tests/          166 CPU unit tests
```

## Design rules (non-negotiable)

- **Stack:** Streamlit + Folium, **pure Python** — no database, no REST API, no JS SPA, no auth (v1).
- **ML:** fine-tune pretrained models only (never from scratch); **PyTorch** only.
- **Resilience = global efficiency** ratio — never a raw average-path-length ratio (it must stay finite when the graph disconnects).
- Training is hardware-agnostic (Colab/Kaggle); graph + dashboard run on **CPU**.
- Raw data and model checkpoints are git-ignored; only small **sample** data is committed so the repo runs out of the box.

Evaluation methodology and numbers live in [`docs/Evaluation.md`](docs/Evaluation.md); how the work is coordinated across the team is in [`docs/Tracker.md`](docs/Tracker.md).

## Tests

```bash
python -m pytest -q        # 166 CPU unit tests
```

## Roadmap

- **Cartosat-3 deployment:** the georeferenced + PAN inference path is ready; fine-tune / validate on the real Cartosat tiles when provided (keep a never-trained held-out set).
- **Push v3 further:** small encoder-unfreeze; graph-first SAM-Road++ spike (judged by APLS).
- Full progress, experiments (incl. negative results), and decisions live in [`docs/Tracker.md`](docs/Tracker.md).

## Team

Built by a three-person team working in separate lanes (coordinated through `docs/Tracker.md`):
**ML / segmentation / integration · graph & resilience · dashboard.**

## Data & licensing

Trained on the DeepGlobe Road Extraction dataset (research use). Sample imagery for demos is fetched from open basemap tiles. Respect each dataset's license; raw imagery and checkpoints are never committed to the repo.
