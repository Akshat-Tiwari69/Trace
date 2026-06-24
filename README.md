# Route Resilience

**Find the road junctions a city can't afford to lose.**

Route Resilience extracts roads from satellite imagery — *even where trees, buildings, and shadows hide them* — heals the gaps into a routable network, then scores which junctions are critical and how gracefully the network degrades when they fail. It ends in an interactive map you can click to simulate a closure and watch the city reroute.

> Status: **v1** — full pipeline closed end-to-end (segmentation → graph → resilience → dashboard), 76 tests green, trained model released.

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

**Segmentation (A4 — held-out DeepGlobe validation):**

| Metric | Value |
|---|---|
| IoU (flip + multi-scale TTA) | **0.670** |
| Occlusion-Recall @ deploy thr 0.44 | **0.793** |

The model is released as [`a4-roadseg-v1`](https://github.com/Akshat-Tiwari69/Trace/releases/tag/a4-roadseg-v1).

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

Grab the trained model from the [release](https://github.com/Akshat-Tiwari69/Trace/releases/tag/a4-roadseg-v1) into `models/`, then one command takes imagery all the way to dashboard-ready artifacts:

```bash
python -m src.pipeline.run_pipeline \
    --image data/raw/your_tile.jpg \
    --checkpoint models/deepglobe_mit_b3_scse_512px_best.pt \
    --aoi your_area
```

Or run the stages individually:

```bash
python -m src.pipeline.p1_segment.predict   --image <tile> --checkpoint <pt> --aoi <id>  # → mask
python -m src.pipeline.p2_graph.build_graph  --aoi <id>                                   # → graph
python -m src.pipeline.p3_analysis.analyze   --aoi <id>                                   # → criticality + resilience
```

Full environment setup (CPU / cloud-GPU training / local-GPU paths) is in [`SETUP.md`](SETUP.md).

---

## Repository layout

```
src/pipeline/
  p1_segment/   segmentation: model, losses, metrics, predict, evaluate, OSM→mask data
  p2_graph/     skeletonize + sknw + MST/Union-Find healing, run_real_mask
  p3_analysis/  betweenness criticality + global-efficiency resilience, evaluate
  run_pipeline.py   A5 walking skeleton — P1→P2→P3→P4 in one command
src/app/        Streamlit + Folium dashboard
notebooks/      Colab/Kaggle training notebook
data/sample/    committed demo artifacts (so the app runs with no GPU)
docs/           Tracker (source of truth) + PRD, TRD, Design, Evaluation, Research, …
tests/          76 CPU unit tests
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
python -m pytest -q        # 76 CPU unit tests
```

## Roadmap

- **A6 — dataset expansion + domain fine-tune:** push segmentation past 0.670 by fine-tuning on domain-matched Indian OSM-labelled tiles (+ optional SpaceNet-3).
- Georeferenced on-map demo (carry the tile's geo-transform through to the dashboard).

## Team

Built by a three-person team working in separate lanes (coordinated through `docs/Tracker.md`):
**ML / segmentation / integration · graph & resilience · dashboard.**

## Data & licensing

Trained on the DeepGlobe Road Extraction dataset (research use). Sample imagery for demos is fetched from open basemap tiles. Respect each dataset's license; raw imagery and checkpoints are never committed to the repo.
