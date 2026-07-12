# Schema.md — Artifact and Data Contracts

> Route Resilience is file-based. This document records the files and fields the current code exchanges; it is not a hypothetical database design.

## Directory lifecycle

| Directory | Purpose | Git policy |
|---|---|---|
| `data/raw/` | Source imagery, OSM and downloaded datasets | ignored except placeholders |
| `data/interim/` | P1 masks and alignment/probability/provenance sidecars | ignored |
| `data/processed/` | P2/P3 graphs, tables and successful run summaries | ignored by default |
| `data/outputs/` | Exports and transient hosted upload jobs | ignored |
| `data/sample/` | Small committed contract fixtures and evidence reports | tracked |
| `models/` | Checkpoints and resumable training state | ignored |

## AOI identity

`aoi` is a sanitized identifier used in paths. It cannot contain traversal or path separators. The source image, checkpoint/config signature and optional georeference—not only the AOI name—determine whether a stage is reusable.

## P1 outputs

### Required mask

`data/interim/{aoi}_mask.png`

- Same raster grid as the inference output.
- Binary road/background when loaded by P2.
- A successful mask alone does not imply a valid graph or completed pipeline.

### Optional/georeferenced sidecars

`data/interim/{aoi}/manifest.json`

- Raster width/height, CRS and affine transform when the source is georeferenced.
- P2 converts geographic/projected raster coordinates into metric graph coordinates; silent degree-as-metre behavior is invalid.

`data/interim/{aoi}/prob.png`

- Same grid as the mask.
- Written by probability-preserving/blended inference.
- Used only as optional corridor support during healing; mask-only inputs remain valid.

`data/interim/{aoi}/provenance.json`

- `checkpoint`, `model_sha256`, `encoder`, `arch`, `threshold`, `image_size`, `git_commit`, `created_utc`.

## P2/P3 graph

Canonical processed forms:

- `data/processed/{aoi}_graph.graphml`
- `data/processed/{aoi}_graph.geojson`

The graph is a NetworkX `MultiGraph`. Parallel edges are identified by `(u, v, edge_key)` and must survive GraphML/GeoJSON round trips.

### Node fields

| Field | Required | Meaning |
|---|---|---|
| `node_id` | GeoJSON | Stable integer identifier within the graph |
| `x`, `y` / Point geometry | yes | Metric or WGS84 position according to artifact metadata |
| `degree`, `type` | yes | Graph degree and endpoint/intersection classification |
| `betweenness` | after P3 | Normalized criticality score |
| `is_critical` | after P3 | Member of configured top critical set |
| `is_articulation` | after P3 | Removing the node increases component count |

### Edge fields

| Field | Required | Meaning |
|---|---|---|
| `u`, `v`, `edge_key` | GeoJSON/topology identity | MultiGraph identity; GraphML may encode the key structurally rather than duplicating every property |
| `geometry` | yes in GeoJSON | LineString following the road/bridge shape |
| `length_m` | yes | Positive route weight in metres, or explicit pixel-space fallback converted by `resolution_m` |
| `is_bridged` | yes | Edge inferred by the P2 healing stage |
| `is_bridge` | after P3 | Graph-theoretic bridge whose removal disconnects the graph |
| `edge_betweenness` | reserved/optional | Normalized edge criticality when an evaluation path computes it; production P3 currently leaves the default |
| `confidence` | when probability exists | Mean P1 support along the edge, clipped for presentation downstream |
| `width_m` | when medial information exists | Approximate road width |

Graph invariants:

- Every edge endpoint exists; `length_m > 0`.
- Geometry endpoints agree with node positions after simplification/consolidation.
- GraphML and GeoJSON describe the same analyzed graph.
- `is_bridged` and `is_bridge` are different concepts and are never interchanged.
- Missing georeference is an explicit pixel-space mode, not an accidental default.

## P3 tables

### Criticality

`data/processed/{aoi}_criticality.csv`

Required columns:

`node_id,betweenness,rank,is_critical,is_articulation,x,y`

### Resilience curve

`data/processed/{aoi}_resilience.csv`

Required columns:

`n_removed,targeted_efficiency,targeted_resilience_index,targeted_largest_cc_fraction,random_efficiency,random_resilience_index,random_largest_cc_fraction`

The product contract requires resilience to use the baseline node universe and remain in `[0, 1]`. The single-scenario path satisfies this; the current multi-step `ablation_curve` removes nodes and therefore has a known shrinking-denominator defect queued for A45. Existing curve artifacts must be regenerated after that fix.

### Successful run summary

`data/processed/{aoi}_run.json`

Contains `status`, AOI, output paths, node/edge counts, per-stage run/skip timings, resolved configuration and available provenance. It is written only after the P4 artifact seam succeeds.

Stage signature manifests are internal reuse metadata. They mean the named stage completed for a specific input/config signature; they are not whole-pipeline success records.

## Dashboard sample contract

The default app reads:

- `data/sample/panaji_demo_graph.geojson`
- `data/sample/panaji_demo_criticality.csv`
- `data/sample/panaji_demo_resilience.csv`

These files are tested as committed fixtures. Evidence JSON derived from the sample must be regenerated whenever the sample graph changes.

## Hosted upload jobs

`data/outputs/upload_jobs/` stores transient single-host queue files:

- `{job_id}.json`: versioned job state, sequence and owner/lease metadata.
- `{job_id}.npy`: binary mask input.
- `{job_id}.claim`: atomic process claim.
- `{job_id}.result.json`: versioned serializable analysis result.
- `sequence.counter`/`sequence.lock`: persistent FIFO allocation.

Jobs are disposable and age-cleaned. This is not a user data store or a horizontally scalable queue.

## Evaluation artifacts

Committed JSON reports in `data/sample/` must name the model/data unit, sample count, thresholds, coverage and metric settings needed to interpret them. Mumbai reports are development-benchmark evidence, not untouched-test evidence.

## Validation ownership

- P1 validates imagery, mask shape, checkpoint metadata and sidecars.
- P2 validates coordinates, geometry and positive edges.
- P3 validates required graph/table fields and bounded metrics.
- P4 validates its committed/runtime input columns and handles missing artifacts gracefully.
- CI guards the sample contract and production upload-analysis dependency path.
