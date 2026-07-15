# UserJourney.md — Current Flows and F9 Parity Contract

## Users and entry points

| User | Entry | Goal |
|---|---|---|
| Planner/disaster analyst | Hosted or local dashboard | Understand chokepoints and compare failures |
| Geospatial reviewer | Dashboard Methodology plus exported evidence | Inspect assumptions, inferred roads and metric limits |
| User with imagery | **Your imagery** tab | Segment one PNG/JPEG, analyze the derived graph and inspect the result |
| Developer/researcher | CLI/notebook | Reproduce or evaluate a pipeline/model change |

The default sample path requires no GPU, checkpoint or secret. The upload path is enabled only when the Modal URL/key are configured.

## Navigation

The current app has four top-level tabs:

```text
Route Resilience
├─ Briefing
│  ├─ baseline explanation and map
│  └─ one-click worst-junction demonstration
├─ Analysis
│  ├─ Scenario
│  ├─ Rankings
│  ├─ Curves
│  └─ Export
├─ Your imagery
│  └─ consent → segmentation → queued CPU analysis → result
└─ Methodology
   └─ pipeline, metrics, evidence and limitations
```

## Flow A — Understand the baseline

1. Open **Briefing** and read what the map and Resilience Index mean.
2. Inspect the Panaji sample; brighter criticality colors indicate higher scores, while inferred/structural states also use labels and line styles.
3. Trigger the worst-junction example or continue to **Analysis**.
4. Compare the intact network with the selected failure and read the plain-language impact.

Success: the user can explain which junction matters and what RI `1.0` represents without reading source code.

## Flow B — Analyze failures

1. Open **Analysis → Scenario**.
2. Select a junction from the accessible control or click it on the map.
3. Optionally draw/select an affected area for a compound failure.
4. The app recomputes global-efficiency retention, component impact and a representative route when meaningful.
5. Use **Rankings** to move between critical nodes and **Curves** to compare targeted/random degradation.
6. Reset to return to the baseline.

Expected behavior:

- A destructive change is not presented as a positive success state.
- A split network remains mathematically valid; unreachable routes are explained rather than divided by infinity.
- Multi-node scenarios do not invent a single travel-time percentage when it has no clear interpretation.
- The map viewport and current scenario survive normal Streamlit reruns.

## Flow C — Upload imagery

1. Open **Your imagery** and read/accept the processing disclosure.
2. Choose a supported PNG/JPEG within the displayed decoded-size and pixel limits. Selection starts the configured Modal segmentation flow; there is no separate submit button.
3. Wait through a possible scale-to-zero cold start. On a retryable transport failure, retry without losing the sample dashboard state.
4. Modal returns a mask. If georeference is unavailable, choose/confirm the ground-sample-distance assumption used for metric graph lengths.
5. The app persists the derived mask and versioned job state, displays queue position, then runs CPU P2/P3. Changing GSD submits a replacement analysis for that mask.
6. Poll in the active browser session until the job is done, then inspect the uploaded-image graph and analysis. The filesystem queue survives normal process restarts, but there is no account/job library for recovering a result after the browser session is lost.

Privacy/retention:

- There is no account or permanent project library.
- Original upload bytes are handled in memory and sent to Modal.
- Derived mask, queue state and result files live on the host temporarily and are age-cleaned (currently about 24 hours).

## Flow D — Export and communicate

1. Open the Analysis export control after choosing a baseline or scenario.
2. Download the current graph as GeoJSON and the summary as PNG.
3. Preserve the visible assumptions/limitations when reusing the output.

Pipeline CSV/JSON evidence remains available to technical users but is not presented as a dashboard download unless the UI explicitly exposes it.

## Flow E — Review the method

1. Open **Methodology**.
2. Follow imagery → P1 → P2 → P3 → dashboard.
3. Read the difference between predicted roads, healed roads, graph-theoretic bridges and resilience.
4. Check the benchmark caveats: Mumbai is development evidence; grayscale is a PAN proxy; A18 is research-only and not deploy-ready.

## Errors and recovery

| Situation | Required response |
|---|---|
| Sample artifacts missing/corrupt | Explain the expected paths; do not show a raw traceback |
| Modal configuration absent | Keep sample mode available and explain that upload is disabled |
| Unsupported/oversized upload | Reject before remote inference and state the accepted limit/type |
| Unauthorized/invalid Modal response | Fail closed, show a safe message and allow retry |
| Cold start or queued work | Show stage/queue progress rather than a frozen page |
| App restarts during a job | Recover queued/running state using claims/leases; do not duplicate live work |
| Empty/degenerate mask graph | Mark the analysis failed; do not report a successful empty network |
| Map tile provider unavailable | Keep controls/evidence usable and explain the basemap failure |
| No nodes in a drawn area | Say that the selection affected no junctions |

## F9 replacement goals

The authorized web-stack replacement must improve this journey without changing its semantics:

- clearer first-run choice between sample exploration and own imagery;
- less tab/sub-tab hunting for the primary scenario flow;
- responsive laptop/narrow layouts;
- keyboard and screen-reader alternatives for every essential map action;
- more legible job progress, evidence, privacy and uncertainty;
- browser-tested completion of Flows A–E.
- measured bundle/payload/Web-Vitals/map-interaction budgets before the Streamlit baseline is removed.
