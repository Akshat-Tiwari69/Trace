# TRACE user journeys

## Primary audience

TRACE is a portfolio-grade technical demonstrator for transport, disaster-resilience, geospatial, and ML reviewers. The interface assumes curiosity, not prior graph-theory knowledge. It must expose evidence honestly enough for an expert without overwhelming a first-time visitor.

## Journey 1 — understand the product quickly

1. The visitor lands on the Panaji field atlas and immediately sees the road network, place, graph size, and current resilience state.
2. A short mode strip communicates the product loop: **Explore → Stress → Compare → Recover**.
3. Selecting the highest-ranked junction from the map or table updates a plain-language field insight explaining whether it is an articulation point or a high-flow junction.
4. The methodology link explains the metric, sample provenance, model boundary, and limitations without interrupting the workspace.

Success: within one minute the visitor can explain that TRACE extracts a road graph and measures how routing efficiency changes when important junctions fail.

## Journey 2 — stress a critical junction

1. The visitor selects a critical junction.
2. They choose **Test this junction**, which moves into Stress mode and stages the failure.
3. Running the stress test sends the normalized node set to the CPU API.
4. The result updates efficiency loss, surviving component fraction, failed links, and either a representative detour or a disconnection.
5. The URL now carries the selected junction, mode, and failed-node set, so the scenario can be reloaded or shared.

Success: the scenario is deterministic, bounded, and derived from the same baseline/sampling policy as the committed evidence.

## Journey 3 — compare and recover

1. The visitor adds several failures and switches to Compare to vary baseline/scenario emphasis on the same map.
2. They switch to Recover and remove junctions in descending criticality rank.
3. Every removal automatically re-runs the scenario and exposes the recovered efficiency.
4. Reset returns to the baseline without reloading the graph.

Success: recovery is clearly described as a criticality-ordered demonstration, not a claim of optimal infrastructure investment.

## Journey 4 — export evidence

1. At baseline or after a scenario, the visitor opens Export.
2. GeoJSON contains the network plus current failed/scenario state.
3. CSV contains ranked junction evidence and failure flags.
4. Files are generated locally in the browser from already-loaded authoritative data.

Success: a reviewer can inspect the exact graph and rankings behind the visible story.

## Journey 5 — analyze personal imagery

1. The visitor opens **Analyze imagery** and selects a PNG or JPEG.
2. The client immediately rejects unsupported type, more than 11 MiB, or more than 4096 px per side.
3. The visitor supplies an estimated metres-per-pixel scale and explicitly confirms authenticated external GPU processing.
4. The UI narrates queued, running, complete, or failed state while polling a capability URL.
5. On completion it shows graph size, critical/articulation counts, worst-junction resilience, and the extracted network over the uploaded image.
6. GeoJSON and JSON results can be downloaded; coordinates remain in honest image space.

Success: the upload reaches the maintained P1→P2→P3 path once, survives a normal app restart through the filesystem queue, never exposes the Modal secret, and never invents a georeference.

## Failure and recovery states

- If initial sample evidence cannot load, show a clear fatal state with retry.
- If a simulation fails, preserve the current graph/selection and announce a dismissible error.
- If basemap tiles fail, keep the authoritative network interactive and show an inline warning.
- If Modal is not configured or wakes slowly, return an honest bounded error/status; never fake an analysis result.
- If the CPU queue is busy, report queue position rather than starting competing analysis workers.
- If an upload job fails, expose a safe public message while retaining internal detail only in server logs/state.

## Keyboard and assistive-technology journey

- A skip link moves directly to the workspace.
- Mode tabs use arrow-key navigation and visible focus.
- Every map-selectable critical junction exists in the semantic ranking table.
- Buttons, ranges, toggles, upload consent, dialog close/Escape, exports, and recovery work without a pointer.
- Status and errors use live regions; decorative graphics are hidden from assistive technology.
- Reduced-motion users receive no essential animation.

Success: automated axe checks find zero serious/critical violations, browser journeys complete at 375/768/1024/1440 px without horizontal overflow, and a manual keyboard pass completes all five journeys.
