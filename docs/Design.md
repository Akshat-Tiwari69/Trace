# Design.md — Current Baseline and Web-Replacement Brief

> Streamlit/Folium describes the current deployed baseline, not the target. On 2026-07-14 Akshat authorized a full web-stack replacement. F9 will preserve the behavioral contract below while replacing the presentation, then rewrite this document with the selected stack, visual system and measured budgets.

## Current information architecture

| View | Primary purpose | Key content |
|---|---|---|
| **Briefing** | Orient a first-time user | Value proposition, baseline metrics, simplified map and worst-failure demonstration |
| **Analysis** | Explore network behavior | Full map, scenario/rankings/curves sub-tabs, metrics and export |
| **Your imagery** | Run the uploaded-image path | Disclosure/consent, validation, Modal P1, queue progress and result |
| **Methodology** | Explain evidence and limits | Pipeline, metric definitions, model/benchmark caveats |

The Analysis view is map-led; the whole application is not a single fixed 65/35 screen.

## Current visual system

- Dark operations-dashboard foundation with navy surfaces and restrained amber/blue accents.
- Fira Sans for interface text and Fira Code/tabular numerals for metrics.
- Satellite imagery is the default basemap; an alternate dark basemap is available.
- Criticality uses a labeled colorblind-conscious sequential ramp.
- Selected, disabled, rerouted, inferred and single-point-of-failure states have separate colors **and** line/marker/label cues.
- The header state chip says `DEMO · PANAJI` at rest and animates only when a simulation is active.
- Streamlit native components inherit the dark theme through `.streamlit/config.toml`; custom CSS must not fight native semantics.

Exact production colors live in the `TOKENS` mapping in `src/app/app.py` until F9 extracts a dedicated theme module. Do not maintain a second hex-value source here.

## Interaction contract

- Map pan/zoom persists through normal control changes.
- A junction can be selected from the map or an accessible list/control.
- Rankings can recenter/highlight the selected junction.
- Area failure has a keyboard-accessible alternative to drawing.
- Disabled links are dashed; reroutes are visually dominant but do not obscure context.
- Healed roads are distinguishable from observed roads.
- RI is explained as retained baseline efficiency; destructive deltas use the correct semantic treatment.
- Export is user-triggered, not eagerly recomputed on every rerun.
- Motion honors `prefers-reduced-motion`.

## Data visualization rules

- Never use color as the only carrier of meaning.
- Label units and assumptions; do not imply measured travel speed when only length at constant speed is available.
- Separate `is_bridged` (inferred edge) from `is_bridge` (structural single point of failure).
- Keep baseline and post-failure values comparable and name the removal count.
- Explain when a route is unreachable or a multi-node travel-time summary is not meaningful.
- Leaflet renders the WGS84 data in Web Mercator; pipeline metric computation remains in the appropriate metric frame.

## Layout and responsive behavior

Current layout uses Streamlit wide mode with a full-width product header. Analysis typically places the map and controls side by side; other tabs use task-appropriate sections.

F9 must define and browser-test at least:

- wide desktop (about 1440 px and above);
- common laptop (about 1280–1366 px);
- narrow/tablet-width fallback where map and controls stack without horizontal clipping.

Mobile field use remains out of scope, but a narrow browser must stay readable and operable.

## Accessibility baseline

- WCAG-AA contrast for normal text and essential controls.
- Visible focus indicators.
- Plain-language labels/help for RI, criticality, inferred roads and routing impact.
- Keyboard access to the full non-map scenario flow.
- Reduced-motion support.
- Maps have nearby text/tabular alternatives for essential findings.

F9 adds browser-level checks for focus order, keyboard completion, narrow layout and meaningful labels.

## Loading, empty and error states

- Sample-load failure names the expected artifact problem.
- Upload remains visibly disabled when Modal configuration is absent; sample use still works.
- Upload validation happens before remote work.
- Cold start, queue position, analysis and completion are distinct states.
- Retryable failures offer a retry; safe errors do not expose secrets or raw tracebacks.
- A degenerate uploaded graph is a failed job, not an empty successful analysis.

## Performance design

- Cache stable graph/data transformations by an input fingerprint.
- Keep graph payloads compact and URL-loaded; graduate to vector tiles only when representative data justifies the extra operational surface.
- Route-split and dynamically load the interactive map so editorial/sample orientation does not pay its JavaScript cost before needed.
- Prefer GPU-backed vector rendering and a small number of data-driven layers over one DOM/Python object per feature.
- Keep map payload/AOI bounded; use measured graph sampling/caching for large analysis.
- Set and verify Core Web Vitals, bundle, payload and map-interaction budgets before visual polish is accepted.

## F9 overhaul objectives

1. Make “Explore the sample” and “Analyze your imagery” unmistakable first-run paths.
2. Reduce nested navigation and keep the primary scenario story visible.
3. Replace the 1,900-line Streamlit presentation with a coherent web component/layout system while keeping Python domain logic behind a thin API.
4. Improve evidence hierarchy: result first, supporting metrics second, method/limitations always reachable.
5. Make upload privacy, cold-start/queue status and uncertainty understandable before submission.
6. Preserve every working analysis/export/recovery behavior through browser and unit tests.

## F9 acceptance evidence

- Before/after screenshots at the three target widths.
- Browser walkthroughs of every flow in `UserJourney.md`.
- Keyboard-only scenario and area-selection completion.
- No regression in existing app/unit/import/production-upload tests.
- Updated design tokens/components and removal of superseded CSS/UI paths.
- Recorded production bundle/payload sizes, Core Web Vitals and representative map frame/interaction measurements.
- Tracker log with the user problems solved, not only aesthetic changes.
