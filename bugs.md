# Trace — Production-Readiness Review (bugs.md)

**Date:** 2026-07-03 · **Branch:** `dev` @ d3edda9 · **Scope:** full P1→P4 pipeline, ML stack, graph generation, UI/UX, deployment.
**Method:** six parallel specialist reviews (ML engineering, graph/code review, UI/UX, architecture, security, silent-failure hunt) + coordinator synthesis. Severity: **P0** correctness/broken, **P1** high-impact, **P2** moderate, **P3** polish. Effort: **S/M/L**.

---

## Status

**Fix-status narratives:** see `docs/Tracker.md` §10 (A36 entries, 2026-07-04; A37 entry, 2026-07-09). This file marks every finding with a checkbox: `[x]` = fixed & verified in the repo, `[ ]` = open (reason stated inline).

### Open items

- [ ] §3 topology-loss retest — **GPU-blocked** (from-scratch retrain experiment)
- [ ] §3 A18 graph-first spike (SAM-Road++ vs v3.2 APLS) — **GPU-blocked**; corpus + harness ready
- [ ] §3 real-PAN validation of the grayscale proxy — **data-blocked** (needs Cartosat chips)
- [ ] §4 probability-map-aware healing (corridor check) — **deferred**: P1↔P2 contract change, sequenced next
- [ ] §5E geopandas major alignment — **operator** (joint dev+prod test)
- [ ] §5H full filesystem job queue — **deferred** until multi-user pilot scale (semaphore covers today)
- [ ] §6 Caddy rate limiting — **operator** (third-party module install)
- [ ] §9.1 four-tab app restructure — **deferred** (product decision; Methodology tab exists)

### Operator checklist (Akshat, on the boxes)

- [ ] Redeploy Modal (`modal deploy deploy/modal_app.py`)
- [ ] Remove the port-8501 ingress rule in the Oracle console + drop the iptables ACCEPT
- [ ] Install updated systemd unit (loopback bind) + journald drop-in + Caddyfile body cap
- [ ] `chmod 600` the env file holding `MODAL_SEG_KEY`
- [ ] Sanity-check blended-inference CPU latency on the ARM box
- [ ] Install the third-party caddy-ratelimit module
- [ ] Run the joint dev+prod geopandas-alignment test before flipping either pin

**96 of 101 checkbox-marked findings fixed; 5 open** (one §6 P3 note is left unmarked — it explicitly says "no action today").

## 0. Executive summary

**~90 findings; 12 P0s.** The project's bones are good — the file-handoff pipeline architecture, the negative-result experiment culture, and the dark-ops design system are all above par for a student team. What's missing is the layer that makes it *trustworthy in front of strangers*: interaction correctness in the dashboard, fail-loud behavior in the pipeline, CI in front of an auto-deploying branch, and statistical rigor in the model-promotion evals.

**The twelve P0s:**

| # | Finding | Where |
|---|---------|-------|
| 1 | Upload re-fires GPU inference on **every** rerun (every click after upload = 5–240 s + GPU cost) | §2B — `app.py:934` |
| 2 | Map viewport resets on every selection/toggle (key churn remounts the iframe) | §2C — `app.py:1053` |
| 3 | Resilience delta colored **green** when the network degrades (`delta_color="inverse"`) | §2D — `app.py:619` |
| 4 | Port 8501 open to the internet in plaintext, bypassing Caddy/TLS | §6 — `deploy/README.md:36` |
| 5 | Modal endpoint: no size cap, no rate limit, key checked *after* body decode → cost abuse | §6 — `modal_app.py:79` |
| 6 | `sknw multi=False` silently drops parallel edges (roundabouts, dual carriageways) → biases the resilience metric | §4 — `skeleton_graph.py:84` |
| 7 | Zero-length edges only pruned in one code path; downstream treats coincident nodes as unreachable | §4 — `resilience.py:69` |
| 8 | Empty-mask runs flow through P2→P3 → RI=0.0 that looks like a real result; pipeline exits 0 | §7 — `analyze.py:57`, `resilience.py:105`, `run_pipeline.py:107` |
| 9 | Modal response bytes decoded outside the try → raw traceback on the public app | §7/§2B — `app.py:947` |
| 10 | No CI while `dev` auto-deploys to production every 2 minutes | §5F — `.github/`, `update.sh` |
| 11 | No AOI sanitization (path traversal/collision) + non-atomic artifact writes | §5A — `config.py`, `graph_io.py` |
| 12 | Unbounded disk/RAM on the deploy box (no `maxUploadSize`, no log rotation) + stale v1-checkpoint defaults in eval/docs | §5C/§3 — `config.toml`, `evaluate.py:26` |

**The three biggest product levers** (beyond bug fixes): (1) **close the upload loop** — uploaded imagery currently dead-ends at a mask preview and never becomes a graph or resilience score, which is the product's entire pitch; (2) **make the demo one click** — a "run the worst-case failure" CTA that showcases the core interaction instantly; (3) **make the evals significance-aware** — several past promotion decisions ride on deltas within plausible sampling noise.

---

## 1. What each stage currently does

### P1 — Segmentation (`src/pipeline/p1_segment/`)
SegFormer MiT-B3 encoder + SCSE U-Net decoder (segmentation_models_pytorch, ~47.5M params), trained on DeepGlobe Roads, fine-tuned on SpaceNet-5 Mumbai with a DeepGlobe anchor and grayscale/gamma augmentation for Cartosat-PAN robustness. Deployed checkpoint **v3.2** (`road_pan.pt`) at threshold 0.52. Inference: sliding-window + Hann blending over tiles (`predict.py`), optional flip/rotate TTA, optional post-processing (component filter, morphological cleanup, spur pruning). Georeferenced GeoTIFF input supported via `raster_io.read_image_any` (PAN 1-band → stretched 3-channel), writing a CRS/transform manifest for P2. Held-out SpaceNet-Mumbai: IoU 0.459 RGB / 0.418 grayscale / APLS 0.499.

### P2 — Graph build (`src/pipeline/p2_graph/`)
Binary mask → skeletonize (skimage) → pixel graph → node/edge extraction (`skeleton_graph.py`, `build_graph.py`) → geometry simplification (`simplify.py`) → gap healing that bridges nearby endpoints (`healing.py`, edges flagged `is_bridged`) → GraphML + GeoJSON export (`graph_io.py`), georeferenced when a manifest exists.

### P3 — Analysis (`src/pipeline/p3_analysis/`)
Betweenness-centrality criticality ranking (`criticality.py`), resilience as a **global-efficiency ratio** (locked metric; finite under disconnection), percolation and flood scenario modules, APLS implementation (`apls.py`) reused by the P1 eval harness. Outputs `{aoi}_criticality.csv` + resilience curve.

### P4 — Dashboard (`src/app/app.py`)
Single-file Streamlit + Folium app: demo AOIs from `data/sample/`, image upload → remote Modal GPU endpoint (shared-secret auth) → P1→P2→P3 in-process → dark-themed Folium map with critical junctions, criticality table, resilience score, in-process `simulate_ablation(graph, node)` what-if. Deployed at trace.tiwaribabu.in (Oracle ARM box, Caddy HTTPS).

---

## 2. UI/UX findings (highest priority)

**Reviewer verdict:** the dark-ops design system (PR #109) is genuinely strong for Streamlit — metric cards, button hierarchy, `:focus-visible`, reduced-motion handling are above-average craft. But there are **two P0 interaction defects** (re-firing GPU inference on every rerun; map viewport reset on every selection), an **inverted metric color semantic**, and several "unfinished product" tells (dead scenario dropdown, disabled region selector, upload that doesn't feed the pipeline). Findings ordered by user journey.

### 2A. First load / first-run experience

#### [x] [P1] No value proposition or guided entry — first 5 seconds are unexplained
- **Where:** first render; header `app.py:590–609`, empty-state hint `app.py:681`
- **What:** A first-time judge sees a dark map on the left and a dense control stack on the right. The only guidance is an `st.info("No simulation running…")` buried below two metrics, a radio, three selectboxes, two buttons, and three checkboxes. Nothing says what the product does.
- **Why:** Judges form the "is this a real product?" judgment before their first click. The headline interaction (click a junction → watch the network degrade) is discoverable only by reading a small caption (line 646).
- **Fix:** One-line subhead + a primary CTA at the top of the panel: `st.button("▶ Run demo: disable the #1 chokepoint", type="primary")` that sets `disabled_nodes` to the top-ranked node and reruns. Optionally a dismissible `st.expander("What am I looking at?")` with 3 bullets. **Effort:** S

#### [x] [P1] Brand header lives inside the 35% right column, not the page top
- **Where:** `app.py:590–609` (rendered inside `render_panel`, inside `panel_column` at `app.py:1095`)
- **What:** Logo, product name, and "LIVE" chip render in the narrow right column; on first paint the page has no title row — the map iframe is the top-left element. On 1366×768 the h1 + chip + subtitle wrap awkwardly.
- **Fix:** Move the `rr-header` markdown to `main()` before `st.columns([6.5, 3.5])` so it spans the page. **Effort:** S

#### [x] [P2] "LIVE" pulsing chip is a false affordance
- **Where:** `app.py:605`, CSS `app.py:791–797`
- **What:** An animated green "LIVE" dot implies streaming data; the app is a read layer over precomputed files. Judges who probe ("live from what feed?") catch the app overselling.
- **Fix:** Change to a truthful label (`DEMO · PANAJI`), or make it stateful — green `SIM ACTIVE` only when `disabled_nodes` is non-empty. **Effort:** S

#### [x] [P1] No About/Methodology view — an entire persona has no entry point
- **Where:** whole app; `UserJourney.md` specifies a `[Map] [About/Methods]` tab pair
- **What:** The methodology evaluator persona has nowhere to read what "Resilience Index = global efficiency ratio" means, or see IoU/APLS numbers. The eval artifacts already exist in `data/sample/` (`panaji_demo_apls.json`, `segmentation_eval.json`, `panaji_demo_graph_eval.json`) but are shown nowhere.
- **Why:** For hackathon judging, the methodology story is half the score.
- **Fix:** `st.tabs(["Map", "Methodology"])` at the top of `main()`; About tab shows the pipeline diagram, metric definitions, and small dataframes of the eval JSONs. **Effort:** M

#### [x] [P2] Two-column scroll imbalance on 1366×768; panels never stack as Design.md promises
- **Where:** `app.py:1012` (`st.columns([6.5, 3.5])`), map fixed at 640px (`app.py:861–866, 1050`)
- **What:** The right panel stacks ~1,800–2,200px of content beside a ~750px map column. On a 768px-tall laptop, reading the degradation curve means the map has scrolled fully off-screen — the core loop (pick junction → simulate → compare) spans two scroll positions. Streamlit columns never wrap at 1366px, so Design.md's "panels stack below the map" never happens.
- **Fix:** (a) collapse low-priority content (degradation curve, top-junctions table, exports) into `st.expander`s, keeping the panel under ~700px; or (b) wrap the panel in `st.container(height=640)` so it scrolls independently beside the map. **Effort:** S–M

#### [x] [P2] Disabled "Region" selectbox signals unfinished software
- **Where:** `app.py:635` — `st.selectbox("Region", ["Panaji demo"], disabled=True)`
- **Fix:** Remove it; the region is already in the subtitle. Advertise multi-region ambition in the Methodology tab instead. **Effort:** S

#### [x] [P1] Scenario dropdown is decorative — "Accident" and "Road closure" behave identically
- **Where:** `app.py:636–637` (`# TODO: wire scenario-specific behavior into routing logic`); only "Flood" changes anything (`app.py:323–337`)
- **What:** The most prominent select changes nothing for 2 of 3 options — pixel-identical results. It also causes a full map remount (scenario baked into the map key, `app.py:1053`), resetting the viewport with no payoff.
- **Fix (minimal, honest):** Cut to what's real: `st.radio("Failure mode", ["Single junction", "Flood area (draw on map)"])`. Don't ship a no-op select. **Effort:** S

### 2B. Upload / live detection flow

#### [x] [P0] GPU inference re-fires on every Streamlit rerun — every click re-pays 5–240 s
- **Where:** `app.py:934–941` (`_call_modal_seg` invoked unconditionally when `upload is not None`; the function at `app.py:890–909` has no cache)
- **What:** After a successful upload, *any* interaction — toggling a checkbox, clicking the map, pressing Simulate — reruns the script and POSTs the image to Modal again under a blocking spinner. Every rerun costs a full GPU round-trip (plus cold start) and GPU dollars.
- **Why:** Makes the dashboard feel broken the moment someone uploads: a 30-second "Segmenting roads on GPU…" spinner after ticking a checkbox. In a live demo this is fatal. Also a cost/DoS amplifier (see §6).
- **Fix:** `@st.cache_data(show_spinner=False, ttl=3600)` on a wrapper taking `image_bytes: bytes` (bytes hash cheaply and correctly), or store `(mask_png, threshold)` in `session_state` keyed by `hash(image_bytes)`. ~5 lines. **Effort:** S

#### [x] [P1] The upload — the product's headline pitch — is buried, conditional, and dead-ends
- **Where:** `app.py:912–958`; rendered at `app.py:1060` *below* the 640px map; hidden entirely unless `MODAL_SEG_URL` is set (`app.py:919–920`)
- **What:** The "upload a satellite image → road network" promise is (a) below the fold, (b) invisible on any box without the env var (a judge running locally per SETUP.md sees no upload at all), and (c) terminates at a 3-image strip — the mask never becomes a graph, never appears on the map, never gets a resilience score.
- **Fix (near-term):** Promote to a clearly labeled section/tab above the fold; when `MODAL_SEG_URL` is unset show the section with the uploader disabled + an info pointing to the hosted demo. **Fix (right):** close the loop — after the mask, run the existing CPU skeletonize→graph→analyze stages and offer "Load this network into the dashboard." That single feature converts the app from "viewer of one canned city" to the actual product. **Effort:** S (promote) / L (close the loop)

#### [x] [P1] No staged progress or cold-start messaging during the long call; the one warning that exists disappears at the worst moment
- **Where:** `app.py:926–932` (the "~30 s warm-up / 0.5 m-px" caption is inside `if upload is None:` so it unmounts the instant a file is chosen); `app.py:935` (single opaque `st.spinner`, 240 s timeout at `app.py:905`)
- **Fix:** `st.status("Extracting road network…", expanded=True)` with staged writes (upload size → "waking the GPU (~30 s after idle)" → complete), and move the resolution caption outside the `if upload is None` branch. A true progress bar would need a polling endpoint on Modal — staged narration is the best Streamlit-compatible workaround. **Effort:** S

#### [x] [P2] No client-side validation before shipping the image to the GPU; constraints not stated up front
- **Where:** `app.py:921–925` (only extension filter), `app.py:896–901` (base64 of raw bytes, no size check)
- **What:** A 40 MB PNG gets base64-inflated (+33%) and POSTed; a 100×100 thumbnail or 12,000px export is accepted without warning even though extraction degrades outside ~0.5 m/px.
- **Fix:** Pre-check bytes length + PIL dimensions with an honest `st.warning`; set `maxUploadSize = 25` in `.streamlit/config.toml`. **Effort:** S

#### [x] [P2] Upload error states: raw exception surfaced, no retry, and an uncaught-traceback path
- **Where:** `app.py:937–940` (`st.error(f"Inference failed: {error}")` surfaces `urllib` internals); `app.py:947–948` (`Image.open` on server-returned bytes sits **outside** the try — a malformed response paints a full traceback on screen)
- **Fix:** Map exception classes to human messages (endpoint unreachable / cold-start timeout), move the PIL decode inside the try, add a retry affordance. **Effort:** S

### 2C. Map & graph visualization

#### [x] [P0] Every selection, toggle, or scenario change destroys the map viewport (zoom/pan reset)
- **Where:** `app.py:1035, 1044, 1053` — `key=f"network_map_{reset_counter}_{selected_node}_{show_critical}_{show_healed}_{show_spof}_{scenario}"`
- **What:** The `st_folium` key encodes five pieces of state, so changing *any* of them remounts the iframe and resets to `zoom_start=15` at the centroid. The flagship interaction — zoom into a neighbourhood, click a junction — immediately throws the user back to city view.
- **Why:** The single most frustrating behavior in the app; it breaks spatial continuity on every interaction. Judges will notice within 30 seconds.
- **Fix:** Stable key + round-trip the viewport: `st_folium` returns `center`/`zoom` — store them in session state and build the map with them; keep only `reset_counter` in the key so Reset intentionally reframes. This is the canonical streamlit-folium workaround. **Effort:** M

#### [x] [P1] FastMarkerCluster is the wrong tool for ~40 markers — it hides the critical junctions it's supposed to showcase
- **Where:** `app.py:388–425`
- **What:** With 36 critical nodes, default zoom collapses junctions into generic numbered cluster bubbles — the color encoding (blue selected, vermillion disabled, ramp-colored critical) is invisible until zoomed in, and cluster clicks can fail the 33 m nearest-node threshold (`app.py:451`).
- **Fix:** Plain `folium.CircleMarker`s in a named `folium.FeatureGroup("Critical junctions")` (also fixes the unnamed LayerControl entry). 40 SVG circles are trivially cheap. **Effort:** S

#### [x] [P2] Legend inaccuracies and split legends
- **Where:** `app.py:249–266` (HTML legend) vs `app.py:302–307, 430` (branca colormap bar) vs actual edge styling `app.py:349–371`
- **What:** (1) Legend says healed = grey dashed, but healed roads render in their criticality ramp color, dashed — the legend teaches a color that never appears. (2) Disabled `#D55E00` vs rerouted `#E69F00` are two adjacent oranges on dark tiles, distinguished mainly by dash/weight, which the legend doesn't show. (3) The encoding is taught in three places (HTML legend, branca bar, caption below map). (4) Side-by-side mode duplicates all chrome in each ~440px pane.
- **Fix:** Truthful legend (dashed mid-ramp swatch labeled "dashed = inferred"; dashed disabled swatch; thick solid reroute swatch), merge the ramp as a CSS gradient bar into the same legend, suppress legend/LayerControl on the baseline pane. **Effort:** M

#### [x] [P2] Clicking empty map / non-critical junction is silently ignored
- **Where:** `app.py:441–453` (`nearest_critical_node` returns None past 33 m or for non-critical nodes) + `app.py:1087–1093`
- **Fix:** `st.toast("No critical junction near that click — the highlighted dots are selectable.")` on unresolved clicks. Also consider allowing selection of *any* node — `simulate_ablation` works on any node; restricting to the pre-blessed 36 makes the tool feel canned. **Effort:** S

#### [x] [P2] Map rebuild is uncached and per-edge; will not survive 10k+ edges
- **Where:** `app.py:339–371` (Python `iterrows()` + one `folium.PolyLine` per edge, every rerun), doubled in side-by-side (`app.py:1016–1026`)
- **What:** Fine at 500 edges (~100–300 ms/rerun); at 10k edges it becomes multi-second Python loops plus a multi-MB iframe payload re-shipped per rerun, and Leaflet stutters with 10k individual layers.
- **Fix:** Single `folium.GeoJson(edges_gdf, style_function=..., tooltip=GeoJsonTooltip)` layer with vectorized style columns; cache rendered map HTML keyed on `(selected_node, disabled_nodes, toggles, scenario)`; `gdf.simplify(tolerance)` for large AOIs. **Effort:** M
- **Status:** Fixed A37 — single `folium.GeoJson` layer with cached vectorised style columns.

#### [x] [P2] Side-by-side comparison mode: unsynchronized panes, doubled clutter
- **Where:** `app.py:1015–1045`
- **What:** The two maps don't share pan/zoom (comparing requires manually navigating both), each carries full chrome, and the baseline pane still shows the selected-junction marker.
- **Fix:** `folium.plugins.DualMap` (synced panes in one figure through a single `st_folium` call); or drop the mode — the orange reroute + dimmed disabled links already *are* the before/after story. **Effort:** M
- **Status:** Resolved A37 by REMOVING the mode (DualMap flaky with streamlit-folium; the reroute overlay already shows before/after).

#### [x] [P3] Fixed `zoom_start=15`, no bounds fitting
- **Where:** `app.py:310–315`. **Fix:** `road_map.fit_bounds(...)` from `nodes.total_bounds`. **Effort:** S

#### [x] [P3] No offline/tile-failure fallback
- **Where:** `app.py:316–321`; UserJourney.md requires a fallback message. If CartoDB/Esri tiles are unreachable (venue Wi-Fi), the graph floats on a grey void.
- **Fix:** Caption under the map + set the map div background to `--rr-bg`. True tile detection isn't feasible from Python. **Effort:** S

### 2D. Panel, metrics & data viz

#### [x] [P0] Resilience Index delta color is semantically inverted — a resilience *drop* renders green
- **Where:** `app.py:615–621`, `delta_color="inverse"`
- **What:** In `st.metric`, `"inverse"` colors *negative deltas green*. When a junction failure drops RI from 1.000 to 0.85, "-15.0%" displays green — the UI congratulates the user on degrading the network.
- **Fix:** `delta_color="normal"`. One word. **Effort:** S

#### [x] [P2] Resilience Index is a naked number with no context, while benchmark data sits unused on disk
- **Where:** `app.py:615–621`; unused artifacts `data/sample/panaji_demo_percolation.json`, `panaji_demo_flood_curve.png`, `betweenness_benchmark.json`
- **Fix:** Contextual caption computed from the degradation curve already loaded at `app.py:96–101` ("equivalent to losing ~k random junctions — targeted vs random"), plus `st.progress(ri, text=f"Network efficiency retained: {ri:.0%}")`. **Effort:** S

#### [x] [P2] "Travel impact" line chart plots exactly two points — a line through (0,0) and (1,Δ)
- **Where:** `app.py:472–485`
- **Fix:** Delete it (the metric card already carries the number); if a visual is wanted, a baseline-vs-rerouted horizontal bar pair is a real comparison. Keep the "Top delay contributors" bar (`app.py:487–493`) — that one is good. **Effort:** S

#### [x] [P2] Top critical junctions table: not linked to the map, only 5 rows, scores stringified
- **Where:** `app.py:699–703`
- **What:** Static 5-row `st.dataframe`; clicking a row does nothing; `betweenness` pre-formatted to strings (`app.py:701`) so sorting is lexicographic; no "show all 36" affordance.
- **Fix:** Streamlit 1.38 row selection: `st.dataframe(df, on_select="rerun", selection_mode="single-row")` → set `selected_node` + recenter (with the viewport fix). Keep values numeric; `st.column_config.ProgressColumn` for in-table score bars; all critical nodes in a `height=240` frame. **Effort:** M

#### [x] [P2] Compound disasters (UserJourney Flow B) are impossible outside Flood-draw
- **Where:** `app.py:649–656` — Simulate always *replaces* `disabled_nodes` with a single node, though `simulate_ablation` already supports multi-node ablation (the flood path uses it)
- **Fix:** Accumulate closures (`set(current) | {selected}`), relabel button "Add closure" once one exists, list active closures as removable chips (row of small ✕ buttons). **Effort:** M

#### [x] [P2] Flood-mode residue: leaving the Flood scenario keeps the flooded ablation with contradictory messaging
- **Where:** `app.py:1065–1079`; messaging `app.py:460–461`
- **What:** Draw a flood polygon, switch scenario to "Road closure": polygons vanish but `disabled_nodes` keeps the flood set — the panel shows "Area-based flood active…" under a selector reading "Road closure".
- **Fix:** `on_change` on the scenario selectbox clears area-based ablations (track an `ablation_source` flag). **Effort:** S

#### [x] [P2] Map-click selection: double rerun flicker and a stale-click dead zone
- **Where:** `app.py:1082–1093`
- **What:** (1) A map click triggers a component rerun, then the code calls `st.rerun()` again — two paint cycles plus the P0 viewport reset. (2) The `last_map_click` signature guard means click node A → pick node B in dropdown → click node A again → ignored.
- **Fix:** Combine with the stable-key fix; clear `last_map_click` in the selectbox's `on_change`. **Effort:** S

#### [x] [P3] `st.success` for a destructive event; disconnection message under-dramatized
- **Where:** `app.py:667–673`
- **Fix:** `st.warning("Junction 142 disabled — simulating failure.")`; when `largest_cc_fraction < 0.99`, escalate to `st.error(f"Network split: {1-frac:.0%} of junctions isolated.")`. **Effort:** S

#### [x] [P3] Exports computed eagerly on every rerun; PNG palette diverges from the design system
- **Where:** `app.py:710` (full 864-feature `.to_json()` per rerun, uncached); `app.py:507–573` (PNG uses the *old* `#1E1E2E`/`#2D2D3D` palette, not the shipped `#0B1220`/`#121C30` tokens, nor Fira Sans)
- **Fix:** `@st.cache_data` keyed on `disabled_nodes`; source PNG colors from the shared token dict (below). **Effort:** S

### 2E. Design system, accessibility, consistency

#### [x] [P2] Design.md drift: font, palette, and the mandated colorblind-safe ramp all diverge undocumented
- **Where:** Design.md §1–2 (Inter; `#121212`/`#1E1E2E`; "criticality ramp is Viridis/cividis — colourblind-safe") vs `app.py:741` (Fira Sans), `config.toml` (`#0B1220`), `app.py:302–307` (custom teal→sky→green→yellow ramp)
- **What:** The shipped ramp was changed for dark-tile legibility but never verified colorblind-safe — sky-blue→green is weak under tritanopia, mid-ramp compresses under deuteranopia — and Design.md still promises Viridis. (Semantic colors *are* Okabe-Ito — good.)
- **Fix:** Either cividis clipped to its upper 70% (`cm.cividis(np.linspace(0.3, 1.0, 4))`) — dark-tile legible *and* CVD-safe — or verify the current ramp in a CVD simulator and update Design.md to record the deviation. **Effort:** S

#### [x] [P2] Color tokens defined once in CSS but duplicated as hardcoded hex across 6+ Python sites
- **Where:** CSS `:root` `app.py:743–754` vs literals in `semantic_legend` (252–264), `build_map` (302–307, 349–371, 394–400), `render_charts` (484, 493), curve chart (695), `generate_summary_png` (516–565)
- **What:** Drift has already happened (the PNG palette). Changing the accent means a dozen edits.
- **Fix:** Module-level `TOKENS = {...}` dict consumed by the CSS f-string, legend HTML, map builder, charts, and PNG. Pure refactor. **Effort:** S

#### [x] [P2] Accessibility gaps: mouse-only map, tooltip-only data, tiny uppercase section labels
- **Where:** Leaflet iframe (no keyboard path to junctions); per-road data only in hover tooltips (`app.py:370, 421`); section headers `.82rem` uppercase (`app.py:770–779`); `.rr-sub` `.8rem` (`app.py:790`)
- **What:** Keyboard/screen-reader users cannot select a junction on the map — the dropdown (`app.py:638–645`) is the saving grace and must stay first-class. Road-level info is unreachable by keyboard and gone on touch. Positives to keep: `:focus-visible` outline, `prefers-reduced-motion`, text labels beside every color, Okabe-Ito semantics.
- **Fix:** Render the selected junction's full details as panel text when selected; bump section headers to `.9rem`, `.rr-sub` to `.85rem`. The Leaflet iframe can't be made keyboard-navigable from Python — dropdown + linked table is the correct workaround; note it in a Methodology accessibility statement. **Effort:** S

#### [x] [P3] Dead code / hygiene in the shipped file
- **Where:** `app.py:17–19` (`matplotlib.pyplot` imported, never used — drags import time on the CPU ARM box), `app.py:20` vs `app.py:942` (duplicate `import io`), `app.py:944–945` (per-call numpy/PIL imports), `app.py:987` (fragile `with st.spinner(...) if disabled_nodes else st.empty():`)
- **Fix:** Drop `plt`, dedupe `io`, hoist imports, replace 987 with a plain if/else. **Effort:** S

#### [x] [P3] Page chrome: no footer, no menu metadata, emoji favicon
- **Where:** `app.py:963`
- **Fix:** `menu_items={"About": ...}` (removes Streamlit's default "Report a bug" link), one-line footer caption with version + data vintage, SVG/PNG favicon. **Effort:** S

#### [x] [P3] `st.cache_resource` graph keyed on nothing — a footgun for the upload-to-graph future
- **Where:** `app.py:117–123` (`_features` underscore-skipped → constant cache key), `app.py:236` (hardcoded `"panaji_demo_v1"` fingerprint)
- **What:** Correct today; the moment uploaded imagery produces a second network, the cache will silently serve Panaji. Also `cache_resource` returns a shared mutable `nx.Graph` — `representative_reroute` copies before mutating (good), but future code that forgets corrupts the cache for all sessions.
- **Fix:** Thread a real fingerprint (hash of source path/bytes) as the first arg of both cached functions now. **Effort:** S

### 2F. UI quick wins (top 10 S-effort, ranked by impact)

1. **Cache the Modal inference call** (`@st.cache_data` on image bytes) — kills the re-inference P0 (`app.py:890`).
2. **`delta_color="normal"`** on the Resilience Index metric (`app.py:619`).
3. **"Run demo scenario" primary CTA** that ablates the #1 chokepoint — one-click wow for judges.
4. **Replace FastMarkerCluster with plain CircleMarkers** in a named FeatureGroup (`app.py:388–425`).
5. **Move the brand header above the columns** to a full-width top bar (`app.py:590`).
6. **Cut the scenario select to the two real modes** (single junction / flood draw) (`app.py:636`).
7. **`st.status` with staged messages + persistent resolution hint** for the GPU call (`app.py:926–935`).
8. **`st.toast` on unresolved map clicks** (`app.py:1087`).
9. **Delete the two-point "Travel impact" chart; add `st.progress(ri)`** under the RI metric (`app.py:472–485, 615`).
10. **Remove the disabled Region selectbox; `st.success` → `st.warning`/`st.error`** for failure states (`app.py:635, 669`).

## 3. ML pipeline findings (P1)

**Reviewer verdict:** the data-centric discipline (A11→A23 pivot, honest negative-result logging, promotion gates) is genuinely strong and should be preserved. The gaps are eval rigor (small-n comparisons with no confidence intervals; threshold tuned on the report set), stale defaults three releases behind, and one structural bet — segmentation as the representation — that has never been challenged despite the scoped, unblocked A18 graph-first spike. Full test suite ran green during review (173 passed).

#### [x] [P0] Defaults/docstrings hardcode the superseded v1 checkpoint — three releases stale
- **Where:** `p1_segment/evaluate.py:26` (`DEFAULT_CKPT = "models/deepglobe_mit_b3_scse_512px_best.pt"` + error text pointing at release `a4-roadseg-v1`); `run_pipeline.py:18,125` (docstring/help say v1); `model.py:3–7` (says "MiT-B0 encoder"; deployed is MiT-B3)
- **What:** Deployed model is v3.2 (`road_pan.pt`, thr 0.52). A teammate following `--help` runs the wrong checkpoint or hits a v1-specific error pointing at the wrong release tag.
- **Fix:** Single source of truth — a `DEPLOYED_CHECKPOINT` constant or `models/CURRENT` pointer file that every script/doc references; update the stale docstrings. **Effort:** S

#### [x] [P1] Uniform random crop wastes signal on a severely imbalanced task
- **Where:** `dataset.py:59` — `A.RandomCrop(size, size)` samples uniformly; road fraction is ~5–8% (lower rural)
- **What:** Tracker describes the recipe as "road-aware crops," but that's only true(-ish) in dead `train_combined.py`; the live `finetune.py` path is plain uniform — many crops carry near-zero road signal. A cheap untried lever (unlike the rejected occlusion/clDice/Massachusetts avenues), directly relevant to the "mediocre on rural Kansbahal" A34 finding.
- **Fix:** Foreground-biased cropping (60–80% of crops centered on a sampled road pixel + jitter, or reject-and-resample empty crops); ablate on held-out Indian IoU/APLS. **Effort:** S

#### [x] [P1] No confidence intervals / significance anywhere in the promotion pipeline
- **Where:** `eval_spacenet.py`, `apls_eval.py` — single point estimates (IoU n=449 tiles, APLS n=50–80; NaN tiles silently skipped, shrinking effective n unreported)
- **What:** Promotion decisions ride on gaps of the same order as plausible sampling noise (A17's "v2 does NOT beat v1" was decided on 0.004 IoU; A24's win is +2–3%). The same trap that made the A12 OSM-agreement metric misleading, now on positive results.
- **Fix:** Paired bootstrap CI (resample tiles, 95% CI on the *delta* between checkpoints); promote only if the CI excludes 0. Cheap — per-tile scores already exist. **Effort:** M

#### [x] [P1] Deploy threshold tuned on the same held-out set used for the reported number
- **Where:** `eval_spacenet.py:133–194` (`threshold_sweep` on the frozen A17 held-out split that also decides promotions)
- **What:** Classic "tuning on test data" — soft (one scalar), but combined with small n, the reported win margin is optimistically biased.
- **Fix:** Split the 127 held-out chips into threshold-selection and report-only subsets, or fix the threshold from training-time validation and never sweep on the held-out set. **Effort:** M

#### [x] [P1] `_read_val_pair` lru_cache is unbounded and never invalidates on data change
- **Where:** `finetune.py:115–124` — `@lru_cache(maxsize=None)` keyed on path strings
- **What:** A long-lived process re-running `finetune()` against regenerated tiles at the same paths silently trains/evals on stale arrays; also unbounded memory growth. Low risk today (fresh process per run), but A19's resumable-training direction pushes toward longer-lived loops.
- **Fix:** Key on `(path, mtime)` or `cache_clear()` at the start of each `finetune()`. **Effort:** S

#### [x] [P2] The default inference path still has no overlap/blending — the seam-artifact fix exists but is opt-in
- **Where:** `model.py:288–313` (`predict_large`, non-overlapping binary stitch) vs `predict_large_prob` (Hann-blended, built specifically to fix tile-seam APLS breaks); `--blend` opt-in in `predict.py`, not exposed at all in `run_pipeline.segment()` (`run_pipeline.py:52`)
- **What:** The no-flags end-to-end run — the deployment common case — still has the seam-break problem that was already fixed.
- **Fix:** Make Hann-blended inference the default (benchmark CPU latency first); `--no-blend` as opt-out. **Effort:** S

#### [x] [P2] No golden regression test on the promotion harness itself
- **Where:** `tests/` — 173 tests cover pieces, none run `eval_spacenet.evaluate_checkpoints` end-to-end against a tiny committed fixture with a known IoU band
- **What:** A regression *in the eval harness* (split off-by-one, silent threshold-default change) would be caught only by a human eyeballing a number — the exact class of issue the A25 Codex audit found after the fact.
- **Fix:** 5–10-tile golden corpus + assert reported IoU falls in a hand-checked band. No GPU needed. **Effort:** M

#### [x] [P2] Grayscale proxy ≠ real Cartosat PAN — the "-9% gap" is unvalidated against the real sensor
- **Where:** `dataset.py:67–70` (`A.ToGray` + `RandomGamma`), grayscale modes in `eval_spacenet.py`/`apls_eval.py`
- **What:** Real PAN has a different spectral response, noise, and MTF at 0.25 m native vs RGB-luma conversion at 0.5 m. Well-reasoned proxy, but the deployment-day gap is unmeasured and could be larger or differently shaped.
- **Fix:** The moment any real PAN chips are available, re-run the grayscale eval logic on them (even unlabeled sanity checks: road-fraction, visual overlays) before any go/no-go; state the proxy limitation in `Evaluation.md`. **Effort:** S (docs) / M (real PAN)
- **Status:** Docs half done (proxy limitation stated); real-PAN re-run is **data-blocked** (needs Cartosat chips).

#### [ ] [P2] Topology-loss hypothesis is open-with-a-confound, and the loss suite has no cheaper alternatives
- **Where:** `losses.py:37–46, 140–178` — full-resolution differentiable skeletonization every step when `cldice_weight > 0` (known 8 GB OOM risk); no boundary/distance-transform loss, no deep supervision
- **What:** A9 rejected clDice-first *via fine-tune* while also zeroing Lovász — its own postmortem says a clean test needs a from-scratch retrain. No cheaper topology proxy has been tried.
- **Fix:** If retested: half-resolution `soft_skeletonize` (scale-tolerant for thin structures) + keep Lovász nonzero; or try an SDT-weighted BCE as a cheap boundary proxy. **Effort:** M (only if retested)
- **Status:** **GPU-blocked** — needs a from-scratch retrain experiment, not a code change.

#### [ ] [P2] Architecture never benchmarked against a topology-first alternative, though APLS is the metric that matters
- **Where:** `model.py:34–64` (`unet`/`manet`/`fpn` over smp encoders); Tracker A14/A15 ⏳ unstarted, A18 🔒 (now unblocked — A16 is done)
- **What:** Four releases deep (v1→v3.2), all APLS gains came from data, none from architecture. SAM-Road++ (frozen SAM ViT-B + topology decoder, reported SpaceNet APLS ≈ 73) fits every locked constraint and sits scoped in the project's own backlog.
- **Fix:** Time-boxed 1–2-day spike of A18 reusing the A16 Indian corpus + the existing A17 APLS harness — just one comparable held-out APLS number to justify (or kill) the pivot discussion. **Effort:** L (see §9)
- **Status:** **GPU-blocked** — the A18 SAM-Road++ spike needs GPU training/eval; A16 corpus + A17 APLS harness are ready.

#### [x] [P3] Implementation hygiene (all S)
- **Inconsistent postprocess flag surfaces:** `predict.py` exposes `--min-component-size`/`--pp-close-radius` but not `--fill-holes`/`--open-radius`; `run_pipeline.py` exposes `--fill-holes` but not open-radius (`postprocess.py:121–137`, `predict.py:36–41`). Fix: shared `add_postprocess_args(parser)` helper.
- **`predict.py:main()` and `run_pipeline.segment()` duplicate ~15 lines** of load→predict→postprocess→save→manifest — and have already drifted (`--blend`/`--tta` exist only in `predict.py`, so the A5 one-command pipeline can't use them). Fix: factor into one shared inference function.
- **Dead-experiment files** (`train_combined.py`, `train_selftrain.py`, `build_massachusetts_data.py`) are excellently labeled but still on the import graph (`train_selftrain` imports `ModelEMA` from `train_combined`) and in every pytest run. Fix (optional hygiene pass): move to an `experiments/` subpackage so `finetune.py` is unambiguously *the* training entry point.

## 4. Graph generation & analysis findings (P2/P3)

#### [x] [P0] `sknw.build_sknw(..., multi=False)` silently drops parallel edges between the same junction pair
- **Where:** `p2_graph/skeleton_graph.py:84`; same keep-one-drop-one pattern at `simplify.py:59` (`collapse_degree2_nodes`) and `simplify.py:186–189` (`consolidate_nearby_nodes`)
- **What:** Building a simple `nx.Graph` means two distinct skeleton branches connecting the same junction pair — a loop/roundabout, a divided carriageway that splits and rejoins — silently lose one physically real segment. Every later stage that checks `graph.has_edge(...)` before adding repeats the loss; "which edge wins" is first-processed, not shortest/most-plausible.
- **Why:** A topology-correctness bug in the exact structure the resilience metric depends on — redundant alternate paths *are* resilience, and this deletes them before criticality/resilience ever run, biasing the headline metric with no visible error.
- **Fix:** Migrate to `build_sknw(skel, multi=True)` + `nx.MultiGraph` through P2/P3 (NetworkX betweenness/efficiency/APLS support MultiGraph). Interim: count + log parallel-edge collisions at each collapse site so the loss is visible; keep the shorter edge. **Effort:** L (MultiGraph) / S (logging)
- **Status:** Fixed A37 — MultiGraph end-to-end (P2 build/simplify/heal/IO, criticality, APLS, dashboard); node-ablation RI provably unchanged (min-weight routing; node removal kills all parallel siblings) — the win is representational fidelity + edge-failure readiness.

#### [x] [P0] Zero-length edges guarded only at one call site, not as a graph-wide invariant
- **Where:** `skeleton_graph.py:105–107` (0.0 fallback), pruned only in `build_graph.py:76`; consumed unguarded by `criticality.py:37` and `resilience.py:69–72` (`dist > 0`)
- **What:** `flood.py`/`percolation.py`/`evaluate.py` load graphs directly and never re-run the prune; a zero-length edge makes `dist > 0` treat coincident nodes as *unreachable* (contributes 0 to global efficiency) instead of maximally close — a silent under-count of the locked headline metric.
- **Fix:** Shared validator in `graph_io.load_geojson_graph`/`load_graphml` asserting `length_m > 0` per edge, called from every P3 entry point. **Effort:** S

#### [x] [P1] Betweenness/global-efficiency exact by default in the interactive path; cache fully invalidates per ablation
- **Where:** `criticality.py:22–39`, `resilience.py:32–73`, called with `k=None` from `analyze.py:59, 69–70`; `BetweennessCache` (`criticality.py:138–162`) keys on a whole-graph fingerprint
- **What:** Exact betweenness is O(V·E), exact all-pairs weighted efficiency O(V·(E+V log V)); every dashboard ablation click on a city-scale graph recomputes both from scratch. The k-sample tooling exists in-repo (benchmarked 16.6× speedup, Spearman 0.98 on a 2,500-node grid) but is not the default anywhere in the live path.
- **Fix:** Auto-select `k` above a node-count threshold (e.g. `k=150` if n>1000) in the interactive path; extend `BetweennessCache` to recognize "baseline minus a few nodes" for partial reuse. **Effort:** M

#### [x] [P1] Skeletonization discards road width — `medial_axis` would give it for free
- **Where:** `skeleton_graph.py:37–44` (plain `skimage.morphology.skeletonize`); zero references to `medial_axis`/`distance_transform`/`width` anywhere in `p2_graph/`
- **What:** `medial_axis(mask, return_distance=True)` yields the skeleton *and* per-pixel half-width at essentially the same cost. Without it, `min_stub_len_m`/`consolidate_tol_m` are fixed constants that can't be road-class-aware, and width-aware routing/capacity work is blocked.
- **Fix:** Swap to `medial_axis`, sample the distance array along each edge polyline → new `width_m` edge attribute. Purely additive. **Effort:** M

#### [ ] [P1] Gap healing has no obstacle/probability awareness — real false-bridge risk between parallel roads
- **Where:** `healing.py:129–185` (`find_candidate_bridges`); `build_graph.py:71` (`_load_mask` reads only the binary PNG — the model's probability map is discarded at the P1→P2 boundary)
- **What:** Healing's only signals are endpoint distance (`gap_max_m=40`) and turn angle (`angle_max_deg=60`). A frontage road parallel to a highway, both broken by the same tree canopy, can bridge the *wrong* component pair. No check that the bridge corridor passes through road-probable terrain, or that it doesn't cross an existing edge.
- **Why:** A false bridge fabricates an alternate route, directly inflating measured resilience.
- **Fix:** (1) Sample the mask (ideally the probability map) along the candidate corridor, require minimum coverage; (2) reject candidates whose segment crosses another component's edge geometry. Plumbing the probability map through the contract is the bigger fix (see §9). **Effort:** M / L
- **Status:** PARTIAL — edge-crossing rejection shipped (A36-M); probability-map corridor check is **deferred** to the P1↔P2 contract change, next pass.

#### [x] [P1] K-sampled efficiency resamples independent source sets at every ablation step — avoidable variance in the resilience curve
- **Where:** `resilience.py:59–65`, `ablation_curve` at `resilience.py:151–204` (fresh sample per removal step; the docstring at 89–93 flags the comparability issue)
- **What:** Latent today (defaults are exact), but the moment k-sampling turns on for large AOIs (recommended above), curve-to-curve noise will reflect resampling, not structure.
- **Fix:** Fix the sampled source-node IDs once from the full node set; reuse across ablation steps, dropping removed ones. **Effort:** S

#### [x] [P1] Douglas-Peucker with `preserve_topology=False` on curved Bézier bridge geometry can self-intersect
- **Where:** `simplify.py:270`
- **What:** The fast DP variant can self-intersect on sharply-bent input — and S6's cubic-Bézier healed bridges are exactly that. Cosmetic (`length_m` untouched), but visible on the *highlighted* `is_bridged` features.
- **Fix:** `preserve_topology=True` at least for `is_bridged` edges. **Effort:** S

#### [x] [P2] `rank_table` sorts purely by betweenness — articulation-point SPOFs with modest traffic rank far down the CSV
- **Where:** `criticality.py:42–70` vs `73–98` (computed independently), `rank_table` at `101–123`
- **What:** "High traffic" and "hard failure point" are distinct concepts (the docstrings say so); the CSV's primary `rank` conflates them. The dashboard does surface `is_articulation` as a layer, so end-to-end it's workable.
- **Fix:** Document that `rank` = traffic-importance; optionally add a combined score (e.g. betweenness percentile + articulation boost) for a single honest ranked list. **Effort:** S

#### [x] [P2] `consolidate_nearby_nodes` merges transitively through chains of sub-tolerance edges — cluster diameter unbounded
- **Where:** `simplify.py:146–194` (Union-Find at 160–163)
- **What:** A and C merge if chained via short edges even when `dist(A,C) > tol_m`; the docstring implies pairwise proximity. Unlikely to bite at `tol=10 m`, but nothing bounds it.
- **Fix:** Bound total cluster span, or document the transitive behavior. **Effort:** S

#### [x] [P2] `prune_short_stubs` can hit `max_iter=20` and silently under-prune
- **Where:** `simplify.py:76–93`. **Fix:** return/log whether the cap was hit. **Effort:** S

#### [x] [P2] `build_osm_truth` (APLS ground truth) has no timeout/retry/graceful failure for new AOIs
- **Where:** `apls.py:176–216` — `ox.graph_from_bbox` with no timeout or clear failure message if Overpass is down/rate-limited; "offline reproducible" only after first cache. **Fix:** try/except with a clear message. **Effort:** S

#### [x] [P2] Point-in-polygon and demand-weighting are O(V) unindexed Shapely loops
- **Where:** `flood.py:30–36`, `percolation.py:36–49`. Fine at demo scale; measurable at tens of thousands of nodes. **Fix:** `shapely` vectorized `contains` / GeoDataFrame `sjoin` / STRtree when city-scale runs happen. **Effort:** S (deferred)
- **Status:** flood.py STRtree shipped A37; the percolation.py:36–49 citation was **stale** — `spatial_demand` has no polygon loop (nothing to index).

#### [x] [P3] Silent pixel-space fallback when no georeference manifest exists (`resolution_m=1.0` default)
- **Where:** `config.py:44`, `run_real_mask.py:64`, consumed at `skeleton_graph.py:91`
- **What:** For a non-georeferenced JPEG/PNG, every metric silently assumes 1 m/px — unlikely to match the true GSD. (See also §7's stronger version of this finding.) **Fix:** loud warning + prompt to set `--resolution-m`. **Effort:** S

#### [x] [P3] No round-trip test against the actual committed `data/sample/*_graph.geojson`
- **Where:** `graph_io.py:27–163`; all tests use synthetic fixtures. Soft-migration defaults (`is_articulation` → `False` if absent) mean schema drift silently degrades rather than erroring. **Fix:** one regression test round-tripping the real committed sample (same fix as §5F's contract test — do together). **Effort:** S

## 5. Architecture & production-readiness findings

**Reviewer verdict:** the core instinct — decoupled stages over file handoffs, thin read-only dashboard, GPU off-box, scale-to-zero — is right. What's missing is the integrity layer around it: atomic writes, provenance, sanitization, CI, and observability. Highest-leverage single fix: **connect the existing 30-module test suite to CI and stop auto-deploying raw `dev`**.

### 5A. Pipeline architecture & provenance

#### [x] [P0] No AOI-id sanitization — path traversal / collision from user or CLI input
- **Where:** `src/pipeline/p2_graph/config.py` (every path interpolates `{aoi}` directly), `run_pipeline.py:57`, `predict.py:70`
- **What:** `--aoi ../../models/road_pan` writes outside the data tree; two people running `--aoi panaji` silently overwrite each other. TRD §Security promises "sanitize any user-provided paths/filenames" — unmet.
- **Fix:** `sanitize_aoi()` (regex `^[a-z0-9_-]{1,64}$`) in `GraphConfig.__post_init__` + every CLI entry. **Effort:** S

#### [x] [P0] Non-atomic artifact writes — a crashed stage leaves a half-written mask/CSV/GraphML the next stage happily consumes
- **Where:** `graph_io.py:38–39, 121`, `analyze.py:38–41`, `raster_io.py:97`, `save_binary_png`
- **What:** All writes go straight to the final contract path. Killed mid-write (routine on a free ARM box — OOM, restart-on-deploy) → truncated artifact consumed downstream. The file-handoff architecture's entire integrity guarantee currently doesn't hold across interruptions.
- **Fix:** One `atomic_write(path, write_fn)` helper (`.tmp` + `os.replace`); route all four writers through it. **Effort:** S

#### [x] [P1] Model provenance is dropped — you cannot answer "which checkpoint produced this mask/graph?"
- **Where:** `raster_io.py:86` (`write_manifest` stores only crs/transform/resolution); `run_pipeline.segment()` has `meta` in scope at line 50 but never persists it
- **What:** Schema.md defines `RoadMask.model_version`/`threshold`; nothing writes them. The graph and CSV carry no lineage to checkpoint, threshold, or commit — the answer lives in someone's shell history. Biggest gap between Schema.md and reality.
- **Fix:** Extend `write_manifest` with `{model_sha256, threshold, encoder, git_commit, created_utc}`; propagate into `graph.graph["provenance"]` (GraphML serializes graph-level dicts) and a `{aoi}_provenance.json`. **Effort:** M

#### [x] [P2] Manifest at `interim/{aoi}/manifest.json` but mask at `interim/{aoi}_mask.png` — split-brain layout
- **Where:** `config.py:53` vs `config.py:58`
- **What:** Delete/copy one and the other orphans; losing the manifest silently degrades `build_graph.py:59` to pixel-space routing (`None,None`) with no warning — a wrong-answer failure, not a crash.
- **Fix:** Co-locate under `interim/{aoi}/{mask.png, manifest.json}`; warn in `build_graph` when a mask has no manifest. **Effort:** S

#### [x] [P2] No idempotency / resume — a rerun redoes every stage; a mid-pipeline failure restarts from P1
- **Where:** `run_pipeline.run()` — linear P1→P2→P3, no existence checks, no `--force`, no stage selection
- **Why:** Forcing a full GPU re-segmentation to iterate on P3 healing params wastes the scarcest resource; bites hard during pilot parameter tuning.
- **Fix:** Per-stage skip-if-fresh (artifact mtime/provenance hash vs inputs) + `--force` + `--from-stage {p1,p2,p3}`. **Effort:** M

### 5B. Orchestration

#### [x] [P1] Zero error handling / logging in the orchestrator — failures are raw tracebacks, progress is `print()`
- **Where:** `run_pipeline.py` throughout; `build_graph.py:45` / `raster_io.py:82` raise `SystemExit`, conflating "bad input" with "bug"
- **Fix:** `logging` with stage/aoi/duration fields; per-stage try/except that re-raises with context; a `{aoi}_run.json` summary (status, timings, provenance). **Effort:** M

#### [x] [P2] Config sprawl: the same tunables duplicated across `run()`'s 17 params, `GraphConfig`, and three argparse blocks
- **What:** `resolution_m`/`tile_size`/`threshold`/healing params have independent defaults in multiple places; `run_pipeline` doesn't even expose the healing knobs (`gap_max_m`, simplify) — the end-to-end CLI can't tune what the per-stage CLI can. → "works via `build_graph`, wrong via `run_pipeline`" bugs, and no single reproducible config per run.
- **Fix:** Make `GraphConfig` (extended with P1 fields) the one config object, loadable from a small YAML/JSON per AOI; every CLI builds it identically; record the resolved config in the run summary. **Effort:** M

#### [x] [P3] No multi-tile input handling — a large GeoTIFF AOI is one monolithic in-RAM array
- **Where:** `run_pipeline.segment()` reads the whole image before `predict_large`
- **What:** 100 km² @ 0.5 m/px ≈ 200k×200k px → OOM in the *reader*, before segmentation. First thing that breaks at pilot scale.
- **Fix:** Pre-tiling stage (windowed rasterio reads → per-tile masks → mosaic). **Effort:** L

### 5C. Deployment (single ARM box + Streamlit)

#### [x] [P0] Disk/RAM exhaustion is unbounded — no upload cap, no log rotation, no disk alarm
- **Where:** `app.py:912` (uploads); `.streamlit/config.toml` (no `maxUploadSize` → default 200 MB straight into RAM and the Modal request body); journald with no size cap; ~47 GB Always-Free disk
- **Why:** A full disk takes down the whole box including Caddy's cert renewal — a silent, hard-to-diagnose outage of the public demo.
- **Fix:** `[server] maxUploadSize = 20`; journald `SystemMaxUse=200M` drop-in; artifact-cleanup timer if the pipeline ever runs on-box. **Effort:** S
- **Status:** Repo side done (maxUploadSize, journald conf file committed); installing the drop-in on the box is on the operator checklist.

#### [x] [P1] Upload concurrency: N simultaneous uploads = N cold Modal containers + N blocked 240 s session threads
- **Where:** `app.py:934–939` (synchronous `_call_modal_seg`, 240 s timeout)
- **Fix:** Module-level `threading.BoundedSemaphore` with a "busy, retry shortly" message when saturated; timeout ~90 s with a cold-start message. (Complements §6's Caddy rate limit and the Modal-side caps.) **Effort:** M
- **Status:** Modal-client semaphore (A36-M) + in-process CPU analysis semaphore & 16 MP ceiling (A37); full filesystem queue deferred (see §5H).

#### [x] [P2] No healthcheck; the 2-minute auto-update timer deploys raw `dev` with no smoke test and no rollback
- **Where:** `deploy/update.sh` (`git pull --ff-only` → pip → restart), `roadresilience-update.timer`
- **What:** `dev` is the integration branch that moves constantly; any in-progress commit black-holes the public demo within 2 minutes. Only signal is `Restart=on-failure` looping.
- **Fix:** Deploy from a tag/`release` ref; after restart, `curl -fsS localhost:8501/_stcore/health` (Streamlit's built-in health path) and `git reset --hard @{1}` + restart on failure. **Effort:** S

#### [x] [P3] `git pull --ff-only` freezes the box permanently if the checkout ever diverges
- **Where:** `update.sh:18`. Silent staleness — the demo serves old code indefinitely with no alert.
- **Fix:** `git fetch` + `git reset --hard @{u}` (the box is a read-only deploy); log/notify on update failure. **Effort:** S

### 5D. Modal GPU endpoint

#### [x] [P1] No retry, no payload cap, and HTTP 200 returned for auth/validation failures
- **Where:** `deploy/modal_app.py:79–94`, client `app.py:890–909`
- **What:** (1) Bad key / missing image → `{"error": ...}` with **200** — the docstring claims "401 on bad key" (drift); monitors and real HTTP clients are misled. (2) No size limit before decode. (3) No client retry distinguishing cold-start 5xx (retry) from 4xx (don't).
- **Fix:** Real status codes (401/400/413); reject `len(image_b64) > ~15 MB` pre-decode; one client retry with backoff. **Effort:** M

#### [x] [P1] No cost controls on the Modal function — unbounded scale-up on student credits
- **Where:** `modal_app.py:48` `@app.cls(gpu="T4", ...)` — no `max_containers`, `timeout`, `scaledown_window`
- **Fix:** `max_containers=2–3`, `timeout=120`, explicit `scaledown_window`, Modal spend alert. **Effort:** S

*(Shared-secret rotation + constant-time compare: see §6 P1.)*

### 5E. Dependencies & reproducibility

#### [x] [P1] Torch version skew across training (2.12.1+cu126), local predict, and Modal (2.4.1) — unpinned and undocumented
- **Where:** `modal_app.py:33` pins `torch==2.4.1`; root `requirements.txt` says "torch installed separately"
- **What:** One `.pt` loaded by three torch versions with no recorded known-good matrix; the `weights_only` default flip in torch 2.6 is a real breakage vector between them.
- **Fix:** Record producing-torch-version in checkpoint provenance; pin `weights_only`/`map_location` explicitly in `load_checkpoint`; document the tested matrix in `deploy/README.md`. **Effort:** S

#### [x] [P2] No checksum on the model download at Modal image build
- **Where:** `modal_app.py:23–28` (`urlretrieve` from GitHub Releases, no hash, no retry)
- **What:** A partial download or retagged release silently bakes a wrong model into the image; you can't guarantee served model == evaluated model. (Overlaps §6 supply-chain finding — same fix.)
- **Fix:** Pin expected `sha256` next to `MODEL_URL`; fail the build on mismatch. **Effort:** S

#### [ ] [P2] Three divergent dependency sets, prose-managed; geopandas a full major apart; no Python pin
- **Where:** `requirements.txt` (`geopandas==0.14.4`) vs `deploy/requirements-app.txt` (`geopandas==1.0.1`) vs `modal_app.py` pins
- **What:** geopandas 1.0 switched fiona→pyogrio as the default reader — `gpd.read_file` on the sample GeoJSON can behave subtly differently dev vs prod, on exactly the path the dashboard depends on. No `.python-version`; TRD says 3.10+ but the box uses 3.12.
- **Fix:** Pin exact Python; align geopandas majors (or test the sample-load under both); long-term a lockfile (`pip-compile`/`uv`). **Effort:** S–M
- **Status:** PARTIAL — `.python-version` (3.11) + reconciled matrix docs shipped (A37); the geopandas major flip itself is **operator** (joint dev+prod test on the box).

### 5F. Testing, CI & observability

#### [x] [P0] No CI — ~30 test modules, zero automation; nothing gates the auto-deploying `dev` branch
- **Where:** `.github/` contains only CODEOWNERS; `update.sh` deploys `dev` on every push
- **What:** The safety net exists but isn't connected: a red commit ships to production untested within 2 minutes. **Highest-leverage single fix in this review.**
- **Fix:** `.github/workflows/ci.yml`: on push/PR to `dev`/`main` → install CI-slim deps → `pytest` → `python -c "import src.app.app"` smoke; deploy only from a passing tag/release. **Effort:** S

#### [x] [P1] No contract test on the committed `data/sample/` artifacts — the deployed app's only data source is unguarded
- **Where:** `app.py:75–90` validates columns at runtime (user-facing error); `test_run_pipeline.py:54` guards the constant, not the committed files
- **What:** A P2/P3 change that regenerates the sample with a renamed column keeps tests green and breaks *production* — discovered by users.
- **Fix:** `test_sample_artifacts.py`: load the committed sample, assert the required column sets + invariants (positive `length_m`, betweenness ∈ [0,1], no dangling edges — Schema.md's own constraints). **Effort:** S

#### [x] [P1] Essentially no observability — no structured logs, no error tracking, no usage signal
- **Where:** whole app/pipeline: `print()` in the CLI; Modal failures surface only in the user's browser (`app.py:938`) then vanish
- **What:** Cannot answer "how many people used it today, how many uploads failed, with what error."
- **Fix (minimal viable, pure Python):** (1) module-level `logging` to journald/rotating file; (2) Sentry free tier wrapping `main()` + `_call_modal_seg`; (3) an append-only usage counter (upload count, sim count). **Effort:** M

### 5G. Docs drift

#### [x] [P2] TRD/Schema promise things the code doesn't do
- Concrete drifts: (1) `RoadMask.model_version`/`threshold` never persisted (see 5A); (2) `GraphNode.is_disabled` never written — runtime-only concept; (3) TRD `simulate_ablation(graph, node)` says one node, code takes a tuple (multi-node flood); (4) TRD deployment table says "Streamlit Community Cloud / HF Spaces" — actual deploy is Oracle ARM + Caddy + Modal, undocumented; (5) modal_app docstring "401 on bad key" vs 200 in code; (6) Schema mentions GeoPackage — never used.
- **Fix:** One drift-reconciliation pass over TRD + Schema. **Effort:** S

#### [x] [P3] `predict.py:12–13` docstring says multiband GeoTIFF is "out of scope" — but `read_image_any` (A26) already handles PAN + multispectral
- **Fix:** Update the stale module docstring. **Effort:** S

### 5H. Scalability — what breaks first, in order

1. **100 km² AOI** → whole-raster read OOMs before segmentation (fix: windowed pre-tiling stage, **L**).
2. **Exact betweenness + exact global efficiency** are O(V·E)/all-pairs — 10k+-node city graphs make P3 untenable. The `k`/`efficiency_k` sampling params exist but default to exact and aren't plumbed through `run_pipeline` (fix: threshold-triggered sampling defaults, **S–M**).
3. **10 concurrent users** → blocking Modal calls + per-session reruns (fixes above; `st.cache_resource` sharing looks correct today).
4. **Concurrency back-pressure without breaking the no-API/no-DB locks:** a filesystem job queue — uploads drop a request file in `data/queue/`, one worker processes serially, the app polls for the result artifact. **Effort:** M–L
- **Status:** 1 [x] (windowed inference, A36-L) · 2 [x] (auto-k, A36-M) · 3 [x] (semaphores) · 4 PARTIAL — semaphore back-pressure shipped (A37); full filesystem queue **deferred** until multi-user pilot scale.

## 6. Security findings

**Reviewer verdict:** no critical, actively-exploitable RCE or data-exfiltration path. The two big gaps are **infrastructure exposure**, not application code: a second plaintext, unauthenticated path to the app (port 8501) alongside the Caddy front door, and real low-effort **cost-abuse exposure on the personally-billed Modal GPU endpoint**. Secrets hygiene is genuinely good (`.env` gitignored, no hardcoded keys/IPs in tracked files, Modal Secret used correctly).

#### [x] [P0] Streamlit port 8501 exposed directly to the internet, bypassing Caddy/TLS entirely
- **Where:** `deploy/README.md:36–45` (iptables `--dport 8501 -j ACCEPT` + Oracle Security List ingress from `0.0.0.0/0`); `deploy/roadresilience.service:10` (`--server.address 0.0.0.0`); `deploy/Caddyfile:6–8`
- **What:** The app is reachable two ways: `https://trace.tiwaribabu.in` (TLS via Caddy) and `http://<public-ip>:8501` (plaintext, Streamlit's own dev-grade server directly internet-facing). Any Caddy-layer mitigation (rate limits, body-size caps) never sees the second path; the stray port also gets found by Shodan/Censys scans.
- **Fix:** Bind Streamlit to `127.0.0.1`, remove the 8501 iptables rule and Security-List ingress; Caddy becomes the sole entrypoint. **Effort:** S

#### [x] [P0] Modal endpoint has no per-request size cap or rate limit — cost abuse / DoS via oversized or repeated payloads
- **Where:** `deploy/modal_app.py:79–94` (`segment` endpoint), `src/app/app.py:890–909`
- **What:** The endpoint accepts any JSON body, base64-decodes `image_b64` with no size bound, and runs a full sliding-window GPU pass on whatever decodes. **The key check happens after the body is received and parsed**, so even bad-key requests cost decode work. The `T4`-backed class sets no `max_containers`/`concurrency_limit`, so Modal spins up (and bills) more containers under concurrent load.
- **Attack scenario:** Scripted POSTs with large payloads burn GPU credits billed to a personal Modal account — a "wake up to a large bill" risk. Compounded by the dashboard's own re-inference-on-rerun bug (§2B P0).
- **Fix:** Check `key` first (cheap short-circuit); reject bodies over ~10 MB before decoding; set `max_containers`/`concurrency_limit` + an aggressive `scaledown_window`; add a Modal spend alert/budget cap as backstop. **Effort:** S–M

#### [x] [P1] Shared-secret compared with `!=` (non-constant-time), never rotated/expires
- **Where:** `deploy/modal_app.py:85` — `if item.get("key") != os.environ.get("ROADSEG_KEY"):`
- **What:** Timing side-channel (weak over the internet, but zero-cost to fix) on the *only* access control of a cost-incurring public endpoint; no rotation procedure documented anywhere.
- **Fix:** `hmac.compare_digest(...)`; document rotation in `deploy/README.md`; ensure the env-file the systemd unit sources is `chmod 600`. **Effort:** S

#### [x] [P1] No upload size/type guard before PIL decode — decompression-bomb and CPU/GPU amplification
- **Where:** `src/app/app.py:919–958` (extension filter only); `deploy/modal_app.py:69`; `.streamlit/config.toml` has no `maxUploadSize` override (default 200 MB)
- **What:** Bytes go straight to `Image.open` on both the ARM box and the Modal container; neither sets `Image.MAX_IMAGE_PIXELS` or checks dimensions. A 200 MB decompression-bomb PNG is a legal upload: gigabytes of RAM on the Always-Free box (single-request OOM of the only app process), and an enormous tile grid on the GPU.
- **Fix:** Set `PIL.Image.MAX_IMAGE_PIXELS` (e.g. 4096²) and catch `DecompressionBombError` on both sides; reject oversized images before the Modal round-trip; `maxUploadSize = 15–20` in config.toml. **Effort:** S

#### [ ] [P1] No concurrency/queueing guard on the ARM box — a few simultaneous uploads starve the single dashboard process
- **Where:** `deploy/roadresilience.service` (one `streamlit run`, no limits); `app.py:934–958` (blocking `urlopen(..., timeout=240)` per session); `deploy/Caddyfile` (no `request_body max_size` or rate limit)
- **What:** Each concurrent visitor's upload blocks a Streamlit thread for up to 240 s and holds a decoded image in RAM — a handful of tabs makes the dashboard unresponsive for everyone.
- **Fix:** Caddy `request_body { max_size 15MB }` + basic rate limiting; drop the client timeout to ~60–120 s. **Effort:** S–M
- **Status:** PARTIAL — Caddy body cap + client timeout shipped; the rate-limit module needs a third-party Caddy plugin install (**operator**).

#### [x] [P2] `torch.load(..., weights_only=False)` for all checkpoint loading, including the Modal-baked model
- **Where:** `src/pipeline/p1_segment/model.py:98,114`; `evaluate.py:68`; `deploy/modal_app.py:23–28` (`urlretrieve` from GitHub Releases at image build)
- **What:** Full-pickle deserialization = RCE if the release asset is ever swapped (GitHub account-takeover blast radius). Current reachable risk is low (fixed HTTPS URL, own account, no user-supplied checkpoints).
- **Fix:** Pin the release asset by SHA-256 in `modal_app.py` (verify after download, fail closed). Longer-term: make checkpoint `meta` JSON-serializable so `weights_only=True` works. **Effort:** S (checksum) / M (migration)
- **Status:** sha256 pin (A36) + weights_only-first loading with legacy fallback (A37); deployed v3.2 verified on the safe path.

#### [x] [P2] Verbose exception surfaced to end users on inference failure
- **Where:** `app.py:936–939` (`st.error(f"Inference failed: {error}")`, bare `Exception`); `modal_app.py` `segment` doesn't catch malformed-base64 errors → Modal 500 traceback body flows into the user-visible string
- **Attack scenario:** Malformed input elicits stack traces revealing paths/versions — free reconnaissance on an unauthenticated endpoint.
- **Fix:** Generic `{"error": "invalid image"}` from the endpoint; generic user message + full exception to server logs on the dashboard side. **Effort:** S

#### [x] [P2] Pillow is untracked in both requirements files despite decoding untrusted images
- **Where:** `requirements.txt`, `deploy/requirements-app.txt` (no Pillow pin — floats transitively); `deploy/modal_app.py:38` pins `pillow==10.4.0` only inside the Modal image
- **What:** The library with the longest history of image-decode CVEs is version-untracked exactly on the paths that decode attacker-controlled bytes; the three environments can silently diverge.
- **Fix:** Explicit `Pillow==` pin in both files matching the Modal image; run `pip-audit` once per dependency-touching task and log it in Tracker §10. **Effort:** S

#### [P3] Notes for the record (no action today)
- `nx.read_graphml` (`p2_graph/graph_io.py:39,46`): stock `xml.etree` doesn't resolve external entities — not XXE-exploitable today, and no user bytes reach it. Re-check (use `defusedxml`) before ever accepting user-supplied GraphML.
- `st.cache_resource`/`simulate_ablation` fingerprints hardcoded to `"panaji_demo_v1"` (`app.py:989, 117–148`): safe single-dataset today; derive keys from real AOI/file hashes when multi-AOI lands (duplicates §2E finding — cross-referenced).

## 7. Silent failures & error handling

**Reviewer verdict:** the pipeline's failure philosophy is "print and continue" — several broken-input paths produce *plausible-looking wrong numbers* instead of errors. The worst compound case: an empty mask flows through P2→P3 into an RI of 0.0 that renders identically to a legitimate "network destroyed" result.

#### [x] [P0] `analyze()` never checks the P2 graph is non-empty — a failed segmentation yields "no critical nodes," not an error
- **Where:** `p3_analysis/analyze.py:57–107`; `run_pipeline.py:101–109`
- **Failure scenario:** Cloudy tile / wrong threshold → ~0% road mask → 0–2-node graph → betweenness returns `{}` without raising → empty criticality CSV written (`analyze.py:36`) → `verify_dashboard_ready` computes `columns_match=False` but `run()` only *prints* it and exits 0.
- **Fix:** Raise in `analyze()` if `graph.number_of_nodes() < 2` ("upstream P1/P2 failure"); in `run()`, raise `RuntimeError` if the contract check fails instead of printing. **Effort:** S

#### [x] [P0] `resilience_index` returns 0.0 silently when baseline efficiency is 0 — indistinguishable from "network destroyed"
- **Where:** `resilience.py:105` (`ri = (eff / base) if base > 0 else 0.0`), same at `resilience.py:200`
- **Failure scenario:** Degenerate baseline graph → every simulated scenario shows RI 0.000 + `travel_time_delta_pct=inf` — a fabricated "zero redundancy" conclusion when the real problem is garbage upstream data.
- **Fix:** Return `None`/`baseline_invalid=True` sentinel when `base <= 0`; callers (`app.py`, `analyze.py`) surface it as an explicit error. **Effort:** S

#### [x] [P0] Modal response body never validated before decode (error-propagation angle of §2B's finding)
- **Where:** `app.py:890–909`, `947–948` — no `resp.status` check, no separate `JSONDecodeError` handling; `Image.open` on returned bytes sits outside the try
- **Failure scenario:** Truncated/proxied 200 response with corrupt base64 → `PIL.UnidentifiedImageError` uncaught → raw traceback on the public dashboard.
- **Fix:** Status check + decode inside the try, distinct messages for "unreachable" vs "invalid response". **Effort:** S

#### [x] [P1] `load_checkpoint` silently defaults to `mit_b0`/`unet` when checkpoint `meta` is missing or partial
- **Where:** `p1_segment/model.py:114–125` — `meta.get("encoder", "mit_b0")`, `meta.get("arch", "unet")`
- **Failure scenario:** A checkpoint saved without `meta` (interrupted save, hand-copied weights) builds the wrong architecture; the incidental `load_state_dict` shape error is the only protection, and a structurally-compatible mismatch would produce plausible garbage masks with no exception.
- **Fix:** `if not meta: raise ValueError(f"{path} has no meta — refusing to guess architecture")`. **Effort:** S

#### [x] [P1] Flood polygon that selects zero nodes gives no feedback — identical to "not drawn yet"
- **Where:** `app.py:1065–1079` — empty `flooded_nodes` → `new_disabled = ()` → equality guard suppresses both rerun and message
- **Fix:** If `drawings` non-empty but `flooded_nodes` empty: `st.warning("No junctions inside the drawn area — try a larger polygon.")`. **Effort:** S

#### [x] [P1] Pipeline exit code is always 0 — CI/scripts can't detect a broken run
- **Where:** `run_pipeline.py:63–77, 107–109` — `verify_dashboard_ready` booleans printed, never enforced; `main()` has no exit-code path
- **Fix:** Fail loudly on contract violation (same fix as the first P0 — one change covers both). **Effort:** S

#### [x] [P2] APLS returns a *perfect 1.0* when every sampled pair is unreachable
- **Where:** `p3_analysis/apls.py:132` — `return sum(contribs)/len(contribs) if contribs else 1.0`; `NetworkXNoPath` pairs are `continue`d past (118–119)
- **Failure scenario:** A badly under-healed, fragmented graph — the exact failure this metric exists to catch — scores 1.0 if no sampled pair is connected.
- **Fix:** Track the skip rate; if `contribs` is empty or >50% skipped, report 0.0/`None` + `insufficient_reachable_pairs=True`. **Effort:** S

#### [x] [P2] `rank_table` reads pre-annotated fields with silent `False` defaults
- **Where:** `criticality.py:101–123` — `data.get("is_critical", False)` etc.
- **Failure scenario:** A refactor that calls `rank_table` before `annotate_criticality` produces a CSV with zero critical nodes and no error.
- **Fix:** Assert `"betweenness" in data` (fail loudly on ordering bugs). **Effort:** S

#### [x] [P2] Missing georeference manifest is indistinguishable from intentional pixel-space mode
- **Where:** `p2_graph/build_graph.py:53–66`; `raster_io.py:92–93` (`write_manifest` silently no-ops when transform/crs is None)
- **Failure scenario:** A Cartosat GeoTIFF processed through a path that drops its transform → graph builds in pixel space → `length_m`-weighted resilience/criticality numbers are quantitatively wrong while looking normal. (Stronger version of §4's P3.)
- **Fix:** Explicit `WARNING: no alignment manifest — graph is in pixel space, length_m is not true metres` in `build_graph` output. **Effort:** S

#### [x] [P2] Corpus-build loop swallows per-city failures with no aggregate threshold
- **Where:** `p1_segment/build_finetune_data.py:233–235`
- **Failure scenario:** Esri endpoint changes format → 18/20 cities fail → script exits 0 with a 2-city corpus; a background launch (per team convention) misses the per-city FAILED prints entirely.
- **Fix:** `SystemExit` if `len(failed)/len(cities) > 0.5`. **Effort:** S

#### [x] [P3] Massachusetts converter silently skips unreadable files (dead-experiment tool; only if ever reused)
- **Where:** `build_massachusetts_data.py:41–44` — `cv2.imread` None → `continue`, conflated with intentional "no roads" skips. **Fix:** count `unreadable` separately. **Effort:** S

---

## 8. Prioritized roadmap (impact × effort)

> Waves 1–2 are fully shipped (A36). Wave 3: items 1 (upload loop), 4a (pre-tiling) shipped in A36-L; item 2's MultiGraph half shipped in A37; the probability-map contract half, the A18 spike (3), the job queue (4b), and the four-tab restructure (5) remain — see the Status section.

### Wave 1 — this week, mostly S-effort, kills every P0
1. **Dashboard correctness trio:** cache the Modal call on image bytes (§2B); stable map key + viewport round-trip (§2C, the one M here); `delta_color="normal"` (§2D).
2. **Lock the front door:** bind Streamlit to `127.0.0.1`, close 8501 (§6); Modal key-check-first + body-size cap + `max_containers`/`timeout` + `hmac.compare_digest` (§6/§5D); `maxUploadSize=20` + `Image.MAX_IMAGE_PIXELS` on both sides (§6); journald cap (§5C).
3. **Fail loud:** empty-graph guard in `analyze()` + RI `baseline_invalid` sentinel + non-zero pipeline exit on contract violation (§7); move the PIL decodes inside the try + human error messages (§7/§2B); `load_checkpoint` refuses missing `meta` (§7); APLS unreachable-pairs guard (§7).
4. **Integrity plumbing:** `sanitize_aoi()` + `atomic_write()` (§5A); zero-length-edge validator in `graph_io` loaders (§4).
5. **CI + deploy gate:** GitHub Actions running pytest + app-import smoke; deploy from tag with post-restart healthcheck + rollback (§5F/§5C).
6. **Freshness:** update stale v1-checkpoint defaults/docstrings via a single `DEPLOYED_CHECKPOINT` source of truth (§3); make Hann-blended inference the default (§3).
7. **UI quick wins 3–10** from §2F (demo CTA, CircleMarkers, header placement, honest scenario radio, `st.status` staged progress, click toast, RI context, remove dead controls).

### Wave 2 — next 2–3 weeks, M-effort, the trust & UX layer
1. **Provenance manifest** (model sha, threshold, commit, timestamp) written by P1 and propagated into the graph + CSV (§5A); checksum-pinned model download in the Modal image (§5E/§6).
2. **Eval rigor:** paired-bootstrap CIs on checkpoint deltas; threshold-selection/report split; golden 5–10-tile harness regression test; sample-artifact contract test (§3, §5F, §4).
3. **Map & panel:** single `folium.GeoJson` edge layer + vectorized styling + cached map HTML; truthful merged legend; selectable rankings table linked to the map; compound-closure accumulation; Methodology tab surfacing the eval JSONs (§2C/§2D).
4. **Healing defenses:** corridor mask-coverage check + edge-crossing rejection (§4); `medial_axis` width capture (§4); k-sampling defaults + fixed source sets above a node threshold (§4).
5. **Observability:** structured logging, Sentry free tier, usage counter (§5F); upload concurrency semaphore + Caddy body cap/rate limit (§5C/§6).
6. **Config consolidation:** one `GraphConfig`-centric config object across all entry points; shared inference function ending the `predict.py`/`run_pipeline` drift (§5B/§3).
7. **Foreground-biased crops ablation** — cheap, untried, directly relevant to the rural push (§3).

### Wave 3 — the L-effort bets (sequence by product value)
1. **Close the upload loop** (upload → mask → graph → resilience on the map) — converts the app from canned demo to actual product. Highest product value in this entire report (§2B, §5C).
2. **MultiGraph migration + probability-map contract** for P2 (§4, §9) — fixes the deepest correctness bias and unlocks confidence-aware everything.
3. **A18 graph-first spike** (SAM-Road++ vs v3.2 on the existing A17 APLS harness) — 1–2 days, decision-grade output (§3, §9).
4. **Pre-tiling stage** for large AOIs; filesystem job queue for concurrent uploads (§5H).
5. **Four-tab app restructure** (§9) once the pieces above exist.

## 9. Beyond the current implementation — redesign proposals

Four redesigns, one per layer. Each is independently adoptable; together they describe the best version of this product. All respect the locked constraints (Streamlit+Folium, pure Python, fine-tune-only PyTorch, global-efficiency metric) except where explicitly flagged as a deliberate, minimal relaxation.

### 9.1 Product: from "one long panel next to a map" to a four-tab narrative app
`st.tabs` (or `st.Page` after a Streamlit bump): **Briefing** — a linear judge/demo mode: full-width hero, one-sentence value prop, map with only ramp + critical dots, three big metrics, and a single amber **"Simulate the worst failure"** button that animates the state change and explains what happened; **Analysis** — the current planner tooling reorganized: stable-viewport map beside a fixed-height scrolling panel with Scenario / Rankings / Curves sub-tabs, exports in a popover; **Your imagery** — the upload flow promoted to first-class, ending in "Analyze this network" which swaps the session's dataset (needs the cache-fingerprint fix); **Methodology** — pipeline diagram, plain-language metric definitions, and the already-shipped eval JSONs rendered as tables. This maps 1:1 onto the four personas UserJourney.md already defines, and every piece is plain Streamlit.

### 9.2 ML: challenge the representation, not just the checkpoint
The one structural bet never challenged since inception is *segmentation as the intermediate representation*, even though the product's real metric is topology (APLS → routable graph → global efficiency). The staged path: (1) **first**, add paired-bootstrap CIs to the eval harness so any comparison is decision-grade (half a day, reuses existing per-tile scores); (2) **spike A18** — SAM-Road++ (frozen SAM ViT-B + topology decoder, fits "fine-tune pretrained + PyTorch") on the A16 Indian corpus, scored on the existing A17 held-out APLS harness; (3) **gate:** if it beats v3.2's APLS 0.499 with a CI excluding zero, add `p1_graph` as a second producer of the *same* §4 graph contract (P2's healing/criticality code is representation-agnostic — it consumes a NetworkX graph), letting P2 skip skeletonization on that path while the mask pipeline stays as the CPU fallback; (4) full migration only after a team decision, per the Tracker's own flag on A18. If the spike loses, it's a documented negative result in the best tradition of A8/A9/A11/A12. Meanwhile the data lever stays primary: hand-corrected rural GT (A35) + foreground-biased crops + real-PAN validation the day Cartosat chips arrive.

### 9.3 Graph: probability-aware, multi-edge extraction
Rebuild P2's front half around one idea — **stop throwing information away at the binarization boundary**. Concretely: pass the probability map alongside the binary mask in the P1→P2 contract (a §4 contract change — propose to the team first); `medial_axis` instead of `skeletonize` to capture `width_m` for free; `MultiGraph` end-to-end so parallel edges survive; healing that validates each candidate bridge against the probability corridor and rejects edge-crossings; a per-edge `confidence` attribute (mean probability along the polyline) that the dashboard renders as opacity and P3 uses for a *complementary* confidence-weighted resilience variant (global efficiency stays the headline, per the lock); STRtree-indexed spatial queries and default-on k-sampling with fixed source sets so the same code serves a 600-node demo and a 50k-node city. This single coherent design removes essentially every §4 P0/P1 rather than patching symptoms.

### 9.4 Platform: provenance-first files, gated deploys
Keep the file-handoff instinct, harden it: every artifact written atomically under `{aoi}/{run_id}/` with a manifest (`model_sha256`, threshold, commit, config hash, timestamp); a thin DAG-style runner with skip-if-fresh, `--from-stage`, structured logs, and a `run.json` summary; Modal with real HTTP semantics, payload caps, spend limits, and a `/version` endpoint; CI-gated tag deploys with healthcheck + auto-rollback; a filesystem job queue for uploads (back-pressure without a REST API or DB). The one deliberate lock relaxation, only when multi-AOI actually arrives: a **SQLite + SpatiaLite read model** — serverless, single-file, repo-shippable — so the region picker becomes real without ever running a database service. Staged: harden-in-place (days) → orchestrator + pre-tiling (1–2 wk) → queue + observability (1 wk) → multi-AOI read store (only if the pilot needs >1 city).

---

*Report compiled 2026-07-03 by six parallel specialist reviews + coordinator synthesis. Companion log entry in `docs/Tracker.md` §10. Nothing in this file has been committed as fixes — each wave above is intended to become tracked tasks on the §6 board.*
