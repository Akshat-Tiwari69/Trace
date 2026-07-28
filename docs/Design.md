# TRACE product design

TRACE is a map-first **route resilience field atlas**. Its interface should feel like a serious field instrument: evidence-dense, calm, and legible, with one strong visual hierarchy instead of a generic dashboard grid.

## Design goals

1. Make the road network the primary workspace, not a decorative chart.
2. Let a reviewer understand the product loop in under a minute: select a junction, stress the network, inspect the loss, then recover it.
3. Keep every metric traceable to the committed P2/P3 evidence or a live API response.
4. Work fully from 375 px phones through large desktop screens without horizontal scrolling.
5. Meet WCAG 2.2 AA and remain useful when map tiles, animation, or a pointing device are unavailable.

## Visual system

The system combines an editorial field-notebook tone with precise cartographic controls.

| Token | Value | Use |
|---|---:|---|
| Ink | `#0B1413` | Primary type, navigation, selected outlines |
| Canvas | `#F4F0E6` | Main paper-like surface |
| Coral | `#F05A3C` | Failed junctions and destructive state |
| Mint | `#71C2A2` | Healthy/low-criticality network state |
| Flood blue | `#2C84A6` | Comparison and contextual information |
| Amber | `#E5B84E` | Articulation points and warnings |

- **Display face:** self-hosted Fraunces, used sparingly for place and narrative headings.
- **Interface face:** self-hosted Public Sans for controls, evidence, tables, and numbers.
- Local WOFF2 files prevent third-party font requests and layout shifts.
- Status is never color-only. Failed links are dashed, inferred links use a separate dash treatment, critical nodes use shape/stroke changes, and every layer has text in the legend/table.
- Icons are small inline SVGs. Emoji are not interface icons.

## Information architecture

The main workspace has four task modes that preserve the same map and selection state:

- **Explore** — inspect network shape, critical junctions, inferred links, and evidence.
- **Stress** — add one or more junction failures and run the CPU resilience simulation.
- **Compare** — compare baseline and scenario emphasis without duplicating the map payload.
- **Recover** — remove failed junctions in criticality order and watch the scenario recover.

On desktop, the map is framed by a control rail, an insight rail, and a bottom resilience timeline. On smaller screens those regions become a natural document flow: place and scenario controls, map, insight, ranking, then timeline. The map remains a useful visual, while every selectable junction and result is also available through semantic HTML.

The active mode, selected junction, and failed-node set are encoded in the URL. A recruiter can share or reload a scenario without losing the current story.

## Core interaction contract

1. Loading reveals a branded evidence-loading state rather than an empty shell.
2. Selecting a map node or ranking row updates the same junction insight.
3. Stressing a junction calls the API and exposes efficiency loss, component survival, and a representative reroute or disconnection.
4. Recovery removes failures in descending criticality rank; it does not pretend to be an infrastructure investment optimizer.
5. Exports are generated from current browser state as GeoJSON and CSV.
6. Imagery analysis requires an explicit external-processing confirmation, reports queue state, and returns an image-space overlay. Unreferenced uploads are never placed on a fake basemap.
7. The methodology page explains the baseline-normalized global-efficiency metric, sample provenance, inference boundary, and known limitations.

## Map behavior

MapLibre GL JS renders the committed GeoJSON once, with data-driven layers for roads, critical junctions, articulation points, inferred bridges, failures, and selection. The map module is loaded dynamically so MapLibre is not part of the initial server-rendered shell. A ranking table is the keyboard and screen-reader alternative to direct map picking.

OpenFreeMap supplies the presentation basemap without an access token. A tile/style failure produces an inline warning while the authoritative network layer remains usable. Cooperative gestures reduce accidental page trapping, navigation controls are labeled, and attribution remains visible.

The approach follows map-first and compact-control lessons from [Felt's interface work](https://felt.com/blog/ui-upgrades), the evidence hierarchy shown in [Stamen's Distressed Communities Index redesign](https://stamen.com/work/redesigning-the-distressed-communities-index/), and MapLibre's [large-data guidance](https://maplibre.org/maplibre-gl-js/docs/guides/large-data/). These are references, not visual templates.

## Accessibility acceptance criteria

- WCAG 2.2 AA contrast for text and essential graphics.
- Visible `:focus-visible` treatment for every interactive element.
- Logical heading order, landmarks, labels, live status, and error announcements.
- Arrow-key tab behavior for the four modes; normal keyboard access for every action.
- Minimum 44 × 44 px pointer targets for primary controls.
- No information available only through hover, animation, or color.
- `prefers-reduced-motion` disables non-essential transitions; map animation duration is zero.
- Layout remains usable at 200% zoom and at 375, 768, 1024, and 1440 px widths.
- Automated browser checks must report zero serious or critical axe violations; manual keyboard and screen-reader-oriented review still applies.

## Performance budgets

Budgets are release gates, not aspirations:

| Surface | Budget |
|---|---:|
| Non-map first-party JavaScript | ≤ 205 KiB gzip (measured framework floor; optimize toward 170 KiB) |
| Map JavaScript | ≤ 350 KiB gzip |
| Total first-party JavaScript | ≤ 520 KiB gzip |
| CSS | ≤ 35 KiB gzip |
| Fonts | ≤ 100 KiB total |
| Initial static shell | ≤ 650 KiB |
| Compressed sample GeoJSON | ≤ 1 MiB |
| Decoded sample GeoJSON | ≤ 5 MiB |

The field targets are p75 LCP ≤ 2.5 s, INP ≤ 200 ms, and CLS ≤ 0.1, matching the [Core Web Vitals thresholds](https://web.dev/articles/defining-core-web-vitals-thresholds). Local gates include a static bundle-budget script, production build, responsive Playwright journeys, and axe checks. Live Web Vitals and Lighthouse measurements are recorded during deployment rather than invented from local builds.

## Product boundaries

- Next.js/React owns presentation and static export; FastAPI owns validation and domain calls.
- The browser never reimplements P2/P3 metrics.
- Modal remains the sole authenticated GPU boundary; P2/P3 and simulations remain CPU-capable.
- File artifacts remain the data store. No database or login product is introduced.
- The licensed `a4-roadseg-v3.2` model remains production until another model passes both the registered metric gate and licensing review.
