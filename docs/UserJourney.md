# UserJourney.md

> **Purpose.** This document maps how people actually move through **Route Resilience** — every entry point, the main flows, what success looks like, and what happens when things go wrong or hit an edge case. It keeps the team focused on the user's experience, not just the algorithms. The personas are summarized from `PRD.md`.

---

## Personas (recap)

- **Meera Nair — Urban Mobility Planner.** Non-technical; wants a clickable map to show a committee what happens if a junction floods.
- **Capt. Rohan Desai — District Disaster-Management Officer.** Needs a fast, ranked list of chokepoints and evacuation-route impact during monsoon.
- **Dr. Aishwarya Rao — NRSC Scientist.** Evaluates whether the methodology and metrics are sound and whether Indian EO data is genuinely used.

(There is also an internal **Analyst/Developer** actor who runs the pipeline to *generate* the artifacts the others view.)

## Entry Points

| Entry point | Who | What they see first |
|---|---|---|
| Open the dashboard (local URL or hosted demo) | Meera, Rohan, Aishwarya | The criticality map of a default city, roads coloured by criticality, legend visible |
| Run the pipeline CLI/notebook for a new area, then open the dashboard | Analyst/Developer | Console progress, then artifacts ready to load |
| Land on the "About / Methodology" tab | Aishwarya | Plain-English explanation of the pipeline + metrics |

## Navigation Map

A deliberately simple **single-page app** with an optional second tab.

```
┌──────────────────────────────────────────────────────┐
│  Route Resilience           [ Map ] [ About/Methods ] │
├────────────────────────────────┬─────────────────────┤
│                                │  Metrics             │
│        INTERACTIVE MAP          │  Scenario selector   │
│   (criticality heatmap;         │  Top Critical Nodes  │
│    click a junction to disable) │  Reset               │
│   ◐ legend                      │                      │
└────────────────────────────────┴─────────────────────┘
```

Keeping navigation flat (no deep menus) is intentional — a planner should never feel "lost" in the tool.

## User Flows

### Flow A — Explore criticality (understand the city)
1. User lands on the map; roads/junctions are coloured by criticality.
2. User reads the legend to learn what the colours mean.
3. User identifies the red ("Gatekeeper") corridors at a glance.
4. User hovers/clicks a node → tooltip shows its name, betweenness score, and rank.
5. User scans the **Top Critical Nodes** list for the ranked chokepoints.

### Flow B — Stress-test a junction (the headline interaction)
1. User optionally picks a **scenario** (flood / accident / closure) or simply chooses a junction.
2. User **clicks a node to disable it**.
3. The app shows a brief spinner, recomputes, and updates: the **rerouted path**, the new **Resilience Index** (global efficiency), and the **travel-time increase (%)**.
4. User compares before/after and can disable **more nodes** to model a compound disaster (cumulative).
5. User hits **Reset** to return to the baseline network.

### Flow C — Capture / export (take it to a meeting or report)
1. User exports the graph + metrics (GeoJSON/CSV) or screenshots the map.
2. User takes the result into a committee deck, situation report, or evaluation.

## Success Flows (what "it worked" means per persona)

| Persona | Success looks like |
|---|---|
| Meera | "I clicked Junction X, the committee saw commute time jump +37%, and we approved the redundancy budget." |
| Rohan | "I have a ranked chokepoint list and an isolation estimate for the monsoon plan in minutes." |
| Aishwarya | "The resilience metric stayed finite when the network split, and IoU/APLS were reported honestly — the method is sound." |

## Error Flows

| Situation | What the app does |
|---|---|
| No graph/artifacts loaded | Friendly message + one-line instruction on how to generate or select a dataset (never a raw stack trace) |
| User clicks empty map area (no node) | Gentle prompt: "Click a junction (a dot) to inspect or disable it." |
| Disabling a node **disconnects** the graph | Handle gracefully — this is expected. Global efficiency stays finite (the whole reason we use it), and the app shows "Network split — X% of nodes isolated." |
| Recompute is slow on a very large graph | Show a spinner; fall back to **k-sample** betweenness; optionally analyze a sub-region |
| Map tiles fail to load (offline) | Show a fallback message; the graph still renders on a plain background |
| Corrupt/unexpected input file | Validation catches it; show "couldn't read this file" rather than crashing |

## Edge Cases

- **Disabling an already-isolated node** → little/no change; the app states that clearly.
- **Disabling the only bridge between two halves** → efficiency drops sharply; highlight this as a maximally critical link (also flag articulation points up front).
- **Selecting a node with no path to the rest** → handle without divide-by-zero.
- **Very large city graph** → performance guardrails (k-sample betweenness, precomputed centralities, optional sub-region).
- **Multiple nodes disabled (compound scenario)** → metrics accumulate; provide a clear running state and an easy reset.
- **Healed/bridged edges** → render distinctly (e.g. dashed) so users know which roads were inferred rather than observed — honesty about uncertainty.

## Exit Points

| Exit | Outcome |
|---|---|
| Close the tab | Session ends (no data persisted — read-only tool) |
| Export then leave | User leaves with a GeoJSON/CSV/screenshot artifact |
| Reset to baseline | User clears a simulation and stays in the tool |

Because the dashboard is read-only and stateless between sessions, leaving never risks data loss — there is nothing to "save."
