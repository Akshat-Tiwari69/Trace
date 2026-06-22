# Rules.md

> **Purpose.** This document defines the working standards for everyone contributing to **Route Resilience** — human developers and AI assistants alike. The goals are a consistent, readable codebase, predictable collaboration across three people on three machines, and outputs that are reproducible and safe to make public. When in doubt, favour **simple and readable over clever**.

---

## Coding Standards

- **Language:** Python 3.10+.
- **Style:** follow PEP 8; auto-format with **black**, sort imports with **isort**, lint with **ruff** (or flake8). Don't hand-argue formatting — let the tools decide.
- **Readability first.** Prefer clear, slightly longer code over dense one-liners. A first-year teammate should be able to read any function and understand it. Clever tricks that save three lines but cost ten minutes of understanding are not worth it.
- **Small functions, one job each.** If a function does three things, split it.
- **Type hints + docstrings** on public functions — a one-line docstring saying what it does, its inputs, and its output is enough.
- **No magic numbers / hardcoded paths.** Put tunables (tile size, thresholds, gap/angle limits, file paths) in a single config file, not scattered through the code.
- **Notebooks for exploration, modules for anything reused.** Once a piece of code is needed twice, move it into a `.py` module.

## Naming Conventions

| Thing | Convention | Example |
|---|---|---|
| Files / modules | `snake_case` | `graph_healing.py` |
| Functions / variables | `snake_case` | `build_graph()` |
| Classes | `PascalCase` | `RoadGraph` |
| Constants | `UPPER_SNAKE` | `GAP_MAX_M` |
| Git branches | `type/short-desc` | `feat/mst-healing`, `fix/crs-mismatch` |
| Commits | imperative, scoped | `feat(graph): add union-find healing` |
| Data artifacts | per `Schema.md` | `graph.graphml`, `criticality.csv` |

## Architecture Rules

- **Respect the phase boundaries.** Each phase (segmentation, graph build/heal, analysis, dashboard) is a module that communicates through **file artifacts**, not by reaching into another phase's internals (see `TRD.md`). This is what lets three people work in parallel.
- **The dashboard reads precomputed artifacts.** It does not run model inference; the only thing it computes live is the cheap node-ablation simulation.
- **Keep it simple — no premature infrastructure.** No database, no auth, no microservices, no JavaScript SPA in v1 (these are explicit project constraints; see below). Add complexity only when a real need is proven.
- **One source of truth for configuration.** All paths and parameters come from one config; don't duplicate constants.

## Documentation Rules

- **Docs-first.** Major decisions are written down (in `docs/`) before or alongside the code, per the framework in `Index.md`.
- **Keep docs in sync with code.** If behaviour changes, update the relevant doc in the same change. A doc that lies is worse than no doc.
- **Update `Tracker.md`** as tasks move; record decisions in its Team Notes.
- **Neutral, public-safe framing.** Documentation is written for a general audience: no private hardware specifics, no secrets, no credentials, jargon explained in plain English.
- **Cross-reference, don't duplicate.** Link to the doc that owns a topic rather than copying its content.

## Testing Standards

Pragmatic, not exhaustive — test the **load-bearing logic**, not everything.

- **Unit tests** for the deterministic graph functions: MST/Union-Find healing, betweenness, the global-efficiency Resilience Index. These have clear inputs/outputs and are the parts most worth protecting.
- **Smoke test** for the full pipeline on one small sample tile (does it run end-to-end without crashing?).
- **Sanity checks / assertions** baked into the code: masks are binary {0,1}; edge weights > 0; betweenness in [0,1]; connected-component count recorded before/after healing; CRS consistent across tile/mask/graph.
- **Visual QC** for masks and graphs — overlay predictions on imagery and eyeball them; numbers alone hide alignment bugs.
- Don't gold-plate tests for throwaway exploration code.

## Security Guidelines

- **No secrets in the repo.** Any data-portal keys go in environment variables / a git-ignored config — never committed.
- **`.gitignore` raw data and checkpoints** (large and/or license-restricted); commit only small sample data so the repo still runs.
- **Respect dataset licenses** (OSM ODbL, OpenSatMap CC BY-NC-SA *non-commercial*, SpaceNet/DeepGlobe research terms, Cartosat-3 restricted). Record source + license for every dataset used; attribute basemap/imagery in the UI.
- **Validate inputs** in the dashboard (accept only expected formats; fail gracefully on bad files).
- **Pin dependency versions**; no PII is handled anywhere in the system.

## AI Instructions

This project uses AI assistants. Any AI agent working in this repo must:
- **Treat the `docs/` as the source of truth**, and keep new work consistent with `PRD.md`, `TRD.md`, `Schema.md`, and these rules.
- **Hold the project constraints** below — do not introduce a database, auth system, heavyweight framework, or JavaScript SPA; do not switch the resilience metric back to raw average-path-length.
- **Prefer simple, readable code** with explanations, over dense or "high-tech" solutions beyond the team's level.
- **Keep the neutral, public-safe framing** — no private hardware details, no secrets, no committing raw/restricted data.
- **Update `Tracker.md`** and explain reasoning for non-trivial choices.
- **Don't invent results or citations.** Where a number isn't known, mark it as a placeholder.

## Project Constraints

The hard boundaries every contribution must respect (from `PRD.md`/`TRD.md`):
- **Compute:** must fit an 8 GB VRAM budget — fine-tune pretrained models only, AMP/FP16, small tiles, gradient accumulation/checkpointing; free cloud for overflow. The graph/dashboard run on CPU.
- **Stack:** pure-Python — Streamlit + Folium for the frontend (no JS SPA); file-based artifact store (no database).
- **Tools:** free / open-source only.
- **Method:** the Resilience Index uses **global efficiency** (finite under disconnection), not the raw average-path-length ratio.
- **Scope:** small team — protect scope; aspirational features stay parked until the core pipeline works end-to-end.