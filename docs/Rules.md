# Rules.md — Engineering Rules

> Prefer simple, evidence-backed changes. `AGENTS.md`/`CLAUDE.md` define the agent entry protocol; `Tracker.md` defines current work and ownership.

## Code

- Target Python 3.11 for development/CI; production currently runs its isolated Python 3.12 environment.
- Match the existing style: PEP 8, type hints and short docstrings on public functions.
- Keep functions focused and names explicit. Do not compress code merely to reduce line count.
- Centralize configuration and paths; avoid duplicated thresholds and hidden defaults.
- Put reusable logic in modules, not notebooks or Streamlit render functions.
- Quarantine rejected experiments from production imports.
- Refactors require characterization tests and, for performance claims, before/after measurements.

## Architecture

- Stable domain seam: P1 mask/probability/provenance → P2 MultiGraph → P3 criticality/resilience → presentation/API adapters.
- The batch pipeline communicates through §4 file artifacts. The hosted upload path may call the authenticated Modal P1 endpoint and run P2/P3 on the CPU host.
- Streamlit/Folium remains only the current deployed presentation until F9 replaces it. Akshat authorized a modern web frontend and a thin Python application API; the migration must preserve domain logic and artifact contracts instead of reimplementing them in the client.
- No database or user-login product is added without a separate requirement and recorded contract/security decision.
- The resilience metric is baseline-normalized global efficiency and must preserve the baseline node universe so it remains finite and in `[0, 1]`. Any path that shrinks the denominator is a correctness defect.
- CPU is the deployment target for P2/P3/dashboard; GPU is optional for local inference and required only for training/remote P1.

## Documentation

- `Tracker.md`: current status, ownership, contracts, decisions and concise daily log.
- `Evaluation.md`: metrics, protocols, tables and model/graph verdicts.
- `Research.md`: literature, hypotheses, experiment rationale and negative results.
- `TRD.md`/`Schema.md`: current architecture and interfaces.
- `Design.md`/`UserJourney.md`: current and next UI behavior.
- `SETUP.md`/`deploy/README.md`: executable environment and operator instructions.
- Cross-reference the owner document; do not maintain the same status table in multiple files.
- Update behavior and its owning documentation in the same PR. Never invent results, deployment state or citations.

## Testing and evidence

- Unit-test deterministic graph, metric, IO, queue and inference-contract behavior.
- Keep an end-to-end sample pipeline test, dashboard import smoke and a local upload-analysis contract smoke under production dependencies.
- Validate binary masks, coordinate frames, positive edge lengths, graph/GeoJSON round trips, bounded metrics and required artifact columns.
- Use visual/geospatial QC for alignment and topology; numerical scores alone cannot expose every frame error.
- Model promotion requires the frozen protocol in `Evaluation.md`, full coverage and paired uncertainty.
- Negative findings are recorded, not rewritten as success or silently discarded.

## Security and data

- Never commit secrets, raw/restricted imagery or checkpoints.
- Validate AOI identifiers, uploads, decoded size/type and remote responses.
- Modal authentication must fail closed and occur before app-level base64 decoding, image parsing and model work.
- Production deploys use immutable refs, health checks and rollback.
- Respect dataset and upstream-code licenses; record provenance and redistribution restrictions.
- Uploaded source bytes are held in memory and sent to Modal; the host persists derived queue masks/state/results for age-based cleanup (currently about 24 hours). Disclosure must remain visible in the UI/docs.

## Git and collaboration

- Branch from current `dev` as `<owner>/<task-id>-<slug>`.
- One coherent task per PR; target `dev`, never `main`.
- Akshat is the only approver. Do not self-merge a newly opened PR.
- Preserve user work, untracked files and unrelated changes.
- Default ownership remains Akshat=P1/integration, Shaivi=P2/P3, Saanvi=P4/design. A documented coordinator authorization may permit cross-lane work, but reviewers and contracts still apply.

## Non-negotiable product constraints

- PyTorch and pretrained fine-tuning only; no training from scratch.
- Global-efficiency Resilience Index, never raw average-path-length ratio.
- Python/PyTorch ML and graph core with a performance-budgeted web presentation; no database or auth/login product scope unless separately authorized.
- Hardware-agnostic training path; no remote access to a teammate’s machine.
- Neutral, public-safe repository with honest, reproducible evidence.
