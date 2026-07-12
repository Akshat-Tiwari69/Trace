# Project Documentation Index

Use this page to find the document that owns a question. The goal is one owner per topic, not duplicated status prose.

## Start here

| Need | Document |
|---|---|
| Current task, owner, contracts or locked decision | [`Tracker.md`](Tracker.md) |
| Environment setup and runnable commands | [`../SETUP.md`](../SETUP.md) |
| Public overview and quickstart | [`../README.md`](../README.md) |
| Contribution and agent rules | [`Rules.md`](Rules.md), [`../AGENTS.md`](../AGENTS.md) |

## Product and system

| Topic | Owner document | Contains |
|---|---|---|
| Product | [`PRD.md`](PRD.md) | Users, problem, current scope, requirements and success criteria |
| Architecture | [`TRD.md`](TRD.md) | Batch and deployed components, boundaries, security and performance |
| Artifact/data contracts | [`Schema.md`](Schema.md) | Required files, fields, invariants and lifecycle |
| User behavior | [`UserJourney.md`](UserJourney.md) | Entry points, current flows, errors and exits |
| UI/UX | [`Design.md`](Design.md) | Current design system and the F9 overhaul brief |

## Delivery and evidence

| Topic | Owner document | Contains |
|---|---|---|
| Execution sequence | [`Implementation.md`](Implementation.md) | A44→A45/A46→F9/O1→X1 and done-gates |
| Evaluation | [`Evaluation.md`](Evaluation.md) | Protocols, metrics, result tables and verdicts |
| Research | [`Research.md`](Research.md) | Literature, hypotheses, experiment history and negative results |
| Risks | [`RiskRegister.md`](RiskRegister.md) | Current unresolved technical, evidence, operational and UX risks |
| Historical A6–A11 review | [`Retrospective-A6-A11.md`](Retrospective-A6-A11.md) | Focused record of the early dataset/model experiments |

## Operations

| Need | Document/file |
|---|---|
| Local CPU/cloud/local-GPU setup | [`../SETUP.md`](../SETUP.md) |
| Oracle/Modal/Caddy deployment | [`../deploy/README.md`](../deploy/README.md) |
| Current open-issue ledger | [`../bugs.md`](../bugs.md) |
| Archived July production-readiness audit | [`ProductionReadinessAudit-2026-07.md`](ProductionReadinessAudit-2026-07.md) |
| CI behavior | [`../.github/workflows/ci.yml`](../.github/workflows/ci.yml) |

## Authority order

When documents disagree:

1. Running code, tests, immutable releases and live remote state are the evidence of what exists.
2. `Tracker.md` is authoritative for current coordination and decisions.
3. The topic-owning document above is authoritative for design intent and contracts.
4. Historical logs and retrospectives explain why, but do not override current state.

Fix contradictions in the same PR that discovers them; do not add another competing summary.
