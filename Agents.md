# AGENTS.md — Agent Entry Point

**Codex (and other coding agents) read this automatically. Read it fully before doing anything.** (Claude Code users: see the identical `CLAUDE.md`.)

This repo (**Route Resilience**) is coordinated through one backbone file: **`docs/Tracker.md`**. It is the source of truth for who owns what, what to do next, and where to wait for others.

## Do this every session, in order:
1. **Open `docs/Tracker.md` and read §0–§2 + §4.**
2. **Identify who you're working for.** If the user hasn't said, ask: *"Which team member am I working as — Akshat, Shaivi, or Saanvi?"*
3. **Follow `docs/Tracker.md` §1 (Agent Operating Protocol):** pick that person's next unblocked task, work **only within their ownership lane**, respect the artifact contracts (§4), and update the Tracker when done.
4. **If asked to touch another person's area or a blocked task, STOP and warn the user** (use the warning templates in §1) — don't silently do it.

## Non-negotiable rules (full list in `docs/Rules.md` / Tracker §2):
- Stack: **Streamlit + Folium, pure Python**. No React/JS-SPA, no database, no REST API, no auth (v1).
- ML: **fine-tune pretrained only**, **PyTorch only**.
- Resilience metric: **global efficiency** (never raw average-path-length ratio).
- Training is **hardware-agnostic (Colab/Kaggle)**; graph + dashboard run on **CPU**. **No remote access** to anyone's machine.
- Keep the repo **neutral/generic**; no secrets; `.gitignore` raw data + checkpoints; respect dataset licenses.
- Prefer **simple, readable code**; explain non-trivial choices; don't invent results.
- **Git:** branch off **`dev`** (never `main`) as `<you>/<task-id>-<slug>`; when done, **open a PR into `dev` and stop** (don't merge on creation). Akshat is the only approver. **Never PR or merge into `main`.** You may merge your *own* PR only if it's already **approved by Akshat and still unmerged**. Full rules in `docs/Tracker.md` §11.

## Setup
Environment setup is self-service in **`SETUP.md`** (pick your path by role).

## Working Principles (how to operate)

**1. Think before coding.** Don't assume; don't hide confusion; surface tradeoffs.
- State your assumptions explicitly; if uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name what's confusing, and ask.

**2. Simplicity first.** The minimum code that solves the problem — nothing speculative.
- No features beyond what was asked; no abstractions for single-use code.
- No "flexibility" / "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you wrote 200 lines and 50 would do, rewrite it. Ask: *"Would a senior engineer call this overcomplicated?"* If yes, simplify.

**3. Surgical changes.** Touch only what you must; clean up only your own mess.
- Don't "improve" adjacent code, comments, or formatting; don't refactor what isn't broken.
- Match existing style, even if you'd do it differently.
- Notice unrelated dead code? Mention it — don't delete it.
- Remove imports/variables/functions that *your* changes made unused; leave pre-existing dead code unless asked.
- The test: every changed line traces directly to the request.

**4. Goal-driven execution.** Define success criteria, then loop until verified.
- Turn tasks into verifiable goals: *"add validation"* → *"write tests for invalid inputs, then make them pass"*; *"fix the bug"* → *"write a failing test that reproduces it, then make it pass"*; *"refactor X"* → *"tests pass before and after"*.
- For multi-step tasks, state a brief plan with a verify-check per step.
- Strong success criteria let you loop independently; weak ones ("make it work") force constant clarification.

**5. Record what you find.** Discoveries belong in the docs, not just the chat — if it isn't written down, it didn't happen.
- Experiment results (**positive _or_ negative**), run findings, progress → `docs/Tracker.md` §10 daily log (+ flip the task status / add a task row).
- Techniques, literature, design rationale, triage of external reviews → `docs/Research.md`.
- Locked decisions → `docs/Tracker.md` §8 Decisions Log.
- A new operating convention (like this one) → **here in the entry-point file**, and **mirror it into the sibling**: `CLAUDE.md` and `AGENTS.md` are the same doc for different agents — keep them **byte-identical except the title line and the "who reads this" line**. Any edit to one is mirrored to the other in the same change.
- **Negative results are first-class** — record them so nobody re-runs a dead end (cf. A8 / A9 / A11).

> One-line summary: **Read `docs/Tracker.md`, work only in your lane, warn before crossing into someone else's, update the Tracker when done, and write down what you learn (incl. negative results) — keeping `CLAUDE.md` ≡ `AGENTS.md`.**