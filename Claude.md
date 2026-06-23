# CLAUDE.md — Agent Entry Point

**Claude Code reads this automatically. Read it fully before doing anything.**

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

## Setup
Environment setup is self-service in **`SETUP.md`** (pick your path by role).

> One-line summary: **Read `docs/Tracker.md`, work only in your lane, warn before crossing into someone else's, update the Tracker when done.**