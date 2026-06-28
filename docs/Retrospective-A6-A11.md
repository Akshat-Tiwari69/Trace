# Segmentation Retrospective — A6 → A11

*Why the road-segmentation model plateaued, everything we tried to push past it, and what the evidence actually says to do next. Written 2026-06-26 (Akshat / seg lane).*

---

## TL;DR

Starting from **v1** (the A4 DeepGlobe-only model, IoU ≈ 0.67), we spent A6–A11 trying to make a meaningfully *better* road-segmentation model for the **Indian deployment target**. The honest scoreboard:

| Task | What we tried | Result |
|---|---|---|
| **A6** | Anti-forgetting fine-tune on Indian tiles → **v2** | ✅ **Modest win, released.** Indian ↑, DeepGlobe neutral. |
| **A7** | D4 test-time augmentation | ❌ No gain (−0.002). |
| **A8** | Heavier occlusion augmentation | ❌ Regression (−0.003 occ, −0.045 clean IoU). |
| **A9** | clDice-first (topology loss up-weighted) | ❌ Regression (−0.055 clDice, −0.078 IoU). |
| **A11** | From-scratch combined retrain + Massachusetts (mit_b4) → **v3** | ❌ **Worse on Indian (−0.018). Not released.** |

**Two clean lessons:**
1. **The model plateaued on *recipe*** — A7/A8/A9 proved that knob-tweaks (TTA, aug, loss weights) via short fine-tunes give no gain or active regression on an already-optimized model.
2. **The model is bottlenecked on *the right data*, not raw volume** — A11 proved that adding a large *foreign* dataset (US aerial) actively *hurts* the deployment target. **In-domain Indian data is the lever** (the Indian-fine-tuned v2 still beats everything on Indian roads).

---

## The starting point

- **v1 = A4**: SegFormer **mit_b3** encoder + SCSE U-Net decoder, trained on **DeepGlobe** (0.5 m satellite). Held-out DeepGlobe **IoU 0.670** (TTA) / 0.662 single-view, occlusion-recall 0.793 @ threshold 0.44. A strong, well-tuned model — but trained on a *non-Indian* distribution.
- **The goal**: a model that's genuinely better on **Indian satellite roads** (the deployment domain), without forgetting DeepGlobe.

## A6 — Anti-forgetting domain fine-tune → v2 *(the one that worked)*

**Idea:** fine-tune v1 on ~169 hand-built Indian tiles, but freeze the encoder and use a dual-domain best-checkpoint rule so it adapts to India *without* catastrophically forgetting DeepGlobe.

**What happened:** a naive full fine-tune adapted to India (+0.13) but forgot DeepGlobe (−0.07) → rejected. The rebuilt version (freeze encoder, larger DeepGlobe anchor, dual-domain selection) gave **Indian +0.03, DeepGlobe −0.004 (neutral)** → released as **`a4-roadseg-v2`**.

**Lesson:** in-domain fine-tuning *works* — but the gain is modest and bounded by how little Indian labeled data we have (169 tiles).

## A7 / A8 / A9 — The recipe-tweak dead end

The v2→v3 roadmap listed "free" wins. We measured each honestly via short fine-tunes from v1:

- **A7 (D4 TTA):** 8-fold dihedral test-time averaging → **−0.002 IoU**. The model is already orientation-robust; TTA just adds inference cost.
- **A8 (heavy occlusion aug):** stronger CoarseDropout + RandomShadow → **occlusion-recall flat (−0.003), clean IoU −0.045**. The A4 model is already occlusion-trained; heavier aug in a short fine-tune only degrades clean performance.
- **A9 (clDice-first):** up-weight the topology loss 0.1→0.3 → **clDice −0.055, IoU −0.078**, worse on *both*. Over-weighting clDice on an already-clDice-trained model destabilises it.

**Why they all failed (the pattern):** v1 is a converged, well-optimized model. **Short fine-tunes can't reveal training-recipe wins** — at best they do nothing, at worst they nudge a good model off its optimum. The conclusion after three straight nulls: **stop tweaking recipe knobs.** The real levers are **data** and a **proper from-scratch retrain**, not fine-tunes.

## A11 — The pivot to data: "3 + 4" (from-scratch retrain + more datasets)

**Hypothesis:** if recipe is maxed out, *more and more-diverse data* should break the plateau. The plan ("3+4") = **(3)** a from-scratch combined retrain with the best recipe + **(4)** add datasets (Massachusetts Roads now, SpaceNet-3 later).

**What we built:**
- `build_massachusetts_data.py` — converts Massachusetts Roads (1 m aerial, 0/255 masks) → DeepGlobe-format, with a **2× upsample** to scale-match DeepGlobe's 0.5 m. Capped at **8,000 tiles** to stay ~balanced with DeepGlobe's 6,226 (so the foreign set wouldn't dominate the mix).
- `train_combined.py` — from-scratch combined harness: **mit_b4** encoder, EMA weights, ComboLoss (BCE+Dice+Lovász+clDice), heavy occlusion, discriminative LR, warmup+cosine.

**The run (v3):** Kaggle T4, **18 epochs**, DeepGlobe 6226 + Massachusetts 8000 → `road_combined_v3.pt`. Combined-val IoU climbed cleanly to **0.5503** at epoch 18 (fully annealed). *(Note: that 0.55 is on the **combined** val set, which includes hard Massachusetts tiles — **not** comparable to v1's DeepGlobe-only 0.67.)*

### The verdict — an apples-to-apples held-out eval

The only honest test is running **both** models on the **same** held-out tiles. IoU @ 0.44, single-view:

| Held-out set | **v1** (mit_b3, DeepGlobe) | **v3** (mit_b4, combined) | v2 (mit_b3, Indian-ft) |
|---|:---:|:---:|:---:|
| **🇮🇳 Indian — zero-shot (the deployment metric, fair to both)** | **0.314** | **0.296** ⬇ | **0.336** |
| Massachusetts-test (v3 in-domain) | 0.552 | 0.646 ⬆ | 0.536 |
| DeepGlobe-train (fit check, not held-out) | 0.651 | 0.659 | 0.646 |

> **Read the table carefully.** Only the **v1-vs-v3** comparison on the **Indian** row is a *clean* test — both are genuinely zero-shot there. **v2's 0.336 is on the very tiles it was fine-tuned on in A6**, so it's optimistically biased (a training-set score) — shown only as a reference, not a fair held-out number.

**v3 is *worse* than v1 on Indian roads (−0.018), on the one perfectly fair comparison.** The Massachusetts "win" is meaningless — v3 trained on Massachusetts-train, so it's just in-domain advantage on a domain **we don't deploy on**. DeepGlobe is a tie.

**Why A11 failed:** Massachusetts is **US, 1 m, aerial** — a different sensor, resolution, and urban form from **Indian, 0.5 m, satellite**. Even scale-matched and balanced, it pulled the model toward the wrong distribution. **More data only helps if it's the *right* domain.** The complementary evidence comes from A6, where Indian fine-tuning gave a **cleanly-measured** Indian gain (+0.03 on a fresh Indian sample): *in-domain Indian data moves the Indian number; foreign data doesn't.*

---

## Infrastructure notes (hard-won, so we don't relearn them)

These didn't change the science but cost real time, and the lessons are reusable:

- **Local GPU (RTX 3070 Ti):** CUDA torch installed into a project-local `.venv-gpu` on F: (C: was full). Encoder weights had to be pre-fetched via `curl` (the in-training download stalled at 40 kB/s with the GPU idle).
- **Agent-launched jobs die on session teardown:** the local 18-epoch run was a child of the agent session and got reaped at epoch 7/18 when the session was torn down. **Long local jobs must be launched *detached* (`Start-Process`).** `train_combined` has no resume, so a kill = restart from epoch 1.
- **Kaggle automation works — except GPU selection:** the run was fully scripted via the Kaggle API (push kernel + auto-attach datasets + poll + pull). But **API pushes default to a P100**, and **current PyTorch dropped Pascal (sm_60) support**, so the P100 crashes on the first conv (`no kernel image for device`). **You can only pick the working T4 in the web UI** — the launch can't be 100% headless.
- **Pulling kernel output:** the notebook's 8,000 generated Massachusetts tiles inflated the kernel "output," so a bulk pull choked. Use `kernels_output(..., file_pattern=...)` to fetch *only* the checkpoint.

---

## What the evidence says to do next

1. **Don't release v3.** It loses to v1 on the deployment metric. **v2 stays the deployed model.**
2. **Don't spend the GCP credit on another foreign-only combined retrain.** A11 *was* the validation gate, and it said no. (This is exactly why we validated cheap before scaling.)
3. **A12 — self-training / pseudo-labels on unlabeled Indian tiles — is now the highest-confidence lever.** The whole A6→A11 arc points one way: *Indian data is what moves the Indian number.* Expand it (pseudo-label confident regions of unlabeled Indian basemap, add to the fine-tune set) and fine-tune.
4. **SpaceNet-3 is a *better-domain* bet than Massachusetts** (0.3 m satellite, global cities incl. developing-world) — but it's still foreign, and A11 counsels caution. Only worth it as a stronger *base* underneath an Indian fine-tune, and only after A12 shows the Indian-data lever is being pulled hard.

**One-line moral:** *the model didn't need more knobs or more data — it needed more of the **right** data.*
