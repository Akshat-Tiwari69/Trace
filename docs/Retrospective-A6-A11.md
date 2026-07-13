# Segmentation Retrospective — A6 → A11

> **Dated archive (2026-06-26).** This retrospective preserves the evidence available after A11. Its “next” recommendations were later tested and are no longer current status; see `Research.md`, `Evaluation.md` and `Tracker.md`. The unreleased A11 model is called the **A11 combined candidate** here to avoid confusion with the later released `a4-roadseg-v3`.

*Why the road-segmentation model plateaued and what the A6–A11 evidence showed at that time.*

---

## TL;DR

Starting from **v1** (the A4 DeepGlobe-only model, IoU ≈ 0.67), we spent A6–A11 trying to make a meaningfully *better* road-segmentation model for the **Indian deployment target**. The honest scoreboard:

| Task | What we tried | Result |
|---|---|---|
| **A6** | Anti-forgetting fine-tune on Indian tiles → **v2** | ✅ **Modest win, released.** Indian ↑, DeepGlobe neutral. |
| **A7** | D4 test-time augmentation | ❌ No gain (−0.002). |
| **A8** | Heavier occlusion augmentation | ❌ Regression (−0.003 occ, −0.045 clean IoU). |
| **A9** | clDice-first (topology loss up-weighted) | ❌ Regression (−0.055 clDice, −0.078 IoU). |
| **A11** | New combined run from ImageNet-pretrained mit_b4 + Massachusetts → **A11 combined candidate** | ❌ **Worse on Indian (−0.018). Not released.** |

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

**Why they all failed (the pattern):** v1 is a converged, well-optimized model. **Short fine-tunes can't reveal training-recipe wins** — at best they do nothing, at worst they nudge a good model off its optimum. The conclusion after three straight nulls: **stop tweaking recipe knobs.** The next tested lever was a new full combined run from a pretrained encoder, not training a backbone from scratch.

## A11 — The pivot to data: “3 + 4” (new pretrained-encoder run + more datasets)

**Hypothesis:** if recipe is maxed out, *more and more-diverse data* should break the plateau. The plan (“3+4”) = **(3)** a new full run with an ImageNet-pretrained encoder and the best recipe + **(4)** add datasets (Massachusetts Roads then, SpaceNet considered later).

**What we built:**
- `build_massachusetts_data.py` — converts Massachusetts Roads (1 m aerial, 0/255 masks) → DeepGlobe-format, with a **2× upsample** to scale-match DeepGlobe's 0.5 m. Capped at **8,000 tiles** to stay ~balanced with DeepGlobe's 6,226 (so the foreign set wouldn't dominate the mix).
- `train_combined.py` — combined harness with an **ImageNet-pretrained mit_b4** encoder, EMA weights, ComboLoss (BCE+Dice+Lovász+clDice), heavy occlusion, discriminative LR, warmup+cosine.

**The run (A11 combined candidate):** Kaggle T4, **18 epochs**, DeepGlobe 6226 + Massachusetts 8000 → `road_combined_v3.pt`. Combined-val IoU climbed cleanly to **0.5503** at epoch 18 (fully annealed). *(Note: that 0.55 is on the **combined** val set, which includes hard Massachusetts tiles — **not** comparable to v1's DeepGlobe-only 0.67.)*

### The verdict — an apples-to-apples held-out eval

The only honest test is running **both** models on the **same** held-out tiles. IoU @ 0.44, single-view:

| Held-out set | **v1** (mit_b3, DeepGlobe) | **A11 combined candidate** (mit_b4) | v2 (mit_b3, Indian-ft) |
|---|:---:|:---:|:---:|
| **🇮🇳 Indian — zero-shot (the deployment metric, fair to both)** | **0.314** | **0.296** ⬇ | **0.336** |
| Massachusetts-test (A11 candidate in-domain) | 0.552 | 0.646 ⬆ | 0.536 |
| DeepGlobe-train (fit check, not held-out) | 0.651 | 0.659 | 0.646 |

> **Read the table carefully.** Only the **v1-vs-A11-candidate** comparison on the **Indian** row is a *clean* test — both are genuinely zero-shot there. **v2's 0.336 is on the very tiles it was fine-tuned on in A6**, so it's optimistically biased (a training-set score) — shown only as a reference, not a fair held-out number.

**The A11 combined candidate is *worse* than v1 on Indian roads (−0.018), on the one fair comparison available then.** The Massachusetts "win" is only in-domain advantage on a domain the project does not deploy on. DeepGlobe is a tie.

**Why A11 failed:** Massachusetts is **US, 1 m, aerial** — a different sensor, resolution, and urban form from **Indian, 0.5 m, satellite**. Even scale-matched and balanced, it pulled the model toward the wrong distribution. **More data only helps if it's the *right* domain.** The complementary evidence comes from A6, where Indian fine-tuning gave a **cleanly-measured** Indian gain (+0.03 on a fresh Indian sample): *in-domain Indian data moves the Indian number; foreign data doesn't.*

---

## Infrastructure notes (hard-won, so we don't relearn them)

These didn't change the science but cost real time, and the lessons are reusable:

- **Local GPU environment:** use an isolated environment and prefetch/cache encoder weights before a long run when network access is unreliable. Keep machine-specific drive paths out of shared instructions.
- **Agent-launched jobs die on session teardown:** the local 18-epoch run was a child of the agent session and got reaped at epoch 7/18 when the session was torn down. **Long local jobs must be launched *detached* (`Start-Process`).** `train_combined` has no resume, so a kill = restart from epoch 1.
- **Kaggle automation works — except GPU selection:** the run was fully scripted via the Kaggle API (push kernel + auto-attach datasets + poll + pull). But **API pushes default to a P100**, and **current PyTorch dropped Pascal (sm_60) support**, so the P100 crashes on the first conv (`no kernel image for device`). **You can only pick the working T4 in the web UI** — the launch can't be 100% headless.
- **Pulling kernel output:** the notebook's 8,000 generated Massachusetts tiles inflated the kernel "output," so a bulk pull choked. Use `kernels_output(..., file_pattern=...)` to fetch *only* the checkpoint.

---

## What happened next

1. The A11 combined candidate was not released.
2. A12 mean-teacher/self-training on weak OSM labels was run and rejected; the OSM-agreement validation signal was misleading.
3. Real SpaceNet-5 Mumbai supervision produced the later released v3/v3.2 checkpoints and the best deployable mask-model development evidence.
4. Repeated Mumbai consultation converted it from “held-out test” to a development benchmark, so a new geography/sensor remains required.
5. A18 direct graph prediction beat v3.2 on the common-unit relative routing gate; LoRA improved A18 again, establishing graph-first as the A46 direction while remaining below deploy quality.

**Durable lesson:** domain-matched, truthful supervision mattered more than more recipe knobs or foreign volume. Current priorities live in `Tracker.md` §6.
