"""Retained negative-result experiments (P1 · segmentation lane).

Per CLAUDE.md's operating principle — *"Negative results are first-class —
record them so nobody re-runs a dead end"* — these training/data-prep scripts
are **not** part of the production pipeline. Each was run, evaluated, and
rejected in favor of a different approach (see `docs/Tracker.md` §6 and
`docs/Retrospective-A6-A11.md` for the full write-ups):

- ``build_massachusetts_data.py`` (A11 data prep) — Massachusetts Roads →
  DeepGlobe-format conversion. Superseded by ``build_spacenet_data.py``.
- ``train_combined.py`` (A11) — from-scratch combined-corpus retrain.
  Rejected: Massachusetts diluted the Indian deployment target. Kept as the
  reference implementation for from-scratch multi-corpus retrains (and
  ``ModelEMA``, reused by ``train_selftrain.py``).
- ``train_selftrain.py`` (A12) — mean-teacher self-training on unlabeled
  Indian tiles. Rejected: both configs peaked at epoch 1 and lost to v1 on
  the honest held-out test.

They are kept (not deleted) so nobody re-derives and re-runs the same dead
ends, and are still covered by CPU smoke tests. The production training path
is ``finetune.py`` (A23/A24), one level up.
"""
