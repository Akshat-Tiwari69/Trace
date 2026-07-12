"""Statistical rigor for model-promotion decisions (bugs.md §3).

Every "v_next beats v_prev" call in this project has ridden on single point
estimates of IoU/APLS over a fixed held-out set — several past gaps (A17's
0.004 IoU, A24's +2-3%) are the same order as plausible sampling noise. This
module supplies the missing piece: a **paired bootstrap** over per-tile scores
that reports a confidence interval on the *delta* between two checkpoints, so a
promotion is only called a win when the CI excludes zero.

Paired (same tiles, per-tile difference resampled) rather than two-sample
because both checkpoints are scored on the *identical* tiles — pairing removes
the tile-difficulty variance that would otherwise swamp the model difference.

Pure Python + numpy, no torch — trivially unit-testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    """Paired-bootstrap result for ``delta = mean(scores_b) - mean(scores_a)``."""

    delta: float          # observed mean difference (b - a)
    ci_low: float         # lower bound of the (1 - alpha) CI on the delta
    ci_high: float        # upper bound
    ci_level: float       # e.g. 0.95
    n: int                # number of paired samples (tiles)
    p_two_sided: float    # bootstrap p-value that delta == 0
    excludes_zero: bool   # True ⇒ CI does not straddle 0 ⇒ a real difference

    @property
    def verdict(self) -> str:
        """Human-readable promotion verdict."""
        if not self.excludes_zero:
            return "inconclusive (CI straddles 0 — within sampling noise)"
        return "b wins" if self.delta > 0 else "a wins"

    def summary(self) -> str:
        # ascii only ("delta"/"->", not Δ/→): this string is printed by the eval
        # CLIs, and redirected stdout on Windows is cp1252 — non-ascii here
        # crashed the A38 APLS stage twice (2026-07-09).
        return (f"delta={self.delta:+.4f}  {int(self.ci_level * 100)}% CI "
                f"[{self.ci_low:+.4f}, {self.ci_high:+.4f}]  n={self.n}  "
                f"p={self.p_two_sided:.3f}  -> {self.verdict}")


def paired_bootstrap_ci(
    scores_a: list[float] | np.ndarray,
    scores_b: list[float] | np.ndarray,
    n_boot: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 17,
) -> BootstrapCI:
    """Paired bootstrap CI on the mean per-tile score difference ``b - a``.

    ``scores_a`` and ``scores_b`` are the two checkpoints' per-tile scores in the
    **same tile order** (element ``i`` of each is the same tile). We resample the
    per-tile differences with replacement ``n_boot`` times and take the empirical
    quantiles of the resampled mean difference.

    Raises ``ValueError`` on length mismatch or empty input — a promotion call on
    zero comparable tiles is a bug, not a valid "inconclusive".
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired scores must align: got {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("no paired scores — cannot bootstrap a promotion decision")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be in (0, 1)")

    diffs = b - a
    n = diffs.size
    rng = np.random.default_rng(seed)
    # (n_boot, n) index matrix → resampled mean differences.
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)

    alpha = 1.0 - ci_level
    ci_low, ci_high = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    # Valid paired randomization p-value under H0: each paired difference is
    # exchangeable in sign. The bootstrap distribution is for the CI only; using
    # its uncentred mass across zero as a p-value is not a null test.
    observed = float(diffs.mean())
    signs = rng.choice((-1.0, 1.0), size=(n_boot, n))
    null_means = (diffs * signs).mean(axis=1)
    p = float((np.count_nonzero(np.abs(null_means) >= abs(observed)) + 1) /
              (n_boot + 1))

    return BootstrapCI(
        delta=observed,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        ci_level=ci_level,
        n=n,
        p_two_sided=p,
        excludes_zero=bool(ci_low > 0.0 or ci_high < 0.0),
    )
