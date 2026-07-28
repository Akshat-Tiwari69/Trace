"""Small data-contract checks for web API evidence."""

from __future__ import annotations

import pytest

from src.app.service import _resilience_curve


def test_load_resilience_curve_rejects_out_of_range_evidence(tmp_path):
    curve = tmp_path / "resilience.csv"
    curve.write_text(
        "n_removed,targeted_resilience_index,random_resilience_index\n"
        "0,1.0,1.0\n"
        "1,1.2,0.8\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="targeted_resilience_index"):
        _resilience_curve(curve)
