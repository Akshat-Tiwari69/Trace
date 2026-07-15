"""Small data-contract checks for dashboard evidence."""

from __future__ import annotations

from src.app import app


def test_load_resilience_curve_rejects_out_of_range_evidence(tmp_path, monkeypatch):
    sample_dir = tmp_path / "data" / "sample"
    sample_dir.mkdir(parents=True)
    (sample_dir / "panaji_demo_resilience.csv").write_text(
        "n_removed,targeted_resilience_index,random_resilience_index\n"
        "0,1.0,1.0\n"
        "1,1.2,0.8\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app, "REPO_ROOT", tmp_path)
    app.load_resilience_curve.clear()

    try:
        assert app.load_resilience_curve() is None
    finally:
        app.load_resilience_curve.clear()
