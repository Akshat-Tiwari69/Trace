"""Dependency-role files stay complete and pin-compatible."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _pins(relative_path: str, seen: set[Path] | None = None) -> dict[str, str]:
    path = (ROOT / relative_path).resolve()
    seen = seen or set()
    if path in seen:
        return {}
    seen.add(path)

    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            included = (path.parent / line[3:].strip()).relative_to(ROOT)
            pins.update(_pins(str(included), seen))
            continue
        name = re.split(r"[<>=!~]", line, maxsplit=1)[0].lower().replace("_", "-")
        assert "==" in line, f"{path.name}: dependency must be pinned: {line}"
        assert name not in pins or pins[name] == line, f"conflicting pin for {name}"
        pins[name] = line
    return pins


def test_dependency_roles_are_isolated_and_aggregate_is_complete():
    app = _pins("deploy/requirements-app.txt")
    train = _pins("requirements-train.txt")
    dev = _pins("requirements-dev.txt")
    aggregate = _pins("requirements.txt")

    assert "streamlit" in app and "streamlit" not in train
    assert "segmentation-models-pytorch" in train and "segmentation-models-pytorch" not in app
    assert set(dev) == {"pytest", "jupyter"}
    assert aggregate == app | train | dev
