"""Artifact provenance: answer "which checkpoint produced this?".

Schema.md defines ``RoadMask.model_version`` / ``threshold`` but nothing ever
persisted them: a mask PNG was an anonymous bitmap and the graph/CSV carried no
lineage back to a checkpoint or commit. This module builds a small, JSON-safe
provenance record at P1 inference time and threads it through the file handoffs
so every downstream artifact can name the exact model, threshold, and code that
made it — a prerequisite for reproducing (or auditing) any result in a pilot.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: str | Path, _chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a (potentially large) checkpoint file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str | None:
    """Current repo commit (short), or ``None`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def build_provenance(
    checkpoint: str | Path,
    meta: dict,
    threshold: float,
    *,
    hash_checkpoint: bool = True,
    checkpoint_sha256: str | None = None,
) -> dict:
    """Assemble the provenance record for a P1 inference run.

    A caller that already scanned the checkpoint can supply ``checkpoint_sha256``.
    Otherwise ``hash_checkpoint`` controls whether this function scans it.
    """
    checkpoint = Path(checkpoint)
    record = {
        "checkpoint": checkpoint.name,
        "model_sha256": checkpoint_sha256 or (
            sha256_file(checkpoint) if (hash_checkpoint and checkpoint.is_file()) else None
        ),
        "encoder": meta.get("encoder"),
        "arch": meta.get("arch"),
        "threshold": float(threshold),
        "image_size": meta.get("image_size"),
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return record


def write_provenance(path: str | Path, record: dict) -> Path:
    """Atomically write a provenance record as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2))
    os.replace(tmp, path)
    return path


def read_provenance(path: str | Path) -> dict | None:
    """Read a provenance record, or ``None`` if absent/unreadable."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
