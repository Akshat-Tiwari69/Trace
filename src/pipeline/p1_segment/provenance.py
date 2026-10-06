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


def dir_fingerprint(path: str | Path) -> dict:
    """Cheap identity of a training-data folder: complete pairs (``<stem>_sat.jpg`` and
    ``<stem>_mask.png`` both present and non-empty, as training pairs them), stems
    missing half a pair, and a SHA-256 over sorted ``(name, size)`` entries. Catches
    added, removed or resized files without reading gigabytes (A51: a mutable data
    volume must not change a resumed run unnoticed); a same-size in-place rewrite is
    out of scope."""
    entries = sorted((e.name, e.stat().st_size) for e in os.scandir(path))
    digest = hashlib.sha256("\n".join(f"{n}\t{s}" for n, s in entries).encode()).hexdigest()
    sats = {n[: -len("_sat.jpg")]: s for n, s in entries if n.endswith("_sat.jpg")}
    masks = {n[: -len("_mask.png")]: s for n, s in entries if n.endswith("_mask.png")}
    complete = {stem for stem in sats.keys() & masks.keys() if sats[stem] > 0 and masks[stem] > 0}
    return {"pairs": len(complete), "incomplete": len((sats.keys() | masks.keys()) - complete),
            "sha256_names_sizes": digest}


def record_run(run_dir: str | Path, record: dict, launch: dict, allow_code_change: bool = False) -> dict:
    """Write or verify ``run_dir/run.json``: one run directory holds one recipe and
    one dataset (A51: a pilot or a changed encoder must never reuse another run's
    ``.done`` stages). Appends ``launch`` (code revision ``commit``/``dirty``, time)
    on every launch. Raises ``RuntimeError`` on a recipe/data mismatch, on
    unrecorded artifacts, or when the code differs from the previous launch (or
    either tree was dirty) unless ``allow_code_change`` -- resuming under changed
    code is then a deliberate, recorded choice. The file is replaced atomically."""
    run_dir = Path(run_dir)
    path = run_dir / "run.json"
    record = json.loads(json.dumps(record))         # compare as JSON, the stored form
    if path.exists():
        saved = json.loads(path.read_text())
        if {k: saved.get(k) for k in record} != record:
            raise RuntimeError(f"{run_dir} was recorded with a different recipe or data; use a new run name")
        launches = saved["launches"]
        prev = launches[-1] if launches else {}
        same_code = (prev.get("commit") == launch.get("commit")
                     and not prev.get("dirty") and not launch.get("dirty"))
        if not same_code and not allow_code_change:
            raise RuntimeError(f"{run_dir}: code changed since the last launch ({prev.get('commit')} -> "
                               f"{launch.get('commit')}, dirty {prev.get('dirty')}/{launch.get('dirty')}); "
                               "pass allow_code_change to resume under the new code")
        launch = {**launch, "allow_code_change": allow_code_change and not same_code}
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"{run_dir} has artifacts but no run.json; use a new run name")
    else:
        launches = []
    record["launches"] = [*launches, launch]
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"run.json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    os.replace(tmp, path)                            # an interrupted write never truncates run.json
    return record


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
