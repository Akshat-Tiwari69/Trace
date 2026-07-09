"""Filesystem-backed FIFO job queue for upload analysis (bugs.md §5H/§9.4).

A37 (upload_analysis.py) added a `BoundedSemaphore(1)` as an interim guard: a
second concurrent upload lost the race and got an outright "busy, retry"
warning. This queues that upload instead, and — the actual restart-survival
win over the semaphore — persists to disk under `data/outputs/upload_jobs/`
so a Streamlit process restart (deploy, crash) mid-analysis doesn't silently
strand the job: `ensure_worker()` re-queues anything left `running` from a
dead process on the next start.

Kept free of Streamlit imports (pure Python) so it's unit-testable without a
UI runtime; app.py drives it from session_state (submit -> poll status/position
-> pull result).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import itertools
import json
import pickle
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from src.pipeline.p2_graph.graph_io import atomic_write

# src/app/job_queue.py -> parents[2] is the repo root; the file's location is
# fixed within the repo so this is simpler than app.py's Tracker.md-marker
# scan (which lives in app.py, and importing that would drag in streamlit).
JOBS_DIR = Path(__file__).resolve().parents[2] / "data" / "outputs" / "upload_jobs"

# FIFO order key, separate from created_utc: created_utc is second-precision
# (readable in the state file) and several jobs can land in the same second,
# so ordering by it alone isn't stable. A simple incrementing counter is.
_seq_counter = itertools.count()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _state_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _mask_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.npy"


def _result_path(job_id: str) -> Path:
    # Pickled AnalysisResult. Internal-only: the only writer is _process_one()
    # below, reading a mask this same module wrote moments earlier — never a
    # user-supplied file — so unpickling it here doesn't cross a trust boundary.
    return JOBS_DIR / f"{job_id}.result.pkl"


def _write_state(job_id: str, state: dict) -> None:
    """Atomic tmp+os.replace (reusing the P2 graph_io helper) so a crash
    mid-write can't leave a half-written/corrupt state file behind."""
    atomic_write(_state_path(job_id), lambda tmp: tmp.write_text(json.dumps(state)))


def _try_read_state(job_id: str) -> dict | None:
    try:
        return json.loads(_state_path(job_id).read_text())
    except (FileNotFoundError, ValueError):
        return None  # raced with cleanup()/a concurrent writer — caller skips it


def _iter_job_ids() -> list[str]:
    if not JOBS_DIR.is_dir():
        return []
    return [p.stem for p in JOBS_DIR.glob("*.json")]


def _save_mask(path: Path, mask: np.ndarray) -> None:
    # np.save(path, ...) appends ".npy" to a bare filename — which would
    # double-suffix atomic_write's "<job_id>.npy.tmp" tmp path. Writing
    # through an already-open file handle skips that auto-suffix behaviour.
    with open(path, "wb") as f:
        np.save(f, mask)


def submit(mask01: np.ndarray, resolution_m: float) -> str:
    """Queue a mask for analysis; returns a job_id to poll via status()/result()."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    atomic_write(_mask_path(job_id), lambda tmp: _save_mask(tmp, np.asarray(mask01)))
    _write_state(job_id, {
        "job_id": job_id,
        "status": "queued",
        "seq": next(_seq_counter),
        "created_utc": _now(),
        "started_utc": None,
        "finished_utc": None,
        "error": None,
        "resolution_m": float(resolution_m),
    })
    return job_id


def status(job_id: str) -> dict:
    """Return the job's current state dict."""
    return json.loads(_state_path(job_id).read_text())


def position(job_id: str) -> int:
    """0 if running (or nothing ahead in the queue); else count of jobs ahead."""
    state = status(job_id)
    if state["status"] != "queued":
        return 0
    ahead = 0
    for other_id in _iter_job_ids():
        if other_id == job_id:
            continue
        other = _try_read_state(other_id)
        if other is None:
            continue
        if other["status"] == "running":
            ahead += 1
        elif other["status"] == "queued" and other["seq"] < state["seq"]:
            ahead += 1
    return ahead


def result(job_id: str) -> Any:
    """Return the finished AnalysisResult. Raises RuntimeError if not done yet."""
    state = status(job_id)
    if state["status"] != "done":
        raise RuntimeError(f"job {job_id} is not done (status={state['status']!r})")
    with open(_result_path(job_id), "rb") as f:
        return pickle.load(f)


def cleanup(max_age_h: float = 24) -> int:
    """Delete any job's files once older than ``max_age_h`` hours old.

    Simple age-based sweep (not status-aware): a job's whole file set — state,
    mask, result — is disposable once it's stale, whatever state it's in.
    Returns the number of jobs removed.
    """
    if not JOBS_DIR.is_dir():
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_h)
    removed = 0
    for job_id in _iter_job_ids():
        state = _try_read_state(job_id)
        if state is None:
            continue
        created = datetime.fromisoformat(state["created_utc"])
        if created < cutoff:
            for path in (_state_path(job_id), _mask_path(job_id), _result_path(job_id)):
                path.unlink(missing_ok=True)
            removed += 1
    return removed


def _process_one() -> bool:
    """Pop the oldest queued job and run it through analyze_mask(). Returns
    whether a job was found and processed (False = queue was empty)."""
    queued = [
        (job_id, state) for job_id in _iter_job_ids()
        if (state := _try_read_state(job_id)) is not None and state["status"] == "queued"
    ]
    if not queued:
        return False
    job_id, state = min(queued, key=lambda item: item[1]["seq"])

    state["status"] = "running"
    state["started_utc"] = _now()
    _write_state(job_id, state)

    from src.app.upload_analysis import analyze_mask  # lazy: heavy P2/P3 imports

    try:
        with open(_mask_path(job_id), "rb") as f:
            mask = np.load(f)
        analysis_result = analyze_mask(mask, resolution_m=state["resolution_m"])
    except Exception as exc:  # noqa: BLE001 — any failure (incl. AnalysisBusyError,
        # which shouldn't occur since this worker is the only app-path caller
        # of analyze_mask, but is recorded rather than killing the worker loop
        # if it somehow does) becomes a failed job, not a dead thread.
        state["status"] = "failed"
        state["finished_utc"] = _now()
        state["error"] = str(exc)
        _write_state(job_id, state)
        return True

    with open(_result_path(job_id), "wb") as f:
        pickle.dump(analysis_result, f)
    state["status"] = "done"
    state["finished_utc"] = _now()
    _write_state(job_id, state)
    return True


def _recover_stale_running() -> int:
    """Re-queue any job stuck in 'running' (bugs.md §5H crash-safety).

    Only _process_one() ever sets 'running', and it runs one job at a time on
    a single worker thread per process — so any 'running' job found before
    that thread has started in *this* process must be left over from a
    process that died mid-job. Returns the count recovered.
    """
    recovered = 0
    for job_id in _iter_job_ids():
        state = _try_read_state(job_id)
        if state is None or state["status"] != "running":
            continue
        state["status"] = "queued"
        state["started_utc"] = None
        _write_state(job_id, state)
        recovered += 1
    return recovered


_worker_lock = threading.Lock()
_worker_started = False


def ensure_worker(poll_interval_s: float = 0.5) -> None:
    """Start the background worker thread once per process (idempotent).

    Recovers stale 'running' jobs first, then spawns a daemon thread looping
    _process_one(), sleeping briefly whenever the queue is empty so it doesn't
    spin a CPU core the analysis pipeline also needs.
    """
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        _recover_stale_running()

        def _loop() -> None:
            while True:
                if not _process_one():
                    time.sleep(poll_interval_s)

        threading.Thread(target=_loop, daemon=True, name="upload-analysis-worker").start()
        _worker_started = True
