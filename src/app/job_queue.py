"""Filesystem-backed FIFO job queue for A39 upload analysis.

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
import json
import os
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
LEASE_SECONDS = 30 * 60
SEQUENCE_LOCK_STALE_SECONDS = 30.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _state_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _mask_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.npy"


def _result_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.result.json"


def _legacy_result_path(job_id: str) -> Path:
    """Pre-v1 pickle path, retained only so cleanup removes stale artifacts."""
    return JOBS_DIR / f"{job_id}.result.pkl"


def _claim_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.claim"


def _next_seq() -> int:
    """Allocate a persistent FIFO sequence under a cross-process lock."""
    counter = JOBS_DIR / "sequence.counter"
    lock = JOBS_DIR / "sequence.lock"
    wait_started = time.monotonic()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
                if age > SEQUENCE_LOCK_STALE_SECONDS:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() - wait_started > SEQUENCE_LOCK_STALE_SECONDS + 5:
                raise RuntimeError("timed out acquiring persistent queue sequence lock")
            time.sleep(0.005)
    try:
        try:
            value = int(counter.read_text()) + 1
        except (FileNotFoundError, ValueError):
            value = 0
        atomic_write(counter, lambda tmp: tmp.write_text(str(value)))
        return value
    finally:
        lock.unlink(missing_ok=True)


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
    return [p.stem for p in JOBS_DIR.glob("*.json") if not p.name.endswith(".result.json")]


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
        "seq": _next_seq(),
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
    return _load_result(_result_path(job_id))


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
            for path in (_state_path(job_id), _mask_path(job_id), _result_path(job_id),
                         _legacy_result_path(job_id), _claim_path(job_id)):
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
    claimed = None
    for job_id, state in sorted(queued, key=lambda item: (item[1]["seq"], item[0])):
        try:
            fd = os.open(_claim_path(job_id), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.close(fd)
        current = _try_read_state(job_id)
        if current is None or current.get("status") != "queued":
            _claim_path(job_id).unlink(missing_ok=True)
            continue
        state = current
        state["status"] = "running"
        state["started_utc"] = _now()
        state["owner_pid"] = os.getpid()
        state["lease_expires_utc"] = (
            datetime.now(timezone.utc) + timedelta(seconds=LEASE_SECONDS)
        ).isoformat(timespec="seconds")
        _write_state(job_id, state)
        claimed = (job_id, state)
        break
    if claimed is None:
        return False
    job_id, state = claimed

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
        _claim_path(job_id).unlink(missing_ok=True)
        return True

    atomic_write(_result_path(job_id), lambda tmp: _dump_result(tmp, analysis_result))
    state["status"] = "done"
    state["finished_utc"] = _now()
    _write_state(job_id, state)
    _claim_path(job_id).unlink(missing_ok=True)
    return True


def _dump_result(path: Path, value: Any) -> None:
    """Persist a versioned, dependency-tolerant JSON contract (never pickle)."""
    import networkx as nx

    payload = {
        "schema_version": 1,
        "graph": nx.node_link_data(value.graph, link="edges"),
        "criticality": {
            "columns": list(value.criticality.columns),
            "records": value.criticality.to_dict(orient="records"),
        },
        "resilience_index": value.resilience_index,
        "top_node": value.top_node,
        "n_nodes": value.n_nodes,
        "n_edges": value.n_edges,
        "resolution_m": value.resolution_m,
        "summary": value.summary,
    }
    path.write_text(json.dumps(payload, default=_json_default, separators=(",", ":")))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported queue-result value: {type(value).__name__}")


def _load_result(path: Path) -> Any:
    import networkx as nx
    import pandas as pd

    from src.app.upload_analysis import AnalysisResult

    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise RuntimeError(
            f"unsupported upload result schema {payload.get('schema_version')!r}"
        )
    table = payload["criticality"]
    criticality = pd.DataFrame(table["records"], columns=table["columns"])
    graph = nx.node_link_graph(payload["graph"], link="edges")
    return AnalysisResult(
        graph=graph,
        criticality=criticality,
        resilience_index=float(payload["resilience_index"]),
        top_node=payload["top_node"],
        n_nodes=int(payload["n_nodes"]),
        n_edges=int(payload["n_edges"]),
        resolution_m=float(payload["resolution_m"]),
        summary=payload.get("summary", {}),
    )


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _recover_stale_running() -> int:
    """Re-queue a job whose running-worker lease is stale.

    Recover only jobs whose owner process is no longer alive. Leases are useful
    crash metadata, but this worker does not renew them during long analyses;
    expiring a live owner's claim would allow duplicate processing and racy
    final writes.
    """
    recovered = 0
    for job_id in _iter_job_ids():
        state = _try_read_state(job_id)
        if state is None or state["status"] != "running":
            continue
        if _pid_alive(state.get("owner_pid")):
            continue
        _claim_path(job_id).unlink(missing_ok=True)
        state["status"] = "queued"
        state["started_utc"] = None
        state["owner_pid"] = None
        state["lease_expires_utc"] = None
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
