"""Tests for the filesystem job queue (bugs.md §5H/§9.4).

Drives `_process_one()` directly rather than spinning up ensure_worker's
background thread — deterministic, no sleeping. Each test monkeypatches
`JOBS_DIR` to an isolated tmp_path so nothing touches the real
data/outputs/upload_jobs/.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os

import numpy as np
import pytest

from src.app import job_queue


@pytest.fixture(autouse=True)
def _isolated_jobs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(job_queue, "JOBS_DIR", tmp_path / "upload_jobs")


def _grid_mask() -> np.ndarray:
    """A real connected road network (same fixture shape as test_l_fixes.py)."""
    mask = np.zeros((256, 256), np.uint8)
    for r in (64, 128, 192):
        mask[r - 1:r + 2, 20:236] = 1
    for c in (64, 128, 192):
        mask[20:236, c - 1:c + 2] = 1
    return mask


def test_submit_process_done_roundtrip():
    job_id = job_queue.submit(_grid_mask(), resolution_m=0.5)
    assert job_queue.status(job_id)["status"] == "queued"

    did_work = job_queue._process_one()
    assert did_work is True

    state = job_queue.status(job_id)
    assert state["status"] == "done"
    assert state["started_utc"] is not None
    assert state["finished_utc"] is not None

    result = job_queue.result(job_id)
    assert result.n_nodes > 0
    assert result.n_edges > 0

    # Queue is empty now.
    assert job_queue._process_one() is False


def test_fifo_order_and_position():
    first = job_queue.submit(_grid_mask(), resolution_m=0.5)
    second = job_queue.submit(_grid_mask(), resolution_m=0.5)
    third = job_queue.submit(_grid_mask(), resolution_m=0.5)

    # Nothing running yet: first is next (0 ahead), second/third queued behind it.
    assert job_queue.position(first) == 0
    assert job_queue.position(second) == 1
    assert job_queue.position(third) == 2

    job_queue._process_one()  # processes `first` (oldest)
    assert job_queue.status(first)["status"] == "done"
    assert job_queue.position(second) == 0
    assert job_queue.position(third) == 1

    job_queue._process_one()  # processes `second`
    assert job_queue.status(second)["status"] == "done"
    assert job_queue.position(third) == 0


def test_failed_job_records_error():
    job_id = job_queue.submit(np.zeros((128, 128), np.uint8), resolution_m=0.5)  # empty -> no roads

    job_queue._process_one()

    state = job_queue.status(job_id)
    assert state["status"] == "failed"
    assert state["error"]  # non-empty message
    with pytest.raises(RuntimeError):
        job_queue.result(job_id)


def test_recover_stale_running_requeues():
    job_id = job_queue.submit(_grid_mask(), resolution_m=0.5)
    state = job_queue.status(job_id)
    # Simulate a process that died mid-job: state left at "running".
    state["status"] = "running"
    state["started_utc"] = job_queue._now()
    job_queue._write_state(job_id, state)

    recovered = job_queue._recover_stale_running()
    assert recovered == 1
    assert job_queue.status(job_id)["status"] == "queued"

    # And it's processable again afterwards.
    assert job_queue._process_one() is True
    assert job_queue.status(job_id)["status"] == "done"


def test_live_lease_is_not_recovered_or_double_claimed():
    job_id = job_queue.submit(_grid_mask(), resolution_m=0.5)
    state = job_queue.status(job_id)
    state.update({
        "status": "running",
        "started_utc": job_queue._now(),
        "owner_pid": os.getpid(),
        "lease_expires_utc": (
            datetime.now(timezone.utc) + timedelta(minutes=5)
        ).isoformat(timespec="seconds"),
    })
    job_queue._write_state(job_id, state)
    job_queue._claim_path(job_id).touch()

    assert job_queue._recover_stale_running() == 0
    assert job_queue.status(job_id)["status"] == "running"
    assert job_queue._process_one() is False


def test_live_owner_is_not_recovered_after_fixed_lease_expires():
    job_id = job_queue.submit(_grid_mask(), resolution_m=0.5)
    state = job_queue.status(job_id)
    state.update({
        "status": "running",
        "started_utc": job_queue._now(),
        "owner_pid": os.getpid(),
        "lease_expires_utc": (
            datetime.now(timezone.utc) - timedelta(minutes=5)
        ).isoformat(timespec="seconds"),
    })
    job_queue._write_state(job_id, state)
    job_queue._claim_path(job_id).touch()

    assert job_queue._recover_stale_running() == 0
    assert job_queue.status(job_id)["status"] == "running"
    assert job_queue._process_one() is False


def test_stale_sequence_lock_is_recovered(monkeypatch):
    job_queue.JOBS_DIR.mkdir(parents=True)
    lock = job_queue.JOBS_DIR / "sequence.lock"
    lock.touch()
    old = datetime.now(timezone.utc).timestamp() - 60
    import os
    os.utime(lock, (old, old))
    monkeypatch.setattr(job_queue, "SEQUENCE_LOCK_STALE_SECONDS", 0.01)
    assert job_queue._next_seq() == 0
    assert not lock.exists()


def test_invalid_lease_timestamp_with_dead_owner_is_recovered(monkeypatch):
    job_id = job_queue.submit(_grid_mask(), resolution_m=0.5)
    state = job_queue.status(job_id)
    state.update({"status": "running", "owner_pid": os.getpid(),
                  "lease_expires_utc": "not-an-iso-date"})
    job_queue._write_state(job_id, state)
    job_queue._claim_path(job_id).touch()
    monkeypatch.setattr(job_queue, "_pid_alive", lambda _pid: False)
    assert job_queue._recover_stale_running() == 1
    assert job_queue.status(job_id)["status"] == "queued"


def test_cleanup_removes_old_files():
    job_id = job_queue.submit(_grid_mask(), resolution_m=0.5)
    state = job_queue.status(job_id)
    # Backdate created_utc well past the default 24h window.
    old = datetime.now(timezone.utc) - timedelta(hours=48)
    state["created_utc"] = old.isoformat(timespec="seconds")
    job_queue._write_state(job_id, state)

    removed = job_queue.cleanup(max_age_h=24)
    assert removed == 1
    assert not job_queue._state_path(job_id).exists()
    assert not job_queue._mask_path(job_id).exists()

    # A fresh job survives the same sweep.
    fresh_id = job_queue.submit(_grid_mask(), resolution_m=0.5)
    assert job_queue.cleanup(max_age_h=24) == 0
    assert job_queue._state_path(fresh_id).exists()
