"""Two rollouts must not append to one zarr; a dead one must not block its own resume.

Both halves matter. Concurrent appends silently duplicate or clobber rows, and these runs
routinely hit a 12 h wall clock and resume from their own output - so a lock left behind by
a killed process must never prevent the restart it exists to enable.
"""

import json
import os
import socket
import time

import pytest

from weathergen.rollout.output_lock import OutputLock, OutputLockError


def _write_lock(path, **over):
    info = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "timestamp": time.time(),
        "run_id": "someone_else",
    }
    info.update(over)
    path.write_text(json.dumps(info))


def test_acquire_and_release(tmp_path):
    out = tmp_path / "amip_x.zarr"
    lock = OutputLock(out, run_id="r1").acquire()
    assert lock.path.exists()
    info = json.loads(lock.path.read_text())
    assert info["pid"] == os.getpid() and info["run_id"] == "r1"
    lock.release()
    assert not lock.path.exists()


def test_live_holder_blocks_with_an_actionable_message(tmp_path):
    """This is the case that nearly corrupted a control rollout."""
    out = tmp_path / "amip_x.zarr"
    _write_lock(OutputLock(out).path)  # our own pid: definitely alive
    with pytest.raises(OutputLockError, match="already being written"):
        OutputLock(out, run_id="mine").acquire()


def test_own_run_reclaims_its_lock(tmp_path):
    """A 12 h-wall kill must not stop the run resuming its own output."""
    out = tmp_path / "amip_x.zarr"
    _write_lock(OutputLock(out).path, run_id="r1", pid=os.getpid())
    OutputLock(out, run_id="r1").acquire()  # same run id -> allowed


def test_dead_pid_on_this_host_is_stale(tmp_path):
    out = tmp_path / "amip_x.zarr"
    # PID 2**22 is above any Linux pid_max default, so it cannot be running
    _write_lock(OutputLock(out).path, pid=2**22)
    OutputLock(out, run_id="mine").acquire()


def test_old_lock_from_another_host_is_stale(tmp_path):
    out = tmp_path / "amip_x.zarr"
    _write_lock(OutputLock(out).path, hostname="some-other-node", timestamp=time.time() - 40 * 3600)
    OutputLock(out, run_id="mine", max_age_hours=24).acquire()


def test_recent_lock_from_another_host_still_blocks(tmp_path):
    """We cannot check a remote pid, so a fresh remote lock must be respected."""
    out = tmp_path / "amip_x.zarr"
    _write_lock(OutputLock(out).path, hostname="some-other-node", timestamp=time.time() - 60)
    with pytest.raises(OutputLockError):
        OutputLock(out, run_id="mine", max_age_hours=24).acquire()


def test_corrupt_lock_does_not_block_forever(tmp_path):
    out = tmp_path / "amip_x.zarr"
    OutputLock(out).path.write_text("{ this is not json")
    OutputLock(out, run_id="mine").acquire()


def test_refresh_keeps_a_long_run_from_looking_dead(tmp_path):
    out = tmp_path / "amip_x.zarr"
    lock = OutputLock(out, run_id="r1").acquire()
    old = json.loads(lock.path.read_text())["timestamp"]
    time.sleep(0.01)
    lock.refresh()
    assert json.loads(lock.path.read_text())["timestamp"] > old


def test_context_manager_releases_on_error(tmp_path):
    out = tmp_path / "amip_x.zarr"
    lock = OutputLock(out, run_id="r1")
    with pytest.raises(ValueError):
        with lock:
            raise ValueError("boom")
    assert not lock.path.exists()
