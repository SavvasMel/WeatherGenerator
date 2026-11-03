"""Advisory lock file beside a rollout output store (one writer at a time)."""

import contextlib
import json
import logging
import os
import socket
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# older than this (and not refreshed) = stale
DEFAULT_MAX_AGE_HOURS = 3.0


class OutputLockError(RuntimeError):
    """Raised when another live process holds the output."""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


class OutputLock:
    """Advisory lock file beside a rollout output store."""

    def __init__(self, output_path, run_id: str = "", max_age_hours: float = DEFAULT_MAX_AGE_HOURS):
        self.path = Path(str(output_path).rstrip("/") + ".lock")
        self.run_id = str(run_id)
        self.max_age_hours = float(max_age_hours)
        self.acquired = False

    def _read(self) -> dict | None:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            # unreadable lock: treat as stale
            return None

    def _stale_reason(self, info: dict) -> str | None:
        if info.get("run_id") and info["run_id"] == self.run_id:
            return "it belongs to this same run (resuming its own output)"
        age_h = (time.time() - float(info.get("timestamp", 0))) / 3600.0
        if info.get("hostname") == socket.gethostname():
            if not _pid_alive(int(info.get("pid", -1))):
                return f"its process ({info.get('pid')}) is gone on this host"
        elif age_h > self.max_age_hours:
            return f"it is {age_h:.1f} h old on another host ({info.get('hostname')})"
        return None

    def acquire(self) -> "OutputLock":
        if self.path.exists():
            info = self._read()
            if info is None:
                logger.warning("unreadable lock at %s; taking it over", self.path)
            else:
                reason = self._stale_reason(info)
                if reason is None:
                    started = time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime(info.get("timestamp", 0))
                    )
                    raise OutputLockError(
                        f"{self.path.with_suffix('')} is already being written by "
                        f"pid {info.get('pid')} on {info.get('hostname')} "
                        f"(run {info.get('run_id') or 'unknown'}, started "
                        f"{started}). "
                        "Two processes appending to one zarr will duplicate or clobber rows. "
                        "Point rollout.output at a path under your own run id, or wait for "
                        "that run to finish."
                    )
                logger.warning("taking over lock at %s: %s", self.path, reason)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "timestamp": time.time(),
            "run_id": self.run_id,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(self.path)  # atomic
        self.acquired = True
        return self

    def refresh(self) -> None:
        """Re-stamp the lock so a long run is never mistaken for a dead one."""
        if not self.acquired:
            return
        try:
            info = self._read() or {}
            info.update({"timestamp": time.time(), "pid": os.getpid()})
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(info))
            tmp.replace(self.path)
        except OSError:
            pass  # a lock we cannot refresh is not worth failing a rollout over

    def release(self) -> None:
        if not self.acquired:
            return
        with contextlib.suppress(OSError):
            self.path.unlink()
        self.acquired = False

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False
