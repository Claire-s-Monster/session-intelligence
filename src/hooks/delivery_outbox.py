"""Durable outbox for hook -> server ``session_track_execution`` deliveries.

Issue #174: the SubagentStart/SubagentStop hooks POST ``session_track_execution``
to the server on :4002. While the server restarts the POST gets
ConnectionRefused; the hook used to log that and give up, so the execution
stayed RUNNING until the startup sweep reaped it ABANDONED.

``spool`` appends a failed payload to a local JSONL file; later hook runs call
``drain`` to replay the backlog in FIFO order (a spooled agent_start must reach
the server before its agent_stop). ``drain`` stops at the first failed
delivery -- the server is still down and continuing would reorder start/stop.

Neither function ever raises: the hook must not break the agent because of the
outbox.

STDLIB ONLY. This module is imported by the hooks in ``~/.claude/hooks``,
which run under plain ``python3`` OUTSIDE this project's pixi environment. It
must not import pydantic, any other third-party package, or anything from this
repo's own ``src`` package tree.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

ENV_VAR = "SESSION_INTELLIGENCE_HOOK_OUTBOX"
MAX_ENTRIES = 1000
DEFAULT_MAX_AGE_S = 7 * 24 * 3600


def default_path() -> Path:
    """Outbox location: ``$SESSION_INTELLIGENCE_HOOK_OUTBOX`` or the default."""
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override)
    return Path.home() / ".claude" / "session-intelligence" / "hook_outbox.jsonl"


def _resolve(path: str | os.PathLike | None) -> Path:
    return Path(path) if path is not None else default_path()


def _atomic_rewrite(path: Path, lines: list[str]) -> None:
    """Replace ``path`` with ``lines`` via temp file + os.replace (same dir)."""
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as tmp:
            tmp.write("".join(line + "\n" for line in lines))
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def spool(payload: dict, path: str | os.PathLike | None = None) -> bool:
    """Append ``payload`` to the outbox. Returns False on any error."""
    try:
        target = _resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        entry = {"spooled_at": datetime.now(UTC).isoformat(), "payload": payload}
        line = json.dumps(entry, default=str)
        # drain() replaces the file (new inode) while holding the lock; a
        # spooler that waited on the old inode must retry on the new one.
        for _attempt in range(10):
            if _append_locked(target, line):
                return True
        return False
    except Exception:
        return False


def _append_locked(target: Path, line: str) -> bool:
    """Append ``line`` under an exclusive lock; False if the file was swapped."""
    with open(target, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            if os.fstat(handle.fileno()).st_ino != os.stat(target).st_ino:
                return False
            handle.write(line + "\n")
            handle.flush()
            handle.seek(0)
            lines = [existing for existing in handle.read().splitlines() if existing.strip()]
            if len(lines) > MAX_ENTRIES:
                # Rewrite in place (same inode) so the held lock stays valid.
                handle.seek(0)
                handle.truncate()
                handle.write("".join(kept + "\n" for kept in lines[-MAX_ENTRIES:]))
                handle.flush()
            return True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _classify(lines: list[str], max_age_s: float) -> tuple[list[tuple[str, dict]], int, int]:
    """Split raw lines into live (line, payload) pairs plus expired/malformed counts."""
    live: list[tuple[str, dict]] = []
    expired = malformed = 0
    now = datetime.now(UTC)
    for line in lines:
        try:
            entry = json.loads(line)
            payload = entry["payload"]
            spooled_at = datetime.fromisoformat(entry["spooled_at"])
            if not isinstance(payload, dict):
                raise TypeError("payload is not an object")
            if spooled_at.tzinfo is None:
                spooled_at = spooled_at.replace(tzinfo=UTC)
        except Exception:
            malformed += 1
            continue
        if (now - spooled_at).total_seconds() > max_age_s:
            expired += 1
            continue
        live.append((line, payload))
    return live, expired, malformed


def drain(
    send: Callable[[dict], bool],
    path: str | os.PathLike | None = None,
    budget_s: float = 1.5,
    max_age_s: float = DEFAULT_MAX_AGE_S,
) -> dict:
    """Replay spooled payloads in FIFO order through ``send``.

    Stops at the first ``send`` that returns False or raises, or when
    ``budget_s`` is spent. Undelivered entries are written back atomically.
    """
    try:
        target = _resolve(path)
        if not target.exists():
            return {"delivered": 0, "remaining": 0, "expired": 0, "malformed": 0}
        deadline = time.monotonic() + budget_s
        with open(target, "a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return {"skipped": "locked"}
            try:
                handle.seek(0)
                lines = [raw for raw in handle.read().splitlines() if raw.strip()]
                live, expired, malformed = _classify(lines, max_age_s)

                delivered = 0
                for _line, payload in live:
                    if time.monotonic() >= deadline:
                        break
                    try:
                        ok = send(payload)
                    except Exception:
                        ok = False
                    if not ok:
                        break
                    delivered += 1

                remainder = [line for line, _payload in live[delivered:]]
                if delivered or expired or malformed:
                    _atomic_rewrite(target, remainder)
                return {
                    "delivered": delivered,
                    "remaining": len(remainder),
                    "expired": expired,
                    "malformed": malformed,
                }
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        return {"error": repr(e)}
