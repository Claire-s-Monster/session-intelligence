"""
Regression tests for issue #174: SubagentStart/SubagentStop hooks lost their
session_track_execution POST whenever the server was restarting, leaving the
execution RUNNING until the startup sweep reaped it ABANDONED.

Fix: a stdlib-only durable outbox (`src/hooks/delivery_outbox.py`). Failed
payloads are spooled to a JSONL file and replayed, in order, by later hook
runs.
"""

from __future__ import annotations

import ast
import fcntl
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hooks import delivery_outbox
from hooks.delivery_outbox import MAX_ENTRIES, drain, spool


def _payloads(path: Path) -> list[dict]:
    return [json.loads(line)["payload"] for line in path.read_text().splitlines()]


def _recorder(fail_on: int | None = None, raise_on: int | None = None):
    """Return (send, sent). Fails (False) or raises on the Nth call (0-based)."""
    sent: list[dict] = []

    def send(payload: dict) -> bool:
        idx = len(sent)
        if raise_on is not None and idx == raise_on:
            raise ConnectionRefusedError("down")
        if fail_on is not None and idx >= fail_on:
            sent.append(payload)  # attempted
            return False
        sent.append(payload)
        return True

    return send, sent


@pytest.mark.regression
def test_spool_drain_round_trip(tmp_path):
    path = tmp_path / "sub" / "dir" / "outbox.jsonl"
    assert spool({"a": 1}, path) is True
    line = json.loads(path.read_text().splitlines()[0])
    assert line["payload"] == {"a": 1}
    datetime.fromisoformat(line["spooled_at"])

    send, sent = _recorder()
    result = drain(send, path)

    assert sent == [{"a": 1}]
    assert result == {"delivered": 1, "remaining": 0, "expired": 0, "malformed": 0}
    assert path.read_text() == ""


@pytest.mark.regression
def test_drain_preserves_fifo_order(tmp_path):
    path = tmp_path / "outbox.jsonl"
    for i in range(5):
        spool({"n": i}, path)
    send, sent = _recorder()

    drain(send, path)

    assert [p["n"] for p in sent] == [0, 1, 2, 3, 4]


@pytest.mark.regression
def test_drain_stops_at_first_failure_and_keeps_remainder_in_order(tmp_path):
    path = tmp_path / "outbox.jsonl"
    for i in range(5):
        spool({"n": i}, path)
    send, sent = _recorder(fail_on=2)

    result = drain(send, path)

    assert len(sent) == 3  # 0, 1 delivered; 2 attempted and failed; 3, 4 never tried
    assert result["delivered"] == 2
    assert result["remaining"] == 3
    assert [p["n"] for p in _payloads(path)] == [2, 3, 4]


@pytest.mark.regression
def test_send_exception_is_a_failure(tmp_path):
    path = tmp_path / "outbox.jsonl"
    for i in range(3):
        spool({"n": i}, path)
    send, _ = _recorder(raise_on=1)

    result = drain(send, path)

    assert result["delivered"] == 1
    assert result["remaining"] == 2
    assert "error" not in result
    assert [p["n"] for p in _payloads(path)] == [1, 2]


@pytest.mark.regression
def test_expired_and_malformed_entries_are_dropped_and_counted(tmp_path):
    path = tmp_path / "outbox.jsonl"
    old = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    fresh = datetime.now(UTC).isoformat()
    lines = [
        json.dumps({"spooled_at": old, "payload": {"n": "old"}}),
        "this is not json",
        json.dumps({"spooled_at": fresh}),  # no payload
        json.dumps({"spooled_at": fresh, "payload": {"n": "ok"}}),
    ]
    path.write_text("\n".join(lines) + "\n")
    send, sent = _recorder()

    result = drain(send, path)

    assert sent == [{"n": "ok"}]
    assert result == {"delivered": 1, "remaining": 0, "expired": 1, "malformed": 2}
    assert path.read_text() == ""


@pytest.mark.regression
def test_drain_skipped_when_lock_held_elsewhere(tmp_path):
    path = tmp_path / "outbox.jsonl"
    spool({"n": 1}, path)
    send, sent = _recorder()

    with open(path, "a") as other:
        fcntl.flock(other.fileno(), fcntl.LOCK_EX)
        result = drain(send, path)
        fcntl.flock(other.fileno(), fcntl.LOCK_UN)

    assert result == {"skipped": "locked"}
    assert sent == []
    assert _payloads(path) == [{"n": 1}]


@pytest.mark.regression
def test_cap_drops_oldest_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(delivery_outbox, "MAX_ENTRIES", 5)
    path = tmp_path / "outbox.jsonl"
    for i in range(8):
        assert spool({"n": i}, path) is True

    assert [p["n"] for p in _payloads(path)] == [3, 4, 5, 6, 7]


@pytest.mark.regression
def test_spool_retries_when_file_replaced_while_waiting_for_lock(tmp_path, monkeypatch):
    """drain() swaps the file's inode; a spooler holding the old inode must
    retry on the new file instead of writing into the orphaned one."""
    path = tmp_path / "outbox.jsonl"
    spool({"n": "existing"}, path)
    real_flock = fcntl.flock
    swapped: list[bool] = []

    def flock_then_swap(fd, op):
        if not swapped:
            # The handle is already open on the old inode; replace the file
            # (new inode) before the lock is granted, as drain() does.
            swapped.append(True)
            replacement = tmp_path / "replacement.tmp"
            replacement.write_text(path.read_text())
            os.replace(replacement, path)
        return real_flock(fd, op)

    monkeypatch.setattr(delivery_outbox.fcntl, "flock", flock_then_swap)

    assert spool({"n": "late"}, path) is True

    assert swapped == [True]
    assert [p["n"] for p in _payloads(path)] == ["existing", "late"]


@pytest.mark.regression
def test_max_entries_default_is_1000():
    assert MAX_ENTRIES == 1000


@pytest.mark.regression
def test_spool_unwritable_path_returns_false(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    # parent "directory" is a regular file -> cannot create/open
    assert spool({"a": 1}, blocker / "outbox.jsonl") is False


@pytest.mark.regression
def test_drain_unreadable_path_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    send, _ = _recorder()
    result = drain(send, blocker / "outbox.jsonl")
    assert isinstance(result, dict)


@pytest.mark.regression
def test_drain_missing_file_is_empty_noop(tmp_path):
    send, sent = _recorder()
    result = drain(send, tmp_path / "nope.jsonl")
    assert sent == []
    assert result.get("delivered", 0) == 0
    assert result.get("remaining", 0) == 0


@pytest.mark.regression
def test_env_var_overrides_default_path(tmp_path, monkeypatch):
    target = tmp_path / "env_outbox.jsonl"
    monkeypatch.setenv("SESSION_INTELLIGENCE_HOOK_OUTBOX", str(target))

    assert spool({"via": "env"}) is True
    assert _payloads(target) == [{"via": "env"}]

    send, sent = _recorder()
    drain(send)
    assert sent == [{"via": "env"}]


@pytest.mark.regression
def test_default_path_is_under_claude_dir(monkeypatch):
    monkeypatch.delenv("SESSION_INTELLIGENCE_HOOK_OUTBOX", raising=False)
    default = Path(delivery_outbox.default_path())
    assert default.name == "hook_outbox.jsonl"
    assert default.parent.name == "session-intelligence"
    assert default.parent.parent.name == ".claude"


@pytest.mark.regression
def test_drain_budget_exhausted_keeps_remainder(tmp_path):
    path = tmp_path / "outbox.jsonl"
    for i in range(3):
        spool({"n": i}, path)
    send, sent = _recorder()

    result = drain(send, path, budget_s=0.0)

    assert result["remaining"] == 3 - result["delivered"]
    assert len(_payloads(path)) == result["remaining"]
    assert [p["n"] for p in sent] == list(range(len(sent)))


_STDLIB_ALLOWLIST = {
    "__future__",
    "collections",
    "datetime",
    "fcntl",
    "json",
    "logging",
    "os",
    "pathlib",
    "tempfile",
    "time",
    "typing",
}


@pytest.mark.regression
def test_delivery_outbox_module_is_stdlib_only():
    """The hook runs under plain python3 outside pixi: stdlib imports only,
    no `src.` / relative imports back into this repo."""
    module_path = Path(__file__).parent.parent.parent / "src" / "hooks" / "delivery_outbox.py"
    tree = ast.parse(module_path.read_text(), filename=str(module_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split(".")[0]
                assert top_level in _STDLIB_ALLOWLIST, f"non-stdlib import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"relative import not allowed: level={node.level}"
            assert node.module is not None, "bare relative import not allowed"
            top_level = node.module.split(".")[0]
            assert top_level in _STDLIB_ALLOWLIST, f"non-stdlib import: {node.module}"
