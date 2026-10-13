"""Tests for scripts/purge_progress_summarizer_rows.py (issue #137)."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from persistence.config import DatabaseConfig
from persistence.sqlite import SQLiteBackend

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "purge_progress_summarizer_rows.py"
spec = importlib.util.spec_from_file_location("purge_progress_summarizer_rows", SCRIPT)
purge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(purge)

T0 = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
STEP = [{"step": "summary"}]


def _row(name, agent_type, status, steps, seconds, sid="s1"):
    completed = None if seconds is None else (T0 + timedelta(seconds=seconds)).isoformat()
    return {
        "id": f"id-{name}",
        "session_id": sid,
        "agent_name": name,
        "agent_type": agent_type,
        "status": status,
        "started_at": T0.isoformat(),
        "completed_at": completed,
        "execution_steps": steps,
    }


ROWS = {
    "A": _row("abcdef0123", "unknown", "success", STEP, 1),
    "A_null": _row("abcdef0124", None, "success", STEP, 0),
    "taskmgr": _row("task-manager", "unknown", "abandoned", STEP * 3, 500),
    "typed": _row("bash-executor", "domain", "success", STEP, 1),
    "B": _row("0123abcdef", "", "abandoned", STEP, 3600),
    "two_steps": _row("abcdef0125", "unknown", "success", STEP * 2, 1),
    "slow": _row("abcdef0126", "unknown", "success", STEP, 10),
}


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("A", "A"),
        ("A_null", "A"),
        ("taskmgr", None),
        ("typed", None),
        ("B", "B"),
        ("two_steps", None),
        ("slow", None),
    ],
)
def test_classify(key, expected):
    row = dict(ROWS[key])
    row["started_at"] = datetime.fromisoformat(row["started_at"])
    row["completed_at"] = datetime.fromisoformat(row["completed_at"])
    assert purge.classify(row) == expected


@pytest.fixture
async def db_path(tmp_path):
    path = tmp_path / "t.db"
    backend = SQLiteBackend(str(path))
    await backend.initialize()
    await backend.close()
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO sessions (id, started_at, project_path, project_name, performance_metrics)"
        " VALUES ('s1', ?, '/p', 'proj', ?)",
        (T0.isoformat(), json.dumps({"agents_executed": 7, "other": 1})),
    )
    for r in ROWS.values():
        conn.execute(
            "INSERT INTO agent_executions (id, session_id, agent_name, agent_type, started_at,"
            " completed_at, status, execution_steps) VALUES (?,?,?,?,?,?,?,?)",
            (
                r["id"],
                r["session_id"],
                r["agent_name"],
                r["agent_type"],
                r["started_at"],
                r["completed_at"],
                r["status"],
                json.dumps(r["execution_steps"]),
            ),
        )
    conn.commit()
    conn.close()
    return path


def _ids(path):
    conn = sqlite3.connect(path)
    ids = {r[0] for r in conn.execute("SELECT id FROM agent_executions")}
    metrics = json.loads(conn.execute("SELECT performance_metrics FROM sessions").fetchone()[0])
    conn.close()
    return ids, metrics


ALL_IDS = {r["id"] for r in ROWS.values()}
A_IDS = {ROWS["A"]["id"], ROWS["A_null"]["id"]}


def test_dry_run_deletes_nothing(db_path, capsys):
    assert purge.main(["--dsn", str(db_path)]) == 0
    assert _ids(db_path)[0] == ALL_IDS
    assert "proj" in capsys.readouterr().out


def test_apply_deletes_bucket_a_and_recomputes(db_path):
    assert purge.main(["--dsn", str(db_path), "--apply"]) == 0
    ids, metrics = _ids(db_path)
    assert ids == ALL_IDS - A_IDS
    assert metrics == {"agents_executed": 5, "other": 1}


def test_apply_with_flag_deletes_bucket_b(db_path):
    args = ["--dsn", f"sqlite:///{db_path}", "--apply", "--include-abandoned-typeless"]
    assert purge.main(args) == 0
    ids, metrics = _ids(db_path)
    assert ids == ALL_IDS - A_IDS - {ROWS["B"]["id"]}
    assert metrics["agents_executed"] == 4


def test_unresolvable_dsn_fails(tmp_path, monkeypatch, db_path):
    monkeypatch.setattr(DatabaseConfig, "load", classmethod(lambda cls: cls()))
    monkeypatch.delenv("SESSION_DB_DSN", raising=False)
    assert purge.main(["--apply"]) != 0
    assert purge.main(["--apply", "--dsn", str(tmp_path / "missing.db")]) != 0
    assert _ids(db_path)[0] == ALL_IDS
