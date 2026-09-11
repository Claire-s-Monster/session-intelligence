"""Regression test for the "_unbound_" session/project binding bug.

Both hook-driven session auto-create paths in SessionIntelligenceEngine used
to file sessions under project_name="_unbound_" whenever they had to invent
a session on the fly (no explicit project_name/session_id resolution
context). Because session_recall / session_search are project-scoped, those
sessions became permanently invisible: as of 2026-08-04 that was 5,797 of
7,286 rows (79.6%) in the production database.

Only the hook-bound site (session_engine.py ~line 886, caller-supplied
working_directory via step_data) now derives a real project_name via
core.project_naming.derive_project_name(). The legacy
_get_or_create_current_session_id auto path (~line 408) deliberately
retains the "_unbound_" sentinel: it only has the server process's own cwd
available -- which under the systemd HTTP deployment is pinned to the
session-intelligence checkout, not the caller's project -- and allow_unbound
callers depend on this sentinel being stable
(see tests/test_session_recall_project_binding.py::
test_log_without_create_with_allow_unbound_uses_sentinel).

This module exercises both sites through their real public entry point
(session_track_execution), not the private _create_session helper directly,
so it actually reproduces the regression and pins the intended fix.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import core.project_naming as project_naming
from core.project_naming import UNBOUND


@pytest.fixture(autouse=True)
def _clear_derive_cache():
    """See tests/unit/test_project_naming.py for why this is required:
    _derive_cached is a process-lifetime lru_cache keyed on resolved path,
    and leaks results across tests without this."""
    project_naming._derive_cached.cache_clear()
    yield
    project_naming._derive_cached.cache_clear()


@pytest.fixture
def fake_remote_git(monkeypatch: pytest.MonkeyPatch):
    """Monkeypatch _run_git so derivation is hermetic: it returns a known
    remote URL instead of depending on the host's actual git state."""

    def fake_run_git(args: list[str], cwd: Path) -> str | None:
        if args[0] == "remote":
            return "https://github.com/Claire-s-Monster/session-intelligence.git"
        return None

    monkeypatch.setattr(project_naming, "_run_git", fake_run_git)
    return fake_run_git


# ---------------------------------------------------------------------------
# Site 1: hook-bound auto-create inside the execution-tracking path
# (session_engine.py ~line 872) -- produced 5,797/7,286 _unbound_ rows.
# ---------------------------------------------------------------------------


async def test_hook_bound_execution_tracking_does_not_produce_unbound_session(
    session_engine, tmp_path: Path, fake_remote_git
) -> None:
    """Drives the real regressed path: session_track_execution() called with
    a session_id the engine has never seen (as a hook supplying Claude
    Code's native subagent session UUID would), forcing the hook-bound
    auto-create branch in _track_execution_sync.
    """
    working_dir = tmp_path / "hook-workdir"
    working_dir.mkdir()

    result = await session_engine.session_track_execution(
        session_id="claude-native-session-abc123",
        agent_name="test-agent",
        step_data={"working_directory": str(working_dir)},
    )

    assert result.status == "success", f"unexpected status: {result.status}"
    assert result.session_id in session_engine.session_cache

    session = session_engine.session_cache[result.session_id]

    assert session.project_name != UNBOUND, (
        "REGRESSION: hook-bound auto-create must not file the session "
        "under the '_unbound_' sentinel -- this is the production bug "
        "that made 5,797/7,286 session rows invisible to project-scoped "
        "session_recall"
    )
    assert session.project_name == "session-intelligence"
    assert session.project_path == str(
        working_dir
    ), "project_path must stay consistent with the derived name's source dir"


# ---------------------------------------------------------------------------
# Site 2 (legacy): _get_or_create_current_session_id auto path
# (session_engine.py ~line 408), driven indirectly through
# session_track_execution(session_id=None). This path deliberately keeps the
# "_unbound_" sentinel -- see module docstring.
# ---------------------------------------------------------------------------


async def test_auto_current_session_id_still_uses_unbound_sentinel(
    session_engine, tmp_path: Path, fake_remote_git, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No session_id supplied and no existing current session -> engine
    auto-creates one via _get_or_create_current_session_id. Even though
    fake_remote_git makes derivation available and would return a real name
    if this path were wired to use it, it must still stamp '_unbound_':
    this path only has the server process's own cwd, not a caller-supplied
    working directory, and allow_unbound=True callers rely on this sentinel
    being stable.

    allow_unbound=True is passed explicitly (issue #77): session_id=None
    with no session_name/project_name now requires it, otherwise
    SessionContextRequiredError is raised instead of silently reaching this
    ambient fallback."""
    working_dir = tmp_path / "auto-workdir"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)

    result = await session_engine.session_track_execution(
        session_id=None,
        agent_name="test-agent",
        step_data={},
        allow_unbound=True,
    )

    assert result.status == "success", f"unexpected status: {result.status}"
    assert result.session_id in session_engine.session_cache

    session = session_engine.session_cache[result.session_id]

    assert session.project_name == UNBOUND, (
        "_get_or_create_current_session_id intentionally preserves the "
        "'_unbound_' sentinel: it only has the server-process cwd "
        "available (not a caller-supplied working directory), and "
        "allow_unbound=True callers depend on this sentinel remaining "
        "stable -- see test_session_recall_project_binding.py::"
        "test_log_without_create_with_allow_unbound_uses_sentinel"
    )
