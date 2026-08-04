"""Unit tests for core.project_naming.

Covers the pure helpers (repo_name_from_remote_url, derive_project_name) that
recover a usable project_name from a filesystem path, replacing the
"_unbound_" sentinel that made 79.6% of production session rows invisible to
project-scoped recall.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import core.project_naming as project_naming
from core.project_naming import (
    UNBOUND,
    _derive_cached,
    derive_project_name,
    repo_name_from_remote_url,
)


@pytest.fixture(autouse=True)
def _clear_derive_cache():
    """Prevent lru_cache leakage between tests.

    _derive_cached is keyed on a resolved directory path; without clearing it
    a result cached by one test (e.g. a monkeypatched _run_git) would silently
    satisfy a later test against the same path, making pass/fail depend on
    test order.
    """
    _derive_cached.cache_clear()
    yield
    _derive_cached.cache_clear()


# ---------------------------------------------------------------------------
# repo_name_from_remote_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://github.com/Claire-s-Monster/session-intelligence.git",
            "session-intelligence",
        ),
        (
            "https://github.com/Claire-s-Monster/session-intelligence",
            "session-intelligence",
        ),
        (
            "git@github.com:Claire-s-Monster/session-intelligence.git",
            "session-intelligence",
        ),
        (
            "ssh://git@github.com/Claire-s-Monster/session-intelligence.git",
            "session-intelligence",
        ),
        (
            "/srv/git/session-intelligence.git",
            "session-intelligence",
        ),
        (
            "https://github.com/Claire-s-Monster/session-intelligence.git/",
            "session-intelligence",
        ),
        ("", None),
        ("   ", None),
    ],
)
def test_repo_name_from_remote_url(url: str, expected: str | None) -> None:
    assert repo_name_from_remote_url(url) == expected


# ---------------------------------------------------------------------------
# derive_project_name - derivation order, with _run_git monkeypatched
# ---------------------------------------------------------------------------


def test_derive_project_name_remote_wins_and_toplevel_not_relied_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a remote URL is present, its repo name wins and the toplevel
    git call is never made (short-circuit)."""
    calls: list[list[str]] = []

    def fake_run_git(args: list[str], cwd: Path) -> str | None:
        calls.append(args)
        if args[0] == "remote":
            return "https://github.com/Claire-s-Monster/session-intelligence.git"
        raise AssertionError(
            f"toplevel git call {args!r} should not have been made when "
            "a remote URL already resolved a name"
        )

    monkeypatch.setattr(project_naming, "_run_git", fake_run_git)

    result = derive_project_name(str(tmp_path))

    assert result == "session-intelligence"
    assert len(calls) == 1
    assert calls[0][0] == "remote"


def test_derive_project_name_falls_back_to_toplevel_when_remote_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No remote origin -> falls back to `git rev-parse --show-toplevel`
    basename."""

    def fake_run_git(args: list[str], cwd: Path) -> str | None:
        if args[0] == "remote":
            return None
        if args[0] == "rev-parse":
            return "/some/checkout/root/my-toplevel-repo"
        raise AssertionError(f"unexpected git call: {args!r}")

    monkeypatch.setattr(project_naming, "_run_git", fake_run_git)

    result = derive_project_name(str(tmp_path))

    assert result == "my-toplevel-repo"


def test_derive_project_name_falls_back_to_path_basename_when_git_yields_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both git calls fail -> falls back to the path's own basename."""
    monkeypatch.setattr(project_naming, "_run_git", lambda args, cwd: None)

    result = derive_project_name(str(tmp_path))

    assert result == tmp_path.name


@pytest.mark.parametrize("bad_input", [None, ""])
def test_derive_project_name_none_or_empty_input_is_unbound(
    bad_input: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_run_git(args: list[str], cwd: Path) -> str | None:
        raise AssertionError("git must not be invoked for a falsy project_path")

    monkeypatch.setattr(project_naming, "_run_git", fail_run_git)

    assert derive_project_name(bad_input) == UNBOUND


def test_derive_project_name_nonexistent_path_falls_back_to_own_basename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path that doesn't exist on disk derives a name from its own
    basename rather than walking up an unbounded chain of parents.

    The walk-up is bounded to at most one level (the path itself, or its
    immediate parent if the path names a file) because the backfill
    migration feeds this function thousands of historical paths whose
    directories no longer exist. An unbounded walk-up would keep climbing
    until it reached the home directory and mislabel all of those rows
    under the same ancestor name."""
    monkeypatch.setattr(project_naming, "_run_git", lambda args, cwd: None)

    missing = tmp_path / "does" / "not" / "exist"

    result = derive_project_name(str(missing))

    assert result == "exist"
    assert result != UNBOUND


# ---------------------------------------------------------------------------
# Integration: real git, real repo (no monkeypatching)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not shutil.which("git"), reason="git binary not available")
@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".git").exists(),
    reason="not a git checkout",
)
def test_derive_project_name_real_repo_resolves_to_session_intelligence() -> None:
    """Resolve the repo root relative to this test file's own location
    (tests/unit/<file> -> repo root) rather than hardcoding a developer
    machine's absolute path, so this test is portable across local
    machines and CI runners alike. On CI, actions/checkout leaves an
    'origin' remote pointing at the repo, and git-remote is the first
    step of the derivation chain, so the assertion holds in both
    environments."""
    repo_root = Path(__file__).resolve().parents[2]
    result = derive_project_name(str(repo_root))
    assert result == "session-intelligence"
