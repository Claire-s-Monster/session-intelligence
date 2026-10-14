"""Regression tests for #216: opt-in ``si.projectName`` override.

Linked worktrees keep deriving the bare remote basename; only an explicit git
config value changes the name. Uses real temporary git repositories.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from core.project_naming import _derive_cached, derive_project_name

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")

REMOTE = "https://github.com/Claire-s-Monster/some-repo.git"


@pytest.fixture(autouse=True)
def _clear_derive_cache():
    _derive_cached.cache_clear()
    yield
    _derive_cached.cache_clear()


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", *args],  # noqa: S607
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    main = tmp_path / "development"
    main.mkdir()
    _git(main, "init", "-q")
    _git(main, "config", "user.email", "t@example.com")
    _git(main, "config", "user.name", "t")
    _git(main, "remote", "add", "origin", REMOTE)
    _git(main, "commit", "-q", "--allow-empty", "-m", "init")
    return main


@pytest.fixture
def linked(repo: Path, tmp_path: Path) -> Path:
    wt = tmp_path / "wt-feature"
    _git(repo, "worktree", "add", "-q", "-b", "feat", str(wt))
    return wt


def test_no_override_uses_remote_basename(repo: Path) -> None:
    assert derive_project_name(repo) == "some-repo"


def test_repo_wide_override(repo: Path) -> None:
    _git(repo, "config", "si.projectName", "custom-name")
    assert derive_project_name(repo) == "custom-name"


def test_linked_worktree_shares_bare_name(repo: Path, linked: Path) -> None:
    assert derive_project_name(linked) == "some-repo"
    assert derive_project_name(repo) == "some-repo"


def test_per_worktree_override_is_isolated(repo: Path, linked: Path) -> None:
    _git(repo, "config", "extensions.worktreeConfig", "true")
    _git(linked, "config", "--worktree", "si.projectName", "wt-only")
    assert derive_project_name(linked) == "wt-only"
    assert derive_project_name(repo) == "some-repo"


@pytest.mark.parametrize("bad", ["../x", "has space", "a/b", "-lead", "x" * 101])
def test_invalid_override_ignored(repo: Path, bad: str) -> None:
    _git(repo, "config", "si.projectName", bad)
    assert derive_project_name(repo) == "some-repo"


def test_whitespace_only_override_ignored(repo: Path) -> None:
    _git(repo, "config", "si.projectName", "   ")
    assert derive_project_name(repo) == "some-repo"


def test_non_git_dir_unaffected(tmp_path: Path) -> None:
    plain = tmp_path / "plainfolder"
    plain.mkdir()
    assert derive_project_name(plain) == "plainfolder"
