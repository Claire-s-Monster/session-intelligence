"""Derive a stable ``project_name`` from a filesystem path.

Historically, sessions created without an explicit ``project_name`` were filed
under the ``_unbound_`` sentinel. Because ``session_recall`` and
``session_search`` are project-scoped, those sessions became invisible: as of
2026-08-04 that was 5,797 of 7,286 rows (79.6%) in the production database.

Both creation sites that produced the sentinel already had a ``project_path``
in hand, so the name is recoverable. This module does the recovery.

Derivation order (first non-empty wins):

1. ``git remote get-url origin`` -> repository basename
2. ``git rev-parse --show-toplevel`` -> directory basename
3. ``Path(project_path).name``
4. ``_unbound_`` (only when the path is unusable and git tells us nothing)

Step 1 leads deliberately. The names already in the database
(``numba-feedstock``, ``hb-event-bus``, ``staged-recipes``,
``session-intelligence``) are *repository* names, and a checkout is frequently
rooted in a directory that describes its role rather than the project -- for
this repo, ``.../session-intelligence/development``, whose basename is
``development``. Steps 2 and 3 are last-resort fallbacks for repositories with
no remote; they are intentionally literal rather than trying to guess which
directory names are role descriptors.

Every git call is best-effort: failures, timeouts, and a missing ``git``
binary all fall through to the next step rather than propagating.
"""

from __future__ import annotations

import logging
import subprocess
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

#: Sentinel used when no name can be derived. Rows carrying this value are
#: invisible to project-scoped recall, which is the bug this module exists to
#: stop reproducing.
UNBOUND = "_unbound_"

#: Git is called on the session-creation hot path, so keep it short. A stalled
#: git call must never become a stalled session write.
_GIT_TIMEOUT_S = 2.0


def _run_git(args: list[str], cwd: Path) -> str | None:
    """Run a git command, returning stripped stdout or None on any failure."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # Missing binary, timeout, permission error -- all non-fatal.
        logger.debug("git %s failed in %s: %s", " ".join(args), cwd, exc)
        return None

    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _sanitize(name: str | None) -> str | None:
    """Reject names that would be useless or misleading as a project scope."""
    if not name:
        return None
    name = name.strip().strip("/")
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        return None
    return name


def repo_name_from_remote_url(url: str) -> str | None:
    """Extract the repository basename from any git remote URL form.

    Handles the three shapes git actually emits::

        https://github.com/Claire-s-Monster/session-intelligence.git
        git@github.com:Claire-s-Monster/session-intelligence.git
        /srv/git/session-intelligence.git

    all of which yield ``session-intelligence``.
    """
    url = (url or "").strip()
    if not url:
        return None

    # Drop any query string or fragment before looking at the tail.
    url = url.split("?", 1)[0].split("#", 1)[0].rstrip("/")

    tail = url.rsplit("/", 1)[-1]
    # scp-like remotes with no path component, e.g. "git@host:repo.git".
    if "/" not in url and ":" in tail:
        tail = tail.rsplit(":", 1)[-1]

    if tail.endswith(".git"):
        tail = tail[: -len(".git")]

    return _sanitize(tail)


@lru_cache(maxsize=256)
def _derive_cached(resolved: str) -> str:
    """Cached derivation keyed on an already-resolved directory path.

    Cached because session creation can happen many times against the same
    working directory and each miss costs up to two subprocess spawns. A
    process-lifetime cache is acceptable: a repository's remote changing
    mid-process is not a case worth paying for on every write.
    """
    path = Path(resolved)

    remote_url = _run_git(["remote", "get-url", "origin"], path)
    if remote_url:
        name = repo_name_from_remote_url(remote_url)
        if name:
            return name

    toplevel = _run_git(["rev-parse", "--show-toplevel"], path)
    if toplevel:
        name = _sanitize(Path(toplevel).name)
        if name:
            return name

    name = _sanitize(path.name)
    if name:
        return name

    return UNBOUND


def derive_project_name(project_path: str | Path | None) -> str:
    """Best-effort project name for ``project_path``.

    Never raises and never returns an empty string; callers can use the result
    directly as a ``project_name``. Returns :data:`UNBOUND` only when the path
    is missing or yields nothing usable.
    """
    if not project_path:
        return UNBOUND

    try:
        path = Path(project_path).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        logger.debug("could not resolve project_path %r: %s", project_path, exc)
        return UNBOUND

    # git needs an existing directory to run in. Probe the path itself, or its
    # immediate parent when the path names a file.
    #
    # Deliberately do NOT walk further up. The backfill migration feeds this
    # function thousands of *historical* project_path values, many pointing at
    # directories that have since been deleted. An arbitrarily distant ancestor
    # belongs to a different project: walking up from a stale ~/work/gone/repo
    # would reach the home directory and derive "memento", confidently
    # mislabelling those rows instead of leaving them honestly unbound. One
    # level covers a path that names a file; past that, fall back to the path's
    # own basename, which is what the row itself claims.
    probe = path if path.is_dir() else path.parent

    if not probe.is_dir():
        return _sanitize(path.name) or UNBOUND

    try:
        return _derive_cached(str(probe))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("project name derivation failed for %s: %s", probe, exc)
        return _sanitize(path.name) or UNBOUND
