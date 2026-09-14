"""
Regression tests for issue #120: `query_project_learnings` cannot scope by
`project_name`, so learnings saved under the sentinel path `_unknown_`
(hook-bound sessions, see project_unbound_binding_fix.md) are invisible to
notebook rollups even though they belong to a known project.

The exact-path query at src/persistence/postgresql.py:1977 and
src/persistence/sqlite.py:2022 (`WHERE project_path = ?`) only ever matches
on the literal `project_path` column. A learning saved with
`project_path="_unknown_"` and a real `project_name` is therefore excluded
from any rollup keyed on the project's real path, even when the caller
supplies `project_name` as an additional, broader scope.

This file pins:
    - the CURRENT (buggy) exact-path-only behaviour (test 1), which must
      keep passing before and after a fix;
    - the DESIRED `project_name`-scoped behaviour (tests 2-3), which fails
      today with a `TypeError` because `query_project_learnings` does not
      accept a `project_name` keyword argument at all;
    - that the `exclude_superseded` and `since_days` filters, once
      `project_name` scoping lands, keep applying to BOTH branches of the
      path/name OR -- guarding against an unparenthesised OR that would
      bind the exclusion to only one branch (tests 4-5).
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

SENTINEL_PATH = "_unknown_"
REAL_PATH = "/real/path"


async def _backdate_learning(backend, learning_id: str, days: int) -> None:
    conn = backend._ensure_connected()
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    await conn.execute(
        "UPDATE project_learnings SET created_at = ? WHERE id = ?", (cutoff, learning_id)
    )
    await conn.commit()


@pytest.mark.regression
class TestIssue120ProjectNameScope:
    async def test_exact_path_query_misses_sentinel_path_learning(self, sqlite_backend):
        sentinel_id = str(uuid4())
        real_id = str(uuid4())

        await sqlite_backend.save_project_learning(
            learning_id=sentinel_id,
            project_path=SENTINEL_PATH,
            category="pattern",
            learning_content="Sentinel-path learning under proj-x",
            project_name="proj-x",
        )
        await sqlite_backend.save_project_learning(
            learning_id=real_id,
            project_path=REAL_PATH,
            category="pattern",
            learning_content="Real-path learning under proj-x",
            project_name="proj-x",
        )

        results = await sqlite_backend.query_project_learnings(REAL_PATH)
        ids = {r["id"] for r in results}

        assert real_id in ids, "issue #120: exact-path query should still return the real-path row"
        assert sentinel_id not in ids, (
            "issue #120: documents CURRENT behaviour -- exact-path query "
            "must NOT surface the sentinel-path row without project_name scoping"
        )

    async def test_project_name_scope_surfaces_sentinel_path_learning(self, sqlite_backend):
        sentinel_id = str(uuid4())
        real_id = str(uuid4())

        await sqlite_backend.save_project_learning(
            learning_id=sentinel_id,
            project_path=SENTINEL_PATH,
            category="pattern",
            learning_content="Sentinel-path learning under proj-x",
            project_name="proj-x",
        )
        await sqlite_backend.save_project_learning(
            learning_id=real_id,
            project_path=REAL_PATH,
            category="pattern",
            learning_content="Real-path learning under proj-x",
            project_name="proj-x",
        )

        results = await sqlite_backend.query_project_learnings(REAL_PATH, project_name="proj-x")
        ids = {r["id"] for r in results}

        assert real_id in ids, "issue #120: project_name scope must still return the real-path row"
        assert sentinel_id in ids, (
            "issue #120: project_name scope must surface the sentinel-path row for the same project"
        )

    async def test_project_name_scope_does_not_leak_other_projects(self, sqlite_backend):
        real_id = str(uuid4())
        other_sentinel_id = str(uuid4())

        await sqlite_backend.save_project_learning(
            learning_id=real_id,
            project_path=REAL_PATH,
            category="pattern",
            learning_content="Real-path learning under proj-x",
            project_name="proj-x",
        )
        await sqlite_backend.save_project_learning(
            learning_id=other_sentinel_id,
            project_path=SENTINEL_PATH,
            category="pattern",
            learning_content="Sentinel-path learning under proj-OTHER",
            project_name="proj-OTHER",
        )

        results = await sqlite_backend.query_project_learnings(REAL_PATH, project_name="proj-x")
        ids = {r["id"] for r in results}

        assert other_sentinel_id not in ids, (
            "issue #120: project_name scoping must not leak learnings "
            "belonging to a different project_name"
        )

    async def test_project_name_scope_still_honours_exclude_superseded(self, sqlite_backend):
        superseded_id = str(uuid4())
        superseder_id = str(uuid4())
        real_superseded_id = str(uuid4())
        real_superseder_id = str(uuid4())

        await sqlite_backend.save_project_learning(
            learning_id=superseded_id,
            project_path=SENTINEL_PATH,
            category="error_fix",
            learning_content="Old, wrong fix under proj-x",
            project_name="proj-x",
        )
        await sqlite_backend.save_project_learning(
            learning_id=superseder_id,
            project_path=SENTINEL_PATH,
            category="error_fix",
            learning_content="Corrected fix under proj-x",
            project_name="proj-x",
            supersedes=superseded_id,
        )
        await sqlite_backend.save_project_learning(
            learning_id=real_superseded_id,
            project_path=REAL_PATH,
            category="error_fix",
            learning_content="Old, wrong fix on the real path under proj-x",
            project_name="proj-x",
        )
        await sqlite_backend.save_project_learning(
            learning_id=real_superseder_id,
            project_path=REAL_PATH,
            category="error_fix",
            learning_content="Corrected fix on the real path under proj-x",
            project_name="proj-x",
            supersedes=real_superseded_id,
        )

        results = await sqlite_backend.query_project_learnings(
            REAL_PATH, project_name="proj-x", exclude_superseded=True
        )
        ids = {r["id"] for r in results}

        assert superseder_id in ids, (
            "issue #120: exclude_superseded must still surface the superseding row"
        )
        assert real_superseder_id in ids, (
            "issue #120: exclude_superseded must still surface the real-path superseding row"
        )
        assert superseded_id not in ids, (
            "issue #120: exclude_superseded must apply across the project_name "
            "branch too -- an unparenthesised OR would let the superseded row leak"
        )
        assert real_superseded_id not in ids, (
            "issue #120: exclude_superseded must apply to the project_path "
            "branch too -- an unparenthesised OR would let project_path rows "
            "bypass exclude_superseded"
        )

    async def test_project_name_scope_still_honours_since_days(self, sqlite_backend):
        stale_id = str(uuid4())
        real_stale_id = str(uuid4())

        await sqlite_backend.save_project_learning(
            learning_id=stale_id,
            project_path=SENTINEL_PATH,
            category="pattern",
            learning_content="Stale sentinel-path learning under proj-x",
            project_name="proj-x",
        )
        await _backdate_learning(sqlite_backend, stale_id, days=30)

        await sqlite_backend.save_project_learning(
            learning_id=real_stale_id,
            project_path=REAL_PATH,
            category="pattern",
            learning_content="Stale real-path learning under proj-x",
            project_name="proj-x",
        )
        await _backdate_learning(sqlite_backend, real_stale_id, days=30)

        results = await sqlite_backend.query_project_learnings(
            REAL_PATH, project_name="proj-x", since_days=1
        )
        ids = {r["id"] for r in results}

        assert stale_id not in ids, (
            "issue #120: since_days must apply across the project_name "
            "branch too -- an unparenthesised OR would let the stale row leak"
        )
        assert real_stale_id not in ids, (
            "issue #120: since_days must apply to the project_path branch too "
            "-- an unparenthesised OR would let project_path rows bypass "
            "since_days"
        )
