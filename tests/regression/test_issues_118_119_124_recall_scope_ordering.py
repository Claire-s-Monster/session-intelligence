"""
Regression tests for issues #118, #119, #124 (recall/search scope + ordering).

Style follows tests/regression/test_issue_120_notebook_learnings_project_name.py:
SQLite-backed `sqlite_backend` fixture (from tests/conftest.py), asyncio_mode =
"auto" (no @pytest.mark.asyncio needed), `regression` marker.

#118: `recall_project`'s NULL-project_name fallback subquery used
`project_path = (SELECT ... LIMIT 1)` with no ORDER BY, so when a project has
multiple distinct session `project_path` values, only ONE of them (chosen
non-deterministically -- in practice whichever row a plain table scan visits
first) was used to rescue NULL-project_name learnings. Fixed by widening to
`project_path IN (SELECT ...)` so ALL of the project's distinct paths match.

#124: `ORDER BY success_count DESC, last_used DESC` ranks NULL `last_used`
FIRST on PostgreSQL (NULLs sort largest) but LAST on SQLite (NULLs sort
smallest) -- a cross-backend behavioural divergence. Fixed by prefixing the
tiebreaker with `(last_used IS NULL)` (0 for non-null, 1 for null; ASC by
default) so never-used learnings rank LAST on both engines. Swept across
`query_project_learnings` and the `recall_project` learnings branch.

#119: the `learnings` branch of `search_sessions` hardcoded
`NULL as project_name` into the result envelope even though
`project_learnings` has a real, populated `project_name` column. Fixed by
selecting the real column.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = pytest.mark.regression

# tests/regression/<this file> -> parents[2] is the repo root.
_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
_POSTGRESQL_SRC = _SRC_DIR / "persistence" / "postgresql.py"
_SQLITE_SRC = _SRC_DIR / "persistence" / "sqlite.py"

# The exact ORDER BY clause both backends must use for the #124 fix. Verified
# against the real file contents (see git blame on this line) rather than
# guessed -- whitespace matters for a substring match.
_EXPECTED_ORDER_BY_CLAUSE = "ORDER BY success_count DESC, (last_used IS NULL), last_used DESC, id"


def _session_row(session_id: str, project_name: str, project_path: str) -> dict:
    started = datetime.now(UTC).isoformat()
    return {
        "id": session_id,
        "started": started,
        "project_path": project_path,
        "project_name": project_name,
        "mode": "local",
        "status": "active",
        "metadata": {},
        "performance_metrics": {},
        "health_status": {},
    }


class TestIssue118RecallProjectMultiPathFallback:
    async def test_recall_project_rescues_null_project_name_learnings_from_all_paths(
        self, sqlite_backend
    ):
        project_name = "proj-118"
        path_a = "/repo/proj-118/checkout-a"
        path_b = "/repo/proj-118/checkout-b"

        # Two distinct sessions under the same project_name, different paths.
        await sqlite_backend.save_session(_session_row("sess-a", project_name, path_a))
        await sqlite_backend.save_session(_session_row("sess-b", project_name, path_b))

        learning_a = str(uuid4())
        learning_b = str(uuid4())

        # Both learnings have NULL project_name (the fallback case) but are
        # tied to different session project_paths under the same project.
        await sqlite_backend.save_project_learning(
            learning_id=learning_a,
            project_path=path_a,
            category="pattern",
            learning_content="Learning tied to checkout-a",
            project_name=None,
        )
        await sqlite_backend.save_project_learning(
            learning_id=learning_b,
            project_path=path_b,
            category="pattern",
            learning_content="Learning tied to checkout-b",
            project_name=None,
        )

        result = await sqlite_backend.recall_project(project_name, include=["learnings"])
        ids = {row["id"] for row in result["learnings"]}

        assert learning_a in ids, (
            "issue #118: recall_project must rescue NULL-project_name learnings "
            "from ALL of the project's distinct session paths, not just one"
        )
        assert learning_b in ids, (
            "issue #118: recall_project must rescue NULL-project_name learnings "
            "from ALL of the project's distinct session paths, not just one"
        )


class TestIssue124NullLastUsedOrdering:
    async def test_query_project_learnings_orders_non_null_last_used_first(self, sqlite_backend):
        project_path = "/repo/proj-124"

        never_used_id = str(uuid4())
        used_id = str(uuid4())

        # Both saved with equal success_count (default 1 from
        # save_project_learning) so ordering is decided purely by last_used.
        await sqlite_backend.save_project_learning(
            learning_id=never_used_id,
            project_path=project_path,
            category="pattern",
            learning_content="Never-used learning",
        )
        await sqlite_backend.save_project_learning(
            learning_id=used_id,
            project_path=project_path,
            category="pattern",
            learning_content="Recently-used learning",
        )

        conn = sqlite_backend._ensure_connected()
        await conn.execute(
            "UPDATE project_learnings SET last_used = NULL WHERE id = ?",
            (never_used_id,),
        )
        await conn.commit()

        results = await sqlite_backend.query_project_learnings(project_path)
        ids = [row["id"] for row in results]

        assert ids.index(used_id) < ids.index(never_used_id), (
            "issue #124: never-used (NULL last_used) learnings must sort LAST, "
            "matching the chosen cross-backend convention"
        )

    async def test_recall_project_orders_non_null_last_used_first(self, sqlite_backend):
        project_name = "proj-124-recall"
        project_path = "/repo/proj-124-recall"

        await sqlite_backend.save_session(_session_row("sess-124", project_name, project_path))

        never_used_id = str(uuid4())
        used_id = str(uuid4())

        await sqlite_backend.save_project_learning(
            learning_id=never_used_id,
            project_path=project_path,
            category="pattern",
            learning_content="Never-used learning",
            project_name=project_name,
        )
        await sqlite_backend.save_project_learning(
            learning_id=used_id,
            project_path=project_path,
            category="pattern",
            learning_content="Recently-used learning",
            project_name=project_name,
        )

        conn = sqlite_backend._ensure_connected()
        await conn.execute(
            "UPDATE project_learnings SET last_used = NULL WHERE id = ?",
            (never_used_id,),
        )
        await conn.commit()

        result = await sqlite_backend.recall_project(project_name, include=["learnings"])
        ids = [row["id"] for row in result["learnings"]]

        assert ids.index(used_id) < ids.index(never_used_id), (
            "issue #124: recall_project must also rank never-used (NULL "
            "last_used) learnings LAST, matching query_project_learnings"
        )


class TestIssue124SourceTextPinsCrossBackendOrdering:
    """SQL-text pin for #124, as suggested by the issue itself.

    WHY THIS ASSERTS ON SOURCE TEXT, NOT BEHAVIOUR: on SQLite, plain
    `ORDER BY ... last_used DESC` and the fixed
    `ORDER BY ..., (last_used IS NULL), last_used DESC, id` produce IDENTICAL
    row order for every input -- SQLite already treats NULL as the smallest
    value, so DESC already puts NULLs last with or without the tiebreaker.
    The two behavioural tests above (TestIssue124NullLastUsedOrdering) pass
    whether or not the fix is present, which means they cannot catch a
    regression that deletes the tiebreaker. The actual divergence this fix
    targets only manifests on PostgreSQL (NULL sorts as largest there), and
    the PostgreSQL contract tests skip locally with no reachable server
    (#114). Reading the SQL text directly is therefore the only local
    mechanism that can catch someone deleting `(last_used IS NULL)` --
    do NOT "simplify" this into a behavioural assertion; that would silently
    remove the only guard this repo has for the #124 fix without a live
    PostgreSQL instance.
    """

    def test_order_by_clause_appears_twice_in_postgresql(self):
        text = _POSTGRESQL_SRC.read_text()
        count = text.count(_EXPECTED_ORDER_BY_CLAUSE)
        assert count == 2, (
            f"expected the fixed ORDER BY clause in postgresql.py exactly twice "
            f"(query_project_learnings and recall_project), found {count}"
        )

    def test_order_by_clause_appears_twice_in_sqlite(self):
        text = _SQLITE_SRC.read_text()
        count = text.count(_EXPECTED_ORDER_BY_CLAUSE)
        assert count == 2, (
            f"expected the fixed ORDER BY clause in sqlite.py exactly twice "
            f"(query_project_learnings and recall_project), found {count}"
        )


class TestIssue119SearchLearningsProjectName:
    async def test_search_sessions_learnings_branch_reports_real_project_name(self, sqlite_backend):
        project_name = "proj-119"
        learning_id = str(uuid4())

        await sqlite_backend.save_project_learning(
            learning_id=learning_id,
            project_path="/repo/proj-119",
            category="pattern",
            learning_content="Unique searchable marker xyzzy119",
            project_name=project_name,
        )

        results = await sqlite_backend.search_sessions("xyzzy119", search_type="learnings")

        assert len(results) == 1
        assert results[0]["project_name"] == project_name, (
            "issue #119: the learnings branch of search_sessions must report "
            "the real project_name column instead of a hardcoded NULL"
        )
