"""
Regression tests for issue #244 part 1: session_transfer_entries.

SQLite is exercised for real; PostgreSQL through a fake asyncpg pool that
records SQL and bound args.
"""

import pytest

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence.postgresql import PostgreSQLBackend
from persistence.sqlite import SQLiteBackend

SRC = "proj-src"
DST = "proj-dst"


@pytest.fixture
async def engine(tmp_path):
    eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
    eng.database = SQLiteBackend(str(tmp_path / "test.db"))
    await eng.database.initialize()
    yield eng
    await eng.database.close()


@pytest.fixture
def paths(tmp_path):
    out = []
    for name in ("wt-a", "wt-b"):
        d = tmp_path / name
        d.mkdir()
        out.append(str(d))
    return out


async def _seed(engine, path, tag, decisions=1, project=SRC):
    created = await engine.session_manage_lifecycle(
        "create", project_name=project, project_path=path, session_name=f"sess-{tag}"
    )
    sid = created.session_id
    dids = []
    for i in range(decisions):
        logged = await engine.session_log_decision(f"decision-{tag}-{i}", session_id=sid)
        dids.append(logged.decision_id)
    learning = await engine.session_log_learning(
        category="pattern",
        learning_content=f"learning-{tag}",
        trigger_context=f"trigger-{tag}",
        session_id=sid,
    )
    await engine.session_create_notebook_async(session_id=sid, title=f"notebook-{tag}", body="b")
    return {"sid": sid, "dids": dids, "lid": learning.id}


async def _texts(engine, project, **kw):
    recalled = await engine.session_recall(project, **kw)
    return {
        "sessions": {s["id"] for s in recalled["sessions"]},
        "decisions": {d["description"] for d in recalled["decisions"]},
        "learnings": {item["learning_content"] for item in recalled["learnings"]},
        "notebooks": {n["title"] for n in recalled["notebooks"]},
    }


async def _raw(engine, sql, *args):
    cursor = await engine.database._connection.execute(sql, args)
    return [dict(r) for r in await cursor.fetchall()]


@pytest.mark.regression
class TestDryRun:
    async def test_default_is_dry_run_and_writes_nothing(self, engine, paths):
        seeded = await _seed(engine, paths[0], "a")
        before = await _texts(engine, SRC)

        result = await engine.session_transfer_entries(
            target_project_name=DST,
            decision_ids=seeded["dids"],
            learning_ids=[seeded["lid"]],
            notebook_session_ids=[seeded["sid"]],
        )
        assert result["status"] == "success"
        assert result["dry_run"] is True
        assert result["counts"] == {"decisions": 1, "learnings": 1, "notebooks": 1, "sessions": 0}
        assert {e["kind"] for e in result["entries"]} == {"decision", "learning", "notebook"}
        decision = next(e for e in result["entries"] if e["kind"] == "decision")
        assert decision["current_project_name"] == SRC
        assert decision["current_project_path"] == paths[0]
        assert decision["summary"] == "decision-a-0"
        assert set(decision) == {
            "kind",
            "id",
            "current_project_name",
            "current_project_path",
            "timestamp",
            "summary",
        }

        assert await _texts(engine, SRC) == before
        assert (await _texts(engine, DST))["decisions"] == set()
        rows = await _raw(engine, "SELECT project_name, project_path FROM decisions")
        assert rows == [{"project_name": None, "project_path": None}]


@pytest.mark.regression
class TestPerEntryMoves:
    async def test_decision_in_mixed_session_moves_alone(self, engine, paths):
        seeded = await _seed(engine, paths[0], "a", decisions=2)
        moved = await engine.session_transfer_entries(
            target_project_name=DST, decision_ids=[seeded["dids"][0]], dry_run=False
        )
        assert moved["dry_run"] is False
        assert moved["counts"]["decisions"] == 1

        src, dst = await _texts(engine, SRC), await _texts(engine, DST)
        assert src["decisions"] == {"decision-a-1"}
        assert dst["decisions"] == {"decision-a-0"}
        # The session itself and its other entries stay put.
        assert src["sessions"] == {seeded["sid"]}
        assert dst["sessions"] == set()
        recalled = await engine.session_recall(DST)
        assert recalled["decisions"][0]["project_path"] == paths[0]

    async def test_target_path_sets_path_override_and_omitted_clears_it(self, engine, paths):
        a, b = paths
        seeded = await _seed(engine, a, "a")
        did = seeded["dids"][0]
        await engine.session_transfer_entries(
            target_project_name=DST, target_project_path=b, decision_ids=[did], dry_run=False
        )
        recalled = await engine.session_recall(DST, project_path=b)
        assert [d["description"] for d in recalled["decisions"]] == ["decision-a-0"]

        await engine.session_transfer_entries(
            target_project_name=SRC, decision_ids=[did], dry_run=False
        )
        rows = await _raw(engine, "SELECT project_name, project_path FROM decisions")
        assert rows == [{"project_name": SRC, "project_path": None}]
        recalled = await engine.session_recall(SRC)
        assert recalled["decisions"][0]["project_path"] == a

    async def test_learning_move_keeps_path_unless_target_path_given(self, engine, paths):
        a, b = paths
        seeded = await _seed(engine, a, "a")
        await engine.session_transfer_entries(
            target_project_name=DST, learning_ids=[seeded["lid"]], dry_run=False
        )
        assert (await _texts(engine, SRC))["learnings"] == set()
        recalled = await engine.session_recall(DST)
        assert [x["learning_content"] for x in recalled["learnings"]] == ["learning-a"]
        assert recalled["learnings"][0]["project_path"] == a

        await engine.session_transfer_entries(
            target_project_name=DST,
            target_project_path=b,
            learning_ids=[seeded["lid"]],
            dry_run=False,
        )
        recalled = await engine.session_recall(DST, project_path=b)
        assert recalled["counts"]["learnings"] == 1

    async def test_notebook_move_visible_via_query_and_recall(self, engine, paths):
        seeded = await _seed(engine, paths[0], "a")
        await engine.session_transfer_entries(
            target_project_name=DST, notebook_session_ids=[seeded["sid"]], dry_run=False
        )
        src_rows = await engine.database.query_session_summaries(project_name=SRC)
        dst_rows = await engine.database.query_session_summaries(project_name=DST)
        assert src_rows == []
        assert [r["session_id"] for r in dst_rows] == [seeded["sid"]]
        assert dst_rows[0]["project_name"] == DST
        assert dst_rows[0]["project_path"] == paths[0]
        assert (await _texts(engine, SRC))["notebooks"] == set()
        assert (await _texts(engine, DST))["notebooks"] == {"notebook-a"}
        notebooks = await engine.session_query_notebooks(project_name=DST)
        assert [n["session_id"] for n in notebooks] == [seeded["sid"]]


@pytest.mark.regression
class TestSessionRehome:
    async def test_rehomes_everything_and_clears_overrides(self, engine, paths):
        a, b = paths
        seeded = await _seed(engine, a, "a", decisions=2)
        # A prior per-entry override that must follow the session afterwards.
        await engine.session_transfer_entries(
            target_project_name="elsewhere", decision_ids=[seeded["dids"][0]], dry_run=False
        )
        await engine.session_transfer_entries(
            target_project_name="elsewhere", notebook_session_ids=[seeded["sid"]], dry_run=False
        )

        preview = await engine.session_transfer_entries(
            target_project_name=DST, target_project_path=b, session_id=seeded["sid"]
        )
        assert preview["counts"] == {"decisions": 2, "learnings": 1, "notebooks": 1, "sessions": 1}

        done = await engine.session_transfer_entries(
            target_project_name=DST,
            target_project_path=b,
            session_id=seeded["sid"],
            dry_run=False,
        )
        assert done["counts"] == preview["counts"]
        assert await _texts(engine, SRC) == {
            k: set() for k in ("sessions", "decisions", "learnings", "notebooks")
        }
        dst = await _texts(engine, DST, project_path=b)
        assert dst == {
            "sessions": {seeded["sid"]},
            "decisions": {"decision-a-0", "decision-a-1"},
            "learnings": {"learning-a"},
            "notebooks": {"notebook-a"},
        }
        assert (
            await _raw(engine, "SELECT project_name, project_path FROM decisions")
            == [{"project_name": None, "project_path": None}] * 2
        )
        assert await _raw(engine, "SELECT project_name, project_path FROM session_summaries") == [
            {"project_name": None, "project_path": None}
        ]

    async def test_session_without_target_path_keeps_session_path(self, engine, paths):
        seeded = await _seed(engine, paths[0], "a")
        await engine.session_transfer_entries(
            target_project_name=DST, session_id=seeded["sid"], dry_run=False
        )
        session = await engine.database.get_session(seeded["sid"])
        assert session["project_name"] == DST
        assert session["project_path"] == paths[0]

    async def test_rehome_updates_cached_session(self, engine, paths):
        a, b = paths
        seeded = await _seed(engine, a, "a")
        cached = engine.session_cache[seeded["sid"]]
        assert (cached.project_name, cached.project_path) == (SRC, a)

        await engine.session_transfer_entries(
            target_project_name=DST, target_project_path=b, session_id=seeded["sid"]
        )
        assert (cached.project_name, cached.project_path) == (SRC, a)  # dry run

        await engine.session_transfer_entries(
            target_project_name=DST,
            target_project_path=b,
            session_id=seeded["sid"],
            dry_run=False,
        )
        assert (cached.project_name, cached.project_path) == (DST, b)

    async def test_rehome_without_path_keeps_cached_path(self, engine, paths):
        seeded = await _seed(engine, paths[0], "a")
        await engine.session_transfer_entries(
            target_project_name=DST, session_id=seeded["sid"], dry_run=False
        )
        cached = engine.session_cache[seeded["sid"]]
        assert (cached.project_name, cached.project_path) == (DST, paths[0])

    async def test_union_is_deduplicated(self, engine, paths):
        seeded = await _seed(engine, paths[0], "a")
        result = await engine.session_transfer_entries(
            target_project_name=DST,
            session_id=seeded["sid"],
            decision_ids=seeded["dids"],
            learning_ids=[seeded["lid"]],
            from_project_name=SRC,
        )
        assert result["counts"] == {"decisions": 1, "learnings": 1, "notebooks": 1, "sessions": 1}
        keys = [(e["kind"], e["id"]) for e in result["entries"]]
        assert len(keys) == len(set(keys))


@pytest.mark.regression
class TestWindowAndCap:
    async def test_from_project_window_includes_superseded(self, engine, paths):
        seeded = await _seed(engine, paths[0], "a", decisions=3)
        old, mid, new = seeded["dids"]
        for did, ts in (
            (old, "2026-10-01T10:00:00+00:00"),
            (mid, "2026-10-05T10:00:00+00:00"),
            (new, "2026-10-09T10:00:00+00:00"),
        ):
            await _raw(engine, "UPDATE decisions SET timestamp = ? WHERE id = ?", ts, did)
        await engine.database._connection.execute(
            "UPDATE decisions SET supersedes = ? WHERE id = ?", (mid, new)
        )
        await engine.database._connection.commit()

        result = await engine.session_transfer_entries(
            target_project_name=DST,
            from_project_name=SRC,
            from_project_path=paths[0],
            since="2026-10-04T00:00:00Z",
            until="2026-10-06T00:00:00",
            dry_run=False,
        )
        moved = {e["id"] for e in result["entries"] if e["kind"] == "decision"}
        assert moved == {mid}  # `mid` is superseded by `new` and still moves
        names = {
            r["id"]: r["project_name"]
            for r in await _raw(engine, "SELECT id, project_name FROM decisions")
        }
        assert names == {old: None, mid: DST, new: None}

    async def test_scope_selector_moves_all_three_kinds(self, engine, paths):
        await _seed(engine, paths[0], "a")
        await _seed(engine, paths[1], "b", project="untouched")
        result = await engine.session_transfer_entries(
            target_project_name=DST, from_project_name=SRC, dry_run=False
        )
        assert result["counts"] == {"decisions": 1, "learnings": 1, "notebooks": 1, "sessions": 0}
        assert (await _texts(engine, DST))["decisions"] == {"decision-a-0"}
        assert (await _texts(engine, "untouched"))["decisions"] == {"decision-b-0"}

    async def test_cap_exceeded_is_error_and_writes_nothing(self, engine, paths):
        await _seed(engine, paths[0], "a", decisions=3)
        for dry_run in (True, False):
            result = await engine.session_transfer_entries(
                target_project_name=DST, from_project_name=SRC, max_entries=2, dry_run=dry_run
            )
            assert result["status"] == "error"
            assert "max_entries=2" in result["message"]
        assert (await _texts(engine, SRC))["decisions"] == {
            "decision-a-0",
            "decision-a-1",
            "decision-a-2",
        }
        assert (await _texts(engine, DST))["decisions"] == set()

    async def test_missing_ids_warn(self, engine, paths):
        seeded = await _seed(engine, paths[0], "a")
        result = await engine.session_transfer_entries(
            target_project_name=DST,
            decision_ids=[seeded["dids"][0], "dec-nope"],
            session_id="sess-nope",
        )
        assert result["status"] == "success"
        joined = " ".join(result["warnings"])
        assert "dec-nope" in joined and "sess-nope" in joined
        assert result["counts"]["decisions"] == 1


@pytest.mark.regression
class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "fragment"),
        [
            ({"target_project_name": DST}, "at least one selector"),
            ({"target_project_name": "  ", "session_id": "x"}, "target_project_name"),
            (
                {"target_project_name": DST, "session_id": "x", "target_project_path": "rel/p"},
                "target_project_path",
            ),
            (
                {"target_project_name": DST, "from_project_name": SRC, "from_project_path": "rel"},
                "from_project_path",
            ),
            (
                {"target_project_name": DST, "from_project_path": "/abs"},
                "requires from_project_name",
            ),
            (
                {"target_project_name": DST, "from_project_name": SRC, "since": "yesterday"},
                "since",
            ),
            (
                {"target_project_name": DST, "from_project_name": SRC, "until": "nope"},
                "until",
            ),
            ({"target_project_name": DST, "session_id": "x", "since": "2026-01-01"}, "require"),
            ({"target_project_name": DST, "session_id": "x", "max_entries": 0}, "max_entries"),
            ({"target_project_name": DST, "session_id": "x", "max_entries": 1001}, "max_entries"),
            ({"target_project_name": DST, "decision_ids": ["ok", ""]}, "decision_ids"),
        ],
    )
    async def test_rejected_without_writing(self, engine, kwargs, fragment):
        result = await engine.session_transfer_entries(dry_run=False, **kwargs)
        assert result["status"] == "error"
        assert fragment in result["message"]


@pytest.mark.regression
class TestEffectiveScopeReads:
    async def test_recent_decision_texts_and_search_use_effective_scope(self, engine, paths):
        seeded = await _seed(engine, paths[0], "a", decisions=2)
        await engine.session_transfer_entries(
            target_project_name=DST,
            target_project_path=paths[1],
            decision_ids=[seeded["dids"][0]],
            dry_run=False,
        )
        src = await engine.database.recent_decision_texts(SRC, 10)
        dst = await engine.database.recent_decision_texts(DST, 10)
        assert [t for _, t in src] == ["decision-a-1"]
        assert [t for _, t in dst] == ["decision-a-0"]

        hits = await engine.database.search_sessions("decision-a-0", "decisions")
        assert [(h["project_name"], h["project_path"]) for h in hits] == [(DST, paths[1])]
        sibling = await engine.database.search_sessions("decision-a-1", "decisions")
        assert [(h["project_name"], h["project_path"]) for h in sibling] == [(SRC, paths[0])]

        await _raw(engine, "UPDATE decisions SET category = 'cat-244'")
        by_category = await engine.database.query_decisions_by_category("cat-244")
        assert len(by_category) == 2
        for row in by_category:
            expected = DST if row["id"] == seeded["dids"][0] else SRC
            assert row["project_name"] == expected

    async def test_notebook_listings_report_effective_scope(self, engine, paths):
        seeded = await _seed(engine, paths[0], "a")
        await engine.session_transfer_entries(
            target_project_name=DST,
            target_project_path=paths[1],
            notebook_session_ids=[seeded["sid"]],
            dry_run=False,
        )
        for rows in (
            await engine.database.query_recent_summaries(),
            await engine.database.query_session_summaries(session_id=seeded["sid"]),
        ):
            assert [(r["project_name"], r["project_path"]) for r in rows] == [(DST, paths[1])]

    async def test_exclude_project_paths_respects_overrides(self, engine, paths):
        a, b = paths
        seeded = await _seed(engine, a, "a")
        await _seed(engine, b, "b")
        # Same project, but this decision/notebook now live under path b.
        await engine.session_transfer_entries(
            target_project_name=SRC,
            target_project_path=b,
            decision_ids=seeded["dids"],
            notebook_session_ids=[seeded["sid"]],
            dry_run=False,
        )
        excl_b = await _texts(engine, SRC, exclude_project_paths=[b])
        assert excl_b["decisions"] == set()
        assert excl_b["notebooks"] == set()
        excl_a = await _texts(engine, SRC, exclude_project_paths=[a])
        assert excl_a["decisions"] == {"decision-a-0", "decision-b-0"}
        assert excl_a["notebooks"] == {"notebook-a", "notebook-b"}
        only_a = await _texts(engine, SRC, project_path=a)
        assert only_a["decisions"] == set()


@pytest.mark.regression
class TestSchema:
    def test_registered_with_dry_run_default_true(self, engine):
        spec = LeanMCPInterface(engine).tool_registry["session_transfer_entries"]["schema"]
        assert spec["required"] == ["target_project_name"]
        assert spec["properties"]["dry_run"]["default"] is True
        assert spec["properties"]["max_entries"]["maximum"] == 1000


class _FakeTx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, decision_rows):
        self.fetches = []
        self.executes = []
        self._decision_rows = decision_rows

    def transaction(self):
        return _FakeTx()

    async def fetch(self, query, *args):
        self.fetches.append((query, args))
        return self._decision_rows if "FROM decisions d" in query else []

    async def execute(self, query, *args):
        self.executes.append((query, args))


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquire(self._conn)


@pytest.mark.regression
class TestPostgresSql:
    def _backend(self, rows):
        backend = PostgreSQLBackend(dsn="postgresql://localhost/fake_issue_244_transfer")
        conn = _FakeConn(rows)
        backend._pool = _FakePool(conn)  # type: ignore[assignment]
        return backend, conn

    @staticmethod
    def _row(did="d1", session="s1"):
        return {
            "id": did,
            "session_id": session,
            "ts": None,
            "label": "a decision",
            "project_name": SRC,
            "project_path": "/wt/a",
        }

    async def test_dry_run_selects_effective_scope_and_writes_nothing(self):
        backend, conn = self._backend([self._row()])
        result = await backend.transfer_entries(
            target_project_name=DST, decision_ids=["d1"], dry_run=True
        )
        query, args = conn.fetches[0]
        assert "COALESCE(d.project_name, s.project_name) AS project_name" in query
        assert "d.id = ANY($1::text[])" in query
        assert args == (["d1"],)
        assert conn.executes == []
        assert result["counts"]["decisions"] == 1

    async def test_apply_sets_overrides_with_bound_args(self):
        backend, conn = self._backend([self._row()])
        await backend.transfer_entries(
            target_project_name=DST,
            target_project_path="/wt/b",
            decision_ids=["d1"],
            dry_run=False,
        )
        assert conn.executes == [
            (
                "UPDATE decisions SET project_name = $1, project_path = $2 "
                "WHERE id = ANY($3::text[])",
                (DST, "/wt/b", ["d1"]),
            )
        ]

    async def test_session_rehome_clears_overrides_and_updates_session(self):
        backend, conn = self._backend([self._row()])
        conn_rows = {
            "id": "s1",
            "session_id": "s1",
            "ts": None,
            "label": "n",
            "project_name": SRC,
            "project_path": "/wt/a",
        }
        original = conn.fetch

        async def fetch(query, *args):
            if "FROM sessions" in query and "JOIN" not in query:
                conn.fetches.append((query, args))
                return [conn_rows]
            return await original(query, *args)

        conn.fetch = fetch  # type: ignore[method-assign]
        await backend.transfer_entries(target_project_name=DST, session_id="s1", dry_run=False)
        sql = [q for q, _ in conn.executes]
        assert any(
            "UPDATE decisions SET project_name = NULL, project_path = NULL" in q for q in sql
        )
        assert any(q.startswith("UPDATE sessions SET project_name = $1") for q in sql)

    async def test_cap_raises_before_any_write(self):
        backend, conn = self._backend([self._row("d1"), self._row("d2")])
        with pytest.raises(ValueError, match="max_entries=1"):
            await backend.transfer_entries(
                target_project_name=DST, decision_ids=["d1", "d2"], dry_run=False, max_entries=1
            )
        assert conn.executes == []

    async def test_recall_reads_effective_scope(self):
        backend, conn = self._backend([])

        async def fetch(query, *args):
            conn.fetches.append((query, args))
            return []

        conn.fetch = fetch  # type: ignore[method-assign]
        await backend.recall_project(SRC)
        decisions = next(q for q, _ in conn.fetches if "FROM decisions d" in q)
        notebooks = next(q for q, _ in conn.fetches if "FROM session_summaries ss" in q)
        assert "COALESCE(d.project_name, s.project_name) = $1" in decisions
        assert "COALESCE(ss.project_name, s.project_name) = $1" in notebooks
