"""Regression tests: session_find_solution is ancestor-aware (follow-up to #244 part 2)."""

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend

ERR = "linker exploded"


@pytest.fixture
async def engine(tmp_path):
    eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
    eng.database = SQLiteBackend(str(tmp_path / "test.db"))
    await eng.database.initialize()
    yield eng
    await eng.database.close()


def _dir(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return str(d)


async def _learn(engine, project, path, tag):
    created = await engine.session_manage_lifecycle(
        "create", project_name=project, project_path=path, session_name=f"sess-{tag}"
    )
    await engine.session_log_learning(
        category="pattern",
        learning_content=f"fix for {ERR} ({tag})",
        trigger_context=f"trigger-{tag}",
        session_id=created.session_id,
    )


@pytest.fixture
async def family(engine, tmp_path):
    paths = {n: _dir(tmp_path, n) for n in ("kid", "mom", "gran")}
    await engine.session_set_project_parent("kid", "mom")
    await engine.session_set_project_parent("mom", "gran")
    for n in ("kid", "mom", "gran"):
        await _learn(engine, n, paths[n], n)
    return paths


@pytest.mark.regression
class TestAncestorSolutions:
    async def test_child_finds_parent_tagged_inherited(self, engine, family):
        res = await engine.session_find_solution(ERR, project_path=family["kid"])
        by_tag = {lr["inherited_from"]: lr for lr in res.learnings if "inherited_from" in lr}
        assert "mom" in by_tag
        assert "(mom)" in by_tag["mom"]["learning_content"]

    async def test_grandparent_reached(self, engine, family):
        res = await engine.session_find_solution(ERR, project_path=family["kid"])
        assert res.ancestors == ["mom", "gran"]
        assert {lr.get("inherited_from") for lr in res.learnings} == {None, "mom", "gran"}

    async def test_include_ancestors_false_excludes(self, engine, family):
        res = await engine.session_find_solution(
            ERR, project_path=family["kid"], include_ancestors=False
        )
        assert all("inherited_from" not in lr for lr in res.learnings)
        assert res.inherited_count == 0
        assert res.ancestors == []

    async def test_own_learning_untagged_and_counted(self, engine, family):
        res = await engine.session_find_solution(ERR, project_path=family["kid"])
        own = [lr for lr in res.learnings if "inherited_from" not in lr]
        assert len(own) == 1 and "(kid)" in own[0]["learning_content"]
        assert res.project_specific_count == 1

    async def test_count_arithmetic(self, engine, family):
        res = await engine.session_find_solution(ERR, project_path=family["kid"])
        assert res.inherited_count == 2
        assert res.total_found == len(res.solutions) + 1 + 2
        assert (
            res.universal_count
            == res.total_found - res.project_specific_count - res.inherited_count
        )

    async def test_explicit_project_name_without_path(self, engine, family):
        res = await engine.session_find_solution(ERR, project_name="kid")
        assert res.ancestors == ["mom", "gran"]
        assert res.inherited_count == 2

    async def test_no_project_row_no_ancestors(self, engine, tmp_path):
        path = _dir(tmp_path, "solo")
        await _learn(engine, "solo", path, "solo")
        res = await engine.session_find_solution(ERR, project_path=path)
        assert res.ancestors == [] and res.inherited_count == 0
        assert not res.degraded
        assert len(res.learnings) == 1

    async def test_ancestor_lookup_failure_keeps_own(self, engine, family, monkeypatch):
        async def boom(name):
            raise RuntimeError("db down")

        monkeypatch.setattr(engine.database, "get_project_ancestors", boom)
        res = await engine.session_find_solution(ERR, project_path=family["kid"])
        assert res.degraded and "ancestor lookup failed" in (res.degraded_reason or "")
        assert len(res.learnings) == 1 and "inherited_from" not in res.learnings[0]


@pytest.mark.regression
class TestQueryProjectLearningsNullPath:
    async def test_name_only_scope(self, engine, family):
        rows = await engine.database.query_project_learnings(project_path=None, project_name="mom")
        assert len(rows) == 1 and rows[0]["project_name"] == "mom"

    async def test_both_none_raises(self, engine):
        with pytest.raises(ValueError):
            await engine.database.query_project_learnings(project_path=None, project_name=None)
