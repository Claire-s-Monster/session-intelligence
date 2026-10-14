"""Regression tests for issue #89: near-duplicate warnings at insert time.

session_log_decision / session_log_learning surface near-duplicates of live
rows in the same project as non-blocking `warnings` on the result.

https://github.com/Claire-s-Monster/session-intelligence/issues/89
"""

from __future__ import annotations

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend

TEXT = "switch the session cache invalidation to a write-through strategy for hydrate"
TEXT_NEAR = "Switch session cache invalidation to write-through strategy for hydrate paths"
TEXT_OTHER = "rotate github token credentials quarterly across deployment hosts"
LEARN = "pixi task needs working_dir otherwise it silently runs against another repository"
LEARN_NEAR = "pixi task needs working_dir, otherwise silently runs against another repository"


@pytest.fixture
async def db(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "issue89.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def engine(tmp_path, db, monkeypatch):
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")
    return SessionIntelligenceEngine(
        repository_path=str(tmp_path), use_filesystem=False, database=db
    )


@pytest.mark.regression
async def test_decision_near_duplicate_warns_with_old_id(engine):
    first = await engine.session_log_decision(TEXT, project_name="proj-a")
    second = await engine.session_log_decision(TEXT_NEAR, project_name="proj-a")
    assert [w["id"] for w in second.warnings] == [first.decision_id]
    assert second.warnings[0]["kind"] == "near_duplicate"
    assert second.decision_id != "error"


@pytest.mark.regression
async def test_decision_other_project_no_warning(engine):
    await engine.session_log_decision(TEXT, project_name="proj-a")
    other = await engine.session_log_decision(TEXT_NEAR, project_name="proj-b")
    assert other.warnings == []


@pytest.mark.regression
async def test_decision_unrelated_text_no_warning(engine):
    await engine.session_log_decision(TEXT, project_name="proj-a")
    res = await engine.session_log_decision(TEXT_OTHER, project_name="proj-a")
    assert res.warnings == []


@pytest.mark.regression
async def test_decision_supersedes_target_not_flagged(engine):
    first = await engine.session_log_decision(TEXT, project_name="proj-a")
    res = await engine.session_log_decision(
        TEXT_NEAR, project_name="proj-a", supersedes=first.decision_id
    )
    assert res.warnings == []


@pytest.mark.regression
async def test_decision_new_row_never_flags_itself(engine):
    first = await engine.session_log_decision(TEXT, project_name="proj-a")
    assert first.warnings == []
    second = await engine.session_log_decision(TEXT_OTHER, project_name="proj-a")
    assert second.decision_id not in {w["id"] for w in second.warnings}
    assert second.warnings == []


@pytest.mark.regression
async def test_learning_near_duplicate_warns_and_category_scoped(engine):
    first = await engine.session_log_learning("pattern", LEARN, project_name="proj-a")
    assert first.warnings == []
    second = await engine.session_log_learning("pattern", LEARN_NEAR, project_name="proj-a")
    assert [w["id"] for w in second.warnings] == [first.id]
    other_cat = await engine.session_log_learning("workflow", LEARN_NEAR, project_name="proj-a")
    assert other_cat.warnings == []
    other_proj = await engine.session_log_learning("pattern", LEARN_NEAR, project_name="proj-b")
    assert other_proj.warnings == []


@pytest.mark.regression
async def test_learning_supersedes_target_not_flagged(engine):
    first = await engine.session_log_learning("pattern", LEARN, project_name="proj-a")
    res = await engine.session_log_learning(
        "pattern", LEARN_NEAR, project_name="proj-a", supersedes=first.id
    )
    assert res.warnings == []


@pytest.mark.regression
async def test_backend_failure_keeps_success_and_empty_warnings(engine, db, monkeypatch):
    async def boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(db, "recent_decision_texts", boom)
    monkeypatch.setattr(db, "recent_learning_texts", boom)
    dec = await engine.session_log_decision(TEXT, project_name="proj-a")
    assert dec.decision_id != "error"
    assert dec.warnings == []
    learn = await engine.session_log_learning("pattern", LEARN, project_name="proj-a")
    assert learn.status == "saved"
    assert learn.warnings == []
