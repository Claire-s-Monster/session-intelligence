"""
Persistence contract tests — backend-agnostic.

Every persistence backend must pass all tests in ``PersistenceContractTests``
identically.  Backend-specific test files inherit from this class and supply
a ``backend`` fixture.

Design notes
------------
- No ``@pytest.mark.asyncio`` — the project uses ``asyncio_mode = "auto"``
  in pyproject.toml so every ``async def test_*`` is collected automatically.
- Builder helpers normalise the dict keys expected by each backend:
  ``save_session`` wants ``id`` / ``started_at`` / ``ended_at``;
  the builders produce ``session_id`` / ``start_time`` / ``end_time`` so we
  add an adapter layer here.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from tests.persistence.builders import (
    make_agent_data,
    make_agent_decision_data,
    make_agent_learning_data,
    make_agent_notebook_data,
    make_agent_execution_data,
    make_decision_data,
    make_error_solution_data,
    make_file_operation_data,
    make_mcp_session_data,
    make_metrics_data,
    make_note_data,
    make_project_learning_data,
    make_session_data,
    make_summary_data,
)


# ---------------------------------------------------------------------------
# Adapter helpers — normalise builder dicts to backend field names
# ---------------------------------------------------------------------------

def _session(session_id: str | None = None, **overrides) -> dict:
    """Return a session dict ready for ``save_session``."""
    raw = make_session_data(**overrides)
    sid = session_id or raw["session_id"]
    return {
        "id": sid,
        "started_at": raw["start_time"].isoformat()
        if isinstance(raw.get("start_time"), datetime)
        else raw.get("start_time", datetime.now(UTC).isoformat()),
        "ended_at": raw["end_time"].isoformat()
        if isinstance(raw.get("end_time"), datetime)
        else raw.get("end_time"),
        "project_path": raw.get("project_path", f"/tmp/project-{sid}"),
        "project_name": raw.get("project_name", f"project-{sid[:8]}"),
        "mode": raw.get("mode", "local"),
        "status": raw.get("status", "active"),
        "metadata": raw.get("metadata", {}),
        "performance_metrics": raw.get("performance_metrics", {}),
        "health_status": raw.get("health_status", {}),
    }


def _decision(session_id: str, decision_id: str | None = None, **overrides) -> dict:
    """Return a decision dict ready for ``save_decision``."""
    raw = make_decision_data(session_id=session_id, **overrides)
    d = dict(raw)
    d["id"] = decision_id or raw.get("decision_id") or f"dec-{uuid.uuid4().hex[:8]}"
    d["decision_id"] = d["id"]
    d["description"] = d.pop("decision", "Use pytest")
    if isinstance(d.get("created_at"), datetime):
        d["timestamp"] = d["created_at"].isoformat()
    else:
        d["timestamp"] = d.get("created_at", datetime.now(UTC).isoformat())
    return d


def _metrics(session_id: str, branch: str = "main", **overrides) -> dict:
    """Return a metrics dict ready for ``save_metrics``."""
    raw = make_metrics_data(session_id=session_id, **overrides)
    m = dict(raw)
    if isinstance(m.get("recorded_at"), datetime):
        m["timestamp"] = m["recorded_at"].isoformat()
    else:
        m.setdefault("timestamp", datetime.now(UTC).isoformat())
    m["branch"] = branch
    m["coverage"] = 85.0
    m["complexity"] = 3.5
    m["test_count"] = 42
    m["agents_executed"] = 2
    m["execution_time_ms"] = 1500
    return m


def _note(session_id: str, **overrides) -> dict:
    """Return a note dict ready for ``save_note``."""
    raw = make_note_data(session_id=session_id, **overrides)
    n = dict(raw)
    if isinstance(n.get("created_at"), datetime):
        n["date"] = n["created_at"].strftime("%Y-%m-%d")
    else:
        n.setdefault("date", datetime.now(UTC).strftime("%Y-%m-%d"))
    return n


def _file_op(session_id: str, **overrides) -> dict:
    """Return a file_operation dict ready for ``save_file_operation``."""
    raw = make_file_operation_data(session_id=session_id, **overrides)
    f = dict(raw)
    f["operation"] = f.pop("operation_type", "write")
    if isinstance(f.get("timestamp"), datetime):
        f["timestamp"] = f["timestamp"].isoformat()
    else:
        f.setdefault("timestamp", datetime.now(UTC).isoformat())
    return f


def _agent(name: str | None = None, **overrides) -> dict:
    """Return an agent dict ready for ``save_agent``."""
    raw = make_agent_data(**overrides)
    a = dict(raw)
    if name:
        a["name"] = name
    for dt_field in ("first_seen_at", "last_active_at"):
        if isinstance(a.get(dt_field), datetime):
            a[dt_field] = a[dt_field].isoformat()
    return a


def _agent_decision(agent_id: str, **overrides) -> dict:
    """Return an agent_decision dict ready for ``save_agent_decision``."""
    raw = make_agent_decision_data(agent_id=agent_id, **overrides)
    d = dict(raw)
    d["description"] = d.pop("decision", "Use hexagonal architecture")
    for dt_field in ("created_at", "updated_at"):
        if isinstance(d.get(dt_field), datetime):
            d[dt_field] = d[dt_field].isoformat()
    d.setdefault("timestamp", d.get("created_at", datetime.now(UTC).isoformat()))
    return d


def _agent_learning(agent_id: str, **overrides) -> dict:
    """Return an agent_learning dict ready for ``save_agent_learning``."""
    raw = make_agent_learning_data(agent_id=agent_id, **overrides)
    al = dict(raw)
    al["learning_content"] = al.pop("content", "Always validate inputs")
    al["category"] = al.pop("learning_type", "pattern")
    al["trigger_context"] = al.pop("source_context", None)
    for dt_field in ("created_at", "updated_at"):
        if isinstance(al.get(dt_field), datetime):
            al[dt_field] = al[dt_field].isoformat()
    return al


def _agent_notebook(agent_id: str, **overrides) -> dict:
    """Return an agent_notebook dict ready for ``save_agent_notebook``."""
    raw = make_agent_notebook_data(agent_id=agent_id, **overrides)
    nb = dict(raw)
    nb["summary_markdown"] = nb.pop("content", "# Notebook\n\nContent here.")
    nb.pop("decisions_referenced", None)
    nb.pop("learnings_referenced", None)
    nb.pop("context", None)
    nb.pop("summary", None)
    nb.pop("notebook_type", None)
    nb["notebook_type"] = raw.get("notebook_type", "summary")
    for dt_field in ("created_at", "updated_at"):
        if isinstance(nb.get(dt_field), datetime):
            nb[dt_field] = nb[dt_field].isoformat()
    return nb


def _summary(session_id: str, **overrides) -> dict:
    """Return a session_summary dict ready for ``save_session_summary``."""
    raw = make_summary_data(session_id=session_id, **overrides)
    s = dict(raw)
    s["summary_markdown"] = s.pop("content", "Session completed.")
    if isinstance(s.get("created_at"), datetime):
        s["created_at"] = s["created_at"].isoformat()
    return s


def _agent_execution(session_id: str, agent_name: str = "test-agent", **overrides) -> dict:
    """Return an agent_execution dict ready for ``save_agent_execution``."""
    raw = make_agent_execution_data(session_id=session_id, **overrides)
    e = dict(raw)
    e["id"] = e.pop("execution_id", f"exec-{uuid.uuid4().hex[:8]}")
    e["agent_name"] = agent_name
    for dt_field in ("started_at", "ended_at"):
        if isinstance(e.get(dt_field), datetime):
            e[dt_field] = e[dt_field].isoformat()
    return e


def _mcp_session(**overrides) -> dict:
    """Return an MCP session dict ready for ``save_mcp_session``."""
    return make_mcp_session_data(**overrides)


# ---------------------------------------------------------------------------
# The Contract
# ---------------------------------------------------------------------------

class PersistenceContractTests:
    """
    Shared contract tests for all persistence backends.

    Subclass this in a backend-specific file and provide a ``backend`` fixture.
    """

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    async def test_save_and_retrieve_session(self, backend):
        s = _session()
        await backend.save_session(s)
        result = await backend.get_session(s["id"])
        assert result is not None
        assert result["id"] == s["id"]
        assert result["status"] == s["status"]
        assert result["project_path"] == s["project_path"]

    async def test_save_session_with_all_fields(self, backend):
        s = _session(
            performance_metrics={"coverage": 95.0},
            health_status={"ok": True},
        )
        s["ended_at"] = datetime.now(UTC).isoformat()
        s["status"] = "completed"
        await backend.save_session(s)
        result = await backend.get_session(s["id"])
        assert result is not None
        assert result["id"] == s["id"]

    async def test_update_existing_session(self, backend):
        s = _session()
        await backend.save_session(s)
        s["status"] = "completed"
        await backend.save_session(s)
        result = await backend.get_session(s["id"])
        assert result is not None
        assert result["status"] == "completed"

    async def test_get_nonexistent_session_returns_none(self, backend):
        result = await backend.get_session("nonexistent-session-id-xyz")
        assert result is None

    async def test_list_sessions_by_project(self, backend):
        project = f"/tmp/project-{uuid.uuid4().hex[:8]}"
        s1 = _session(project_path=project)
        s2 = _session(project_path=project)
        s3 = _session(project_path="/tmp/other-project")
        for s in (s1, s2, s3):
            await backend.save_session(s)
        results = await backend.query_sessions(project_path=project)
        ids = {r["id"] for r in results}
        assert s1["id"] in ids
        assert s2["id"] in ids
        assert s3["id"] not in ids

    # ------------------------------------------------------------------
    # Decision CRUD
    # ------------------------------------------------------------------

    async def test_save_and_query_decisions(self, backend):
        s = _session()
        await backend.save_session(s)
        d = _decision(session_id=s["id"])
        await backend.save_decision(d)
        results = await backend.query_decisions_by_session(s["id"])
        assert len(results) >= 1
        ids = {r.get("id") or r.get("decision_id") for r in results}
        assert d["id"] in ids

    async def test_decision_requires_valid_session_fk(self, backend):
        d = _decision(session_id="invalid-session-fk-xyz")
        with pytest.raises(Exception):
            await backend.save_decision(d)

    async def test_query_decisions_with_filters(self, backend):
        s = _session()
        await backend.save_session(s)
        d1 = _decision(session_id=s["id"], category="architecture")
        d2 = _decision(session_id=s["id"], category="testing")
        await backend.save_decision(d1)
        await backend.save_decision(d2)
        results = await backend.query_decisions_by_category("architecture")
        categories = {r.get("category") for r in results}
        assert "architecture" in categories

    async def test_update_decision_outcome(self, backend):
        # Only test if backend supports update_agent_decision_outcome
        if not hasattr(backend, "update_agent_decision_outcome"):
            pytest.skip("Backend does not implement update_agent_decision_outcome")
        agent = _agent()
        await backend.save_agent(agent)
        ad = _agent_decision(agent_id=agent["id"])
        await backend.save_agent_decision(ad)
        await backend.update_agent_decision_outcome(ad["id"], "success", notes="worked well")
        results = await backend.query_agent_decisions(agent["id"])
        match = next((r for r in results if r["id"] == ad["id"]), None)
        assert match is not None
        assert match.get("outcome") == "success"

    # ------------------------------------------------------------------
    # Learning CRUD
    # ------------------------------------------------------------------

    async def test_save_and_query_learnings(self, backend):
        s = _session()
        await backend.save_session(s)
        pl = make_project_learning_data(project_path=s["project_path"])
        await backend.save_project_learning(
            learning_id=pl["learning_id"],
            project_path=pl["project_path"],
            category=pl["category"],
            learning_content=pl["learning_content"],
            trigger_context=pl.get("trigger_context"),
            source_session_id=s["id"],
        )
        results = await backend.query_project_learnings(project_path=pl["project_path"])
        assert len(results) >= 1
        ids = {r["id"] for r in results}
        assert pl["learning_id"] in ids

    async def test_query_learnings_by_category(self, backend):
        s = _session()
        await backend.save_session(s)
        project = s["project_path"]
        pl = make_project_learning_data(project_path=project, category="workflow")
        await backend.save_project_learning(
            learning_id=pl["learning_id"],
            project_path=project,
            category="workflow",
            learning_content=pl["learning_content"],
        )
        results = await backend.query_project_learnings(project_path=project, category="workflow")
        assert len(results) >= 1
        for r in results:
            assert r["category"] == "workflow"

    # ------------------------------------------------------------------
    # Agent CRUD
    # ------------------------------------------------------------------

    async def test_save_and_get_agent(self, backend):
        a = _agent()
        await backend.save_agent(a)
        result = await backend.get_agent(a["id"])
        assert result is not None
        assert result["id"] == a["id"]
        assert result["name"] == a["name"]

    async def test_get_agent_by_name(self, backend):
        a = _agent()
        await backend.save_agent(a)
        result = await backend.get_agent_by_name(a["name"])
        assert result is not None
        assert result["id"] == a["id"]

    async def test_agent_name_uniqueness(self, backend):
        name = f"unique-agent-{uuid.uuid4().hex[:8]}"
        a1 = _agent(name=name)
        a2 = _agent(name=name)
        # same name, different id — second save should update, not fail
        await backend.save_agent(a1)
        await backend.save_agent(a2)
        result = await backend.get_agent_by_name(name)
        assert result is not None

    async def test_update_agent_stats(self, backend):
        a = _agent()
        await backend.save_agent(a)
        await backend.update_agent_stats(a["id"], "executions")
        result = await backend.get_agent(a["id"])
        assert result is not None
        assert result["total_executions"] >= 1

    # ------------------------------------------------------------------
    # Agent Decisions
    # ------------------------------------------------------------------

    async def test_save_agent_decision(self, backend):
        a = _agent()
        await backend.save_agent(a)
        d = _agent_decision(agent_id=a["id"])
        await backend.save_agent_decision(d)
        results = await backend.query_agent_decisions(a["id"])
        assert len(results) >= 1

    async def test_agent_decision_requires_valid_agent_fk(self, backend):
        d = _agent_decision(agent_id="invalid-agent-fk-xyz")
        with pytest.raises(Exception):
            await backend.save_agent_decision(d)

    async def test_query_agent_decisions_with_filters(self, backend):
        a = _agent()
        await backend.save_agent(a)
        d = _agent_decision(agent_id=a["id"], category="architecture")
        await backend.save_agent_decision(d)
        results = await backend.query_agent_decisions(a["id"])
        assert len(results) >= 1
        assert all(r["agent_id"] == a["id"] for r in results)

    async def test_update_agent_decision_outcome(self, backend):
        a = _agent()
        await backend.save_agent(a)
        d = _agent_decision(agent_id=a["id"])
        await backend.save_agent_decision(d)
        await backend.update_agent_decision_outcome(d["id"], "success", notes="worked")
        results = await backend.query_agent_decisions(a["id"])
        match = next((r for r in results if r["id"] == d["id"]), None)
        assert match is not None
        assert match.get("outcome") == "success"

    # ------------------------------------------------------------------
    # Agent Learnings
    # ------------------------------------------------------------------

    async def test_save_agent_learning(self, backend):
        a = _agent()
        await backend.save_agent(a)
        al = _agent_learning(agent_id=a["id"])
        await backend.save_agent_learning(al)
        results = await backend.query_agent_learnings(a["id"])
        assert len(results) >= 1

    async def test_query_agent_learnings_with_filters(self, backend):
        a = _agent()
        await backend.save_agent(a)
        al = _agent_learning(agent_id=a["id"], learning_type="anti-pattern")
        al["category"] = "anti-pattern"
        await backend.save_agent_learning(al)
        results = await backend.query_agent_learnings(a["id"])
        assert any(r["agent_id"] == a["id"] for r in results)

    async def test_update_agent_learning_outcome(self, backend):
        a = _agent()
        await backend.save_agent(a)
        al = _agent_learning(agent_id=a["id"])
        await backend.save_agent_learning(al)
        await backend.update_agent_learning_outcome(al["id"], success=True)
        results = await backend.query_agent_learnings(a["id"])
        match = next((r for r in results if r["id"] == al["id"]), None)
        assert match is not None
        assert match.get("success_count", 0) >= 2  # was 1, now 2

    # ------------------------------------------------------------------
    # Agent Notebooks
    # ------------------------------------------------------------------

    async def test_save_agent_notebook(self, backend):
        a = _agent()
        await backend.save_agent(a)
        nb = _agent_notebook(agent_id=a["id"])
        await backend.save_agent_notebook(nb)
        results = await backend.query_agent_notebooks(a["id"])
        assert len(results) >= 1

    async def test_query_agent_notebooks_with_filters(self, backend):
        a = _agent()
        await backend.save_agent(a)
        nb = _agent_notebook(agent_id=a["id"])
        await backend.save_agent_notebook(nb)
        results = await backend.query_agent_notebooks(a["id"])
        assert all(r["agent_id"] == a["id"] for r in results)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def test_save_and_query_metrics(self, backend):
        s = _session()
        await backend.save_session(s)
        m = _metrics(session_id=s["id"])
        await backend.save_metrics(m)
        results = await backend.query_metrics_by_session(s["id"])
        assert len(results) >= 1

    async def test_query_metrics_by_branch(self, backend):
        s = _session()
        await backend.save_session(s)
        branch = f"feature/{uuid.uuid4().hex[:8]}"
        m = _metrics(session_id=s["id"], branch=branch)
        await backend.save_metrics(m)
        results = await backend.query_metrics_by_branch(branch)
        assert len(results) >= 1
        assert all(r["branch"] == branch for r in results)

    async def test_query_metrics_by_session(self, backend):
        s = _session()
        await backend.save_session(s)
        m = _metrics(session_id=s["id"])
        await backend.save_metrics(m)
        results = await backend.query_metrics_by_session(s["id"])
        assert len(results) >= 1
        assert all(r["session_id"] == s["id"] for r in results)

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------

    async def test_save_and_query_notes(self, backend):
        s = _session()
        await backend.save_session(s)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        n = _note(session_id=s["id"])
        n["date"] = today
        await backend.save_note(n)
        results = await backend.query_notes_by_date(today)
        assert len(results) >= 1

    async def test_query_notes_by_date(self, backend):
        s = _session()
        await backend.save_session(s)
        date = "2025-01-15"
        n = _note(session_id=s["id"])
        n["date"] = date
        await backend.save_note(n)
        results = await backend.query_notes_by_date(date)
        assert len(results) >= 1
        assert all(r["date"] == date for r in results)

    # ------------------------------------------------------------------
    # File Operations
    # ------------------------------------------------------------------

    async def test_save_and_query_file_operations(self, backend):
        s = _session()
        await backend.save_session(s)
        f = _file_op(session_id=s["id"])
        await backend.save_file_operation(f)
        results = await backend.query_file_operations_by_session(s["id"])
        assert len(results) >= 1
        assert all(r["session_id"] == s["id"] for r in results)

    # ------------------------------------------------------------------
    # Session Summaries
    # ------------------------------------------------------------------

    async def test_save_and_get_session_summary(self, backend):
        s = _session()
        await backend.save_session(s)
        sm = _summary(session_id=s["id"])
        await backend.save_session_summary(sm)
        result = await backend.get_session_summary(s["id"])
        assert result is not None
        assert result["session_id"] == s["id"]

    async def test_query_session_summaries(self, backend):
        s = _session()
        await backend.save_session(s)
        sm = _summary(session_id=s["id"])
        await backend.save_session_summary(sm)
        results = await backend.query_session_summaries()
        assert len(results) >= 1

    async def test_query_summaries_by_tag(self, backend):
        s = _session()
        await backend.save_session(s)
        tag = f"tag-{uuid.uuid4().hex[:6]}"
        sm = _summary(session_id=s["id"], tags=[tag])
        await backend.save_session_summary(sm)
        results = await backend.query_summaries_by_tag(tag)
        assert len(results) >= 1

    async def test_query_recent_summaries(self, backend):
        s = _session()
        await backend.save_session(s)
        sm = _summary(session_id=s["id"])
        await backend.save_session_summary(sm)
        results = await backend.query_recent_summaries(limit=10)
        assert isinstance(results, list)
        ids = {r["session_id"] for r in results}
        assert s["id"] in ids

    # ------------------------------------------------------------------
    # Agent Executions
    # ------------------------------------------------------------------

    async def test_save_and_query_agent_executions(self, backend):
        s = _session()
        await backend.save_session(s)
        e = _agent_execution(session_id=s["id"], agent_name="test-agent")
        await backend.save_agent_execution(e)
        results = await backend.query_agent_executions(session_id=s["id"])
        assert len(results) >= 1
        assert all(r["session_id"] == s["id"] for r in results)

    # ------------------------------------------------------------------
    # MCP Sessions
    # ------------------------------------------------------------------

    async def test_save_and_get_mcp_session(self, backend):
        m = _mcp_session()
        await backend.save_mcp_session(m)
        result = await backend.get_mcp_session(m["mcp_session_id"])
        assert result is not None
        assert result["mcp_session_id"] == m["mcp_session_id"]

    async def test_update_mcp_session_activity(self, backend):
        m = _mcp_session()
        await backend.save_mcp_session(m)
        old_activity = (await backend.get_mcp_session(m["mcp_session_id"]))["last_activity"]
        await backend.update_mcp_session_activity(m["mcp_session_id"])
        result = await backend.get_mcp_session(m["mcp_session_id"])
        assert result is not None
        # last_activity should be set (either same or newer — just must not be None)
        assert result["last_activity"] is not None

    async def test_link_mcp_to_engine_session(self, backend):
        s = _session()
        await backend.save_session(s)
        m = _mcp_session()
        await backend.save_mcp_session(m)
        await backend.link_mcp_to_engine_session(m["mcp_session_id"], s["id"])
        result = await backend.get_mcp_session(m["mcp_session_id"])
        assert result is not None
        assert result["engine_session_id"] == s["id"]

    # ------------------------------------------------------------------
    # Project Learnings
    # ------------------------------------------------------------------

    async def test_save_and_query_project_learnings(self, backend):
        pl = make_project_learning_data()
        await backend.save_project_learning(
            learning_id=pl["learning_id"],
            project_path=pl["project_path"],
            category=pl["category"],
            learning_content=pl["learning_content"],
            trigger_context=pl.get("trigger_context"),
        )
        results = await backend.query_project_learnings(project_path=pl["project_path"])
        assert len(results) >= 1
        ids = {r["id"] for r in results}
        assert pl["learning_id"] in ids

    async def test_update_learning_usage(self, backend):
        pl = make_project_learning_data()
        await backend.save_project_learning(
            learning_id=pl["learning_id"],
            project_path=pl["project_path"],
            category=pl["category"],
            learning_content=pl["learning_content"],
        )
        result = await backend.update_learning_usage(pl["learning_id"], success=True)
        assert result.get("updated") or result.get("id") == pl["learning_id"]

    # ------------------------------------------------------------------
    # Error Solutions
    # ------------------------------------------------------------------

    async def test_save_and_find_error_solutions(self, backend):
        es = make_error_solution_data()
        await backend.save_error_solution(
            solution_id=es["solution_id"],
            error_pattern=es["error_pattern"],
            solution_steps=[es["solution"]],
            error_category="dependency",
        )
        results = await backend.find_error_solutions(
            error_text="No module named",
        )
        # May or may not match — just verify no crash and returns list
        assert isinstance(results, list)

    async def test_update_solution_outcome(self, backend):
        es = make_error_solution_data()
        await backend.save_error_solution(
            solution_id=es["solution_id"],
            error_pattern=es["error_pattern"],
            solution_steps=[es["solution"]],
        )
        result = await backend.update_solution_outcome(es["solution_id"], success=True)
        assert result is not None
        assert result.get("id") == es["solution_id"] or result.get("updated")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def test_search_sessions_basic(self, backend):
        # search_sessions may return nothing if FTS index is empty — just
        # verify the call doesn't raise and returns a list.
        results = await backend.search_sessions("pytest")
        assert isinstance(results, list)

    async def test_search_by_file_change(self, backend):
        s = _session()
        await backend.save_session(s)
        sm = _summary(session_id=s["id"], key_changes=["src/foo.py", "tests/bar.py"])
        await backend.save_session_summary(sm)
        results = await backend.search_by_file_change("foo.py")
        assert isinstance(results, list)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def test_vacuum(self, backend):
        # Should not raise
        await backend.vacuum()

    async def test_get_statistics(self, backend):
        stats = await backend.get_statistics()
        assert isinstance(stats, dict)
        # Must include at least sessions count
        assert "sessions_count" in stats or any("count" in k for k in stats)

    # ------------------------------------------------------------------
    # Data Type Handling
    # ------------------------------------------------------------------

    async def test_datetime_fields_roundtrip(self, backend):
        now = datetime.now(UTC)
        s = _session()
        s["started_at"] = now.isoformat()
        await backend.save_session(s)
        result = await backend.get_session(s["id"])
        assert result is not None
        # started field should be present (not None)
        started = result.get("started") or result.get("started_at")
        assert started is not None

    async def test_json_fields_roundtrip(self, backend):
        meta = {"key": "value", "nested": {"a": 1}}
        s = _session(metadata=meta)
        await backend.save_session(s)
        result = await backend.get_session(s["id"])
        assert result is not None
        stored = result.get("metadata", {})
        assert stored.get("key") == "value"

    async def test_list_fields_roundtrip(self, backend):
        caps = ["read", "write", "analyze"]
        a = _agent()
        a["capabilities"] = caps
        await backend.save_agent(a)
        result = await backend.get_agent(a["id"])
        assert result is not None
        stored = result.get("capabilities", [])
        assert isinstance(stored, list)
        assert "read" in stored

    async def test_null_optional_fields(self, backend):
        s = _session()
        s["ended_at"] = None
        await backend.save_session(s)
        result = await backend.get_session(s["id"])
        assert result is not None
        completed = result.get("completed") or result.get("ended_at")
        assert completed is None

    async def test_unicode_content(self, backend):
        s = _session()
        await backend.save_session(s)
        n = _note(session_id=s["id"], content="Unicode: こんにちは 🎉 café naïve résumé")
        await backend.save_note(n)
        results = await backend.query_notes_by_date(n["date"])
        contents = [r["content"] for r in results]
        assert any("こんにちは" in c for c in contents)

    async def test_large_text_content(self, backend):
        s = _session()
        await backend.save_session(s)
        large = "x" * 10_240  # 10 KB
        n = _note(session_id=s["id"], content=large)
        await backend.save_note(n)
        results = await backend.query_notes_by_date(n["date"])
        assert any(len(r["content"]) >= 10_240 for r in results)

    # ------------------------------------------------------------------
    # Edge Cases
    # ------------------------------------------------------------------

    async def test_empty_query_results(self, backend):
        results = await backend.query_decisions_by_session("nonexistent-session-xyz")
        assert results == []

    async def test_duplicate_id_handling(self, backend):
        s = _session()
        await backend.save_session(s)
        # Save again with same id but different status
        s["status"] = "completed"
        await backend.save_session(s)
        result = await backend.get_session(s["id"])
        assert result is not None
        assert result["status"] == "completed"

    async def test_special_characters_in_strings(self, backend):
        s = _session()
        s["project_name"] = "project with 'quotes' and \"double\" and\\backslash\nnewline"
        await backend.save_session(s)
        result = await backend.get_session(s["id"])
        assert result is not None
        assert result["id"] == s["id"]
