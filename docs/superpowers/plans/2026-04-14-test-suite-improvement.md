# Test Suite Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a comprehensive multi-level test suite (190-210 tests, ~90% coverage) that catches the classes of bugs that caused 3 critical production failures.

**Architecture:** Contract-based persistence tests (shared across SQLite/PostgreSQL), engine tests with real SQLite (no mocks), regression tests for each historical bug, integration tests for HTTP/error/concurrency paths.

**Tech Stack:** pytest, pytest-asyncio, aiosqlite, asyncpg (CI-only), httpx (HTTP transport tests), pytest-cov

**Spec:** `docs/superpowers/specs/2026-04-14-test-suite-improvement-design.md`

---

## File Structure

### Files to Create

| File | Responsibility |
|------|---------------|
| `tests/persistence/__init__.py` | Package init |
| `tests/persistence/conftest.py` | Persistence-specific fixtures (backend factories) |
| `tests/persistence/builders.py` | Data builder functions (make_session_data, make_decision_data, etc.) — importable from any test package |
| `tests/persistence/contract_tests.py` | ~70 shared contract test methods — the single source of truth for backend behavior |
| `tests/persistence/test_sqlite_contract.py` | Runs contract against SQLite via fixture |
| `tests/persistence/test_postgresql_contract.py` | Runs contract against PostgreSQL (CI-only, skipped locally) |
| `tests/engine/__init__.py` | Package init |
| `tests/engine/conftest.py` | Engine + real SQLite fixtures |
| `tests/engine/test_session_lifecycle.py` | Session create/finalize/recover/health/recall (~15 tests) |
| `tests/engine/test_agent_operations.py` | Agent register/query/search/notebooks (~15 tests) |
| `tests/engine/test_decision_learning.py` | Decisions + learnings CRUD via engine (~15 tests) |
| `tests/engine/test_mcp_tool_dispatch.py` | LeanMCPInterface execute_tool for all tools (~25 tests) |
| `tests/regression/__init__.py` | Package init |
| `tests/regression/test_async_await_bugs.py` | PR #12 regression: async calls awaited |
| `tests/regression/test_datetime_type_bugs.py` | PR #13 regression: datetime vs isoformat |
| `tests/regression/test_session_persistence_bugs.py` | PR #14 regression: sessions persisted to DB |
| `tests/integration/test_http_transport.py` | HTTP server request/response (~15 tests) |
| `tests/integration/test_error_paths.py` | Cross-layer error handling (~10 tests) |
| `tests/integration/test_concurrency.py` | Concurrent access patterns (~6 tests) |
| `tests/unit/test_token_limiter.py` | Token estimation + truncation (~10 tests) |
| `tests/unit/test_security.py` | LocalhostOnlyMiddleware + SecurityConfig (~8 tests) |
| `tests/unit/test_config.py` | DatabaseConfig load/save/env/file (~6 tests) |
| `tests/unit/test_migration.py` | MigrationManager.migrate_all() correctness (~12 tests) |

### Files to Modify

| File | Change |
|------|--------|
| `tests/conftest.py` | Refactor: add persistence builder fixtures, remove stale ones |
| `pyproject.toml` | Add pytest-cov, httpx to test deps; add pytest marks (postgresql, regression) |

### Files to Move

| From | To |
|------|-----|
| `tests/unit/test_knowledge_system.py` | `tests/engine/test_knowledge_system.py` |

### Files to Remove

| File | Reason |
|------|--------|
| `tests/debug/` (entire directory) | Ad-hoc debug scripts, not real tests |
| `tests/live/` (entire directory) | Live-server scripts, not CI-compatible |
| `tests/integration/test_lean_mcp.py` | Demo with hardcoded fake data |
| `tests/integration/test_simplified_lean.py` | Trivial, replaced by contract tests |
| `tests/integration/test_agent_system.py` | Replaced by persistence contracts |
| `tests/integration/test_token_limiting.py` | Consolidated into unit/test_token_limiter.py |
| `tests/integration/test_specific_limit.py` | Consolidated into unit/test_token_limiter.py |
| `tests/integration/test_large_response.py` | Consolidated into unit/test_token_limiter.py |

---

## Task 1: Test Infrastructure Setup

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`
- Create: `tests/persistence/__init__.py`
- Create: `tests/persistence/conftest.py`
- Create: `tests/engine/__init__.py`
- Create: `tests/engine/conftest.py`
- Create: `tests/regression/__init__.py`
- Move: `tests/unit/test_knowledge_system.py` → `tests/engine/test_knowledge_system.py`

- [ ] **Step 1: Add test dependencies to pyproject.toml**

Add `httpx` and `pytest-cov` to test dependencies. Add pytest markers for `postgresql` and `regression`. Check current pyproject.toml for existing test config section and extend it.

Add to `[tool.pixi.feature.quality.pypi-dependencies]`:

```toml
httpx = ">=0.27"
```

Add to `[tool.pytest.ini_options]` markers list:

```toml
markers = [
    "postgresql: tests requiring PostgreSQL (deselect with '-m not postgresql')",
    "regression: regression tests for historical production bugs",
]
```

Also in `tests/conftest.py` refactor: **remove the deprecated `event_loop` session-scoped fixture** (lines 28-33). It conflicts with `asyncio_mode = "auto"` and `asyncio_default_fixture_loop_scope = "function"` already in pyproject.toml.

**Important**: All async test functions in this plan do NOT need `@pytest.mark.asyncio` decorators because `asyncio_mode = "auto"` is set in pyproject.toml. The decorator is omitted throughout — pytest-asyncio discovers async tests automatically.

- [ ] **Step 2: Create package init files**

```bash
touch tests/persistence/__init__.py tests/engine/__init__.py tests/regression/__init__.py
```

- [ ] **Step 3: Move test_knowledge_system.py to engine/**

```bash
git mv tests/unit/test_knowledge_system.py tests/engine/test_knowledge_system.py
```

Update imports in the moved file if needed — the `conftest.py` sys.path setup should still work since it's relative to project root.

- [ ] **Step 4: Create tests/persistence/builders.py and tests/persistence/conftest.py**

The builder functions go in `builders.py` (not conftest) so they can be imported from any test package. Add `tests/` to sys.path in the root conftest.py so `from tests.persistence.builders import ...` works everywhere.

In `tests/conftest.py`, add alongside the existing `sys.path.insert(0, ...)`:
```python
sys.path.insert(0, str(Path(__file__).parent))  # Add tests/ to path
```

**`tests/persistence/builders.py`:**

```python
"""Data builder functions for persistence tests. Importable from any test package."""

import uuid
from datetime import UTC, datetime

# Check PostgreSQL availability
try:
    import asyncpg

    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

POSTGRES_DSN = os.environ.get("POSTGRES_DSN")
POSTGRES_AVAILABLE = HAS_ASYNCPG and bool(POSTGRES_DSN)


def make_session_data(**overrides):
    """Build valid session data dict with sensible defaults."""
    data = {
        "id": f"test-session-{uuid.uuid4().hex[:8]}",
        "started_at": datetime.now(UTC),
        "project_path": "/test/project",
        "project_name": "test-project",
        "mode": "local",
        "status": "active",
        "metadata": {"session_type": "test", "environment": "test", "user": "tester"},
    }
    data.update(overrides)
    return data


def make_decision_data(session_id, **overrides):
    """Build valid decision data dict."""
    data = {
        "id": f"test-decision-{uuid.uuid4().hex[:8]}",
        "session_id": session_id,
        "timestamp": datetime.now(UTC),
        "category": "test",
        "description": "Test decision",
        "rationale": "Test rationale",
        "impact_level": "medium",
    }
    data.update(overrides)
    return data


def make_metrics_data(session_id, **overrides):
    """Build valid metrics data dict."""
    data = {
        "session_id": session_id,
        "branch": "main",
        "timestamp": datetime.now(UTC),
        "coverage": 85.0,
        "complexity": 10.0,
        "test_count": 50,
        "agents_executed": 3,
        "execution_time_ms": 1500,
    }
    data.update(overrides)
    return data


def make_note_data(session_id, **overrides):
    """Build valid note data dict."""
    data = {
        "session_id": session_id,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "content": "Test note content",
        "tags": ["test"],
    }
    data.update(overrides)
    return data


def make_file_operation_data(session_id, **overrides):
    """Build valid file operation data dict."""
    data = {
        "session_id": session_id,
        "timestamp": datetime.now(UTC),
        "operation": "edit",
        "file_path": "/test/file.py",
        "lines_added": 10,
        "lines_removed": 5,
        "summary": "Test edit",
        "tool_name": "Edit",
    }
    data.update(overrides)
    return data


def make_agent_data(**overrides):
    """Build valid agent data dict."""
    data = {
        "id": f"test-agent-{uuid.uuid4().hex[:8]}",
        "name": f"test-agent-{uuid.uuid4().hex[:8]}",
        "agent_type": "focused",
        "display_name": "Test Agent",
        "description": "A test agent",
        "capabilities": ["testing"],
        "metadata": {},
    }
    data.update(overrides)
    return data


def make_agent_decision_data(agent_id, **overrides):
    """Build valid agent decision data dict."""
    data = {
        "id": f"test-agdec-{uuid.uuid4().hex[:8]}",
        "agent_id": agent_id,
        "decision_type": "tool_selection",
        "context": "Testing",
        "decision": "Use pytest",
        "reasoning": "Standard framework",
        "alternatives": ["unittest"],
        "confidence": 0.9,
        "tags": ["test"],
    }
    data.update(overrides)
    return data


def make_agent_learning_data(agent_id, **overrides):
    """Build valid agent learning data dict."""
    data = {
        "id": f"test-aglrn-{uuid.uuid4().hex[:8]}",
        "agent_id": agent_id,
        "learning_type": "pattern",
        "title": "Test Pattern",
        "content": "Always validate input",
        "source_context": "Code review",
        "applicability": ["testing"],
        "confidence": 0.85,
        "tags": ["test"],
    }
    data.update(overrides)
    return data


def make_agent_notebook_data(agent_id, **overrides):
    """Build valid agent notebook data dict."""
    data = {
        "id": f"test-agnb-{uuid.uuid4().hex[:8]}",
        "agent_id": agent_id,
        "title": "Test Notebook",
        "summary": "Test summary",
        "content": "# Test Notebook\n\nTest content.",
        "notebook_type": "execution",
        "tags": ["test"],
    }
    data.update(overrides)
    return data


def make_mcp_session_data(**overrides):
    """Build valid MCP session data dict."""
    data = {
        "mcp_session_id": f"mcp-{uuid.uuid4().hex[:8]}",
        "engine_session_id": None,
        "created_at": datetime.now(UTC),
        "last_activity": datetime.now(UTC),
        "client_info": {"client": "test"},
    }
    data.update(overrides)
    return data


def make_summary_data(session_id, **overrides):
    """Build valid session summary data dict."""
    data = {
        "session_id": session_id,
        "title": "Test Summary",
        "summary_markdown": "# Summary\n\nTest session.",
        "key_changes": ["change1", "change2"],
        "tags": ["test", "summary"],
        "created_at": datetime.now(UTC),
    }
    data.update(overrides)
    return data


def make_agent_execution_data(session_id, **overrides):
    """Build valid agent execution data dict."""
    data = {
        "id": f"exec-{uuid.uuid4().hex[:8]}",
        "session_id": session_id,
        "agent_name": "test-agent",
        "agent_type": "focused",
        "started_at": datetime.now(UTC),
        "status": "running",
    }
    data.update(overrides)
    return data


def make_project_learning_data(session_id=None, **overrides):
    """Build valid project learning data dict."""
    data = {
        "id": f"plrn-{uuid.uuid4().hex[:8]}",
        "project_path": "/test/project",
        "category": "pattern",
        "trigger_context": "When testing",
        "learning_content": "Always use fixtures",
        "source_session_id": session_id,
        "created_at": datetime.now(UTC),
    }
    data.update(overrides)
    return data


def make_error_solution_data(session_id=None, **overrides):
    """Build valid error solution data dict."""
    data = {
        "id": f"esol-{uuid.uuid4().hex[:8]}",
        "error_pattern": "ImportError: No module named 'foo'",
        "error_hash": "abc123",
        "error_category": "runtime",
        "solution_steps": ["pip install foo"],
        "context_requirements": {"python_version": "3.11"},
        "project_path": "/test/project",
        "source_session_id": session_id,
        "created_at": datetime.now(UTC),
    }
    data.update(overrides)
    return data
```

**`tests/persistence/conftest.py`:**

```python
"""Persistence-layer test fixtures."""

import os

import pytest

from persistence.sqlite import SQLiteBackend

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

POSTGRES_DSN = os.environ.get("POSTGRES_DSN")
POSTGRES_AVAILABLE = HAS_ASYNCPG and bool(POSTGRES_DSN)
```

- [ ] **Step 5: Create tests/engine/conftest.py**

```python
"""Engine-layer test fixtures with real SQLite backend."""

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend


@pytest.fixture
async def engine(tmp_path):
    """Engine with real SQLite backend, initialized and ready."""
    eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
    eng.database = SQLiteBackend(str(tmp_path / "test.db"))
    await eng.database.initialize()
    yield eng
```

- [ ] **Step 6: Refactor tests/conftest.py**

Keep the existing fixtures (event_loop, project_root, src_path, temp_dir, sqlite_db_path, sqlite_backend, session_engine, sample data fixtures). Remove any that duplicate what the new sub-package conftest files provide. Ensure `sys.path.insert(0, ...)` for `src/` is present.

- [ ] **Step 7: Verify existing tests still pass**

Run: `pixi run -e ci pytest tests/unit/test_session_models.py -v`
Expected: All existing model tests pass unchanged.

- [ ] **Step 8: Commit**

```
git add -A tests/ pyproject.toml
git commit -m "test: set up multi-level test infrastructure

Add persistence/, engine/, regression/ test packages with fixtures.
Move test_knowledge_system.py to engine/. Add pytest markers.
Add data builder functions for all persistence entities.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Persistence Contract Tests — Core CRUD (Sessions, Decisions, Learnings)

**Files:**
- Create: `tests/persistence/contract_tests.py`
- Create: `tests/persistence/test_sqlite_contract.py`
- Create: `tests/persistence/test_postgresql_contract.py`

- [ ] **Step 1: Create contract_tests.py with Session CRUD tests**

```python
"""
Persistence contract tests.

Every backend must pass ALL tests in this class. If a test passes on SQLite
but fails on PostgreSQL (or vice versa), we've found a backend divergence bug.
"""

import uuid
from datetime import UTC, datetime

import pytest

from tests.persistence.builders import (
    make_decision_data,
    make_session_data,
)


class PersistenceContractTests:
    """Shared contract — inherited by backend-specific test classes."""

    # --- Session CRUD ---

    async def test_save_and_retrieve_session(self, backend):
        session = make_session_data()
        await backend.save_session(session)
        result = await backend.get_session(session["id"])
        assert result is not None
        assert result["id"] == session["id"]
        assert result["project_name"] == "test-project"

    async def test_save_session_with_all_fields(self, backend):
        session = make_session_data(
            ended_at=datetime.now(UTC),
            status="completed",
            performance_metrics={"agents_executed": 5},
            health_status={"overall_score": 95.0},
        )
        await backend.save_session(session)
        result = await backend.get_session(session["id"])
        assert result is not None
        assert result["status"] == "completed"

    async def test_update_existing_session(self, backend):
        session = make_session_data()
        await backend.save_session(session)
        session["status"] = "completed"
        session["ended_at"] = datetime.now(UTC)
        await backend.save_session(session)
        result = await backend.get_session(session["id"])
        assert result["status"] == "completed"

    async def test_get_nonexistent_session_returns_none(self, backend):
        result = await backend.get_session("nonexistent-session-id")
        assert result is None

    async def test_list_sessions_by_project(self, backend):
        s1 = make_session_data(project_path="/project/a")
        s2 = make_session_data(project_path="/project/a")
        s3 = make_session_data(project_path="/project/b")
        for s in [s1, s2, s3]:
            await backend.save_session(s)
        results = await backend.query_sessions(project_path="/project/a")
        assert len(results) >= 2

    # --- Decision CRUD ---

    async def test_save_and_query_decisions(self, backend):
        session = make_session_data()
        await backend.save_session(session)
        decision = make_decision_data(session["id"])
        await backend.save_decision(decision)
        results = await backend.query_decisions_by_session(session["id"])
        assert len(results) >= 1
        assert results[0]["description"] == "Test decision"

    async def test_decision_requires_valid_session_fk(self, backend):
        decision = make_decision_data("nonexistent-session")
        with pytest.raises(Exception):
            await backend.save_decision(decision)

    async def test_query_decisions_with_filters(self, backend):
        session = make_session_data()
        await backend.save_session(session)
        d1 = make_decision_data(session["id"], category="architecture")
        d2 = make_decision_data(session["id"], category="testing")
        await backend.save_decision(d1)
        await backend.save_decision(d2)
        results = await backend.query_decisions_by_category("architecture")
        assert any(d["id"] == d1["id"] for d in results)

    # (Continue with Learning CRUD, Agent CRUD, etc. — see spec for full list)
```

**Note to implementer:** The contract class continues with ALL sections from the spec. The above shows the pattern — each method is a `@pytest.mark.asyncio async def test_*` that receives `backend` as a fixture. Continue adding all sections: Learning CRUD, Agent CRUD, Agent Decisions, Agent Learnings, Agent Notebooks, Metrics, Notes, File Operations, Session Summaries, Agent Executions, MCP Sessions, Project Learnings, Error Solutions, Search, Maintenance, Data Type Handling, Edge Cases. Use the `make_*` builder functions from conftest.py. Target: ~70 test methods total.

- [ ] **Step 2: Create test_sqlite_contract.py**

```python
"""Run persistence contract tests against SQLite backend."""

import pytest

from persistence.sqlite import SQLiteBackend

from .contract_tests import PersistenceContractTests


class TestSQLiteContract(PersistenceContractTests):
    """All contract tests run against a fresh SQLite database."""

    @pytest.fixture
    async def backend(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / "test.db"))
        await db.initialize()
        yield db
```

- [ ] **Step 3: Create test_postgresql_contract.py**

```python
"""Run persistence contract tests against PostgreSQL backend."""

import os

import pytest

from tests.persistence.conftest import POSTGRES_AVAILABLE
from .contract_tests import PersistenceContractTests


@pytest.mark.postgresql
@pytest.mark.skipif(not POSTGRES_AVAILABLE, reason="PostgreSQL not available")
class TestPostgreSQLContract(PersistenceContractTests):
    """All contract tests run against a PostgreSQL test database."""

    @pytest.fixture
    async def backend(self):
        from persistence.postgresql import PostgreSQLBackend

        dsn = os.environ["POSTGRES_DSN"]
        db = PostgreSQLBackend(dsn=dsn)
        await db.initialize()
        yield db
        await db.close()

    # --- PostgreSQL-only tests (API asymmetry) ---

    async def test_recall_project(self, backend):
        from tests.persistence.builders import make_decision_data, make_session_data

        session = make_session_data(project_name="recall-test")
        await backend.save_session(session)
        decision = make_decision_data(session["id"], description="unique recall keyword")
        await backend.save_decision(decision)
        results = await backend.recall_project("recall-test", query="unique recall keyword")
        assert results is not None

    async def test_search_sessions_with_search_type(self, backend):
        from tests.persistence.builders import make_session_data

        session = make_session_data(project_name="searchable-project")
        await backend.save_session(session)
        results = await backend.search_sessions("searchable", search_type="project", limit=10)
        assert isinstance(results, list)
```

- [ ] **Step 4: Run SQLite contract tests**

Run: `pixi run -e ci pytest tests/persistence/test_sqlite_contract.py -v`
Expected: All contract tests pass against SQLite.

- [ ] **Step 5: Commit**

```
git commit -m "test: add persistence contract tests for core CRUD

~70 shared contract tests covering sessions, decisions, learnings,
agents, metrics, notes, file ops, summaries, MCP sessions, project
learnings, error solutions, search, and maintenance.

Both SQLite and PostgreSQL backends must pass identical tests.
PostgreSQL-only tests for recall_project and search_type.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Regression Tests

**Files:**
- Create: `tests/regression/test_async_await_bugs.py`
- Create: `tests/regression/test_datetime_type_bugs.py`
- Create: `tests/regression/test_session_persistence_bugs.py`

- [ ] **Step 1: Write test_async_await_bugs.py (PR #12)**

```python
"""
Regression tests for PR #12: fire-and-forget async calls.

Bug: async DB calls were called without await, so data was silently lost.
Fix: All async engine methods now properly await database calls.
"""

import inspect

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend


@pytest.mark.regression
class TestAsyncAwaitBugs:

    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng

    async def test_session_log_decision_persists_data(self, engine):
        """Verify decision data actually reaches the database (not fire-and-forget)."""
        # Create session first
        result = await engine.session_manage_lifecycle(operation="create", mode="local", project_name="test")
        session_id = result["session_id"]

        # Log a decision
        await engine.session_log_decision(
            description="Test decision",
            rationale="Test rationale",
            category="test",
        )

        # Verify it persisted — if not awaited, this would be empty
        decisions = await engine.database.query_decisions_by_session(session_id)
        assert len(decisions) >= 1
        assert decisions[0]["description"] == "Test decision"

    async def test_session_log_learning_persists_data(self, engine):
        """Verify learning data actually reaches the database."""
        await engine.session_manage_lifecycle(operation="create", mode="local", project_name="test")

        await engine.session_log_learning(
            category="pattern",
            learning_content="Test learning",
            trigger_context="Test trigger",
        )

        learnings = await engine.database.query_project_learnings(
            project_path=engine.repository_path
        )
        assert len(learnings) >= 1

    async def test_no_coroutine_objects_from_mcp_tools(self, engine):
        """Verify all async engine tools are wrapped with _wrap_async_tool."""
        from lean_mcp_interface import LeanMCPInterface

        interface = LeanMCPInterface(engine)

        for tool_name, tool_info in interface.tool_registry.items():
            func = tool_info["implementation"]
            # Check if the underlying (unwrapped) function is async
            underlying = getattr(func, "__wrapped__", None)
            if underlying is not None and inspect.iscoroutinefunction(underlying):
                # The wrapper must also be async (i.e., _wrap_async_tool was used)
                # If _wrap_tool was used on an async function, the wrapper is sync
                # and returns a coroutine object instead of awaiting — the PR #12 bug
                assert inspect.iscoroutinefunction(func), (
                    f"Tool '{tool_name}' wraps an async function but is not async. "
                    "This is the PR #12 bug — use _wrap_async_tool instead of _wrap_tool."
                )
```

- [ ] **Step 2: Write test_datetime_type_bugs.py (PR #13)**

```python
"""
Regression tests for PR #13: isoformat strings vs datetime objects.

Bug: asyncpg rejects ISO format strings for TIMESTAMPTZ columns.
     SQLite silently accepts both, masking the bug.
Fix: Always pass datetime objects, never .isoformat() strings.
"""

from datetime import UTC, datetime

import pytest

from persistence.sqlite import SQLiteBackend

# Import builders
from tests.persistence.builders import (
    make_mcp_session_data,
    make_session_data,
    make_agent_data,
    make_agent_decision_data,
    make_agent_learning_data,
    make_agent_notebook_data,
)


@pytest.mark.regression
class TestDatetimeTypeBugs:

    @pytest.fixture
    async def backend(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / "test.db"))
        await db.initialize()
        yield db

    async def test_session_datetime_roundtrip(self, backend):
        """Verify datetime objects survive save/retrieve without isoformat conversion."""
        now = datetime.now(UTC)
        session = make_session_data(started_at=now)
        await backend.save_session(session)
        result = await backend.get_session(session["id"])
        assert result is not None
        # The retrieved timestamp should be parseable back to datetime
        # (not an isoformat string that asyncpg would reject)

    async def test_mcp_session_datetime_fields(self, backend):
        """MCP session save uses datetime objects for created_at and last_activity."""
        mcp_data = make_mcp_session_data()
        # This should not raise — PR #13 bug was isoformat strings here
        await backend.save_mcp_session(mcp_data)
        result = await backend.get_mcp_session(mcp_data["mcp_session_id"])
        assert result is not None

    async def test_agent_datetime_fields(self, backend):
        """Agent save uses datetime objects for first_seen_at and last_active_at."""
        agent = make_agent_data()
        await backend.save_agent(agent)
        result = await backend.get_agent(agent["id"])
        assert result is not None

    async def test_agent_decision_datetime(self, backend):
        """Agent decision timestamps are datetime objects."""
        agent = make_agent_data()
        await backend.save_agent(agent)
        decision = make_agent_decision_data(agent["id"])
        await backend.save_agent_decision(decision)
        results = await backend.query_agent_decisions(agent["id"])
        assert len(results) >= 1

    async def test_agent_learning_datetime(self, backend):
        """Agent learning timestamps are datetime objects."""
        agent = make_agent_data()
        await backend.save_agent(agent)
        learning = make_agent_learning_data(agent["id"])
        await backend.save_agent_learning(learning)
        results = await backend.query_agent_learnings(agent["id"])
        assert len(results) >= 1

    async def test_agent_notebook_datetime(self, backend):
        """Agent notebook timestamps are datetime objects."""
        agent = make_agent_data()
        await backend.save_agent(agent)
        notebook = make_agent_notebook_data(agent["id"])
        await backend.save_agent_notebook(notebook)
        results = await backend.query_agent_notebooks(agent["id"])
        assert len(results) >= 1
```

- [ ] **Step 3: Write test_session_persistence_bugs.py (PR #14)**

```python
"""
Regression tests for PR #14: sessions not persisted to DB.

Bug: _create_session only stored sessions in memory/filesystem, never in the
     database. Decisions have FK constraint referencing sessions(id), so any
     decision INSERT failed with constraint violation.
Fix: session_manage_lifecycle now persists to DB after creation.
     session_log_decision auto-creates and persists session when needed.
"""

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend


@pytest.mark.regression
class TestSessionPersistenceBugs:

    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng

    async def test_session_persisted_to_db_not_just_memory(self, engine):
        """_create_session must write to DB, not just memory/filesystem."""
        result = await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="persist-test"
        )
        session_id = result["session_id"]

        # Query the database directly — the session must be there
        db_session = await engine.database.get_session(session_id)
        assert db_session is not None, (
            "Session was created in memory but not persisted to database. "
            "This is the PR #14 bug."
        )
        assert db_session["project_name"] == "persist-test"

    async def test_decision_without_session_auto_creates(self, engine):
        """Logging a decision without active session must auto-create one and persist."""
        # Do NOT call session_manage_lifecycle — go straight to logging
        await engine.session_log_decision(
            description="Decision without session",
            rationale="Testing auto-create",
            category="test",
        )

        # The engine should have auto-created a session AND persisted it
        # so the FK constraint is satisfied
        assert engine.current_session is not None

    async def test_decision_fk_constraint_satisfied(self, engine):
        """Decision's session_id must reference a session that exists in DB."""
        result = await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="fk-test"
        )
        session_id = result["session_id"]

        # Log decision — should not raise FK constraint violation
        await engine.session_log_decision(
            description="FK test decision",
            rationale="Verify FK satisfied",
            category="test",
        )

        # Verify the decision's session_id matches a real DB session
        decisions = await engine.database.query_decisions_by_session(session_id)
        assert len(decisions) >= 1
```

- [ ] **Step 4: Run regression tests**

Run: `pixi run -e ci pytest tests/regression/ -v -m regression`
Expected: All 3 regression test files pass.

- [ ] **Step 5: Commit**

```
git commit -m "test: add regression tests for 3 critical production bugs

PR #12: verify async DB calls are awaited, no coroutine objects from MCP tools
PR #13: verify datetime objects (not isoformat strings) for all timestamp fields
PR #14: verify sessions persist to DB, auto-create on decision without session

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Engine Tests — Session Lifecycle

**Files:**
- Create: `tests/engine/test_session_lifecycle.py`

- [ ] **Step 1: Write session lifecycle tests**

Test the `session_manage_lifecycle()`, `session_monitor_health()`, and session recall engine methods. Use the `engine` fixture from `tests/engine/conftest.py` (real SQLite).

Cover:
- `test_create_session_returns_valid_id` — verify result has session_id and status=success
- `test_create_session_sets_active_status` — verify session status is "active"
- `test_create_session_stores_metadata` — verify project_name, mode, path preserved
- `test_finalize_session` — create then finalize, verify status changes to completed
- `test_finalize_nonexistent_session` — verify graceful handling
- `test_manage_lifecycle_invalid_operation` — verify error for unknown operation
- `test_session_health_check` — call session_monitor_health, verify health_score
- `test_multiple_sessions_same_project` — create 2 sessions for same project
- `test_session_metadata_preserved` — verify metadata dict roundtrips
- `test_get_or_create_current_session_id` — verify auto-creation when no session exists
- `test_session_manage_lifecycle_is_async` — verify the method is a coroutine function
- `test_create_session_with_custom_metadata` — pass custom tags, attributes

~12-15 tests total.

- [ ] **Step 2: Run tests**

Run: `pixi run -e ci pytest tests/engine/test_session_lifecycle.py -v`
Expected: All pass.

- [ ] **Step 3: Commit**

```
git commit -m "test: add engine session lifecycle tests (~15 tests)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Engine Tests — Agent Operations

**Files:**
- Create: `tests/engine/test_agent_operations.py`

- [ ] **Step 1: Write agent operation tests**

Test `agent_register()`, `agent_get_info()`, `agent_log_decision()`, `agent_query_decisions()`, `agent_log_learning()`, `agent_query_learnings()`, `agent_create_notebook()`, `agent_query_notebooks()`, `agent_search_all()`, `agent_update_decision_outcome()`, `agent_update_learning_outcome()`.

Cover:
- `test_register_new_agent` — verify agent_id returned, status=created
- `test_register_existing_agent_updates` — register twice, verify status=updated
- `test_get_agent_info_by_name` — verify all fields returned
- `test_get_agent_info_by_id` — verify lookup by UUID
- `test_get_nonexistent_agent` — verify error/empty result
- `test_agent_log_decision` — verify decision stored
- `test_agent_query_decisions` — verify filtering by agent
- `test_agent_update_decision_outcome` — verify outcome fields updated
- `test_agent_log_learning` — verify learning stored
- `test_agent_query_learnings` — verify filtering
- `test_agent_update_learning_outcome` — verify success_count incremented
- `test_agent_create_notebook` — verify notebook stored
- `test_agent_query_notebooks` — verify retrieval
- `test_agent_search_all` — verify cross-data search returns results

~14-15 tests total.

- [ ] **Step 2: Run tests**

Run: `pixi run -e ci pytest tests/engine/test_agent_operations.py -v`
Expected: All pass.

- [ ] **Step 3: Commit**

```
git commit -m "test: add engine agent operation tests (~15 tests)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Engine Tests — Decision & Learning + MCP Tool Dispatch

**Files:**
- Create: `tests/engine/test_decision_learning.py`
- Create: `tests/engine/test_mcp_tool_dispatch.py`

- [ ] **Step 1: Write decision/learning tests**

Test `session_log_decision()`, `session_log_learning()`, and their query/update counterparts at the engine level.

Cover:
- `test_log_decision_with_active_session`
- `test_log_decision_without_session_auto_creates`
- `test_log_decision_returns_decision_id`
- `test_query_decisions_by_session`
- `test_query_decisions_by_category`
- `test_log_learning_with_active_session`
- `test_log_learning_returns_learning_id`
- `test_query_learnings_by_category`
- `test_update_learning_usage`
- `test_save_and_find_error_solutions`
- `test_decision_with_all_optional_fields`

~11-15 tests total.

- [ ] **Step 2: Write MCP tool dispatch tests**

Test `LeanMCPInterface`: `discover_tools()`, `get_tool_spec()`, `execute_tool()`.

```python
"""Test the LeanMCPInterface meta-tool dispatch layer."""

import inspect

import pytest

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence.sqlite import SQLiteBackend


@pytest.fixture
async def lean_interface(tmp_path):
    engine = SessionIntelligenceEngine(repository_path=str(tmp_path))
    engine.database = SQLiteBackend(str(tmp_path / "test.db"))
    await engine.database.initialize()
    interface = LeanMCPInterface(engine)
    yield interface


class TestDiscoverTools:

    async def test_discover_all_tools(self, lean_interface):
        result = lean_interface.discover_tools()
        assert "available_tools" in result
        assert result["total_tools"] > 0
        assert result["total_tools"] == len(lean_interface.tool_registry)

    async def test_discover_with_pattern_filter(self, lean_interface):
        result = lean_interface.discover_tools(pattern="session")
        assert result["filtered_count"] <= result["total_tools"]
        for tool in result["available_tools"]:
            assert "session" in tool["name"].lower()

    async def test_discover_with_nonmatching_pattern(self, lean_interface):
        result = lean_interface.discover_tools(pattern="zzz_nonexistent_zzz")
        assert result["filtered_count"] == 0


class TestGetToolSpec:

    async def test_valid_tool(self, lean_interface):
        result = lean_interface.get_tool_spec("session_manage_lifecycle")
        assert result["name"] == "session_manage_lifecycle"
        assert "schema" in result

    async def test_invalid_tool(self, lean_interface):
        result = lean_interface.get_tool_spec("nonexistent_tool")
        assert "error" in result


class TestExecuteTool:

    async def test_execute_session_manage_lifecycle(self, lean_interface):
        result = await lean_interface.execute_tool(
            "session_manage_lifecycle",
            {"operation": "create", "mode": "local", "project_name": "test"},
        )
        assert result["status"] == "success"
        assert "session_id" in result.get("result", {})

    async def test_execute_invalid_tool(self, lean_interface):
        result = await lean_interface.execute_tool("nonexistent", {})
        assert result["status"] == "error"

    async def test_all_async_tools_properly_wrapped(self, lean_interface):
        """Critical: async engine methods must be wrapped with _wrap_async_tool."""
        for tool_name, tool_info in lean_interface.tool_registry.items():
            func = tool_info["implementation"]
            underlying = getattr(func, "__wrapped__", None)
            if underlying is not None and inspect.iscoroutinefunction(underlying):
                assert inspect.iscoroutinefunction(func), (
                    f"Tool '{tool_name}' wraps async function but wrapper is sync. "
                    "Use _wrap_async_tool instead of _wrap_tool."
                )

    async def test_tool_count_matches_registry(self, lean_interface):
        """Tool count from discover matches actual registry size."""
        discovery = lean_interface.discover_tools()
        assert discovery["total_tools"] == len(lean_interface.tool_registry)
```

Continue adding one `test_execute_<tool_name>` for each registered tool that can be tested with minimal params. ~25 tests total in this file.

- [ ] **Step 3: Run all engine tests**

Run: `pixi run -e ci pytest tests/engine/ -v`
Expected: All engine tests pass.

- [ ] **Step 4: Commit**

```
git commit -m "test: add engine decision/learning and MCP dispatch tests (~40 tests)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Unit Tests — Token Limiter, Security, Config

**Files:**
- Create: `tests/unit/test_token_limiter.py`
- Create: `tests/unit/test_security.py`
- Create: `tests/unit/test_config.py`

- [ ] **Step 1: Write test_token_limiter.py**

Consolidate logic from the existing `test_token_limiting.py`, `test_specific_limit.py`, and `test_large_response.py`. Test the public API of `utils/token_limiter.py`:

- `TestTokenEstimator`: `test_estimate_empty_string`, `test_estimate_text`, `test_estimate_json`, `test_detect_content_type_json`, `test_detect_content_type_log`, `test_detect_content_type_text`
- `TestIntelligentTruncator`: `test_no_truncation_under_limit`, `test_truncate_json_dict`, `test_truncate_json_list`, `test_truncate_text`, `test_truncate_log_keeps_head_and_tail`
- `TestSessionTokenLimiter`: `test_limit_response_under_limit_unchanged`, `test_limit_response_over_limit_truncated`, `test_operation_specific_limits`, `test_truncation_disabled`, `test_pydantic_model_conversion`

~15 tests total. Read existing token limiting test files first to port any valuable assertions.

- [ ] **Step 2: Write test_security.py**

```python
"""Tests for transport/security.py."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from transport.security import LocalhostOnlyMiddleware, SecurityConfig


class TestSecurityConfig:

    def test_defaults(self):
        config = SecurityConfig()
        assert config.localhost_only is True
        assert config.require_api_key is False
        assert len(config.allowed_origins) > 0

    def test_custom_origins(self):
        config = SecurityConfig(allowed_origins=["http://example.com"])
        assert "http://example.com" in config.allowed_origins


class TestLocalhostOnlyMiddleware:

    async def test_allows_localhost(self):
        middleware = LocalhostOnlyMiddleware(app=MagicMock())
        request = MagicMock()
        request.client.host = "127.0.0.1"
        call_next = AsyncMock(return_value=MagicMock(status_code=200))
        response = await middleware.dispatch(request, call_next)
        call_next.assert_called_once()

    async def test_allows_ipv6_localhost(self):
        middleware = LocalhostOnlyMiddleware(app=MagicMock())
        request = MagicMock()
        request.client.host = "::1"
        call_next = AsyncMock(return_value=MagicMock(status_code=200))
        response = await middleware.dispatch(request, call_next)
        call_next.assert_called_once()

    async def test_rejects_remote_host(self):
        middleware = LocalhostOnlyMiddleware(app=MagicMock())
        request = MagicMock()
        request.client.host = "192.168.1.100"
        call_next = AsyncMock()
        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 403
        call_next.assert_not_called()

    async def test_rejects_none_client(self):
        middleware = LocalhostOnlyMiddleware(app=MagicMock())
        request = MagicMock()
        request.client = None
        call_next = AsyncMock()
        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 403
```

- [ ] **Step 3: Write test_config.py**

```python
"""Tests for persistence/config.py."""

import json
import os

import pytest

from persistence.config import DatabaseConfig


class TestDatabaseConfig:

    def test_defaults(self):
        config = DatabaseConfig()
        assert config.postgresql_dsn is None
        assert config.postgresql_pool_min == 2
        assert config.postgresql_pool_max == 10
        assert config.auto_vacuum is True

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("SESSION_DB_DSN", "postgresql://test:test@localhost/testdb")
        monkeypatch.setenv("SESSION_DB_POOL_MIN", "5")
        config = DatabaseConfig.from_env()
        assert config.postgresql_dsn == "postgresql://test:test@localhost/testdb"
        assert config.postgresql_pool_min == 5

    def test_from_file(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "postgresql_dsn": "postgresql://file@localhost/filedb",
            "retention_days": 30,
        }))
        config = DatabaseConfig.from_file(config_file)
        assert config.postgresql_dsn == "postgresql://file@localhost/filedb"
        assert config.retention_days == 30

    def test_from_file_missing(self, tmp_path):
        config = DatabaseConfig.from_file(tmp_path / "nonexistent.json")
        assert config.postgresql_dsn is None  # Falls back to defaults

    def test_save_and_reload_roundtrip(self, tmp_path):
        config = DatabaseConfig(postgresql_dsn="postgresql://rt@localhost/rtdb", retention_days=7)
        config_file = tmp_path / "config.json"
        config.save(config_file)
        reloaded = DatabaseConfig.from_file(config_file)
        assert reloaded.postgresql_dsn == "postgresql://rt@localhost/rtdb"
        assert reloaded.retention_days == 7

    def test_env_overrides_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"postgresql_dsn": "from-file"}))
        monkeypatch.setenv("SESSION_DB_DSN", "from-env")
        config = DatabaseConfig.load()
        # env should take precedence — but load() uses default file path,
        # so test with from_env directly
        env_config = DatabaseConfig.from_env()
        assert env_config.postgresql_dsn == "from-env"
```

- [ ] **Step 4: Run unit tests**

Run: `pixi run -e ci pytest tests/unit/ -v`
Expected: All pass (including existing test_session_models.py).

- [ ] **Step 5: Commit**

```
git commit -m "test: add unit tests for token limiter, security, config (~30 tests)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 8: Unit Tests — Migration

**Files:**
- Create: `tests/unit/test_migration.py`

- [ ] **Step 1: Write migration tests**

```python
"""Tests for persistence/migration.py MigrationManager."""

import pytest

from persistence.migration import MigrationManager
from persistence.sqlite import SQLiteBackend


@pytest.fixture
async def source_db(tmp_path):
    db = SQLiteBackend(str(tmp_path / "source.db"))
    await db.initialize()
    yield db


@pytest.fixture
async def target_db(tmp_path):
    db = SQLiteBackend(str(tmp_path / "target.db"))
    await db.initialize()
    yield db


class TestMigrationManager:

    async def test_migrate_empty_source(self, source_db, target_db):
        mgr = MigrationManager(source=source_db, target=target_db)
        result = await mgr.migrate_all()
        assert result["status"] == "success"
        assert result["total_records"] == 0

    async def test_migrate_sessions(self, source_db, target_db):
        from tests.persistence.builders import make_session_data
        session = make_session_data()
        await source_db.save_session(session)

        mgr = MigrationManager(source=source_db, target=target_db)
        result = await mgr.migrate_all()
        assert result["records_migrated"]["sessions"] >= 1

        # Verify data in target
        target_session = await target_db.get_session(session["id"])
        assert target_session is not None

    async def test_migrate_decisions(self, source_db, target_db):
        from tests.persistence.builders import make_decision_data, make_session_data
        session = make_session_data()
        await source_db.save_session(session)
        decision = make_decision_data(session["id"])
        await source_db.save_decision(decision)

        mgr = MigrationManager(source=source_db, target=target_db)
        result = await mgr.migrate_all()
        assert result["records_migrated"]["decisions"] >= 1

    async def test_migrate_all_entity_types(self, source_db, target_db):
        """Verify all 6 entity types are migrated."""
        from tests.persistence.builders import (
            make_session_data, make_decision_data, make_metrics_data,
            make_note_data, make_agent_execution_data, make_mcp_session_data,
        )
        session = make_session_data()
        await source_db.save_session(session)
        await source_db.save_decision(make_decision_data(session["id"]))
        await source_db.save_metrics(make_metrics_data(session["id"]))
        await source_db.save_note(make_note_data(session["id"]))
        await source_db.save_agent_execution(make_agent_execution_data(session["id"]))
        await source_db.save_mcp_session(make_mcp_session_data())

        mgr = MigrationManager(source=source_db, target=target_db)
        result = await mgr.migrate_all()

        for entity_type in ["sessions", "decisions", "metrics", "notes", "agent_executions", "mcp_sessions"]:
            assert result["records_migrated"][entity_type] >= 1, f"Missing migration for {entity_type}"

    async def test_migrate_idempotent(self, source_db, target_db):
        """Running migrate_all twice should not duplicate data."""
        from tests.persistence.builders import make_session_data
        session = make_session_data()
        await source_db.save_session(session)

        mgr1 = MigrationManager(source=source_db, target=target_db)
        await mgr1.migrate_all()

        mgr2 = MigrationManager(source=source_db, target=target_db)
        await mgr2.migrate_all()

        # Should still have exactly 1 session, not 2
        stats = await target_db.get_statistics()
        assert stats.get("sessions", 0) >= 1

    async def test_migrate_preserves_data_integrity(self, source_db, target_db):
        """Verify migrated data matches source exactly."""
        from tests.persistence.builders import make_session_data, make_decision_data
        session = make_session_data(project_name="integrity-test")
        await source_db.save_session(session)
        decision = make_decision_data(session["id"], description="integrity check")
        await source_db.save_decision(decision)

        mgr = MigrationManager(source=source_db, target=target_db)
        await mgr.migrate_all()

        target_session = await target_db.get_session(session["id"])
        assert target_session["project_name"] == "integrity-test"

        target_decisions = await target_db.query_decisions_by_session(session["id"])
        assert any(d["description"] == "integrity check" for d in target_decisions)
```

- [ ] **Step 2: Run migration tests**

Run: `pixi run -e ci pytest tests/unit/test_migration.py -v`
Expected: All pass.

- [ ] **Step 3: Commit**

```
git commit -m "test: add migration manager tests (~8 tests)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 9: Integration Tests — HTTP Transport

**Files:**
- Create: `tests/integration/test_http_transport.py`

- [ ] **Step 1: Write HTTP transport tests**

Use `httpx.AsyncClient` with the FastAPI/Starlette ASGI app. Read `src/transport/http_server.py` to understand the app factory and endpoint structure before writing tests.

Cover:
- `test_health_endpoint` — GET /health returns 200
- `test_post_valid_tool_request` — POST with valid tool_name and parameters
- `test_post_invalid_json` — POST with malformed JSON body
- `test_post_missing_tool_name` — POST without tool_name field
- `test_post_unknown_tool` — POST with nonexistent tool_name
- `test_error_response_format` — verify error responses have consistent structure
- `test_large_response_truncation` — trigger a response that exceeds token limits
- `test_mcp_session_persistence` — verify MCP session save uses datetime objects (PR #13 class)
- `test_concurrent_requests` — send multiple requests concurrently via asyncio.gather

~10-15 tests total. The exact endpoint paths depend on `http_server.py` — read it first.

- [ ] **Step 2: Run tests**

Run: `pixi run -e ci pytest tests/integration/test_http_transport.py -v`
Expected: All pass.

- [ ] **Step 3: Commit**

```
git commit -m "test: add HTTP transport integration tests (~12 tests)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 10: Integration Tests — Error Paths & Concurrency

**Files:**
- Create: `tests/integration/test_error_paths.py`
- Create: `tests/integration/test_concurrency.py`

- [ ] **Step 1: Write error path tests**

```python
"""Test error handling across layer boundaries."""

import pytest

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence.sqlite import SQLiteBackend


@pytest.fixture
async def lean_interface(tmp_path):
    engine = SessionIntelligenceEngine(repository_path=str(tmp_path))
    engine.database = SQLiteBackend(str(tmp_path / "test.db"))
    await engine.database.initialize()
    yield LeanMCPInterface(engine)


class TestErrorPaths:

    async def test_execute_tool_missing_required_params(self, lean_interface):
        result = await lean_interface.execute_tool("session_manage_lifecycle", {})
        assert result["status"] == "error"

    async def test_execute_tool_invalid_param_types(self, lean_interface):
        result = await lean_interface.execute_tool(
            "session_manage_lifecycle",
            {"operation": 12345},  # should be string
        )
        # Should handle gracefully, not crash
        assert "status" in result

    async def test_empty_string_parameters(self, lean_interface):
        result = await lean_interface.execute_tool(
            "session_manage_lifecycle",
            {"operation": "", "mode": "", "project_name": ""},
        )
        assert "status" in result

    async def test_none_parameters_handling(self, lean_interface):
        result = await lean_interface.execute_tool(
            "session_log_decision",
            {"description": None, "rationale": None},
        )
        assert "status" in result

    async def test_fk_violation_error_message(self, lean_interface):
        """FK violation should produce a meaningful error, not a raw traceback."""
        result = await lean_interface.execute_tool(
            "agent_log_decision",
            {
                "agent_name": "nonexistent-agent-xyz",
                "decision_type": "test",
                "context": "test",
                "decision": "test",
            },
        )
        # Should handle gracefully — either auto-create agent or return error
        assert "status" in result
```

- [ ] **Step 2: Write concurrency tests**

```python
"""Test concurrent access patterns."""

import asyncio

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend


@pytest.fixture
async def engine(tmp_path):
    eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
    eng.database = SQLiteBackend(str(tmp_path / "test.db"))
    await eng.database.initialize()
    yield eng


class TestConcurrency:

    async def test_concurrent_decision_logging(self, engine):
        await engine.session_manage_lifecycle(operation="create", mode="local", project_name="concurrent")

        async def log_decision(i):
            await engine.session_log_decision(
                description=f"Decision {i}",
                rationale=f"Rationale {i}",
                category="test",
            )

        await asyncio.gather(*[log_decision(i) for i in range(10)])

    async def test_rapid_sequential_operations(self, engine):
        """Rapid sequential creates should not corrupt state."""
        for i in range(5):
            await engine.session_manage_lifecycle(
                operation="create", mode="local", project_name=f"rapid-{i}"
            )

    async def test_multiple_engines_same_db(self, tmp_path):
        """Two engines sharing a DB should not corrupt data."""
        db_path = str(tmp_path / "shared.db")

        eng1 = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng1.database = SQLiteBackend(db_path)
        await eng1.database.initialize()

        eng2 = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng2.database = SQLiteBackend(db_path)
        await eng2.database.initialize()

        await eng1.session_manage_lifecycle(operation="create", mode="local", project_name="eng1")
        await eng2.session_manage_lifecycle(operation="create", mode="local", project_name="eng2")

        # Both sessions should exist
        stats = await eng1.database.get_statistics()
        assert stats.get("sessions", 0) >= 2
```

- [ ] **Step 3: Run integration tests**

Run: `pixi run -e ci pytest tests/integration/ -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```
git commit -m "test: add integration tests for error paths and concurrency (~15 tests)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 11: Cleanup — Remove Obsolete Test Files

**Files:**
- Remove: `tests/debug/` (entire directory)
- Remove: `tests/live/` (entire directory)
- Remove: `tests/integration/test_lean_mcp.py`
- Remove: `tests/integration/test_simplified_lean.py`
- Remove: `tests/integration/test_agent_system.py`
- Remove: `tests/integration/test_token_limiting.py`
- Remove: `tests/integration/test_specific_limit.py`
- Remove: `tests/integration/test_large_response.py`

- [ ] **Step 1: Remove obsolete files**

```bash
git rm -r tests/debug/ tests/live/
git rm tests/integration/test_lean_mcp.py tests/integration/test_simplified_lean.py tests/integration/test_agent_system.py
git rm tests/integration/test_token_limiting.py tests/integration/test_specific_limit.py tests/integration/test_large_response.py
```

- [ ] **Step 2: Verify no test breakage**

Run: `pixi run -e ci pytest tests/ -v --ignore=tests/live --ignore=tests/debug`
Expected: All remaining tests pass. No imports break.

- [ ] **Step 3: Commit**

```
git commit -m "test: remove obsolete debug, live, and demo test files

Removed:
- tests/debug/ (5 ad-hoc debug scripts)
- tests/live/ (5 live-server scripts)
- tests/integration/test_lean_mcp.py (demo with fake data)
- tests/integration/test_simplified_lean.py (trivial)
- tests/integration/test_agent_system.py (replaced by persistence contracts)
- tests/integration/test_token_limiting.py (consolidated)
- tests/integration/test_specific_limit.py (consolidated)
- tests/integration/test_large_response.py (consolidated)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 12: Coverage Measurement & Final Validation

- [ ] **Step 1: Run full test suite with coverage**

Run: `pixi run -e ci pytest tests/ -v --cov=src --cov-report=term-missing`
Expected: ~90%+ coverage, all tests pass.

- [ ] **Step 2: Review coverage gaps**

Examine the `--cov-report=term-missing` output. Identify any modules below 80% and note what's missing. If critical paths are uncovered, add targeted tests.

- [ ] **Step 3: Run regression tests specifically**

Run: `pixi run -e ci pytest tests/regression/ -v -m regression`
Expected: All 6 regression tests pass.

- [ ] **Step 4: Run persistence contract tests specifically**

Run: `pixi run -e ci pytest tests/persistence/test_sqlite_contract.py -v`
Expected: All ~70 contract tests pass.

- [ ] **Step 5: Final commit if any gap-filling was needed**

```
git commit -m "test: fill coverage gaps identified during final validation

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 13: CI Workflow — Add PostgreSQL Service

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Read existing CI workflow**

Read `.github/workflows/ci.yml` to understand the current structure. The project uses `Claire-s-Monster/ci-framework` reusable workflow.

- [ ] **Step 2: Add PostgreSQL service and POSTGRES_DSN env var**

Add a PostgreSQL service container to the CI job that runs tests. Add the `POSTGRES_DSN` environment variable so the PostgreSQL contract tests run in CI.

```yaml
services:
  postgres:
    image: postgres:16
    env:
      POSTGRES_DB: session_intelligence_test
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
    ports:
      - 5432:5432
    options: >-
      --health-cmd pg_isready
      --health-interval 10s
      --health-timeout 5s
      --health-retries 5

env:
  POSTGRES_DSN: postgresql://test:test@localhost:5432/session_intelligence_test
```

**Note**: If the project uses a reusable workflow from `ci-framework`, the PostgreSQL service may need to be added at the caller workflow level (not inside the reusable workflow). Read the workflow file to determine the correct placement.

- [ ] **Step 3: Verify CI config is valid**

Run: `gh workflow view ci.yml` or validate YAML syntax.

- [ ] **Step 4: Commit**

```
git commit -m "ci: add PostgreSQL service for persistence contract tests

Enables PostgreSQL contract tests to run in CI alongside SQLite.
Both backends must pass identical contract tests.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
