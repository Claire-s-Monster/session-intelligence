# Test Suite Improvement Design

## Problem

Session-intelligence has had 3 critical production failures (PRs #12, #13, #14) that the existing test suite failed to catch:

1. **PR #12**: Fire-and-forget `create_task` — async DB calls never awaited, data silently lost
2. **PR #13**: `isoformat()` strings passed to asyncpg TIMESTAMPTZ columns — asyncpg requires `datetime` objects
3. **PR #14**: `_create_session` only wrote to memory/filesystem, never to DB — FK constraint violations when logging decisions

The current test suite has a 5:1 code-to-test ratio (10,750 source lines vs 2,100 test lines), with coverage concentrated on Pydantic model validation and no coverage of persistence, engine, or MCP interface layers.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Priority ordering | Persistence-first | All 3 critical bugs were in the persistence chain |
| Database testing | Real SQLite locally + real PostgreSQL in CI | Mock-based tests wouldn't have caught PR #13 (type mismatch) |
| Existing test handling | Refactor in place | Keep solid files (models, knowledge), remove debug/demo scripts |
| Coverage target | ~90%+ | Warranted by severity of production failures |
| Architecture | Contract-based + regression | Contracts ensure backend parity; regression tests prevent recurrence |

## Test Directory Structure

```
tests/
├── conftest.py                          # Shared fixtures (refactored)
├── unit/
│   ├── test_session_models.py           # KEEP (existing, solid)
│   ├── test_token_limiter.py            # NEW
│   ├── test_security.py                 # NEW
│   ├── test_config.py                   # NEW
│   └── test_migration.py               # NEW
├── persistence/
│   ├── conftest.py                      # Persistence-specific fixtures
│   ├── contract_tests.py               # Shared contract test cases
│   ├── test_sqlite_contract.py          # Contract -> SQLite
│   └── test_postgresql_contract.py      # Contract -> PostgreSQL (CI-only)
├── engine/
│   ├── conftest.py                      # Engine fixtures (real SQLite)
│   ├── test_session_lifecycle.py        # create/finalize/recover sessions
│   ├── test_agent_operations.py         # register/query/search agents
│   ├── test_decision_learning.py        # decisions + learnings CRUD
│   ├── test_knowledge_system.py         # KEEP (move from unit/)
│   └── test_mcp_tool_dispatch.py        # execute_tool -> engine -> DB
├── integration/
│   ├── test_http_transport.py           # HTTP server request/response
│   ├── test_error_paths.py              # Error handling across layers
│   └── test_concurrency.py             # Concurrent access patterns
├── regression/
│   ├── test_async_await_bugs.py         # PR #12: fire-and-forget async
│   ├── test_datetime_type_bugs.py       # PR #13: isoformat vs datetime
│   └── test_session_persistence_bugs.py # PR #14: sessions not in DB
```

### Files Removed

- `tests/debug/` (5 ad-hoc debug scripts — not real tests)
- `tests/live/` (5 live-server scripts — manual tests, not CI)
- `tests/integration/test_lean_mcp.py` (demo script with hardcoded fake data)
- `tests/integration/test_simplified_lean.py` (trivial)
- `tests/integration/test_agent_system.py` (replaced by persistence contracts)

### Files Kept

- `tests/unit/test_session_models.py` — solid Pydantic model validation
- `tests/unit/test_knowledge_system.py` — moved to `tests/engine/`
- `tests/integration/test_token_limiting.py` + `test_specific_limit.py` + `test_large_response.py` — consolidated into `tests/unit/test_token_limiter.py`

## Design Details

### 1. Persistence Contract Pattern

A single abstract base class defines ~40-50 test methods that every persistence backend must pass identically. Both SQLite and PostgreSQL run the same contract.

```python
class PersistenceContractTests:
    """Every persistence backend must pass ALL of these tests."""

    # --- Session CRUD ---
    async def test_save_and_retrieve_session(self, backend): ...
    async def test_save_session_with_all_fields(self, backend): ...
    async def test_update_existing_session(self, backend): ...
    async def test_get_nonexistent_session_returns_none(self, backend): ...
    async def test_list_sessions_by_project(self, backend): ...

    # --- Decision CRUD ---
    async def test_save_and_query_decisions(self, backend): ...
    async def test_decision_requires_valid_session_fk(self, backend): ...
    async def test_query_decisions_with_filters(self, backend): ...
    async def test_update_decision_outcome(self, backend): ...

    # --- Learning CRUD ---
    async def test_save_and_query_learnings(self, backend): ...
    async def test_learning_requires_valid_session_fk(self, backend): ...
    async def test_query_learnings_by_category(self, backend): ...

    # --- Agent CRUD ---
    async def test_save_and_get_agent(self, backend): ...
    async def test_get_agent_by_name(self, backend): ...
    async def test_agent_name_uniqueness(self, backend): ...
    async def test_update_agent_stats(self, backend): ...

    # --- Agent Decisions ---
    async def test_save_agent_decision(self, backend): ...
    async def test_agent_decision_requires_valid_agent_fk(self, backend): ...
    async def test_query_agent_decisions_with_filters(self, backend): ...
    async def test_update_agent_decision_outcome(self, backend): ...

    # --- Agent Learnings ---
    async def test_save_agent_learning(self, backend): ...
    async def test_query_agent_learnings_with_filters(self, backend): ...
    async def test_update_agent_learning_outcome(self, backend): ...

    # --- Agent Notebooks ---
    async def test_save_agent_notebook(self, backend): ...
    async def test_query_agent_notebooks_with_filters(self, backend): ...

    # --- Data Type Handling ---
    async def test_datetime_fields_roundtrip(self, backend): ...
    async def test_json_fields_roundtrip(self, backend): ...
    async def test_list_fields_roundtrip(self, backend): ...
    async def test_null_optional_fields(self, backend): ...
    async def test_unicode_content(self, backend): ...
    async def test_large_text_content(self, backend): ...

    # --- Metrics ---
    async def test_save_and_query_metrics(self, backend): ...
    async def test_query_metrics_by_branch(self, backend): ...
    async def test_query_metrics_by_session(self, backend): ...

    # --- Notes ---
    async def test_save_and_query_notes(self, backend): ...
    async def test_query_notes_by_date(self, backend): ...

    # --- File Operations ---
    async def test_save_and_query_file_operations(self, backend): ...

    # --- Session Summaries ---
    async def test_save_and_get_session_summary(self, backend): ...
    async def test_query_session_summaries(self, backend): ...
    async def test_query_summaries_by_tag(self, backend): ...
    async def test_query_recent_summaries(self, backend): ...

    # --- Agent Executions ---
    async def test_save_and_query_agent_executions(self, backend): ...

    # --- MCP Sessions ---
    async def test_save_and_get_mcp_session(self, backend): ...
    async def test_update_mcp_session_activity(self, backend): ...
    async def test_link_mcp_to_engine_session(self, backend): ...

    # --- Project Learnings ---
    async def test_save_and_query_project_learnings(self, backend): ...
    async def test_update_learning_usage(self, backend): ...

    # --- Error Solutions ---
    async def test_save_and_find_error_solutions(self, backend): ...
    async def test_update_solution_outcome(self, backend): ...

    # --- Search ---
    async def test_search_sessions_basic(self, backend): ...
    async def test_search_by_file_change(self, backend): ...

    # --- Maintenance ---
    async def test_vacuum(self, backend): ...
    async def test_get_statistics(self, backend): ...

    # --- Edge Cases ---
    async def test_empty_query_results(self, backend): ...
    async def test_duplicate_id_handling(self, backend): ...
    async def test_special_characters_in_strings(self, backend): ...
```

**Note on API asymmetry**: The following methods exist only in PostgreSQL and require PostgreSQL-only tests in `test_postgresql_contract.py` (not in the shared contract):

- `recall_project(project_name, ...)` — full-text search across sessions, decisions, learnings
- `search_sessions(query, search_type, limit)` — the `search_type` parameter is PostgreSQL-only; SQLite's `search_sessions(query, limit)` uses FTS5 without search_type

The shared contract's `test_search_sessions_basic` tests the common subset (query + limit). `test_postgresql_contract.py` adds:
```python
async def test_recall_project(self, backend): ...
async def test_search_sessions_with_search_type(self, backend): ...
async def test_search_sessions_signature_parity(self, backend): ...
```

Backend-specific test files inherit the contract and provide only a fixture:

```python
# test_sqlite_contract.py
class TestSQLiteContract(PersistenceContractTests):
    @pytest.fixture
    async def backend(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / "test.db"))
        await db.initialize()
        yield db

# test_postgresql_contract.py
@pytest.mark.skipif(not POSTGRES_AVAILABLE, reason="PostgreSQL not available")
class TestPostgreSQLContract(PersistenceContractTests):
    @pytest.fixture
    async def backend(self):
        db = PostgreSQLBackend(dsn=TEST_POSTGRES_DSN)
        await db.initialize()
        yield db
        await db.close()
```

**Key principle**: If a test passes on SQLite but fails on PostgreSQL (or vice versa), we've found a backend divergence bug.

### 2. Engine Tests

Engine tests use a **real SQLite backend** (no mocks). This catches actual type/async bugs.

**`tests/engine/conftest.py`**:
```python
@pytest.fixture
async def engine(tmp_path):
    engine = SessionIntelligenceEngine(repository_path=str(tmp_path))
    engine.database = SQLiteBackend(str(tmp_path / "test.db"))
    await engine.database.initialize()
    yield engine
```

**Test files and approximate counts**:

- `test_session_lifecycle.py` (~15 tests): create, finalize, recover, health check, recall, metadata preservation, multiple sessions per project
- `test_agent_operations.py` (~15 tests): register, update, get by name/id, search, notebooks, stats increment
- `test_decision_learning.py` (~15 tests): log with/without active session, query with filters, update outcomes, string param coercion (PR #9 bug class)
- `test_knowledge_system.py` (existing, ~15 tests): moved from `tests/unit/`
- `test_mcp_tool_dispatch.py` (~25 tests): discover_tools, get_tool_spec, execute_tool for all registered tools (count verified dynamically via `discover_tools()`), invalid tool names, missing params, string-to-dict coercion. **Critical assertion**: every tool execution must return a dict (not a coroutine object) — this catches `_wrap_tool` vs `_wrap_async_tool` mismatches, the same bug class as PR #12

### 3. Integration Tests

**`test_http_transport.py`** (~15 tests): Uses `httpx.AsyncClient` against the ASGI app directly (no live server). Tests request/response format, MCP session headers, initialize handshake, error responses, large response truncation, concurrent requests, MCP session DB persistence (save/get/link), SSE notification subscribe/broadcast paths. Includes `test_mcp_session_persistence` to verify `save_mcp_session` passes datetime objects (not ISO strings) — same PR #13 bug class in the HTTP layer.

**`test_error_paths.py`** (~10 tests): Deliberately triggers failures across layer boundaries — DB connection failure, corrupt JSON, FK violations, missing required fields, invalid IDs, empty/None parameters.

**`test_concurrency.py`** (~6 tests): Concurrent session creates, decision logging, agent registration, reads during writes, rapid sequential operations, multiple engines sharing one DB.

### 4. Regression Tests

Each file reproduces the exact production bug scenario and verifies the fix holds:

**`test_async_await_bugs.py`** (PR #12): Verify async DB calls are awaited and data persists. Scan for remaining fire-and-forget patterns.

**`test_datetime_type_bugs.py`** (PR #13): Verify datetime objects (not ISO strings) are used for TIMESTAMPTZ. Verify SQLite's leniency (documents why this bug wasn't caught before).

**`test_session_persistence_bugs.py`** (PR #14): Verify `_create_session` writes to DB (not just memory). Verify decision logging without active session auto-creates and persists the session.

### 5. Unit Tests (Supplementary)

- `test_token_limiter.py` (~10 tests): Consolidates existing token limiting tests. Tests `apply_token_limits()` for various sizes, truncation behavior, JSON structure preservation.
- `test_security.py` (~8 tests): Input validation, sanitization rules.
- `test_config.py` (~6 tests): Config loading, defaults, environment variable overrides.
- `test_migration.py` (~12 tests): Schema migration paths, idempotent `initialize()`, `MigrationManager.migrate_all()` data-correctness validation (migrate SQLite-to-SQLite, verify row counts, verify data integrity across sessions/decisions/metrics/notes/agent_executions/mcp_sessions), migration idempotency on re-run, migration with empty source DB.

## CI Configuration

GitHub Actions workflow needs a PostgreSQL service for contract tests:

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

## Pixi Task Configuration

```toml
[feature.ci.tasks.test]
cmd = "pytest tests/ -v --ignore=tests/live --ignore=tests/debug"

[feature.ci.tasks.test-unit]
cmd = "pytest tests/unit/ -v"

[feature.ci.tasks.test-persistence]
cmd = "pytest tests/persistence/ -v"

[feature.ci.tasks.test-engine]
cmd = "pytest tests/engine/ -v"

[feature.ci.tasks.test-integration]
cmd = "pytest tests/integration/ -v"

[feature.ci.tasks.test-regression]
cmd = "pytest tests/regression/ -v"
```

## Estimated Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Test count | ~45 | ~190-210 |
| Test lines | 2,100 | ~8,000-9,000 |
| Code:test ratio | 5:1 | ~1.2:1 |
| Coverage | ~20% (models only) | ~90%+ |
| Layers tested | 1 (models) | All 6 (models, persistence, engine, MCP, HTTP, utils) |
| Regression tests | 0 | 6 (covering 3 critical bugs) |
| Contract tests | 0 | ~70 shared + ~5 PostgreSQL-only |

## Success Criteria

1. All persistence contract tests pass on both SQLite and PostgreSQL
2. All 3 historical production bugs have dedicated regression tests
3. Engine tests use real SQLite (no mocks) and cover all registered MCP tools (verified dynamically)
4. CI runs full suite including PostgreSQL service
5. Coverage reaches 90%+ as measured by `pytest-cov`
6. No test requires a live server or manual intervention
