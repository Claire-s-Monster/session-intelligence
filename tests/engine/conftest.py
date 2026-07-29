"""
Fixtures for engine-layer tests.

Provides a real SessionIntelligenceEngine backed by an in-process SQLite
database (created fresh for each test via pytest's tmp_path).
"""

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    """
    Yield a fully-initialised SessionIntelligenceEngine backed by SQLite.

    A fresh database file is created under pytest's tmp_path for each test,
    so tests are fully isolated from one another.

    Agent-name validation is disabled so tests using synthetic names like
    "test-agent" do not require matching files under ~/.claude/agents/.
    """
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")

    db_path = str(tmp_path / "test_engine.db")
    db = SQLiteBackend(db_path)
    await db.initialize()

    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()
