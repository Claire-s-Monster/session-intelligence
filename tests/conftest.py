"""
Shared pytest fixtures for session-intelligence tests.

This module provides common fixtures used across all test categories:
- Unit tests (tests/unit/)
- Integration tests (tests/integration/)
- Debug tests (tests/debug/)
- Live tests (tests/live/)
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# Async Support
# ============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Path Fixtures
# ============================================================================

@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def src_path(project_root: Path) -> Path:
    """Return the src directory path."""
    return project_root / "src"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_session_dir(temp_dir: Path) -> Path:
    """Create a temporary session directory structure."""
    session_dir = temp_dir / ".claude" / "session-intelligence" / "test-session"
    session_dir.mkdir(parents=True)
    return session_dir


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
def sqlite_db_path(temp_dir: Path) -> Path:
    """Create a temporary SQLite database path."""
    return temp_dir / "test_session.db"


@pytest.fixture
async def sqlite_backend(sqlite_db_path: Path) -> AsyncGenerator:
    """Create a temporary SQLite backend for testing."""
    from persistence.sqlite import SQLiteBackend

    backend = SQLiteBackend(str(sqlite_db_path))
    await backend.initialize()
    yield backend
    # Cleanup handled by temp_dir fixture


# ============================================================================
# Engine Fixtures
# ============================================================================

@pytest.fixture
def session_engine(temp_dir: Path):
    """Create a session engine for testing."""
    from core.session_engine import SessionIntelligenceEngine

    engine = SessionIntelligenceEngine(repository_path=str(temp_dir))
    return engine


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture
def clean_env() -> Generator[None, None, None]:
    """Provide a clean environment without session-intelligence env vars."""
    original_env = os.environ.copy()

    # Remove any session-intelligence related env vars
    keys_to_remove = [k for k in os.environ if k.startswith("SESSION_INTELLIGENCE_")]
    for key in keys_to_remove:
        del os.environ[key]

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def postgresql_env() -> Generator[None, None, None]:
    """Set up environment for PostgreSQL testing (if available)."""
    original_env = os.environ.copy()

    # Set PostgreSQL connection string if not already set
    if "SESSION_INTELLIGENCE_POSTGRES_URL" not in os.environ:
        os.environ["SESSION_INTELLIGENCE_POSTGRES_URL"] = (
            "postgresql://localhost:5432/session_intelligence_test"
        )

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# ============================================================================
# Mock Data Fixtures
# ============================================================================

@pytest.fixture
def sample_session_metadata() -> dict:
    """Provide sample session metadata for testing."""
    return {
        "session_type": "development",
        "environment": "test",
        "user": "test-user",
        "git_branch": "test-branch",
        "git_commit": "abc123",
        "tags": ["test", "unit"],
        "custom_attributes": {"test_key": "test_value"},
    }


@pytest.fixture
def sample_agent_data() -> dict:
    """Provide sample agent data for testing."""
    return {
        "name": "test-agent",
        "agent_type": "focused",
        "display_name": "Test Agent",
        "description": "A test agent for unit testing",
        "capabilities": ["test", "validate"],
        "metadata": {"version": "1.0.0"},
    }


@pytest.fixture
def sample_decision_data() -> dict:
    """Provide sample decision data for testing."""
    return {
        "decision_type": "tool_selection",
        "context": "Testing decision logging",
        "decision": "Use pytest for testing",
        "reasoning": "Standard Python testing framework",
        "alternatives": ["unittest", "nose"],
        "confidence": 0.9,
        "tags": ["testing", "tool"],
    }


@pytest.fixture
def sample_learning_data() -> dict:
    """Provide sample learning data for testing."""
    return {
        "learning_type": "pattern",
        "title": "Test Pattern",
        "content": "Always use fixtures for shared test data",
        "source_context": "Code review",
        "applicability": ["pytest", "testing"],
        "confidence": 0.85,
        "tags": ["testing", "best-practice"],
    }
