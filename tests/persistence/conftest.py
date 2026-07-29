"""
Fixtures for persistence-layer tests.

PostgreSQL availability is determined by:
1. asyncpg being importable
2. POSTGRES_DSN environment variable being set
"""

import os

import pytest

# ---------------------------------------------------------------------------
# PostgreSQL availability check
# ---------------------------------------------------------------------------

try:
    import asyncpg  # noqa: F401

    _ASYNCPG_AVAILABLE = True
except ImportError:
    _ASYNCPG_AVAILABLE = False

POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "")
POSTGRES_AVAILABLE = _ASYNCPG_AVAILABLE and bool(POSTGRES_DSN)


def pytest_configure(config):
    """Register persistence-specific markers (if not already registered)."""
    # Markers are registered in pyproject.toml; this is a no-op guard.
    pass
