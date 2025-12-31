"""
Session Intelligence Persistence Layer.

Provides database abstraction supporting multiple backends:
- SQLite (default, development)
- PostgreSQL (production, multi-agent analysis)

Quick Start:
    from persistence import get_database, DatabaseConfig

    # Auto-detect backend from environment
    db = await get_database()

    # Or explicit configuration
    config = DatabaseConfig(backend="postgresql", postgresql_dsn="...")
    db = await get_database(config=config)

Environment Variables:
    SESSION_DB_BACKEND: sqlite | postgresql
    SESSION_DB_PATH: SQLite file path
    SESSION_DB_DSN: PostgreSQL connection string

Default Data Location:
    ~/.claude/session-intelligence/sessions.db (SQLite)
"""

from .base import (
    DEFAULT_DATA_DIR,
    DEFAULT_POSTGRES_DSN,
    DEFAULT_SQLITE_PATH,
    BaseDatabaseBackend,
    DatabaseBackend,
    get_default_data_dir,
)
from .config import DatabaseConfig, create_database, get_database
from .sqlite import SQLiteBackend

# PostgreSQL is optional (requires asyncpg)
try:
    from .postgresql import PostgreSQLBackend
except ImportError:
    PostgreSQLBackend = None  # type: ignore

# Backwards compatibility
Database = SQLiteBackend

__all__ = [
    # Protocols and base
    "DatabaseBackend",
    "BaseDatabaseBackend",
    # Backends
    "SQLiteBackend",
    "PostgreSQLBackend",
    "Database",  # Backwards compatibility alias
    # Configuration
    "DatabaseConfig",
    "create_database",
    "get_database",
    # Constants
    "DEFAULT_DATA_DIR",
    "DEFAULT_SQLITE_PATH",
    "DEFAULT_POSTGRES_DSN",
    "get_default_data_dir",
]
