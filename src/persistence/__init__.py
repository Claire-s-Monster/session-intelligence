"""
Session Intelligence Persistence Layer.

Uses PostgreSQL for production-grade session management with
connection pooling, concurrent access, and cross-session analytics.

Quick Start:
    from persistence import get_database, DatabaseConfig

    # Auto-detect from environment/config
    db = await get_database()

    # Or explicit configuration
    config = DatabaseConfig(postgresql_dsn="postgresql://localhost/session_intelligence")
    db = await get_database(config=config)

Environment Variables:
    SESSION_DB_DSN: PostgreSQL connection string
    SESSION_DB_POOL_MIN: Connection pool minimum size (default: 2)
    SESSION_DB_POOL_MAX: Connection pool maximum size (default: 10)

Default Connection:
    postgresql://localhost/session_intelligence
"""

from .base import (
    DEFAULT_DATA_DIR,
    DEFAULT_POSTGRES_DSN,
    BaseDatabaseBackend,
    DatabaseBackend,
    db_retry,
    get_default_data_dir,
    sanitize_dsn,
)
from .config import DatabaseConfig, create_database, get_database
from .postgresql import PostgreSQLBackend

# Backwards compatibility alias
Database = PostgreSQLBackend

__all__ = [
    # Protocols and base
    "DatabaseBackend",
    "BaseDatabaseBackend",
    # Backend
    "PostgreSQLBackend",
    "Database",  # Backwards compatibility alias
    # Configuration
    "DatabaseConfig",
    "create_database",
    "get_database",
    # Constants
    "DEFAULT_DATA_DIR",
    "DEFAULT_POSTGRES_DSN",
    # Utilities
    "get_default_data_dir",
    "sanitize_dsn",
    "db_retry",
]
