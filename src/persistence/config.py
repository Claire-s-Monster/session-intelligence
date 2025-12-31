"""
Database configuration and factory for session persistence.

Supports configuration via:
- Environment variables
- Configuration file (~/.claude/session-intelligence/config.json)
- Constructor arguments

Environment Variables:
    SESSION_DB_BACKEND: sqlite | postgresql (default: sqlite)
    SESSION_DB_PATH: SQLite file path (default: ~/.claude/session-intelligence/sessions.db)
    SESSION_DB_DSN: PostgreSQL connection string
    SESSION_DB_POOL_MIN: PostgreSQL pool minimum size (default: 2)
    SESSION_DB_POOL_MAX: PostgreSQL pool maximum size (default: 10)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .base import (
    DEFAULT_DATA_DIR,
    DEFAULT_POSTGRES_DSN,
    DatabaseBackend,
    get_default_data_dir,
)

logger = logging.getLogger(__name__)

BackendType = Literal["sqlite", "postgresql"]


@dataclass
class DatabaseConfig:
    """Configuration for database backend."""

    # Backend selection
    backend: BackendType = "sqlite"

    # SQLite settings
    sqlite_path: Path | None = None

    # PostgreSQL settings
    postgresql_dsn: str | None = None
    postgresql_pool_min: int = 2
    postgresql_pool_max: int = 10

    # Shared settings
    auto_vacuum: bool = True
    retention_days: int | None = None  # None = no retention policy

    @classmethod
    def from_env(cls) -> DatabaseConfig:
        """Create configuration from environment variables."""
        backend = os.environ.get("SESSION_DB_BACKEND", "sqlite").lower()
        if backend not in ("sqlite", "postgresql"):
            logger.warning(f"Invalid backend '{backend}', defaulting to sqlite")
            backend = "sqlite"

        config = cls(backend=backend)  # type: ignore

        # SQLite settings
        if sqlite_path := os.environ.get("SESSION_DB_PATH"):
            config.sqlite_path = Path(sqlite_path)

        # PostgreSQL settings
        if dsn := os.environ.get("SESSION_DB_DSN"):
            config.postgresql_dsn = dsn
        if pool_min := os.environ.get("SESSION_DB_POOL_MIN"):
            config.postgresql_pool_min = int(pool_min)
        if pool_max := os.environ.get("SESSION_DB_POOL_MAX"):
            config.postgresql_pool_max = int(pool_max)

        # Shared settings
        if retention := os.environ.get("SESSION_DB_RETENTION_DAYS"):
            config.retention_days = int(retention)

        return config

    @classmethod
    def from_file(cls, config_path: Path | None = None) -> DatabaseConfig:
        """Load configuration from JSON file."""
        if config_path is None:
            config_path = DEFAULT_DATA_DIR / "config.json"

        if not config_path.exists():
            logger.debug(f"Config file not found: {config_path}")
            return cls()

        try:
            with open(config_path) as f:
                data = json.load(f)

            config = cls()

            if backend := data.get("backend"):
                if backend in ("sqlite", "postgresql"):
                    config.backend = backend

            if sqlite_path := data.get("sqlite_path"):
                config.sqlite_path = Path(sqlite_path)

            if dsn := data.get("postgresql_dsn"):
                config.postgresql_dsn = dsn
            if pool_min := data.get("postgresql_pool_min"):
                config.postgresql_pool_min = pool_min
            if pool_max := data.get("postgresql_pool_max"):
                config.postgresql_pool_max = pool_max

            if retention := data.get("retention_days"):
                config.retention_days = retention

            return config

        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return cls()

    @classmethod
    def load(cls) -> DatabaseConfig:
        """Load configuration with precedence: env > file > defaults."""
        # Start with file config
        config = cls.from_file()

        # Override with environment variables
        env_config = cls.from_env()

        # Merge (env takes precedence)
        if os.environ.get("SESSION_DB_BACKEND"):
            config.backend = env_config.backend
        if os.environ.get("SESSION_DB_PATH"):
            config.sqlite_path = env_config.sqlite_path
        if os.environ.get("SESSION_DB_DSN"):
            config.postgresql_dsn = env_config.postgresql_dsn
        if os.environ.get("SESSION_DB_POOL_MIN"):
            config.postgresql_pool_min = env_config.postgresql_pool_min
        if os.environ.get("SESSION_DB_POOL_MAX"):
            config.postgresql_pool_max = env_config.postgresql_pool_max
        if os.environ.get("SESSION_DB_RETENTION_DAYS"):
            config.retention_days = env_config.retention_days

        return config

    def save(self, config_path: Path | None = None) -> None:
        """Save configuration to JSON file."""
        if config_path is None:
            config_path = get_default_data_dir() / "config.json"

        data = {
            "backend": self.backend,
            "sqlite_path": str(self.sqlite_path) if self.sqlite_path else None,
            "postgresql_dsn": self.postgresql_dsn,
            "postgresql_pool_min": self.postgresql_pool_min,
            "postgresql_pool_max": self.postgresql_pool_max,
            "retention_days": self.retention_days,
        }

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Configuration saved to {config_path}")


def create_database(
    config: DatabaseConfig | None = None,
    backend: BackendType | None = None,
    **kwargs: Any,
) -> DatabaseBackend:
    """Factory function to create the appropriate database backend.

    Args:
        config: DatabaseConfig instance. If None, loads from env/file.
        backend: Override backend type.
        **kwargs: Additional arguments passed to backend constructor.

    Returns:
        Configured database backend (not yet initialized).

    Usage:
        # Auto-detect from environment
        db = create_database()
        await db.initialize()

        # Explicit SQLite
        db = create_database(backend="sqlite", db_path="/path/to/db.sqlite")

        # Explicit PostgreSQL
        db = create_database(backend="postgresql", dsn="postgresql://...")
    """
    if config is None:
        config = DatabaseConfig.load()

    # Allow override
    actual_backend = backend or config.backend

    if actual_backend == "postgresql":
        from .postgresql import PostgreSQLBackend

        dsn = kwargs.pop("dsn", None) or config.postgresql_dsn or DEFAULT_POSTGRES_DSN
        pool_kwargs = {
            "min_size": kwargs.pop("min_size", config.postgresql_pool_min),
            "max_size": kwargs.pop("max_size", config.postgresql_pool_max),
        }
        pool_kwargs.update(kwargs)
        return PostgreSQLBackend(dsn=dsn, **pool_kwargs)

    else:  # sqlite (default)
        from .sqlite import SQLiteBackend

        db_path = kwargs.pop("db_path", None)
        if db_path is None:
            db_path = str(config.sqlite_path) if config.sqlite_path else None
        return SQLiteBackend(db_path=db_path)


async def get_database(
    config: DatabaseConfig | None = None,
    backend: BackendType | None = None,
    **kwargs: Any,
) -> DatabaseBackend:
    """Create and initialize database backend.

    Convenience function that creates and initializes in one call.

    Usage:
        db = await get_database()
        sessions = await db.query_sessions()
    """
    db = create_database(config=config, backend=backend, **kwargs)
    await db.initialize()
    return db
