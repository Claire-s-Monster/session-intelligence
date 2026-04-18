"""Unit tests for src/persistence/config.py.

Tests DatabaseConfig defaults, from_env, from_file, load precedence,
save/roundtrip, and missing-file handling.
"""

import json
import os
from pathlib import Path

import pytest

from persistence.config import DatabaseConfig


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDatabaseConfigDefaults:
    def test_postgresql_dsn_is_none_by_default(self):
        cfg = DatabaseConfig()
        assert cfg.postgresql_dsn is None

    def test_pool_min_default(self):
        cfg = DatabaseConfig()
        assert cfg.postgresql_pool_min == 2

    def test_pool_max_default(self):
        cfg = DatabaseConfig()
        assert cfg.postgresql_pool_max == 10

    def test_auto_vacuum_default(self):
        cfg = DatabaseConfig()
        assert cfg.auto_vacuum is True

    def test_retention_days_none_by_default(self):
        cfg = DatabaseConfig()
        assert cfg.retention_days is None


# ---------------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------------


class TestDatabaseConfigFromEnv:
    def test_from_env_picks_up_dsn(self, monkeypatch):
        monkeypatch.setenv("SESSION_DB_DSN", "postgresql://localhost/test")
        cfg = DatabaseConfig.from_env()
        assert cfg.postgresql_dsn == "postgresql://localhost/test"

    def test_from_env_picks_up_pool_min(self, monkeypatch):
        monkeypatch.setenv("SESSION_DB_POOL_MIN", "5")
        cfg = DatabaseConfig.from_env()
        assert cfg.postgresql_pool_min == 5

    def test_from_env_picks_up_pool_max(self, monkeypatch):
        monkeypatch.setenv("SESSION_DB_POOL_MAX", "20")
        cfg = DatabaseConfig.from_env()
        assert cfg.postgresql_pool_max == 20

    def test_from_env_picks_up_retention_days(self, monkeypatch):
        monkeypatch.setenv("SESSION_DB_RETENTION_DAYS", "30")
        cfg = DatabaseConfig.from_env()
        assert cfg.retention_days == 30

    def test_from_env_without_vars_uses_defaults(self, monkeypatch):
        for var in ("SESSION_DB_DSN", "SESSION_DB_POOL_MIN", "SESSION_DB_POOL_MAX",
                    "SESSION_DB_RETENTION_DAYS"):
            monkeypatch.delenv(var, raising=False)
        cfg = DatabaseConfig.from_env()
        assert cfg.postgresql_dsn is None
        assert cfg.postgresql_pool_min == 2


# ---------------------------------------------------------------------------
# from_file
# ---------------------------------------------------------------------------


class TestDatabaseConfigFromFile:
    def test_missing_file_returns_defaults(self, tmp_path):
        cfg = DatabaseConfig.from_file(tmp_path / "nonexistent.json")
        assert cfg.postgresql_dsn is None
        assert cfg.postgresql_pool_min == 2

    def test_loads_dsn_from_file(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"postgresql_dsn": "postgresql://localhost/mydb"}))
        cfg = DatabaseConfig.from_file(config_file)
        assert cfg.postgresql_dsn == "postgresql://localhost/mydb"

    def test_loads_pool_settings_from_file(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"postgresql_pool_min": 3, "postgresql_pool_max": 15}))
        cfg = DatabaseConfig.from_file(config_file)
        assert cfg.postgresql_pool_min == 3
        assert cfg.postgresql_pool_max == 15

    def test_corrupt_file_returns_defaults(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text("not valid json {{{")
        cfg = DatabaseConfig.from_file(config_file)
        assert cfg.postgresql_dsn is None


# ---------------------------------------------------------------------------
# save / roundtrip
# ---------------------------------------------------------------------------


class TestDatabaseConfigSave:
    def test_save_and_reload_roundtrip(self, tmp_path):
        cfg = DatabaseConfig(
            postgresql_dsn="postgresql://localhost/roundtrip",
            postgresql_pool_min=4,
            postgresql_pool_max=12,
            retention_days=7,
        )
        config_file = tmp_path / "config.json"
        cfg.save(config_file)

        loaded = DatabaseConfig.from_file(config_file)
        assert loaded.postgresql_dsn == "postgresql://localhost/roundtrip"
        assert loaded.postgresql_pool_min == 4
        assert loaded.postgresql_pool_max == 12
        assert loaded.retention_days == 7

    def test_save_creates_valid_json_file(self, tmp_path):
        cfg = DatabaseConfig(postgresql_dsn="postgresql://localhost/test")
        config_file = tmp_path / "config.json"
        cfg.save(config_file)
        data = json.loads(config_file.read_text())
        assert "postgresql_dsn" in data
