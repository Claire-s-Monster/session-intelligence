"""Tests for opt-in, size-capped debug logging (issue #68)."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

import pytest

from core.debug_logging import (
    BACKUP_COUNT,
    DEBUG_ENV_VAR,
    LOG_PATH_ENV_VAR,
    MAX_BYTES,
    configure_debug_logger,
)


def _cleanup_logger(name: str) -> None:
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


@pytest.fixture
def logger_name(request):
    """Unique logger name per test to avoid leaking handler state."""
    name = f"test_debug_logger.{request.node.name}"
    yield name
    _cleanup_logger(name)


@pytest.mark.unit
@pytest.mark.regression
def test_disabled_by_default_never_creates_file(monkeypatch, tmp_path, logger_name):
    monkeypatch.delenv(DEBUG_ENV_VAR, raising=False)
    log_path = tmp_path / "debug.log"
    monkeypatch.setenv(LOG_PATH_ENV_VAR, str(log_path))

    logger = configure_debug_logger(logger_name, "%(message)s")
    logger.info("this should not be written anywhere")

    assert not log_path.exists()
    assert logger.getEffectiveLevel() == logging.WARNING


@pytest.mark.unit
@pytest.mark.regression
def test_disabled_errors_still_propagate_to_root(monkeypatch, tmp_path, logger_name, caplog):
    monkeypatch.delenv(DEBUG_ENV_VAR, raising=False)
    monkeypatch.setenv(LOG_PATH_ENV_VAR, str(tmp_path / "debug.log"))

    logger = configure_debug_logger(logger_name, "%(message)s")

    with caplog.at_level(logging.WARNING):
        logger.error("genuine failure: something broke")

    assert any("genuine failure: something broke" in r.message for r in caplog.records)


@pytest.mark.unit
@pytest.mark.regression
def test_enabled_writes_to_file(monkeypatch, tmp_path, logger_name):
    monkeypatch.setenv(DEBUG_ENV_VAR, "1")
    log_path = tmp_path / "debug.log"
    monkeypatch.setenv(LOG_PATH_ENV_VAR, str(log_path))

    logger = configure_debug_logger(logger_name, "%(message)s")
    logger.info("hello debug world")
    for handler in logger.handlers:
        handler.flush()

    assert log_path.exists()
    assert "hello debug world" in log_path.read_text()
    assert logger.getEffectiveLevel() == logging.INFO


@pytest.mark.unit
@pytest.mark.regression
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("True", True),
        ("YES", True),
        ("yes", True),
        ("on", True),
        ("ON", True),
        ("", False),
        ("0", False),
        ("false", False),
        ("nope", False),
    ],
)
def test_truthiness_of_env_var(monkeypatch, tmp_path, logger_name, value, expected):
    monkeypatch.setenv(DEBUG_ENV_VAR, value)
    log_path = tmp_path / "debug.log"
    monkeypatch.setenv(LOG_PATH_ENV_VAR, str(log_path))

    logger = configure_debug_logger(logger_name, "%(message)s")
    logger.info("probe")
    for handler in logger.handlers:
        handler.flush()

    assert log_path.exists() is expected


@pytest.mark.unit
@pytest.mark.regression
def test_propagate_is_true_when_disabled(monkeypatch, tmp_path, logger_name):
    monkeypatch.delenv(DEBUG_ENV_VAR, raising=False)
    monkeypatch.setenv(LOG_PATH_ENV_VAR, str(tmp_path / "debug.log"))

    logger = configure_debug_logger(logger_name, "%(message)s")

    assert logger.propagate is True


@pytest.mark.unit
@pytest.mark.regression
def test_propagate_is_true_when_enabled(monkeypatch, tmp_path, logger_name):
    monkeypatch.setenv(DEBUG_ENV_VAR, "1")
    monkeypatch.setenv(LOG_PATH_ENV_VAR, str(tmp_path / "debug.log"))

    logger = configure_debug_logger(logger_name, "%(message)s")

    assert logger.propagate is True


@pytest.mark.unit
@pytest.mark.regression
def test_enabled_handler_is_rotating_with_expected_limits(monkeypatch, tmp_path, logger_name):
    monkeypatch.setenv(DEBUG_ENV_VAR, "1")
    monkeypatch.setenv(LOG_PATH_ENV_VAR, str(tmp_path / "debug.log"))

    logger = configure_debug_logger(logger_name, "%(message)s")

    rotating_handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
    assert len(rotating_handlers) == 1
    handler = rotating_handlers[0]
    assert handler.maxBytes == MAX_BYTES
    assert handler.backupCount == BACKUP_COUNT


@pytest.mark.unit
@pytest.mark.regression
def test_idempotent_configuration_leaves_single_handler(monkeypatch, tmp_path, logger_name):
    monkeypatch.setenv(DEBUG_ENV_VAR, "1")
    monkeypatch.setenv(LOG_PATH_ENV_VAR, str(tmp_path / "debug.log"))

    configure_debug_logger(logger_name, "%(message)s")
    logger = configure_debug_logger(logger_name, "%(message)s")

    non_null_handlers = [h for h in logger.handlers if not isinstance(h, logging.NullHandler)]
    assert len(non_null_handlers) == 1


@pytest.mark.unit
@pytest.mark.regression
def test_idempotent_configuration_when_disabled_leaves_single_null_handler(
    monkeypatch, tmp_path, logger_name
):
    monkeypatch.delenv(DEBUG_ENV_VAR, raising=False)
    monkeypatch.setenv(LOG_PATH_ENV_VAR, str(tmp_path / "debug.log"))

    configure_debug_logger(logger_name, "%(message)s")
    logger = configure_debug_logger(logger_name, "%(message)s")

    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.NullHandler)
