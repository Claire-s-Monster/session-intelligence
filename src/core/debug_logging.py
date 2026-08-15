"""Opt-in debug logging for session-intelligence.

Debug logging is OFF unless SESSION_INTELLIGENCE_DEBUG is set to a truthy
value. When enabled it writes to a size-capped rotating file so it cannot
grow without bound (issue #68: the ungated handler reached 103 MB / 720k
lines over 45 days).
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEBUG_ENV_VAR = "SESSION_INTELLIGENCE_DEBUG"
LOG_PATH_ENV_VAR = "SESSION_INTELLIGENCE_DEBUG_LOG"
DEFAULT_LOG_PATH = Path("/tmp/session-intelligence-debug.log")
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 2

_TRUTHY = {"1", "true", "yes", "on"}


def debug_enabled() -> bool:
    """Return True only when debug logging is explicitly switched on."""
    return os.environ.get(DEBUG_ENV_VAR, "").strip().lower() in _TRUTHY


def debug_log_path() -> Path:
    """Destination for the debug log; overridable for tests and packaging."""
    return Path(os.environ.get(LOG_PATH_ENV_VAR) or DEFAULT_LOG_PATH)


def configure_debug_logger(name: str, fmt: str) -> logging.Logger:
    """Return a debug logger whose file sink is opt-in.

    Idempotent: existing handlers are cleared first, so repeated calls (or
    module reloads) cannot stack duplicate sinks on the same file.

    When debug is off the logger keeps propagating at WARNING, so genuine
    failures still reach the root handlers (journal). Only the high-volume
    INFO tracing is dropped -- that tracing is what made the log unbounded,
    not the error records.
    """
    logger = logging.getLogger(name)

    # Propagation stays ON in both states: these loggers carry real
    # exception records, and the root handlers are their only sink when
    # the debug file is disabled.
    logger.propagate = True

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if not debug_enabled():
        # No file sink; WARNING drops the INFO tracing that caused issue #68
        # while leaving warnings/errors visible via the root handlers.
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.WARNING)
        return logger

    path = debug_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
