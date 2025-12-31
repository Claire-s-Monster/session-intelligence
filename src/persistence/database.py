"""
DEPRECATED: This module is maintained for backwards compatibility only.

Use the new persistence module instead:
    from persistence import SQLiteBackend, PostgreSQLBackend, create_database

This file will be removed in a future version.
"""

from .sqlite import SQLiteBackend as Database

__all__ = ["Database"]

# Re-export for backwards compatibility
# The new location is persistence.sqlite.SQLiteBackend
