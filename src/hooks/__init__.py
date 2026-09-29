"""Stdlib-only helper modules used by external Claude Code hooks.

Hooks under ``~/.claude/hooks/`` run as plain ``python3`` scripts outside
this project's pixi environment, so anything imported by them (including
this package) must not depend on third-party packages or the rest of the
``src`` tree. See ``src/hooks/transcript_summary.py`` (issue #138).
"""
