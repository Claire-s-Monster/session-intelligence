"""Content-digest change detection for session persistence (issue #67).

``_persist_sessions_to_database`` used to walk the entire session cache and
unconditionally re-upsert every session, decision and agent execution it held,
on *every* session-modifying tool call. That produced ~1,435 UPDATEs per row on
``agent_executions`` and kept autovacuum near-continuously busy.

This module records a content digest per persisted entity so a payload that is
byte-identical to what was last written successfully can be skipped. Digests are
committed only *after* a successful write, so a failed or retried write is
attempted again on the next pass rather than being silently dropped.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["PersistDigestTracker", "compute_digest"]


def compute_digest(payload: Mapping[str, Any]) -> str:
    """Return a stable digest of ``payload``.

    Falls back to a unique value when the payload cannot be canonicalised, which
    makes the entity look changed and preserves the pre-#67 always-write
    behaviour rather than skipping a write we cannot prove is redundant.
    """
    return _canonical_digest(payload)


def compute_steps_digest(steps: list[Any]) -> str:
    """Digest of a serialized ``execution_steps`` list (or a prefix of one)."""
    return _canonical_digest(steps)


def _canonical_digest(obj: Any) -> str:
    try:
        canonical = json.dumps(obj, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return f"uncomparable:{uuid.uuid4().hex}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PersistDigestTracker:
    """Track the last successfully persisted digest per entity, grouped by session.

    Entities are addressed by ``(session_id, entity_key)``. Grouping by session
    lets :meth:`retain` drop every digest belonging to a session that has left
    the cache, so the tracker cannot outgrow the cache it shadows.
    """

    def __init__(self) -> None:
        self._digests: dict[str, dict[str, str]] = {}
        # Issue #209: (persisted step count, digest of those steps) per execution.
        self._steps: dict[str, dict[str, tuple[int, str]]] = {}

    def append_plan(self, session_id: str, entity_key: str, steps: list[Any]) -> int | None:
        """Return the persisted step count if only new steps were appended since.

        ``None`` means a full rewrite is required: no recorded state, the list
        did not grow, or the already-persisted prefix changed.
        """
        state = self._steps.get(session_id, {}).get(entity_key)
        if state is None:
            return None
        count, prefix_digest = state
        if len(steps) <= count or compute_steps_digest(steps[:count]) != prefix_digest:
            return None
        return count

    def commit_steps(self, session_id: str, entity_key: str, count: int, digest: str) -> None:
        """Record the persisted step prefix. Call only after the write."""
        self._steps.setdefault(session_id, {})[entity_key] = (count, digest)

    def digest_if_changed(
        self, session_id: str, entity_key: str, payload: Mapping[str, Any]
    ) -> str | None:
        """Return the digest to commit if ``payload`` differs from the last write.

        Returns ``None`` when the payload is unchanged and the write can be
        skipped.
        """
        digest = compute_digest(payload)
        if self._digests.get(session_id, {}).get(entity_key) == digest:
            return None
        return digest

    def commit(self, session_id: str, entity_key: str, digest: str) -> None:
        """Record ``digest`` as successfully persisted. Call only after the write."""
        self._digests.setdefault(session_id, {})[entity_key] = digest

    def retain(self, session_ids: Iterable[str]) -> None:
        """Drop digests for every session not in ``session_ids``."""
        keep = set(session_ids)
        for session_id in list(self._digests):
            if session_id not in keep:
                del self._digests[session_id]
        for session_id in list(self._steps):
            if session_id not in keep:
                del self._steps[session_id]

    def forget(self, session_id: str) -> None:
        """Drop all digests for a single session, forcing a rewrite next pass."""
        self._digests.pop(session_id, None)
        self._steps.pop(session_id, None)

    def tracked_entity_count(self) -> int:
        """Total number of tracked entities. Intended for tests and diagnostics."""
        return sum(len(entities) for entities in self._digests.values())
