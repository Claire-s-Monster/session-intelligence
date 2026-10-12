"""Collect and write phases of the HTTP persist sweep (issue #190).

``HTTPSessionIntelligenceServer._persist_sessions_to_database`` used to digest
and write each entity inline, one autocommit statement per entity. It is now
split in two so the two costs can be measured separately and the writes can
share one transaction:

* collect: ``model_dump`` + digest per entity, building a :class:`PendingBatch`
  of only the entities whose payload changed (the #67 digest filter);
* write: one ``persist_batch`` transaction; if it fails, a per-entity retry so a
  single poison row cannot block every other write.

Digests are committed only after the matching write is durable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from transport.persist_tracker import PersistDigestTracker, compute_steps_digest

logger = logging.getLogger(__name__)

__all__ = ["PendingBatch", "PendingWrite", "WriteOutcome", "collect_session", "write_pending"]

_SESSION_CHILDREN = ("decisions", "agents_executed")


@dataclass(slots=True)
class PendingWrite:
    kind: str  # "session" | "decision" | "execution"
    session_id: str
    entity_key: str
    digest: str
    data: dict[str, Any]
    # Issue #209: (count, digest) of the execution steps this write persists.
    steps_state: tuple[int, str] | None = None


@dataclass
class PendingBatch:
    """Changed entities awaiting a write, plus collect-phase bookkeeping."""

    sessions: list[PendingWrite] = field(default_factory=list)
    decisions: list[PendingWrite] = field(default_factory=list)
    executions: list[PendingWrite] = field(default_factory=list)
    digested: int = 0
    failed_sessions: set[str] = field(default_factory=set)

    def all(self) -> list[PendingWrite]:
        return [*self.sessions, *self.decisions, *self.executions]


@dataclass
class WriteOutcome:
    written: int = 0
    batch_failed: bool = False
    failed_sessions: set[str] = field(default_factory=set)


def _dump(entity: Any) -> dict[str, Any]:
    return entity.model_dump() if hasattr(entity, "model_dump") else entity


def collect_session(
    tracker: PersistDigestTracker, session_id: str, session: Any, batch: PendingBatch
) -> None:
    """Digest one cached session and its children; queue the changed ones.

    Never raises: a failure marks the session in ``batch.failed_sessions`` so it
    stays dirty and is retried, without aborting the sweep for other sessions.
    """
    try:
        session_data = session.model_dump()
        # Children are persisted separately and are not columns of the sessions
        # row, so they must not influence the session digest.
        session_row = {k: v for k, v in session_data.items() if k not in _SESSION_CHILDREN}
        _queue(tracker, batch, "session", session_id, "session", session_row)

        for index, decision in enumerate(session.decisions):
            _collect_child(tracker, batch, "decision", session_id, index, decision)
        for index, agent_exec in enumerate(session.agents_executed):
            _collect_child(tracker, batch, "execution", session_id, index, agent_exec)
    except Exception as e:
        logger.error(f"Failed to persist session {session_id}: {e}")
        batch.failed_sessions.add(session_id)


def _collect_child(
    tracker: PersistDigestTracker,
    batch: PendingBatch,
    kind: str,
    session_id: str,
    index: int,
    entity: Any,
) -> None:
    try:
        data = _dump(entity)
        data["session_id"] = session_id
        if kind == "decision":
            # #217: canonical stored shape is the flat caller dict, not the
            # whole DecisionContext dump (hydrate would re-wrap it every cycle).
            ctx = data.get("context")
            if isinstance(ctx, dict) and isinstance(ctx.get("project_state"), dict):
                data["context"] = ctx["project_state"]
            key = f"decision:{data.get('id') or index}"
        else:
            key = f"execution:{data.get('id') or data.get('execution_id') or index}"
        pending = _queue(tracker, batch, kind, session_id, key, data)
        if pending is not None and kind == "execution":
            _plan_step_append(tracker, pending)
    except Exception as e:
        label = "decision" if kind == "decision" else "agent execution"
        logger.warning(f"Failed to persist {label}: {e}")
        batch.failed_sessions.add(session_id)


def _queue(
    tracker: PersistDigestTracker,
    batch: PendingBatch,
    kind: str,
    session_id: str,
    entity_key: str,
    data: dict[str, Any],
) -> PendingWrite | None:
    digest = tracker.digest_if_changed(session_id, entity_key, data)
    batch.digested += 1
    if digest is None:
        return None
    target = {"session": batch.sessions, "decision": batch.decisions}.get(kind, batch.executions)
    pending = PendingWrite(kind, session_id, entity_key, digest, data)
    target.append(pending)
    return pending


def _plan_step_append(tracker: PersistDigestTracker, pending: PendingWrite) -> None:
    """Issue #209: mark an execution payload for an append-only steps write.

    The payload keeps the full ``execution_steps`` (the backend's fallback needs
    it); ``_append_from``/``_append_steps`` tell a backend that supports it to
    send only the new tail. Backends that do not (SQLite) ignore both keys.
    """
    steps = pending.data.get("execution_steps")
    if not isinstance(steps, list):
        return
    pending.steps_state = (len(steps), compute_steps_digest(steps))
    persisted = tracker.append_plan(pending.session_id, pending.entity_key, steps)
    if persisted is not None:
        pending.data["_append_from"] = persisted
        pending.data["_append_steps"] = steps[persisted:]


def _commit(tracker: PersistDigestTracker, w: PendingWrite) -> None:
    tracker.commit(w.session_id, w.entity_key, w.digest)
    if w.steps_state is not None:
        tracker.commit_steps(w.session_id, w.entity_key, *w.steps_state)


async def write_pending(
    database: Any, tracker: PersistDigestTracker, batch: PendingBatch
) -> WriteOutcome:
    """Write ``batch`` in one transaction, falling back to entity-by-entity."""
    pending = batch.all()
    if not pending:
        return WriteOutcome()

    persist_batch = getattr(database, "persist_batch", None)
    batch_failed = False
    if persist_batch is not None:
        try:
            await persist_batch(
                [w.data for w in batch.sessions],
                [w.data for w in batch.decisions],
                [w.data for w in batch.executions],
            )
        except Exception as e:
            batch_failed = True
            logger.warning(
                f"Batch persist of {len(pending)} entities failed ({e}); retrying entity by entity"
            )
        else:
            for w in pending:
                _commit(tracker, w)
            return WriteOutcome(written=len(pending))

    outcome = await _write_one_by_one(database, tracker, pending)
    outcome.batch_failed = batch_failed
    return outcome


async def _write_one_by_one(
    database: Any, tracker: PersistDigestTracker, pending: list[PendingWrite]
) -> WriteOutcome:
    savers = {
        "session": database.save_session,
        "decision": database.save_decision,
        "execution": database.save_agent_execution,
    }
    outcome = WriteOutcome()
    for w in pending:
        try:
            await savers[w.kind](w.data)
        except Exception as e:
            logger.warning(f"Failed to persist {w.kind} {w.entity_key} of {w.session_id}: {e}")
            outcome.failed_sessions.add(w.session_id)
            continue
        _commit(tracker, w)
        outcome.written += 1
    return outcome
