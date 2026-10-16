"""Backend-neutral planning for session_transfer_entries (issue #244 part 1).

The backends differ only in placeholder syntax, so selection is expressed as
condition templates: ``%s`` is a scalar parameter and ``@IN(col)`` is a list
membership test. ``render`` turns a template into backend SQL.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_SUMMARY_CLIP = 80

# Effective scope = per-entry override, else the owning session's value.
DECISION_SELECT = """
    SELECT d.id AS id, d.session_id AS session_id, d.timestamp AS ts,
           COALESCE(d.summary, d.description) AS label,
           COALESCE(d.project_name, s.project_name) AS project_name,
           COALESCE(d.project_path, s.project_path) AS project_path
    FROM decisions d JOIN sessions s ON d.session_id = s.id
"""
NOTEBOOK_SELECT = """
    SELECT ss.session_id AS id, ss.session_id AS session_id, ss.created_at AS ts,
           COALESCE(ss.title, ss.summary_markdown) AS label,
           COALESCE(ss.project_name, s.project_name) AS project_name,
           COALESCE(ss.project_path, s.project_path) AS project_path
    FROM session_summaries ss JOIN sessions s ON ss.session_id = s.id
"""
LEARNING_SELECT = """
    SELECT id AS id, source_session_id AS session_id, created_at AS ts,
           COALESCE(summary, learning_content) AS label,
           project_name AS project_name, project_path AS project_path
    FROM project_learnings
"""
SESSION_SELECT = """
    SELECT id AS id, id AS session_id, started_at AS ts,
           COALESCE(session_name, id) AS label,
           project_name AS project_name, project_path AS project_path
    FROM sessions
"""
SELECTS = {
    "decision": DECISION_SELECT,
    "notebook": NOTEBOOK_SELECT,
    "learning": LEARNING_SELECT,
    "session": SESSION_SELECT,
}
_KIND_KEYS = {
    "decision": "decisions",
    "notebook": "notebooks",
    "learning": "learnings",
    "session": "sessions",
}

_TOKEN = re.compile(r"%s|@IN\(([\w.]+)\)")


@dataclass
class Selection:
    kind: str
    cond: str
    params: list[Any]
    requested: list[str] = field(default_factory=list)


def _scope_conds(kind: str, name: str, path: str | None, since: Any, until: Any) -> Selection:
    """Window selection (from_project_name + optional path/since/until)."""
    params: list[Any] = [name]
    if kind == "learning":
        conds = [
            "(project_name = %s OR (project_name IS NULL AND project_path IN ("
            "SELECT project_path FROM sessions WHERE project_name = %s "
            "AND project_path IS NOT NULL)))"
        ]
        params.append(name)
        path_col, ts_col = "project_path", "created_at"
    else:
        alias = "d" if kind == "decision" else "ss"
        conds = [f"COALESCE({alias}.project_name, s.project_name) = %s"]
        path_col = f"COALESCE({alias}.project_path, s.project_path)"
        ts_col = f"{alias}.timestamp" if kind == "decision" else f"{alias}.created_at"
    for value, expr in (
        (path, f"{path_col} = %s"),
        (since, f"{ts_col} >= %s"),
        (until, f"{ts_col} <= %s"),
    ):
        if value is not None:
            conds.append(expr)
            params.append(value)
    return Selection(kind, " AND ".join(conds), params)


def plan_selections(
    *,
    decision_ids: list[str] | None,
    learning_ids: list[str] | None,
    notebook_session_ids: list[str] | None,
    session_id: str | None,
    from_project_name: str | None,
    from_project_path: str | None,
    since: Any,
    until: Any,
) -> list[Selection]:
    """Every query whose union of rows is the transfer set."""
    plan: list[Selection] = []
    for kind, ids, col in (
        ("decision", decision_ids, "d.id"),
        ("learning", learning_ids, "id"),
        ("notebook", notebook_session_ids, "ss.session_id"),
    ):
        if ids:
            plan.append(Selection(kind, f"@IN({col})", [list(ids)], list(ids)))
    if session_id:
        plan.append(Selection("session", "id = %s", [session_id], [session_id]))
        plan.append(Selection("decision", "d.session_id = %s", [session_id]))
        plan.append(Selection("notebook", "ss.session_id = %s", [session_id]))
        plan.append(Selection("learning", "source_session_id = %s", [session_id]))
    if from_project_name:
        for kind in ("decision", "notebook", "learning"):
            plan.append(_scope_conds(kind, from_project_name, from_project_path, since, until))
    return plan


def render(cond: str, params: list[Any], dialect: str, start: int = 1) -> tuple[str, list[Any]]:
    """Render a condition template for ``sqlite`` (?) or ``postgres`` ($n)."""
    args: list[Any] = []
    values = iter(params)

    def sub(match: re.Match[str]) -> str:
        value = next(values)
        col = match.group(1)
        n = start + len(args)
        if dialect == "sqlite":
            args.append(json.dumps(value) if col else value)
            return f"{col} IN (SELECT value FROM json_each(?))" if col else "?"
        args.append(value)
        return f"{col} = ANY(${n}::text[])" if col else f"${n}"

    return _TOKEN.sub(sub, cond), args


def make_entry(kind: str, row: dict[str, Any]) -> dict[str, Any]:
    ts = row["ts"]
    label = (row["label"] or "").strip().replace("\n", " ")
    return {
        "kind": kind,
        "id": row["id"],
        "current_project_name": row["project_name"],
        "current_project_path": row["project_path"],
        "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else ts,
        "summary": label[:_SUMMARY_CLIP],
        "_session_id": row["session_id"],
    }


def finalize_entries(
    found: dict[tuple[str, str], dict[str, Any]],
    plan: list[Selection],
    max_entries: int,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Cap-check the deduplicated set; report requested ids that matched nothing."""
    if len(found) > max_entries:
        raise ValueError(
            f"matched {len(found)} entries, exceeding max_entries={max_entries}; "
            "narrow the selectors or raise max_entries (max 1000). Nothing was written."
        )
    missing: dict[str, list[str]] = {}
    for sel in plan:
        absent = [i for i in sel.requested if (sel.kind, i) not in found]
        if absent:
            missing.setdefault(_KIND_KEYS[sel.kind], []).extend(absent)
    return list(found.values()), missing


def split_for_apply(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Group ids by write action. Entries owned by a re-homed session just
    clear their overrides so they follow the session."""
    rehomed = {e["id"] for e in entries if e["kind"] == "session"}
    out: dict[str, Any] = {
        "sessions": sorted(rehomed),
        "decisions_set": [],
        "decisions_clear": [],
        "notebooks_set": [],
        "notebooks_clear": [],
        "learnings": [],
    }
    for e in entries:
        kind = e["kind"]
        if kind == "learning":
            out["learnings"].append(e["id"])
        elif kind in ("decision", "notebook"):
            action = "clear" if e["_session_id"] in rehomed else "set"
            out[f"{kind}s_{action}"].append(e["id"])
    return out


_UPDATES = (
    ("decisions_set", "UPDATE decisions SET project_name = %s, project_path = %s WHERE @IN(id)"),
    (
        "decisions_clear",
        "UPDATE decisions SET project_name = NULL, project_path = NULL WHERE @IN(id)",
    ),
    (
        "notebooks_set",
        "UPDATE session_summaries SET project_name = %s, project_path = %s WHERE @IN(session_id)",
    ),
    (
        "notebooks_clear",
        "UPDATE session_summaries SET project_name = NULL, project_path = NULL "
        "WHERE @IN(session_id)",
    ),
    (
        "learnings",
        "UPDATE project_learnings SET project_name = %s, "
        "project_path = COALESCE(%s, project_path) WHERE @IN(id)",
    ),
    (
        "sessions",
        "UPDATE sessions SET project_name = %s, "
        "project_path = COALESCE(%s, project_path) WHERE @IN(id)",
    ),
)


def apply_statements(
    groups: dict[str, Any], name: str, path: str | None, dialect: str
) -> list[tuple[str, list[Any]]]:
    """UPDATE statements (already rendered) for the non-empty groups."""
    out: list[tuple[str, list[Any]]] = []
    for key, template in _UPDATES:
        ids = groups[key]
        if not ids:
            continue
        params: list[Any] = [] if key.endswith("_clear") else [name, path]
        out.append(render(template, [*params, ids], dialect))
    return out


def build_result(
    dry_run: bool, entries: list[dict[str, Any]], missing: dict[str, list[str]]
) -> dict[str, Any]:
    counts = dict.fromkeys(_KIND_KEYS.values(), 0)
    for e in entries:
        counts[_KIND_KEYS[e["kind"]]] += 1
    public = [{k: v for k, v in e.items() if not k.startswith("_")} for e in entries]
    return {"dry_run": dry_run, "counts": counts, "entries": public, "not_found": missing}
