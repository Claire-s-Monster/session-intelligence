"""Pure helpers for outline / section / window retrieval of notebook bodies (issue #141)."""

from __future__ import annotations

import re
from bisect import bisect_right
from typing import Any

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_MAX_LISTED_HEADINGS = 50


def validate_window_params(offset: int, max_chars: int | None, search: str | None = None) -> None:
    """Raise ValueError for a negative offset, non-positive max_chars or blank search."""
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    if max_chars is not None and max_chars < 1:
        raise ValueError(f"max_chars must be >= 1, got {max_chars}")
    if search is not None and not search.strip():
        raise ValueError("search must be a non-empty, non-whitespace string")


def find_matches(
    body: str, query: str, context_chars: int = 120, max_matches: int = 20
) -> tuple[list[dict[str, Any]], int]:
    """Find case-insensitive literal, non-overlapping occurrences of ``query``.

    Returns (matches, total_found); ``matches`` holds at most ``max_matches``
    dicts of {offset (absolute, in ``body``), heading (nearest enclosing
    heading or None), snippet}.
    """
    outline = parse_outline(body)
    heading_offsets = [h["offset"] for h in outline]
    matches: list[dict[str, Any]] = []
    total = 0
    for hit in re.finditer(re.escape(query), body, re.IGNORECASE):
        total += 1
        if len(matches) >= max_matches:
            continue
        start = hit.start()
        idx = bisect_right(heading_offsets, start) - 1
        matches.append(
            {
                "offset": start,
                "heading": outline[idx]["heading"] if idx >= 0 else None,
                "snippet": body[max(0, start - context_chars) : hit.end() + context_chars],
            }
        )
    return matches, total


def parse_outline(body: str) -> list[dict[str, Any]]:
    """Return ATX headings as {heading, level, offset, chars}, skipping fenced code.

    ``chars`` spans from the heading line to the next heading of the same or
    higher level (fewer-or-equal ``#``), or to the end of the body.
    """
    headings: list[dict[str, Any]] = []
    in_fence = False
    pos = 0
    for line in body.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            match = _HEADING_RE.match(line.rstrip("\r\n"))
            if match:
                headings.append(
                    {
                        "heading": match.group(2),
                        "level": len(match.group(1)),
                        "offset": pos,
                        "chars": 0,
                    }
                )
        pos += len(line)

    for i, current in enumerate(headings):
        end = len(body)
        for later in headings[i + 1 :]:
            if later["level"] <= current["level"]:
                end = later["offset"]
                break
        current["chars"] = end - current["offset"]
    return headings


def extract_section(body: str, name: str) -> tuple[str, str | None]:
    """Return (section_text, error). On no/ambiguous match text is "" and error is set."""
    outline = parse_outline(body)
    wanted = name.strip().lower()
    exact = [h for h in outline if h["heading"].strip().lower() == wanted]
    candidates = exact or [h for h in outline if wanted in h["heading"].strip().lower()]
    if len(candidates) == 1:
        hit = candidates[0]
        return body[hit["offset"] : hit["offset"] + hit["chars"]], None

    problem = "ambiguous" if candidates else "not found"
    names = [h["heading"] for h in outline[:_MAX_LISTED_HEADINGS]]
    more = len(outline) - len(names)
    suffix = f" (+{more} more)" if more > 0 else ""
    return "", f"Section {name!r} {problem}. Available headings: {names}{suffix}"


def apply_window(
    text: str, offset: int, max_chars: int | None
) -> tuple[str, dict[str, int | None]]:
    """Slice ``text`` and return (window, metadata) with elision accounting."""
    total = len(text)
    end = total if max_chars is None else offset + max_chars
    window = text[offset:end]
    returned = len(window)
    elided = max(0, total - offset - returned)
    return window, {
        "total_chars": total,
        "offset": offset,
        "returned_chars": returned,
        "elided_chars": elided,
        "next_offset": offset + returned if elided > 0 else None,
    }
