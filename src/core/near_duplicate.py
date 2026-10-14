"""Near-duplicate detection for decisions and learnings (issue #89).

Pure functions: token-containment similarity over significant tokens.
"""

from __future__ import annotations

import re

NEAR_DUPLICATE_THRESHOLD = 0.6
NEAR_DUPLICATE_CANDIDATES = 200

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

_STOPWORDS = frozenset(
    {
        "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
        "had", "her", "was", "one", "our", "out", "has", "have", "this",
        "that", "with", "from", "they", "will", "would", "there", "their",
        "what", "which", "when", "were", "been", "than", "then", "into",
        "its", "also", "use", "used", "using", "should",
    }
)  # fmt: skip


def significant_tokens(text: str) -> set[str]:
    """Lowercase tokens of length >= 3 excluding common stopwords."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 3 and t not in _STOPWORDS}


def similarity(a: set[str], b: set[str]) -> float:
    """Containment: |A & B| / min(|A|, |B|); 0.0 if either is empty."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def find_near_duplicates(
    text: str,
    candidates: list[tuple[str, str]],
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
    limit: int = 3,
    min_tokens: int = 3,
) -> list[dict[str, object]]:
    """Return the top near-duplicate candidates for text as warning dicts.

    Both the new text and each candidate need >= min_tokens significant
    tokens; shorter ones are skipped (containment would be a false positive).
    """
    tokens = significant_tokens(text)
    if len(tokens) < min_tokens:
        return []
    scored = []
    for cid, ctext in candidates:
        ctokens = significant_tokens(ctext)
        if len(ctokens) >= min_tokens:
            scored.append((cid, similarity(tokens, ctokens)))
    hits = sorted((s for s in scored if s[1] >= threshold), key=lambda s: s[1], reverse=True)
    return [
        {
            "kind": "near_duplicate",
            "id": cid,
            "score": round(score, 2),
            "hint": f"consider supersedes={cid} rather than re-logging",
        }
        for cid, score in hits[:limit]
    ]
