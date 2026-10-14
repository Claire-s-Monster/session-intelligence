"""Unit tests for core.near_duplicate (issue #89)."""

from __future__ import annotations

from core.near_duplicate import (
    NEAR_DUPLICATE_CANDIDATES,
    NEAR_DUPLICATE_THRESHOLD,
    find_near_duplicates,
    significant_tokens,
    similarity,
)


def test_constants():
    assert NEAR_DUPLICATE_THRESHOLD == 0.6
    assert NEAR_DUPLICATE_CANDIDATES == 200


def test_tokens_lowercase_and_drop_short_and_stopwords():
    toks = significant_tokens("The Quick_fox, is in the BOX and uses db42!")
    assert toks == {"quick_fox", "box", "uses", "db42"}


def test_similarity_containment():
    assert similarity({"a", "b"}, {"a", "b", "c", "d"}) == 1.0
    assert similarity({"a", "b", "c", "d"}, {"a", "x"}) == 0.5
    assert similarity(set(), {"a"}) == 0.0
    assert similarity({"a"}, set()) == 0.0


def test_short_text_skipped():
    assert find_near_duplicates("redis cache", [("x", "redis cache layer")]) == []


def test_short_candidate_skipped():
    text = "migrating the postgres backend connection handling to asyncpg pooling"
    assert find_near_duplicates(text, [("short", "postgres backend")]) == []


def test_threshold_filters():
    text = "postgres connection pool sizing"
    cands = [("hit", "postgres connection pool sizing tuned"), ("miss", "unrelated words here")]
    out = find_near_duplicates(text, cands)
    assert [w["id"] for w in out] == ["hit"]
    assert out[0]["kind"] == "near_duplicate"
    assert out[0]["score"] == 1.0
    assert out[0]["hint"] == "consider supersedes=hit rather than re-logging"


def test_limit_and_ordering():
    text = "alpha beta gamma delta"
    cands = [
        ("c1", "alpha beta gamma zzz"),  # 0.75
        ("c2", "alpha beta gamma delta"),  # 1.0
        ("c3", "alpha beta gamma delta eps"),  # 1.0
        ("c4", "alpha beta qqq www"),  # 0.5, below threshold
        ("c5", "alpha beta gamma delta zeta"),  # 1.0
    ]
    out = find_near_duplicates(text, cands, limit=3)
    assert len(out) == 3
    scores = [w["score"] for w in out]
    assert scores == sorted(scores, reverse=True)
    assert "c1" not in {w["id"] for w in out}
    out_all = find_near_duplicates(text, cands, limit=10)
    assert [w["id"] for w in out_all][-1] == "c1"
    assert "c4" not in {w["id"] for w in out_all}
