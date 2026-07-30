"""Unit tests for src/utils/token_limiter.py.

Tests TokenEstimator, IntelligentTruncator, SessionTokenLimiter,
and the apply_token_limits convenience function.
"""

import json

import pytest

from utils.token_limiter import (
    ContentType,
    IntelligentTruncator,
    SessionTokenLimiter,
    TokenEstimate,
    TokenEstimator,
    TruncationResult,
    apply_token_limits,
)


# ---------------------------------------------------------------------------
# TokenEstimator
# ---------------------------------------------------------------------------


class TestTokenEstimator:
    def setup_method(self):
        self.estimator = TokenEstimator()

    def test_empty_content_returns_zero(self):
        result = self.estimator.estimate_tokens("", ContentType.TEXT)
        assert result.estimated_tokens == 0
        assert result.char_count == 0

    def test_returns_token_estimate_instance(self):
        result = self.estimator.estimate_tokens("hello world", ContentType.TEXT)
        assert isinstance(result, TokenEstimate)

    def test_text_ratio_approx_4_chars_per_token(self):
        content = "a" * 400
        result = self.estimator.estimate_tokens(content, ContentType.TEXT)
        assert result.estimated_tokens == 100

    def test_json_ratio_denser_than_text(self):
        content = "a" * 350
        result = self.estimator.estimate_tokens(content, ContentType.JSON)
        assert result.estimated_tokens == 100

    def test_log_ratio_denser_than_json(self):
        content = "a" * 300
        result = self.estimator.estimate_tokens(content, ContentType.LOG)
        assert result.estimated_tokens == 100

    def test_detect_json_content(self):
        content = json.dumps({"key": "value", "count": 42})
        assert self.estimator.detect_content_type(content) == ContentType.JSON

    def test_detect_log_content(self):
        content = "2025-01-01 12:00:00 INFO: Starting process"
        assert self.estimator.detect_content_type(content) == ContentType.LOG

    def test_detect_plain_text_fallback(self):
        content = "Just some plain text without any special structure."
        assert self.estimator.detect_content_type(content) == ContentType.TEXT

    def test_detect_empty_string_returns_text(self):
        assert self.estimator.detect_content_type("") == ContentType.TEXT

    def test_char_count_matches_len(self):
        content = "hello"
        result = self.estimator.estimate_tokens(content, ContentType.TEXT)
        assert result.char_count == 5

    def test_content_type_preserved_in_estimate(self):
        result = self.estimator.estimate_tokens("data", ContentType.METRICS)
        assert result.content_type == ContentType.METRICS


# ---------------------------------------------------------------------------
# IntelligentTruncator
# ---------------------------------------------------------------------------


class TestIntelligentTruncator:
    def setup_method(self):
        self.truncator = IntelligentTruncator()

    def test_empty_content_returns_empty_result(self):
        result = self.truncator.truncate_content("", 1000)
        assert result.content == ""
        assert result.truncated is False

    def test_returns_truncation_result_instance(self):
        result = self.truncator.truncate_content("hello", 1000)
        assert isinstance(result, TruncationResult)

    def test_small_content_not_truncated(self):
        content = "short content"
        result = self.truncator.truncate_content(content, 10000, ContentType.TEXT)
        assert result.truncated is False
        assert result.content == content

    def test_large_text_is_truncated(self):
        content = "word " * 5000  # ~5000 * 5 chars = 25000 chars ~ 6250 tokens
        result = self.truncator.truncate_content(content, 100, ContentType.TEXT)
        assert result.truncated is True
        assert len(result.content) < len(content)

    def test_json_dict_truncation_preserves_priority_keys(self):
        data = {
            "session_id": "abc123",
            "status": "ok",
            "some_other_key": "x" * 10000,
        }
        content = json.dumps(data, indent=2)
        result = self.truncator.truncate_content(content, 50, ContentType.JSON)
        truncated = json.loads(result.content)
        assert "session_id" in truncated
        assert "status" in truncated

    def test_json_list_truncation_adds_indicator(self):
        data = [f"item_{i}" for i in range(200)]
        content = json.dumps(data, indent=2)
        result = self.truncator.truncate_content(content, 50, ContentType.JSON)
        assert result.truncated is True
        truncated = json.loads(result.content)
        assert any("more items" in str(item) for item in truncated)

    def test_log_truncation_keeps_head_and_tail(self):
        lines = [f"2025-01-01 12:{i:02d}:00 INFO: log entry {i}" for i in range(50)]
        content = "\n".join(lines)
        result = self.truncator.truncate_content(content, 50, ContentType.LOG)
        assert result.truncated is True
        assert "truncated" in result.content

    def test_truncation_summary_is_informative(self):
        content = "word " * 5000
        result = self.truncator.truncate_content(content, 100, ContentType.TEXT)
        assert result.truncation_summary != ""
        assert "->" in result.truncation_summary or "tokens" in result.truncation_summary

    def test_final_tokens_less_than_original_when_truncated(self):
        content = "word " * 5000
        result = self.truncator.truncate_content(content, 100, ContentType.TEXT)
        assert result.final_tokens < result.original_tokens


# ---------------------------------------------------------------------------
# SessionTokenLimiter
# ---------------------------------------------------------------------------


class TestSessionTokenLimiter:
    def setup_method(self):
        self.limiter = SessionTokenLimiter(default_limit=25000)

    def test_small_response_passes_through_unchanged(self):
        response = {"session_id": "x", "status": "ok", "message": "small"}
        result = self.limiter.limit_response(response, "test_op")
        assert "_token_limit_info" not in result
        assert result["status"] == "ok"

    def test_large_response_is_truncated(self):
        response = {
            "session_id": "abc",
            "status": "success",
            "data": {"items": [f"item {i} " * 100 for i in range(200)]},
        }
        limiter = SessionTokenLimiter(default_limit=500)
        result = limiter.limit_response(response, "test_op")
        assert "_token_limit_info" in result

    def test_truncated_response_contains_metadata(self):
        response = {"data": "x" * 50000}
        limiter = SessionTokenLimiter(default_limit=500)
        result = limiter.limit_response(response, "my_op")
        if "_token_limit_info" in result:
            info = result["_token_limit_info"]
            assert "original_tokens" in info
            assert "final_tokens" in info
            assert info["operation"] == "my_op"

    def test_operation_specific_limit_overrides_default(self):
        # session_get_dashboard has limit 20000 by default
        limiter = SessionTokenLimiter(default_limit=25000)
        assert limiter.operation_limits.get("session_get_dashboard") == 20000

    def test_update_limits_changes_operation_limit(self):
        self.limiter.update_limits(my_custom_op=5000)
        assert self.limiter.operation_limits["my_custom_op"] == 5000

    def test_truncation_disabled_returns_large_response_intact(self):
        limiter = SessionTokenLimiter(default_limit=1, enable_truncation=False)
        response = {"data": "x" * 10000}
        result = limiter.limit_response(response, "op")
        assert "_token_limit_info" not in result

    def test_pydantic_model_converted_to_dict(self):
        from pydantic import BaseModel

        class MyModel(BaseModel):
            name: str
            value: int

        model = MyModel(name="test", value=42)
        result = self.limiter.limit_response(model, "test_op")  # type: ignore[arg-type]
        assert isinstance(result, dict)
        assert result["name"] == "test"

    def test_apply_token_limits_convenience_function(self):
        response = {"status": "ok", "msg": "small"}
        result = apply_token_limits(response, "some_op")
        assert isinstance(result, dict)
        assert result["status"] == "ok"
