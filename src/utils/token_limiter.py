"""
Token limiting utilities for Session Intelligence MCP Server.

This module provides token estimation and intelligent content truncation
to prevent responses that exceed MCP token limits while preserving
semantic meaning and important information.
"""

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of content for different truncation strategies."""

    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"
    LOG = "log"
    METRICS = "metrics"


@dataclass
class TokenEstimate:
    """Token estimation result."""

    estimated_tokens: int
    char_count: int
    content_type: ContentType
    confidence: float = 0.85


@dataclass
class TruncationResult:
    """Result of content truncation operation."""

    content: str
    original_tokens: int
    final_tokens: int
    truncated: bool
    truncation_summary: str


class TokenEstimator:
    """Estimates token usage for different types of content."""

    # Token estimation ratios (characters per token) by content type
    RATIOS = {
        ContentType.TEXT: 4.0,  # Regular text ~4 chars/token
        ContentType.JSON: 3.5,  # JSON is more dense ~3.5 chars/token
        ContentType.STRUCTURED: 4.5,  # Structured data ~4.5 chars/token
        ContentType.LOG: 3.0,  # Logs are dense ~3 chars/token
        ContentType.METRICS: 2.5,  # Metrics/numbers are very dense ~2.5 chars/token
    }

    def estimate_tokens(
        self, content: str, content_type: ContentType = ContentType.TEXT
    ) -> TokenEstimate:
        """Estimate token count for content."""
        if not content:
            return TokenEstimate(0, 0, content_type)

        char_count = len(content)
        ratio = self.RATIOS.get(content_type, 4.0)
        estimated_tokens = int(char_count / ratio)

        return TokenEstimate(
            estimated_tokens=estimated_tokens, char_count=char_count, content_type=content_type
        )

    def detect_content_type(self, content: str) -> ContentType:
        """Auto-detect content type from string."""
        if not content.strip():
            return ContentType.TEXT

        # Try to parse as JSON
        try:
            json.loads(content)
            return ContentType.JSON
        except (json.JSONDecodeError, TypeError):
            pass

        # Check for common patterns
        if re.search(r"\d{4}-\d{2}-\d{2}.*\d{2}:\d{2}:\d{2}", content):
            return ContentType.LOG

        if re.search(r'("[\w_]+"\s*:\s*([\d.]+|"[^"]*"))', content):
            return ContentType.METRICS

        if re.search(r"^[\s]*[\{\[]", content.strip()):
            return ContentType.STRUCTURED

        return ContentType.TEXT


class IntelligentTruncator:
    """Intelligently truncates content while preserving important information."""

    def __init__(self):
        self.token_estimator = TokenEstimator()

    def truncate_content(
        self, content: str, max_tokens: int, content_type: ContentType = None
    ) -> TruncationResult:
        """Truncate content intelligently to fit within token limit."""
        if not content:
            return TruncationResult(
                content="",
                original_tokens=0,
                final_tokens=0,
                truncated=False,
                truncation_summary="",
            )

        # Auto-detect content type if not provided
        if content_type is None:
            content_type = self.token_estimator.detect_content_type(content)

        original_estimate = self.token_estimator.estimate_tokens(content, content_type)

        # If already under limit, return as-is
        if original_estimate.estimated_tokens <= max_tokens:
            return TruncationResult(
                content=content,
                original_tokens=original_estimate.estimated_tokens,
                final_tokens=original_estimate.estimated_tokens,
                truncated=False,
                truncation_summary="",
            )

        # Apply content-type specific truncation
        if content_type == ContentType.JSON:
            truncated_content = self._truncate_json(content, max_tokens)
        elif content_type == ContentType.STRUCTURED:
            truncated_content = self._truncate_structured(content, max_tokens)
        elif content_type == ContentType.LOG:
            truncated_content = self._truncate_log(content, max_tokens)
        elif content_type == ContentType.METRICS:
            truncated_content = self._truncate_metrics(content, max_tokens)
        else:
            truncated_content = self._truncate_text(content, max_tokens)

        final_estimate = self.token_estimator.estimate_tokens(truncated_content, content_type)

        # Calculate savings
        tokens_saved = original_estimate.estimated_tokens - final_estimate.estimated_tokens
        truncation_summary = f"Content truncated: {tokens_saved} tokens saved ({original_estimate.estimated_tokens} -> {final_estimate.estimated_tokens})"

        return TruncationResult(
            content=truncated_content,
            original_tokens=original_estimate.estimated_tokens,
            final_tokens=final_estimate.estimated_tokens,
            truncated=True,
            truncation_summary=truncation_summary,
        )

    def _truncate_json(self, content: str, max_tokens: int) -> str:
        """Truncate JSON content intelligently."""
        try:
            data = json.loads(content)

            # For dictionaries, prioritize certain keys
            if isinstance(data, dict):
                priority_keys = [
                    "session_id",
                    "status",
                    "error",
                    "message",
                    "operation",
                    "summary",
                    "health_score",
                    "recommendations",
                    "next_steps",
                ]

                # Keep priority keys first
                truncated_data = {}
                for key in priority_keys:
                    if key in data:
                        truncated_data[key] = data[key]

                # Add other keys until we hit the limit
                remaining_keys = [k for k in data.keys() if k not in priority_keys]
                for key in remaining_keys:
                    test_data = {**truncated_data, key: data[key]}
                    test_content = json.dumps(test_data, indent=2)
                    if (
                        self.token_estimator.estimate_tokens(
                            test_content, ContentType.JSON
                        ).estimated_tokens
                        > max_tokens
                    ):
                        break
                    truncated_data[key] = data[key]

                # Add truncation indicator
                if len(truncated_data) < len(data):
                    truncated_data["_truncated"] = (
                        f"... {len(data) - len(truncated_data)} more fields"
                    )

                return json.dumps(truncated_data, indent=2)

            elif isinstance(data, list):
                # For lists, keep first N items
                truncated_list = []
                for i, item in enumerate(data):
                    test_list = truncated_list + [item]
                    test_content = json.dumps(test_list, indent=2)
                    if (
                        self.token_estimator.estimate_tokens(
                            test_content, ContentType.JSON
                        ).estimated_tokens
                        > max_tokens
                    ):
                        break
                    truncated_list.append(item)

                # Add truncation indicator
                if len(truncated_list) < len(data):
                    truncated_list.append(f"... {len(data) - len(truncated_list)} more items")

                return json.dumps(truncated_list, indent=2)

        except (json.JSONDecodeError, TypeError):
            # Fall back to text truncation
            return self._truncate_text(content, max_tokens)

        return json.dumps(data, indent=2)

    def _truncate_structured(self, content: str, max_tokens: int) -> str:
        """Truncate structured content by lines."""
        lines = content.split("\n")
        truncated_lines = []

        for line in lines:
            test_content = "\n".join(truncated_lines + [line])
            if (
                self.token_estimator.estimate_tokens(
                    test_content, ContentType.STRUCTURED
                ).estimated_tokens
                > max_tokens
            ):
                break
            truncated_lines.append(line)

        if len(truncated_lines) < len(lines):
            truncated_lines.append(f"... {len(lines) - len(truncated_lines)} more lines truncated")

        return "\n".join(truncated_lines)

    def _truncate_log(self, content: str, max_tokens: int) -> str:
        """Truncate log content, keeping beginning and end."""
        lines = content.split("\n")

        if len(lines) <= 20:  # Small logs, truncate normally
            return self._truncate_structured(content, max_tokens)

        # For large logs, keep first 10 and last 5 lines with summary
        head_lines = lines[:10]
        tail_lines = lines[-5:]

        truncated_content = "\n".join(head_lines)
        truncated_content += f"\n\n... {len(lines) - 15} lines truncated ...\n\n"
        truncated_content += "\n".join(tail_lines)

        # If still too long, fall back to basic truncation
        if (
            self.token_estimator.estimate_tokens(
                truncated_content, ContentType.LOG
            ).estimated_tokens
            > max_tokens
        ):
            return self._truncate_text(truncated_content, max_tokens)

        return truncated_content

    def _truncate_metrics(self, content: str, max_tokens: int) -> str:
        """Truncate metrics content, prioritizing key metrics."""
        # Try to parse as JSON first
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                # Keep key metrics
                return self._truncate_json(content, max_tokens)
        except (json.JSONDecodeError, TypeError):
            pass

        # Fall back to structured truncation
        return self._truncate_structured(content, max_tokens)

    def _truncate_text(self, content: str, max_tokens: int) -> str:
        """Basic text truncation with word boundaries."""
        # Calculate approximate character limit
        char_limit = int(max_tokens * 4.0)  # ~4 chars per token

        if len(content) <= char_limit:
            return content

        # Find last word boundary before limit
        truncation_point = char_limit
        while truncation_point > 0 and content[truncation_point] != " ":
            truncation_point -= 1

        if truncation_point == 0:  # No word boundary found, hard truncate
            truncation_point = char_limit - 20  # Leave room for indicator

        truncated = content[:truncation_point].rstrip()
        truncated += (
            f"\n\n... [Content truncated: {len(content) - truncation_point} more characters]"
        )

        return truncated


class SessionTokenLimiter:
    """Main token limiter for session intelligence responses."""

    def __init__(self, default_limit: int = 25000, enable_truncation: bool = True):
        self.default_limit = default_limit
        self.enable_truncation = enable_truncation
        self.truncator = IntelligentTruncator()
        self.token_estimator = TokenEstimator()

        # Operation-specific limits (lower limits for typically verbose operations)
        self.operation_limits = {
            "session_get_dashboard": 20000,
            "session_analyze_patterns": 15000,
            "session_monitor_health": 20000,
            "quality_execute_comprehensive_suite": 15000,  # This was the problematic one
        }

        logger.info(f"SessionTokenLimiter initialized with default limit: {default_limit}")

    def _to_dict(self, obj: Any) -> Any:
        """Convert Pydantic models and nested structures to plain dicts."""
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, dict):
            return {k: self._to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_dict(item) for item in obj]
        return obj

    def limit_response(
        self, response: dict[str, Any] | BaseModel, operation: str = "unknown"
    ) -> dict[str, Any]:
        """Limit response size by truncating content if necessary."""
        # Convert Pydantic models to dicts first
        response = self._to_dict(response)

        if not self.enable_truncation:
            return response

        # Get operation-specific limit or default
        token_limit = self.operation_limits.get(operation, self.default_limit)

        # Convert response to JSON to estimate size
        response_json = json.dumps(response, indent=2, default=str)
        estimate = self.token_estimator.estimate_tokens(response_json, ContentType.JSON)

        # If under limit, return as-is
        if estimate.estimated_tokens <= token_limit:
            logger.debug(
                f"Response for {operation}: {estimate.estimated_tokens} tokens (under limit)"
            )
            return response

        logger.info(
            f"Response for {operation}: {estimate.estimated_tokens} tokens exceeds limit of {token_limit}, truncating..."
        )

        # Truncate the response
        truncation_result = self.truncator.truncate_content(
            response_json, token_limit, ContentType.JSON
        )

        try:
            # Parse back to dict
            truncated_response = json.loads(truncation_result.content)

            # Add metadata about truncation
            if isinstance(truncated_response, dict):
                truncated_response["_token_limit_info"] = {
                    "original_tokens": truncation_result.original_tokens,
                    "final_tokens": truncation_result.final_tokens,
                    "truncated": truncation_result.truncated,
                    "operation": operation,
                    "limit": token_limit,
                    "summary": truncation_result.truncation_summary,
                }

            logger.info(
                f"Successfully truncated {operation}: {truncation_result.original_tokens} -> {truncation_result.final_tokens} tokens"
            )
            return truncated_response

        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse truncated response: {e}")
            # Return minimal error response
            return {
                "error": "Response too large and truncation failed",
                "original_size_tokens": estimate.estimated_tokens,
                "limit": token_limit,
                "operation": operation,
            }

    def update_limits(self, **operation_limits):
        """Update operation-specific limits."""
        self.operation_limits.update(operation_limits)
        logger.info(f"Updated operation limits: {operation_limits}")


# Global token limiter instance
token_limiter = SessionTokenLimiter()


def apply_token_limits(response: dict[str, Any], operation: str = "unknown") -> dict[str, Any]:
    """Convenience function to apply token limits to responses."""
    return token_limiter.limit_response(response, operation)
