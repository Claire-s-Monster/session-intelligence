#!/usr/bin/env python3
"""Test script to verify token limiting functionality."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.token_limiter import SessionTokenLimiter, apply_token_limits, ContentType


def test_token_estimation():
    """Test token estimation for different content types."""
    print("Testing token estimation...")

    limiter = SessionTokenLimiter()

    # Test various content types
    test_cases = [
        ("Short text", ContentType.TEXT),
        ('{"key": "value", "count": 42}', ContentType.JSON),
        ("Line 1\nLine 2\nLine 3\n", ContentType.STRUCTURED),
        ("2025-01-01 12:00:00 INFO: Starting process", ContentType.LOG),
        ('{"cpu": 85.2, "memory": 67.1, "disk": 45.8}', ContentType.METRICS),
    ]

    for content, content_type in test_cases:
        estimate = limiter.token_estimator.estimate_tokens(content, content_type)
        print(
            f"{content_type.value:10}: {estimate.estimated_tokens:3} tokens | {estimate.char_count:3} chars | '{content[:30]}...'"
        )


def test_large_response_truncation():
    """Test truncation of large responses."""
    print("\nTesting large response truncation...")

    # Create a large response that exceeds token limits
    large_response = {
        "session_id": "test-session-123",
        "status": "success",
        "operation": "test_operation",
        "data": {
            "large_list": [
                f"Item {i} with detailed description and lots of text to make it bigger"
                for i in range(1000)
            ],
            "metadata": {
                "processing_time": 1.25,
                "records_processed": 1000,
                "details": "This is a very long detailed response with lots of nested data structures and content that would normally exceed token limits in MCP responses, causing errors like the one we're trying to fix.",
            },
        },
        "recommendations": [
            f"Recommendation {i}: This is a detailed recommendation with extensive explanation and context"
            for i in range(50)
        ],
    }

    # Test with default limits
    print("Original response size:")
    original_json = json.dumps(large_response, indent=2)
    print(f"  Characters: {len(original_json)}")

    # Apply token limits
    limited_response = apply_token_limits(large_response, "test_operation")

    print("\nTruncated response size:")
    limited_json = json.dumps(limited_response, indent=2)
    print(f"  Characters: {len(limited_json)}")

    # Check if truncation info was added
    if "_token_limit_info" in limited_response:
        info = limited_response["_token_limit_info"]
        print(f"  Token reduction: {info['original_tokens']} -> {info['final_tokens']}")
        print(f"  Summary: {info['summary']}")

    return limited_response


def test_small_response_passthrough():
    """Test that small responses pass through unchanged."""
    print("\nTesting small response passthrough...")

    small_response = {
        "session_id": "test-session-456",
        "status": "success",
        "message": "This is a small response that should pass through unchanged.",
    }

    # Apply token limits
    result = apply_token_limits(small_response, "test_small_operation")

    # Should be unchanged (no truncation info added)
    if "_token_limit_info" not in result:
        print("✅ Small response passed through unchanged")
    else:
        print("❌ Small response was modified unexpectedly")

    return result


def main():
    """Run all tests."""
    print("🧪 Testing Session Intelligence Token Limiting")
    print("=" * 50)

    try:
        test_token_estimation()
        large_result = test_large_response_truncation()
        small_result = test_small_response_passthrough()

        print("\n✅ All token limiting tests completed successfully!")
        print(f"Large response truncated: {'_token_limit_info' in large_result}")
        print(f"Small response unchanged: {'_token_limit_info' not in small_result}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
