#!/usr/bin/env python3
"""Test token limiting with specific limits."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.token_limiter import SessionTokenLimiter, ContentType


def test_with_low_limit():
    """Test truncation with a very low limit."""
    print("Testing with low token limit (1000 tokens)...")

    # Create limiter with very low limit
    limiter = SessionTokenLimiter(default_limit=1000)

    # Create large response
    large_response = {
        "session_id": "test-session-123",
        "status": "success",
        "operation": "quality_execute_comprehensive_suite",
        "data": {
            "large_list": [
                f"Item {i} with detailed description and lots of text" for i in range(100)
            ],
            "metrics": {"count": 100, "processed": True},
            "recommendations": [f"Recommendation {i}: Detailed explanation" for i in range(20)],
        },
    }

    # Estimate original size
    original_json = json.dumps(large_response, indent=2)
    original_estimate = limiter.token_estimator.estimate_tokens(original_json, ContentType.JSON)
    print(f"Original: {original_estimate.estimated_tokens} tokens, {len(original_json)} chars")

    # Apply limits
    limited = limiter.limit_response(large_response, "quality_execute_comprehensive_suite")

    # Check result
    limited_json = json.dumps(limited, indent=2)
    final_estimate = limiter.token_estimator.estimate_tokens(limited_json, ContentType.JSON)
    print(f"Limited: {final_estimate.estimated_tokens} tokens, {len(limited_json)} chars")

    if "_token_limit_info" in limited:
        info = limited["_token_limit_info"]
        print(f"Truncation: {info['original_tokens']} -> {info['final_tokens']} tokens")
        print(f"Summary: {info['summary']}")
        return True
    else:
        print("No truncation occurred")
        return False


def test_json_truncation():
    """Test JSON-specific truncation logic."""
    print("\nTesting JSON truncation logic...")

    limiter = SessionTokenLimiter(default_limit=500)  # Very small limit

    # Create a response that mimics the problematic quality response
    quality_response = {
        "overall_health": "good",
        "health_score": 95.5,
        "test_results": {"passed": 45, "failed": 0, "skipped": 2, "coverage": 87.3},
        "lint_results": {"errors": 0, "warnings": 3, "style_issues": 12},
        "detailed_analysis": [
            {"file": f"test_file_{i}.py", "issues": [f"Issue {j}" for j in range(5)]}
            for i in range(20)
        ],
        "recommendations": [
            f"Recommendation {i}: This is a detailed recommendation with extensive explanation"
            for i in range(15)
        ],
    }

    # Apply truncation
    truncated = limiter.limit_response(quality_response, "quality_execute_comprehensive_suite")

    # Show results
    print("Keys in original:", list(quality_response.keys()))
    print("Keys in truncated:", list(truncated.keys()))

    if "_token_limit_info" in truncated:
        info = truncated["_token_limit_info"]
        print(f"Successfully truncated: {info['original_tokens']} -> {info['final_tokens']} tokens")
        return True

    return False


if __name__ == "__main__":
    success1 = test_with_low_limit()
    success2 = test_json_truncation()

    if success1 and success2:
        print("\n✅ Token limiting is working correctly!")
    else:
        print("\n❌ Token limiting needs adjustment")
