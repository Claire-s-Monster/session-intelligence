#!/usr/bin/env python3
"""Create a test that should definitely trigger token limiting."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.token_limiter import SessionTokenLimiter


def test_guaranteed_truncation():
    """Test with guaranteed large response."""

    print("=== TESTING GUARANTEED TRUNCATION ===")

    # Create limiter with very low limit for testing
    limiter = SessionTokenLimiter(default_limit=500)  # Very low limit

    # Create a definitely large response
    massive_response = {
        "operation": "test_massive_response",
        "status": "success",
        "data": {
            "large_array": [
                f"This is item {i} with lots of padding text " * 20 for i in range(200)
            ],
            "detailed_analysis": {
                f"analysis_section_{i}": f"Detailed analysis content {i} " * 50 for i in range(50)
            },
            "logs": [f"Log entry {i}: " + "x" * 100 for i in range(100)],
            "metadata": {
                "processing_time": 2.5,
                "records_processed": 10000,
                "full_report": "This is a very detailed report. " * 500,
            },
        },
    }

    # Calculate original size
    original_json = json.dumps(massive_response, indent=2)
    original_estimate = limiter.token_estimator.estimate_tokens(
        original_json, limiter.token_estimator.detect_content_type(original_json)
    )

    print(f"Original response:")
    print(f"  Characters: {len(original_json)}")
    print(f"  Estimated tokens: {original_estimate.estimated_tokens}")
    print(f"  Token limit: {limiter.default_limit}")
    print(f"  Will truncate: {original_estimate.estimated_tokens > limiter.default_limit}")

    # Apply token limiting
    limited_response = limiter.limit_response(massive_response, "test_massive_response")

    # Show results
    limited_json = json.dumps(limited_response, indent=2)
    final_estimate = limiter.token_estimator.estimate_tokens(
        limited_json, limiter.token_estimator.detect_content_type(limited_json)
    )

    print(f"\nLimited response:")
    print(f"  Characters: {len(limited_json)}")
    print(f"  Estimated tokens: {final_estimate.estimated_tokens}")

    if "_token_limit_info" in limited_response:
        info = limited_response["_token_limit_info"]
        print(f"\n🎉 TRUNCATION SUCCESS!")
        print(f'  Original tokens: {info["original_tokens"]}')
        print(f'  Final tokens: {info["final_tokens"]}')
        print(f'  Tokens saved: {info["original_tokens"] - info["final_tokens"]}')
        print(f'  Operation: {info["operation"]}')
        print(f'  Summary: {info["summary"]}')

        # Show what keys survived
        print(f"\nKeys in original: {list(massive_response.keys())}")
        print(f"Keys in limited: {list(limited_response.keys())}")

        if "data" in limited_response:
            print(
                f'Data keys survived: {list(limited_response["data"].keys()) if isinstance(limited_response["data"], dict) else "not dict"}'
            )

        return True
    else:
        print("\n❌ No truncation occurred!")
        return False


if __name__ == "__main__":
    print("🧪 Testing Guaranteed Truncation")
    print("=" * 50)

    success = test_guaranteed_truncation()

    print("\n" + "=" * 50)
    if success:
        print("✅ Token limiting working perfectly!")
    else:
        print("❌ Token limiting needs investigation")
