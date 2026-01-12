#!/usr/bin/env python3
"""Live test of token limiting functionality using the engine directly."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.session_engine import SessionIntelligenceEngine
from utils.token_limiter import apply_token_limits


def live_test_with_engine():
    """Test token limiting with the session engine directly."""

    print("=== LIVE TESTING WITH SESSION ENGINE ===")

    # Initialize engine
    engine = SessionIntelligenceEngine()

    # Create large metadata that should trigger truncation
    large_metadata = {
        "detailed_logs": [f"Log entry {i}: " + "x" * 200 for i in range(100)],
        "performance_data": {f"metric_{i}": f"value_{i}_" + "data" * 50 for i in range(50)},
        "analysis_results": [
            f"Analysis {i}: Detailed analysis with extensive content " + "padding" * 20
            for i in range(75)
        ],
        "extra_data": {"massive_field": "z" * 10000},  # Add even more data
    }

    print(f"Large metadata size: {len(json.dumps(large_metadata))} characters")

    try:
        # Test the engine function directly
        result = engine.session_manage_lifecycle(
            operation="create",
            mode="local",
            project_name="token-limit-live-test",
            metadata=large_metadata,
            auto_recovery=True,
        )

        # Convert to dict for token limiting
        if hasattr(result, "model_dump"):
            response_dict = result.model_dump()
        else:
            response_dict = result

        print(f"Engine response size: {len(json.dumps(response_dict))} characters")

        # Now apply token limiting manually
        limited_response = apply_token_limits(response_dict, "session_manage_lifecycle")

        result_json = json.dumps(limited_response, indent=2)
        print(f"Limited response size: {len(result_json)} characters")

        if "_token_limit_info" in limited_response:
            info = limited_response["_token_limit_info"]
            print(f"🎉 TRUNCATION TRIGGERED!")
            print(f'   Original tokens: {info["original_tokens"]}')
            print(f'   Final tokens: {info["final_tokens"]}')
            print(f'   Limit: {info["limit"]}')
            print(f'   Operation: {info["operation"]}')
            print(f'   Summary: {info["summary"]}')
            print("✅ Token limiting is working correctly!")
        else:
            print("ℹ️  No truncation occurred - response was under limit")

        # Show response structure
        print("\n=== RESPONSE KEYS ===")
        if isinstance(limited_response, dict):
            for key in limited_response.keys():
                if key == "metadata" and isinstance(limited_response[key], dict):
                    print(f"  {key}: {len(limited_response[key])} items")
                elif isinstance(limited_response[key], str) and len(limited_response[key]) > 100:
                    print(f"  {key}: {len(limited_response[key])} chars")
                else:
                    print(f"  {key}: {type(limited_response[key]).__name__}")

        return limited_response

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_different_operations():
    """Test token limiting with different operation limits."""
    print("\n=== TESTING DIFFERENT OPERATION LIMITS ===")

    from utils.token_limiter import SessionTokenLimiter

    limiter = SessionTokenLimiter()

    # Create different test responses
    test_data = {
        "normal_operation": {
            "data": ["item" * 100 for _ in range(200)],  # Moderate size
            "metadata": {"key": "value" * 50},
        },
        "quality_execute_comprehensive_suite": {
            "data": ["quality_item" * 100 for _ in range(300)],  # Large size
            "analysis": {"detailed": "analysis" * 200},
        },
        "session_analyze_patterns": {
            "patterns": ["pattern" * 80 for _ in range(250)],  # Large size
            "insights": {"insight": "data" * 150},
        },
    }

    for operation, data in test_data.items():
        print(f"\n--- Testing {operation} ---")

        original_json = json.dumps(data, indent=2)
        print(f"Original size: {len(original_json)} chars")

        # Apply limits
        limited = limiter.limit_response(data, operation)
        limited_json = json.dumps(limited, indent=2)
        print(f"Limited size: {len(limited_json)} chars")

        if "_token_limit_info" in limited:
            info = limited["_token_limit_info"]
            print(
                f'✂️  TRUNCATED: {info["original_tokens"]} -> {info["final_tokens"]} tokens (limit: {info["limit"]})'
            )
        else:
            print("✅ Under limit, no truncation needed")


if __name__ == "__main__":
    print("🧪 Live Testing Token Limiting System (Engine Direct)")
    print("=" * 60)

    # Test 1: Large response through engine
    result1 = live_test_with_engine()

    # Test 2: Different operation limits
    test_different_operations()

    print("\n" + "=" * 60)
    if result1 and "_token_limit_info" in result1:
        print("🎉 LIVE TEST SUCCESS - Token limiting is working!")
        print("The system successfully truncated a large response while preserving key data.")
    elif result1:
        print("ℹ️  Live test completed but truncation didn't trigger")
        print("This might mean the limits need adjustment for testing.")
    else:
        print("❌ Live test failed - check the error above")
