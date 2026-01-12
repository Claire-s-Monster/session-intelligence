#!/usr/bin/env python3
"""Live test of token limiting functionality."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from session.server import session_manage_lifecycle


def live_test_token_limiting():
    """Test token limiting with actual MCP function call."""

    # Create large metadata that should trigger truncation
    large_metadata = {
        "detailed_logs": [f"Log entry {i}: " + "x" * 200 for i in range(100)],
        "performance_data": {f"metric_{i}": f"value_{i}_" + "data" * 50 for i in range(50)},
        "analysis_results": [
            f"Analysis {i}: Detailed analysis with extensive content " + "padding" * 20
            for i in range(75)
        ],
    }

    print("=== LIVE TESTING TOKEN LIMITING ===")
    print(f"Large metadata size: {len(json.dumps(large_metadata))} characters")

    try:
        result = session_manage_lifecycle(
            operation="create",
            mode="local",
            project_name="token-limit-live-test",
            metadata=large_metadata,
            auto_recovery=True,
        )

        result_json = json.dumps(result, indent=2)
        print(f"Response size: {len(result_json)} characters")

        if "_token_limit_info" in result:
            info = result["_token_limit_info"]
            print(f"🎉 TRUNCATION TRIGGERED!")
            print(f'   Original tokens: {info["original_tokens"]}')
            print(f'   Final tokens: {info["final_tokens"]}')
            print(f'   Limit: {info["limit"]}')
            print(f'   Operation: {info["operation"]}')
            print(f'   Summary: {info["summary"]}')
            print("✅ Token limiting is working correctly!")
        else:
            print("ℹ️  No truncation occurred - response was under limit")
            print("   This might indicate the response wasn't large enough")

        # Show response structure
        print("\n=== RESPONSE KEYS ===")
        if isinstance(result, dict):
            for key in result.keys():
                if key == "metadata" and isinstance(result[key], dict):
                    print(f"  {key}: {len(result[key])} items")
                else:
                    print(f"  {key}: {type(result[key]).__name__}")

        # Show truncated metadata if present
        if "metadata" in result and isinstance(result["metadata"], dict):
            print("\n=== METADATA KEYS (after truncation) ===")
            for key in result["metadata"].keys():
                value = result["metadata"][key]
                if isinstance(value, list):
                    print(f"  {key}: {len(value)} items")
                elif isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {len(value)} chars")
                else:
                    print(f"  {key}: {type(value).__name__}")

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_session_get_dashboard():
    """Test the dashboard function which should have lower limits."""
    print("\n=== TESTING DASHBOARD FUNCTION ===")

    try:
        # Import the dashboard function
        from session.server import session_get_dashboard

        result = session_get_dashboard(
            dashboard_type="overview", session_id="test-dashboard-session", real_time=False
        )

        result_json = json.dumps(result, indent=2)
        print(f"Dashboard response size: {len(result_json)} characters")

        if "_token_limit_info" in result:
            info = result["_token_limit_info"]
            print(f"🎉 DASHBOARD TRUNCATION!")
            print(f'   Limit was: {info["limit"]} tokens')
            print(f'   Operation: {info["operation"]}')
        else:
            print("ℹ️  Dashboard response was under limit")

        return result

    except Exception as e:
        print(f"❌ Dashboard test error: {e}")
        return None


if __name__ == "__main__":
    print("🧪 Live Testing Token Limiting System")
    print("=" * 50)

    # Test 1: Large metadata payload
    result1 = live_test_token_limiting()

    # Test 2: Dashboard function
    result2 = test_session_get_dashboard()

    print("\n" + "=" * 50)
    if result1 and "_token_limit_info" in result1:
        print("✅ Live test successful - Token limiting is working!")
    elif result1:
        print("ℹ️  Live test completed but truncation didn't trigger")
    else:
        print("❌ Live test failed")
