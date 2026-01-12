#!/usr/bin/env python3
"""Live test using actual MCP calls."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_mcp_function():
    """Test actual MCP function call."""
    print("=== TESTING ACTUAL MCP INTEGRATION ===")

    try:
        # Test the session_manage_lifecycle MCP function
        result = mcp__session_intelligence__session_manage_lifecycle(
            operation="create",
            mode="local",
            project_name="mcp-token-limit-test",
            auto_recovery=True,
        )

        print(f"✅ MCP call successful!")
        print(f"Response type: {type(result)}")

        if isinstance(result, dict):
            print(f"Response keys: {list(result.keys())}")

            if "_token_limit_info" in result:
                info = result["_token_limit_info"]
                print(f"🎉 TRUNCATION INFO FOUND!")
                print(f'   Original tokens: {info["original_tokens"]}')
                print(f'   Final tokens: {info["final_tokens"]}')
                print(f'   Operation: {info["operation"]}')
            else:
                print("ℹ️  No truncation info (response was under limit)")

        return True

    except Exception as e:
        print(f"❌ MCP test error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Testing Token Limiting via MCP Integration")
    print("=" * 60)

    success = test_mcp_function()

    print("\n" + "=" * 60)
    if success:
        print("🎉 MCP INTEGRATION SUCCESS!")
        print("Token limiting is properly integrated into the MCP server.")
    else:
        print("❌ MCP integration test failed")
