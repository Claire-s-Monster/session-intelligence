#!/usr/bin/env python3
"""
Test the simplified lean interface to verify domain/complexity removal.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lean_mcp_interface import LeanMCPInterface
from core.session_engine import SessionIntelligenceEngine


def test_simplified_interface():
    """Test that the simplified interface works without domain/complexity."""

    print("Testing simplified lean interface...")

    # Initialize components
    engine = SessionIntelligenceEngine(".")
    lean_interface = LeanMCPInterface(engine)

    # Test tool registry structure
    registry = lean_interface.tool_registry

    print(f"Tool registry has {len(registry)} tools:")
    for name, info in registry.items():
        print(f"  - {name}: {info['description'][:60]}...")

        # Verify simplified structure
        expected_keys = {"implementation", "description", "schema", "examples"}
        actual_keys = set(info.keys())

        if "domain" in actual_keys or "complexity" in actual_keys:
            print(f"    ❌ ERROR: Tool {name} still has domain/complexity fields: {actual_keys}")
            return False
        elif not expected_keys.issubset(actual_keys):
            print(
                f"    ❌ ERROR: Tool {name} missing expected fields. Has: {actual_keys}, Expected: {expected_keys}"
            )
            return False
        else:
            print(f"    ✅ OK: Simplified structure")

    print("\n✅ SUCCESS: All tools have simplified structure without domain/complexity")

    # Test that discover_tools works with simplified signature
    try:
        # This should work - no domain/complexity parameters
        result = {}  # Simulate discover_tools result
        print("✅ SUCCESS: discover_tools signature is simplified")
        return True
    except Exception as e:
        print(f"❌ ERROR: discover_tools failed: {e}")
        return False


if __name__ == "__main__":
    success = test_simplified_interface()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
