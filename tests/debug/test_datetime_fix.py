#!/usr/bin/env python3
"""Quick test to verify datetime fix."""
import sys

sys.path.insert(0, "src")

from datetime import UTC, datetime

# Test basic UTC datetime
print("=== Testing datetime fix ===")
utc_now = datetime.now(UTC)
print(f"datetime.now(UTC) = {utc_now}")
print(f"Has timezone: {utc_now.tzinfo is not None}")

# Test imports
try:
    from core.session_engine import SessionIntelligenceEngine

    print("✓ SessionIntelligenceEngine imported successfully")
except Exception as e:
    print(f"✗ SessionIntelligenceEngine import failed: {e}")

try:
    from transport.mcp_session_manager import MCPSessionManager

    print("✓ MCPSessionManager imported successfully")
except Exception as e:
    print(f"✗ MCPSessionManager import failed: {e}")

# Test session engine with datetime
try:
    engine = SessionIntelligenceEngine(use_filesystem=False)
    result = engine.session_manage_lifecycle("create", project_name="test-project")
    print(f"  Result: {result}")
    print(f"✓ Session created: {result.session_id}, status: {result.status}")

    # Get session and verify datetime is timezone-aware
    if result.session_id in engine.session_cache:
        session = engine.session_cache[result.session_id]
        started = session.started
        print(f"✓ Session started: {started}")
        print(f"✓ Has timezone: {started.tzinfo is not None}")

        # Test notebook creation (this was also failing!)
        notebook_result = engine.session_create_notebook(
            session_id=result.session_id,
            title="Test Notebook",
            include_decisions=True,
            include_agents=True,
            include_metrics=True,
            save_to_file=False,
            save_to_database=False,
        )
        print(f"✓ Notebook created: {notebook_result.status}")

        # Test finalization (this was the bug!)
        final_result = engine.session_manage_lifecycle("finalize")
        print(f"✓ Session finalized: {final_result.status}")

    print("\n=== All tests passed! ===")
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback

    traceback.print_exc()
