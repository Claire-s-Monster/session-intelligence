#!/usr/bin/env python3
"""
Quick test to check if the lean server exposes the correct 3 tools instead of 10.
"""

import json
import subprocess
import sys
import time


def test_lean_server_tools():
    """Test that lean server only exposes 3 meta-tools."""

    print("Testing lean MCP server tool count...")

    # Start the lean server as a subprocess
    proc = subprocess.Popen(
        ["pixi", "run", "mcp-server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=".",
    )

    try:
        # Send MCP initialization and tools list request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        tools_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}

        # Send requests
        proc.stdin.write(json.dumps(init_request) + "\n")
        proc.stdin.write(json.dumps(tools_request) + "\n")
        proc.stdin.flush()

        # Read responses (with timeout)
        responses = []
        for _ in range(10):  # Try to read multiple lines
            try:
                line = proc.stdout.readline()
                if line.strip():
                    response = json.loads(line)
                    responses.append(response)

                    if "tools" in response.get("result", {}):
                        tools = response["result"]["tools"]
                        tool_names = [tool["name"] for tool in tools]

                        print(f"Found {len(tools)} tools:")
                        for name in tool_names:
                            print(f"  - {name}")

                        expected_tools = {"discover_tools", "get_tool_spec", "execute_tool"}
                        actual_tools = set(tool_names)

                        if actual_tools == expected_tools:
                            print("✅ SUCCESS: Lean server exposes exactly 3 meta-tools!")
                            return True
                        else:
                            print(f"❌ FAILURE: Expected {expected_tools}, got {actual_tools}")
                            return False

            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error reading response: {e}")
                break

        print("❌ FAILURE: No tools response received")
        return False

    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    success = test_lean_server_tools()
    sys.exit(0 if success else 1)
