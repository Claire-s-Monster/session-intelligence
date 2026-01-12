#!/usr/bin/env python3
"""Test async notebook generation with file operations."""
import requests
import json

BASE = "http://127.0.0.1:4002/mcp"
MCP_SESSION_ID = None


def call(method, params=None, id_=1):
    global MCP_SESSION_ID
    headers = {"Content-Type": "application/json"}
    if MCP_SESSION_ID:
        headers["MCP-Session-Id"] = MCP_SESSION_ID
    data = {"jsonrpc": "2.0", "method": method, "id": id_}
    if params:
        data["params"] = params
    r = requests.post(BASE, headers=headers, json=data)
    # Capture session ID from response
    if "MCP-Session-Id" in r.headers:
        MCP_SESSION_ID = r.headers["MCP-Session-Id"]
    return r.json()


def tool_call(tool, args):
    return call("tools/call", {"name": tool, "arguments": args})


# Initialize
resp = call(
    "initialize",
    {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0"},
    },
)
print("1. Initialize:", resp.get("result", {}).get("serverInfo", {}).get("name"))
print(f"   MCP Session: {MCP_SESSION_ID}")

# Create session
print("\n2. Create session:")
r = tool_call(
    "execute_tool",
    {
        "tool_name": "session_manage_lifecycle",
        "parameters": {"operation": "create", "project_name": "notebook-async-test"},
    },
)
print(f"   Raw: {json.dumps(r, indent=2)[:500]}")
result = r.get("result", {})
session_id = None
if isinstance(result, dict) and "content" in result:
    content = json.loads(result["content"][0]["text"])
    session_id = content.get("result", {}).get("session_id")
    print(f"   Session: {session_id}")
elif "error" in r:
    print(f"   Error: {r['error']}")

if session_id:
    # Track file operation
    print("\n3. Track file operation:")
    r = tool_call(
        "execute_tool",
        {
            "tool_name": "session_track_file_operation",
            "parameters": {
                "session_id": session_id,
                "operation": "create",
                "file_path": "/test/example.py",
                "summary": "Created test file",
                "lines_added": 50,
            },
        },
    )
    result = r.get("result", {})
    if isinstance(result, dict) and "content" in result:
        import json

        content = json.loads(result["content"][0]["text"])
        print(f"   Status: {content.get('status')}")

    # Log decision
    print("\n4. Log decision:")
    r = tool_call(
        "execute_tool",
        {
            "tool_name": "session_log_decision",
            "parameters": {
                "session_id": session_id,
                "decision": "Use async for database queries",
                "context": {"rationale": "Sync context cannot await DB"},
            },
        },
    )
    result = r.get("result", {})
    if isinstance(result, dict) and "content" in result:
        import json

        content = json.loads(result["content"][0]["text"])
        print(f"   Status: {content.get('status')}")

    # Create notebook (async)
    print("\n5. Create notebook (async):")
    r = tool_call(
        "execute_tool",
        {
            "tool_name": "session_create_notebook",
            "parameters": {
                "session_id": session_id,
                "title": "Async Test Notebook",
                "save_to_file": False,
                "save_to_database": False,
            },
        },
    )
    result = r.get("result", {})
    if isinstance(result, dict) and "content" in result:
        import json

        content = json.loads(result["content"][0]["text"])
        notebook = content.get("result", {})
        print(f"   Status: {notebook.get('status')}")
        if "notebook" in notebook:
            nb = notebook["notebook"]
            sections = nb.get("sections", [])
            headings = [s.get("heading") for s in sections]
            print(f"   Sections: {headings}")

            # Check Work Completed (files)
            files_sec = next((s for s in sections if s.get("heading") == "Work Completed"), None)
            if files_sec:
                content = files_sec.get("content", "")
                if "No files" in content:
                    print("   Files: EMPTY (PRD bug still present)")
                else:
                    print(f"   Files: {len(content)} chars - SUCCESS!")
                    print(f"   Preview: {content[:200]}")
            else:
                print("   Files section: NOT FOUND")

            # Check Decisions
            decisions_sec = next(
                (s for s in sections if s.get("heading") == "Decisions Made"), None
            )
            if decisions_sec:
                content = decisions_sec.get("content", "")
                if "No decisions" in content:
                    print("   Decisions: EMPTY")
                else:
                    print(f"   Decisions: {len(content)} chars - SUCCESS!")
            else:
                print("   Decisions section: NOT FOUND")

            # Check Project Learnings
            learnings_sec = next(
                (s for s in sections if s.get("heading") == "Project Learnings"), None
            )
            if learnings_sec:
                print(f"   Learnings: Found ({len(learnings_sec.get('content', ''))} chars)")
            else:
                print("   Learnings section: NOT FOUND")
    else:
        print(f"   Error: {r}")
