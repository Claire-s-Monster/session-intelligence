#!/usr/bin/env python3
"""Test script to verify local project directory detection."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.session_engine import SessionIntelligenceEngine


def test_local_path():
    """Test the local project path detection."""
    engine = SessionIntelligenceEngine()

    print(f"Current working directory: {Path.cwd()}")
    print(f"Session path: {engine.claude_sessions_path}")
    print(f"Path exists: {engine.claude_sessions_path.exists()}")
    print(
        f"Is relative to current project: {str(engine.claude_sessions_path).startswith(str(Path.cwd()))}"
    )

    # Test project root detection
    project_path = engine._get_project_session_path()
    print(f"Detected project session path: {project_path}")

    # Show what markers were found
    current_path = Path.cwd()
    project_markers = [
        ".git",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "go.mod",
        ".project",
        "composer.json",
    ]

    print("\nProject markers found:")
    for path in [current_path] + list(current_path.parents):
        for marker in project_markers:
            if (path / marker).exists():
                print(f"  {marker} found at: {path}")
                return  # Exit after first match (same logic as the engine)


if __name__ == "__main__":
    test_local_path()
