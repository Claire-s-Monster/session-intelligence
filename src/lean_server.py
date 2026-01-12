"""
Lean MCP Server Entry Point.

This is the new lean entry point that exposes only 3 meta-tools instead of 10+ verbose tools.
Reduces context consumption from 20-50K tokens down to ~500 tokens while maintaining
full functionality through dynamic tool discovery.

Context Consumption Comparison:
- Traditional MCP: 10 tools × 2-5K tokens each = 20-50K tokens
- Lean MCP: 3 meta-tools × ~150 tokens each = ~500 tokens
- Savings: 95%+ reduction in context consumption

Usage:
- Agents use discover_tools() to find relevant tools
- Agents use get_tool_spec() to get full schemas only when needed
- Agents use execute_tool() for actual execution
- Zero functionality loss, massive context savings
"""

import argparse
import logging
from pathlib import Path

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import create_lean_interface

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Lean Session Intelligence MCP Server")
    parser.add_argument(
        "--repository",
        type=str,
        default=".",
        help="Repository root path (default: current directory)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def setup_logging(log_level: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main():
    """Main entry point for the lean MCP server."""
    try:
        args = parse_args()
        setup_logging(args.log_level)

        # Resolve repository path
        repository_path = Path(args.repository).resolve()
        logger.info(f"Initializing lean MCP server for repository: {repository_path}")

        # Initialize session intelligence engine
        session_engine = SessionIntelligenceEngine(repository_path=str(repository_path))

        # Create lean interface with 3 meta-tools
        app = create_lean_interface(session_engine)

        logger.info("Starting lean MCP server with meta-tool pattern")
        logger.info("Context consumption: ~500 tokens (vs 20-50K tokens for traditional MCP)")

        # Run the server
        app.run()

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
