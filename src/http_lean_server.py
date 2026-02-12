#!/usr/bin/env python3
"""
HTTP entry point for Session Intelligence MCP Server.

Provides HTTP transport with:
- POST /mcp for JSON-RPC 2.0 MCP requests
- GET /mcp for SSE streaming notifications
- GET /health for health checks
- REST API for session queries

Usage:
    pixi run http-server
    pixi run http-server --port 4002 --api-key secret
    python src/http_lean_server.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports (must be before local imports)
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# ruff: noqa: E402
from transport.http_server import HTTPSessionIntelligenceServer
from transport.security import SecurityConfig


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the HTTP server."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Session Intelligence HTTP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Start server with defaults (localhost:4002, PostgreSQL):
    pixi run http-server

  Start with custom DSN:
    pixi run http-server --dsn "postgresql://localhost/sessions"

  Start with custom port:
    pixi run http-server --port 5000

  Start with API key authentication:
    pixi run http-server --api-key mysecretkey

Environment Variables:
  SESSION_DB_DSN: PostgreSQL connection string (default: postgresql://localhost/session_intelligence)
  SESSION_DB_POOL_MIN: Connection pool minimum size (default: 2)
  SESSION_DB_POOL_MAX: Connection pool maximum size (default: 10)
  SESSION_INTELLIGENCE_API_KEY: API key for authentication
        """,
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1, localhost only for security)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4002,
        help="Port to bind to (default: 4002)",
    )
    parser.add_argument(
        "--repository",
        default=".",
        help="Repository path for session engine (default: current directory)",
    )
    parser.add_argument(
        "--dsn",
        help="PostgreSQL connection string (default: postgresql://localhost/session_intelligence)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for authentication (optional)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow network connections (DANGEROUS - only use in trusted networks)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the HTTP server."""
    args = parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Security warning if network access enabled
    if args.allow_network:
        logger.warning(
            "Network access enabled! This exposes the server to all network interfaces. "
            "Only use this in trusted networks."
        )

    # Build security config
    security_config = SecurityConfig(
        localhost_only=not args.allow_network,
        require_api_key=bool(args.api_key),
        api_key=args.api_key,
    )

    # Build database config from args (overrides env/file)
    from persistence import DatabaseConfig

    db_config = DatabaseConfig.load()  # Start with env/file config

    if args.dsn:
        db_config.postgresql_dsn = args.dsn

    # Create and run server
    server = HTTPSessionIntelligenceServer(
        host=args.host if args.allow_network else "127.0.0.1",
        port=args.port,
        repository_path=args.repository,
        db_config=db_config,
        security_config=security_config,
    )

    logger.info("Starting Session Intelligence HTTP Server")
    logger.info(f"  Host: {server.host}")
    logger.info(f"  Port: {server.port}")
    logger.info("  Backend: postgresql")
    logger.info(f"  Database: {db_config.postgresql_dsn}")
    logger.info(f"  API Key: {'enabled' if args.api_key else 'disabled'}")
    logger.info(f"  Network Access: {'enabled' if args.allow_network else 'localhost only'}")
    logger.info("")
    logger.info("REST API Endpoints (direct curl/HTTP access, no MCP protocol needed):")
    logger.info(f"  GET  http://{server.host}:{server.port}/health - Health check")
    logger.info(f"  GET  http://{server.host}:{server.port}/api/sessions - List sessions")
    logger.info(f"  GET  http://{server.host}:{server.port}/api/sessions/{{id}} - Get session")
    logger.info(
        f"  POST http://{server.host}:{server.port}/tools/agent_query_learnings - Query learnings"
    )
    logger.info(
        f"  POST http://{server.host}:{server.port}/tools/session_find_solution - Find solutions"
    )
    logger.info(
        f"  POST http://{server.host}:{server.port}/tools/session_log_learning - Log learning"
    )
    logger.info("")
    logger.info("MCP Protocol Endpoints (JSON-RPC 2.0):")
    logger.info(f"  POST http://{server.host}:{server.port}/mcp - MCP requests")
    logger.info(f"  GET  http://{server.host}:{server.port}/mcp - SSE notifications")
    logger.info("")

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
