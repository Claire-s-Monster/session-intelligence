#!/usr/bin/env bash
#
# Daemon management for session-intelligence HTTP server.
#
# Usage:
#   ./session-intelligence-daemon.sh start [options]
#   ./session-intelligence-daemon.sh stop
#   ./session-intelligence-daemon.sh status
#   ./session-intelligence-daemon.sh restart [options]
#
# Options:
#   --port PORT           Port to bind to (default: 4002)
#   --api-key KEY         API key for authentication
#   --backend TYPE        Database backend: sqlite|postgresql (default: sqlite)
#   --dsn DSN             PostgreSQL connection string
#   --db-path PATH        SQLite database path
#
# Environment Variables:
#   SESSION_DB_BACKEND    Database backend (sqlite|postgresql)
#   SESSION_DB_DSN        PostgreSQL connection string
#   SESSION_DB_PATH       SQLite database path
#
# The server runs as a background daemon, enabling cross-session state
# sharing between multiple Claude Code instances.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Global state directory (cross-project)
STATE_DIR="${HOME}/.claude/session-intelligence"
PID_FILE="${STATE_DIR}/server.pid"
LOG_FILE="${STATE_DIR}/server.log"
DEFAULT_PORT=4002

# Ensure state directory exists
mkdir -p "$STATE_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    else
        echo ""
    fi
}

start_server() {
    local port="${1:-$DEFAULT_PORT}"
    local api_key="${2:-}"
    local backend="${3:-}"
    local dsn="${4:-}"
    local db_path="${5:-}"

    # Check if already running
    if is_running; then
        local pid
        pid=$(get_pid)
        log "Server already running with PID $pid"
        exit 1
    fi

    # Clean up stale PID file
    if [ -f "$PID_FILE" ]; then
        rm -f "$PID_FILE"
    fi

    # Build command
    local cmd="pixi run http-server --port $port"

    if [ -n "$api_key" ]; then
        cmd="$cmd --api-key $api_key"
    fi

    if [ -n "$backend" ]; then
        cmd="$cmd --backend $backend"
    fi

    if [ -n "$dsn" ]; then
        cmd="$cmd --dsn $dsn"
    fi

    if [ -n "$db_path" ]; then
        cmd="$cmd --db-path $db_path"
    fi

    # Determine backend for logging
    local display_backend="${backend:-${SESSION_DB_BACKEND:-sqlite}}"

    log "Starting session-intelligence HTTP server..."
    log "  Port: $port"
    log "  Backend: $display_backend"
    if [ "$display_backend" = "postgresql" ]; then
        log "  DSN: ${dsn:-${SESSION_DB_DSN:-postgresql://localhost/session_intelligence}}"
    else
        log "  Database: ${db_path:-${SESSION_DB_PATH:-$STATE_DIR/sessions.db}}"
    fi

    # Start server in background
    cd "$PROJECT_DIR"
    nohup $cmd >> "$LOG_FILE" 2>&1 &
    local pid=$!

    echo "$pid" > "$PID_FILE"

    # Wait briefly for startup
    sleep 2

    # Verify it's running
    if is_running; then
        log "Server started successfully (PID: $pid)"
        log "Log file: $LOG_FILE"
        log ""
        log "Endpoints:"
        log "  POST http://127.0.0.1:$port/mcp - MCP requests"
        log "  GET  http://127.0.0.1:$port/mcp - SSE notifications"
        log "  GET  http://127.0.0.1:$port/health - Health check"
        log "  GET  http://127.0.0.1:$port/api/sessions - Session list"
    else
        log "ERROR: Server failed to start. Check log file: $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
}

stop_server() {
    if ! is_running; then
        log "Server not running"
        rm -f "$PID_FILE"
        exit 0
    fi

    local pid
    pid=$(get_pid)

    log "Stopping server (PID: $pid)..."

    # Graceful shutdown
    kill "$pid" 2>/dev/null || true

    # Wait for graceful shutdown (up to 10 seconds)
    local count=0
    while is_running && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done

    # Force kill if still running
    if is_running; then
        log "Force killing..."
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi

    rm -f "$PID_FILE"
    log "Server stopped"
}

status_server() {
    if ! is_running; then
        log "Server not running"
        if [ -f "$PID_FILE" ]; then
            log "(Stale PID file found, cleaning up)"
            rm -f "$PID_FILE"
        fi
        exit 1
    fi

    local pid
    pid=$(get_pid)

    log "Server running (PID: $pid)"

    # Try health check
    local health_url="http://127.0.0.1:$DEFAULT_PORT/health"
    if command -v curl > /dev/null 2>&1; then
        if curl -s "$health_url" > /dev/null 2>&1; then
            log "Health check: OK"
            curl -s "$health_url" | python3 -m json.tool 2>/dev/null || true
        else
            log "Health check: FAILED (server may be starting up)"
        fi
    else
        log "Health check: curl not available"
    fi

    log ""
    log "Log file: $LOG_FILE"
    log "Recent log entries:"
    tail -5 "$LOG_FILE" 2>/dev/null || echo "  (no log entries)"
}

restart_server() {
    log "Restarting server..."
    stop_server
    sleep 1
    start_server "$@"
}

show_logs() {
    local lines="${1:-50}"
    if [ -f "$LOG_FILE" ]; then
        tail -"$lines" "$LOG_FILE"
    else
        log "No log file found"
    fi
}

# Parse arguments
PORT="$DEFAULT_PORT"
API_KEY=""
BACKEND=""
DSN=""
DB_PATH=""
LINES="50"
ACTION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        start|stop|status|restart|logs)
            ACTION="$1"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --dsn)
            DSN="$2"
            shift 2
            ;;
        --db-path)
            DB_PATH="$2"
            shift 2
            ;;
        --lines)
            LINES="$2"
            shift 2
            ;;
        -h|--help)
            cat << 'EOF'
Usage: session-intelligence-daemon.sh {start|stop|restart|status|logs} [options]

Commands:
  start     Start the HTTP server as a daemon
  stop      Stop the running server
  restart   Restart the server
  status    Check server status and health
  logs      Show recent log entries

Options:
  --port PORT         Port to bind to (default: 4002)
  --api-key KEY       API key for authentication
  --backend TYPE      Database backend: sqlite|postgresql (default: sqlite)
  --dsn DSN           PostgreSQL connection string
  --db-path PATH      SQLite database path (default: ~/.claude/session-intelligence/sessions.db)
  --lines N           Number of log lines to show (default: 50)

Environment Variables:
  SESSION_DB_BACKEND  Database backend (sqlite|postgresql)
  SESSION_DB_DSN      PostgreSQL connection string
  SESSION_DB_PATH     SQLite database path

Examples:
  # Start with SQLite (default)
  ./session-intelligence-daemon.sh start

  # Start with PostgreSQL
  ./session-intelligence-daemon.sh start --backend postgresql --dsn "postgresql://localhost/sessions"

  # Start with custom port and API key
  ./session-intelligence-daemon.sh start --port 5000 --api-key mysecret

  # Check status
  ./session-intelligence-daemon.sh status

  # View logs
  ./session-intelligence-daemon.sh logs --lines 100
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Execute action
case "${ACTION:-}" in
    start)
        start_server "$PORT" "$API_KEY" "$BACKEND" "$DSN" "$DB_PATH"
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server "$PORT" "$API_KEY" "$BACKEND" "$DSN" "$DB_PATH"
        ;;
    status)
        status_server
        ;;
    logs)
        show_logs "$LINES"
        ;;
    "")
        echo "Usage: $0 {start|stop|restart|status|logs} [options]"
        echo "Use --help for more information"
        exit 1
        ;;
    *)
        echo "Unknown command: $ACTION"
        exit 1
        ;;
esac
