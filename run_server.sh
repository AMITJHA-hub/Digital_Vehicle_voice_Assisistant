#!/usr/bin/env bash
# In-Vehicle AI Voice Assistant - Unified Server Launcher
# Automatically manages ports, virtual environment, and hardware mode.

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
PORT=8080
SSL_PORT=8443
MODE="physical"
DAEMON=false

# Print header
echo "=========================================================="
echo "    IN-VEHICLE AI VOICE ASSISTANT - SERVER LAUNCHER      "
echo "=========================================================="

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --sim|--simulation)
            MODE="simulation"
            shift
            ;;
        --physical)
            MODE="physical"
            shift
            ;;
        --bg|--background)
            DAEMON=true
            shift
            ;;
        --stop)
            echo "[*] Stopping any running voice assistant server instances..."
            fuser -k -n tcp "$PORT" 2>/dev/null || true
            fuser -k -n tcp "$SSL_PORT" 2>/dev/null || true
            pkill -f "app/main.py.*--web" 2>/dev/null || true
            echo "[+] Voice assistant stopped."
            exit 0
            ;;
        --help|-h)
            echo "Usage: ./run_server.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  (no args)       Run in Physical mode (real GPIO, default)"
            echo "  --sim           Run in Simulation mode (no GPIO required)"
            echo "  --bg            Run in background (detached, logs to server.log)"
            echo "  --stop          Stop any running server"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo "[!] Unknown option: $1"
            echo "    Run './run_server.sh --help' for usage."
            exit 1
            ;;
    esac
done

# Check virtual environment
if [ ! -f "$VENV_PYTHON" ]; then
    echo "[ERROR] Python virtual environment not found at $VENV_PYTHON"
    echo "Please ensure the .venv directory is configured."
    exit 1
fi

# Clean up any existing process holding ports 8080 or 8443
echo "[*] Checking for existing server instances..."
EXISTING_PIDS=$(lsof -t -i :$PORT -i :$SSL_PORT 2>/dev/null || true)
if [ -n "$EXISTING_PIDS" ]; then
    echo "[*] Cleaning up stale process(es): $EXISTING_PIDS"
    fuser -k -n tcp "$PORT" 2>/dev/null || true
    fuser -k -n tcp "$SSL_PORT" 2>/dev/null || true
    pkill -f "app/main.py.*--web" 2>/dev/null || true
    sleep 1
fi

# Export PYTHONPATH so all internal modules resolve cleanly
export PYTHONPATH="$PROJECT_DIR"

# Get current LAN IP address
LAN_IP=$(hostname -I | awk '{print $1}')
if [ -z "$LAN_IP" ]; then
    LAN_IP="127.0.0.1"
fi

echo "[+] Hardware Mode : ${MODE^^}"
echo "[+] Local LAN IP  : $LAN_IP"
echo "[+] Web Dashboard : http://$LAN_IP:$PORT"
echo "[+] Phone Mic SSL : https://$LAN_IP:$SSL_PORT"
echo "----------------------------------------------------------"

if [ "$DAEMON" = true ]; then
    echo "[*] Starting server in the background..."
    nohup "$VENV_PYTHON" app/main.py --web --mode "$MODE" --port "$PORT" > "$PROJECT_DIR/server.log" 2>&1 &
    NEW_PID=$!
    disown "$NEW_PID" 2>/dev/null || true
    sleep 2
    if ps -p "$NEW_PID" > /dev/null; then
        echo "[SUCCESS] Server is running in the background! (PID: $NEW_PID)"
        echo "Logs are available at: tail -f $PROJECT_DIR/server.log"
        echo "To stop the server: ./run_server.sh --stop"
    else
        echo "[ERROR] Server failed to start. Last log lines:"
        cat "$PROJECT_DIR/server.log"
        exit 1
    fi
else
    echo "[*] Launching server in the foreground..."
    echo "    (Press Ctrl+C to stop)"
    echo "=========================================================="
    exec "$VENV_PYTHON" app/main.py --web --mode "$MODE" --port "$PORT"
fi
