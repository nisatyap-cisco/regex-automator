#!/usr/bin/env bash
# Start the Regex Discovery System (Flask UI + pipeline server)
# Usage: bash start.sh [--port 5001]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${1:-5001}"
if [[ "$1" == "--port" ]] 2>/dev/null; then
    PORT="${2:-5001}"
fi

SESSION="regex-system"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already running. Attaching..."
    tmux attach -t "$SESSION"
    exit 0
fi

tmux new-session -d -s "$SESSION" -n server

tmux send-keys -t "$SESSION:server" "cd $SCRIPT_DIR && source .venv/bin/activate && python web/app.py --port $PORT" Enter

tmux split-window -t "$SESSION:server" -v -p 30
tmux send-keys -t "$SESSION:server.1" "cd $SCRIPT_DIR && source .venv/bin/activate" Enter
tmux send-keys -t "$SESSION:server.1" "echo '──────────────────────────────────────────'" Enter
tmux send-keys -t "$SESSION:server.1" "echo 'CLI pane — run pipelines manually here:'" Enter
tmux send-keys -t "$SESSION:server.1" "echo '  python main.py --condition \"california zip code\"'" Enter
tmux send-keys -t "$SESSION:server.1" "echo '──────────────────────────────────────────'" Enter

tmux select-pane -t "$SESSION:server.0"

echo "╔═══════════════════════════════════════════╗"
echo "║  Regex Discovery System started!          ║"
echo "║                                           ║"
echo "║  UI: http://0.0.0.0:$PORT                 ║"
echo "║                                           ║"
echo "║  Top pane  = Flask server                 ║"
echo "║  Bottom pane = CLI (optional manual runs) ║"
echo "║                                           ║"
echo "║  Ctrl+B then arrow keys = switch panes    ║"
echo "║  Ctrl+B then d = detach session           ║"
echo "╚═══════════════════════════════════════════╝"

tmux attach -t "$SESSION"
