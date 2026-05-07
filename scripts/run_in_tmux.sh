#!/usr/bin/env bash
# Wrapper to launch a long-running command inside a detached tmux session.
# - Creates logs/<name>.log with full output (tee'd, so still visible if attached).
# - Session name = first argument; rest is the command.
# - Idempotent: if a session with that name is already running, refuses
#   to overwrite and tells you how to attach.
#
# Usage:
#   bash scripts/run_in_tmux.sh transformer_v1 \
#       python -m src.models.train_transformer --gpu --name v1
#
#   tmux attach -t transformer_v1     # to view output / interact
#   tmux ls                           # list active sessions
#   tail -f logs/transformer_v1.log   # watch from anywhere

set -euo pipefail

if [ $# -lt 2 ]; then
    cat <<EOF >&2
Usage: $0 <session_name> <command...>

Examples:
  $0 transformer_v1 python -m src.models.train_transformer --gpu --name v1
  $0 catch22 python -m src.features.catch22_features
  $0 round1 bash -c 'python -m src.features.catch22_features && \\
      python -m src.models.train_xgb_cat --gpu --models xgb cat && \\
      python -m src.models.train_minirocket --name v1'

After launching:
  tmux ls                             list active sessions
  tmux attach -t <session_name>       reattach (Ctrl+b d to detach)
  tail -f logs/<session_name>.log     follow log without attaching
  tmux kill-session -t <session_name> stop the session
EOF
    exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux not installed. On Debian/Ubuntu: sudo apt install tmux" >&2
    exit 1
fi

SESSION="$1"; shift
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${SESSION}.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists." >&2
    echo "Attach with: tmux attach -t '$SESSION'" >&2
    echo "Kill with:   tmux kill-session -t '$SESSION'" >&2
    exit 2
fi

# Build the command, tee'd to the log so output is captured even if the
# session is detached.
INNER_CMD="cd '$ROOT' && exec $* 2>&1 | tee '$LOG_FILE'"

tmux new-session -d -s "$SESSION" "bash -lc \"$INNER_CMD\""

cat <<EOF
Launched in detached tmux session '$SESSION'.
  Attach:        tmux attach -t '$SESSION'
  Detach inside: Ctrl+b then d
  Follow log:    tail -f '$LOG_FILE'
  Kill:          tmux kill-session -t '$SESSION'
EOF
