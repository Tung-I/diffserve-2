#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config (override via env)
# =========================
NUM_WORKERS="${NUM_WORKERS:-16}"
WORKER_PORT_START="${WORKER_PORT_START:-50051}"
SINK_PORT="${SINK_PORT:-50100}"
CASCADE="${CASCADE:-multi}"            # sdturbo | sdxs | sdxlltn | multi
PYTHON_BIN="${PYTHON_BIN:-python}"
IP_ADDRESS="${IP_ADDRESS:-$(hostname -I | awk '{print $1}')}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_DIR="$SCRIPT_DIR/src/worker"
LOG_DIR="$SCRIPT_DIR/logs"
PID_DIR="$LOG_DIR/pids"
PID_FILE="$PID_DIR/workers.pids"

ensure_dirs() { mkdir -p "$LOG_DIR" "$PID_DIR"; }

alive() { local pid="${1:-}"; [[ -n "${pid}" ]] && kill -0 "$pid" 2>/dev/null; }

# ---- Start ----
start_workers() {
  ensure_dirs
  if [[ -f "$PID_FILE" ]]; then
    echo "[WARN] $PID_FILE exists. Use '$0 stop' first if workers are running."
  fi
  : > "$PID_FILE"

  echo "[INFO] Starting $NUM_WORKERS simulated workers on $IP_ADDRESS (cascade=$CASCADE)..."
  for ((i=0; i<NUM_WORKERS; i++)); do
    port=$((WORKER_PORT_START + i))
    (
      cd "$WORKER_DIR"
      nohup "$PYTHON_BIN" worker.py -cip "$IP_ADDRESS" -p "$port" -c "$CASCADE" --do_simulate \
        > "$LOG_DIR/worker_${port}.out" 2>&1 &
      echo $! >> "$PID_FILE"
    )
    echo "  - worker port $port started"
  done

  echo "[INFO] Starting sink worker on port $SINK_PORT..."
  (
    cd "$WORKER_DIR"
    nohup "$PYTHON_BIN" worker.py -cip "$IP_ADDRESS" -p "$SINK_PORT" -c "$CASCADE" --is_sink \
      > "$LOG_DIR/worker_sink_${SINK_PORT}.out" 2>&1 &
    echo $! >> "$PID_FILE"
  )
  echo "  - sink worker port $SINK_PORT started"

  echo "[OK] Launched $(wc -l < "$PID_FILE") worker processes. PIDs saved in $PID_FILE"
}

# Collect PIDs from multiple sources to be safe
collect_pids() {
  local pids=()

  # 1) From pidfile if present
  if [[ -f "$PID_FILE" ]]; then
    while read -r pid; do [[ -n "$pid" ]] && pids+=("$pid"); done < "$PID_FILE"
  fi

  # 2) From command line pattern
  #    (matches this repo's worker.py; adjust if running from different path)
  while read -r line; do
    [[ -n "$line" ]] && pids+=("$line")
  done < <(pgrep -f "$WORKER_DIR/worker.py" || true)

  # 3) From listening ports
  for p in $(seq "$WORKER_PORT_START" $((WORKER_PORT_START + NUM_WORKERS - 1))) "$SINK_PORT"; do
    # lsof returns just PIDs with -t; okay if absent
    while read -r pid; do
      [[ -n "$pid" ]] && pids+=("$pid")
    done < <(lsof -ti tcp:"$p" 2>/dev/null || true)
  done

  # 4) From open files in logs directory (processes that still hold logs open)
  if command -v lsof >/dev/null 2>&1; then
    while read -r pid; do
      [[ -n "$pid" ]] && pids+=("$pid")
    done < <(lsof +D "$LOG_DIR" 2>/dev/null | awk 'NR>1{print $2}' | sort -u || true)
  fi

  # Dedup
  printf "%s\n" "${pids[@]}" | awk 'NF' | sort -u
}

# ---- Stop ----
stop_workers() {
  echo "[INFO] Stopping workers (best-effort, even without pidfile)..."

  mapfile -t PIDS < <(collect_pids || true)
  if [[ "${#PIDS[@]}" -eq 0 ]]; then
    echo "[INFO] No worker PIDs found."
  else
    echo "[INFO] Sending TERM to: ${PIDS[*]}"
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    sleep 2
    # Force kill any stragglers
    mapfile -t PIDS2 < <(collect_pids || true)
    if [[ "${#PIDS2[@]}" -gt 0 ]]; then
      echo "[INFO] Sending KILL to: ${PIDS2[*]}"
      for pid in "${PIDS2[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
    fi
  fi

#   rm -f "$PID_FILE"
  echo "[OK] Workers stopped. If .nfs* remain, wait a few seconds and try 'rm -f logs/.nfs*'."
}

# ---- Status ----
status_workers() {
  if [[ -f "$PID_FILE" ]]; then
    echo "[INFO] PIDs in $PID_FILE:"
    nl -ba "$PID_FILE"
  else
    echo "[INFO] No pidfile at $PID_FILE."
  fi

  echo "[INFO] Processes matching $WORKER_DIR/worker.py:"
  pgrep -af "$WORKER_DIR/worker.py" || echo "  (none)"

  echo "[INFO] Listeners on worker/sink ports:"
  for p in $(seq "$WORKER_PORT_START" $((WORKER_PORT_START + NUM_WORKERS - 1))) "$SINK_PORT"; do
    lsof -iTCP:"$p" -sTCP:LISTEN 2>/dev/null || true
  done
}

# ---- Purge logs (after stop) ----
purge_logs() {
  echo "[INFO] Purging logs/* (after ensuring no processes hold files)..."
  # One more sweep for any file holders
  if command -v lsof >/dev/null 2>&1; then
    lsof +D "$LOG_DIR" 2>/dev/null | awk 'NR>1{print $2}' | sort -u | xargs -r kill
    sleep 2
    lsof +D "$LOG_DIR" 2>/dev/null | awk 'NR>1{print $2}' | sort -u | xargs -r kill -9
  fi
  rm -f "$LOG_DIR"/worker_*.out "$LOG_DIR"/worker_sink_*.out "$LOG_DIR"/.nfs*
  echo "[OK] logs/ cleaned."
}

usage() {
  cat <<EOF
Usage: $(basename "$0") start|stop|restart|status|purge-logs

Environment overrides:
  NUM_WORKERS        (default: 16)
  WORKER_PORT_START  (default: 50051)
  SINK_PORT          (default: 50100)
  CASCADE            (default: multi)
  PYTHON_BIN         (default: python)
  IP_ADDRESS         (default: auto from 'hostname -I')

Examples:
  ./workers.sh start
  ./workers.sh status
  ./workers.sh stop
  ./workers.sh purge-logs
  NUM_WORKERS=12 WORKER_PORT_START=50051 ./workers.sh restart
EOF
}

cmd="${1:-}"
case "$cmd" in
  start)   start_workers ;;
  stop)    stop_workers ;;
  restart) stop_workers; start_workers ;;
  status)  status_workers ;;
  purge-logs) purge_logs ;;
  *) usage; exit 1 ;;
esac