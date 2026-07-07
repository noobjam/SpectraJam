#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
CONFIG=plain_tessera_incremental/config.yaml
LOG_DIR="$REPO/logs"
PREFLIGHT_LOG="$LOG_DIR/plain_tessera_preflight_v2.log"
V2_LOG="$LOG_DIR/plain_tessera_incremental_v2.log"
V2_PID_FILE="$LOG_DIR/plain_tessera_incremental_v2.pid"
CUTOVER_LOCK="$LOG_DIR/plain_tessera_cutover.lock"

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4))); then
  echo "ABORT: exact argv parsing requires Bash 4.4 or newer" >&2
  exit 1
fi
if test "$(type -t mapfile)" != "builtin"; then
  echo "ABORT: this Bash build lacks the required mapfile builtin" >&2
  exit 1
fi

process_start_time() {
  local pid=$1
  local stat_line
  local stat_fields

  IFS= read -r stat_line < "/proc/$pid/stat" || return 1
  stat_fields=${stat_line##*) }
  set -- $stat_fields
  test "$#" -ge 20 || return 1
  printf '%s\n' "${20}"
}

process_state() {
  local pid=$1
  local stat_line
  local stat_fields

  IFS= read -r stat_line < "/proc/$pid/stat" || return 1
  stat_fields=${stat_line##*) }
  set -- $stat_fields
  test "$#" -ge 1 || return 1
  printf '%s\n' "$1"
}

is_plain_tessera_job() {
  local pid=$1
  local proc_dir="/proc/$pid"
  local process_cwd
  local process_exe
  local -a argv=()

  test -r "$proc_dir/cmdline" || return 1
  process_cwd=$(readlink -f "$proc_dir/cwd" 2>/dev/null) || return 1
  test "$process_cwd" = "$REPO" || return 1
  process_exe=$(basename "$(readlink -f "$proc_dir/exe" 2>/dev/null)") || return 1
  [[ "$process_exe" =~ ^python([0-9]+(\.[0-9]+)*)?$ ]] || return 1

  mapfile -d '' -t argv < "$proc_dir/cmdline" || return 1
  test "${#argv[@]}" -eq 6 || return 1
  test "${argv[1]}" = "-u" || return 1
  test "${argv[2]}" = "-m" || return 1
  test "${argv[3]}" = "plain_tessera_incremental" || return 1
  test "${argv[4]}" = "--config" || return 1
  test "${argv[5]}" = "$CONFIG" || return 1
}

find_existing_job() {
  local proc_dir
  local pid
  local start_time
  local matched_pid=
  local matched_start=

  for proc_dir in /proc/[0-9]*; do
    pid=${proc_dir##*/}
    if is_plain_tessera_job "$pid"; then
      start_time=$(process_start_time "$pid") || continue
      if test -n "$matched_pid"; then
        echo "ABORT: multiple matching plain-TESSERA jobs: $matched_pid and $pid" >&2
        return 1
      fi
      matched_pid=$pid
      matched_start=$start_time
    fi
  done

  printf '%s %s\n' "$matched_pid" "$matched_start"
}

stop_existing_job() {
  local pid=$1
  local expected_start=$2
  local current_start
  local attempt

  test -n "$pid" || {
    echo "No existing matching plain-TESSERA process"
    return 0
  }

  echo "Stopping verified existing plain-TESSERA PID $pid"
  is_plain_tessera_job "$pid" || {
    echo "ABORT: PID $pid changed identity before shutdown" >&2
    return 1
  }
  current_start=$(process_start_time "$pid") || {
    echo "Existing PID $pid exited before shutdown"
    return 0
  }
  test "$current_start" = "$expected_start" || {
    echo "ABORT: PID $pid was reused before shutdown" >&2
    return 1
  }

  kill -TERM "$pid"
  for attempt in $(seq 1 30); do
    current_start=$(process_start_time "$pid" 2>/dev/null || true)
    test "$current_start" = "$expected_start" || return 0
    sleep 1
  done
  echo "ABORT: PID $pid is still alive after 30 seconds" >&2
  return 1
}

finish_v2_child() {
  local status

  set +e
  wait "$V2_PID"
  status=$?
  set -e
  rm -f "$V2_PID_FILE"
  if test "$status" -eq 0; then
    echo "v2 completed successfully during the startup check"
    echo "output: $OUTPUT_DIR"
    exit 0
  fi
  echo "ABORT: v2 exited with status $status; last log lines:" >&2
  tail -n 100 "$V2_LOG" >&2
  exit 1
}

test -d /proc || {
  echo "ABORT: this VM cutover script requires Linux /proc" >&2
  exit 1
}
test -f "$REPO/.venv/bin/activate" || {
  echo "ABORT: missing VM environment: $REPO/.venv" >&2
  exit 1
}

cd "$REPO"
mkdir -p "$LOG_DIR"
command -v flock >/dev/null 2>&1 || {
  echo "ABORT: safe cutover requires the Linux flock command" >&2
  exit 1
}
exec 9>"$CUTOVER_LOCK"
flock -n 9 || {
  echo "ABORT: another plain-TESSERA cutover is already running" >&2
  exit 1
}
source .venv/bin/activate
OUTPUT_DIR=$(
  python -c \
    'import sys; from plain_tessera_incremental.config import load_config; print(load_config(sys.argv[1]).output_dir)' \
    "$CONFIG"
)

echo "Running v2 preflight; the existing job remains untouched if this fails"
python -u -m plain_tessera_incremental \
  --config "$CONFIG" \
  --preflight-only \
  2>&1 | tee "$PREFLIGHT_LOG"

mkdir -p "$OUTPUT_DIR"
test -w "$OUTPUT_DIR" || {
  echo "ABORT: v2 output directory is not writable: $OUTPUT_DIR" >&2
  exit 1
}

read -r EXISTING_PID EXISTING_START < <(find_existing_job)
stop_existing_job "$EXISTING_PID" "$EXISTING_START"
read -r REMAINING_PID _ < <(find_existing_job)
test -z "$REMAINING_PID" || {
  echo "ABORT: matching PID $REMAINING_PID appeared before v2 launch" >&2
  exit 1
}

nohup env PYTHONUNBUFFERED=1 python -u -m plain_tessera_incremental \
  --config "$CONFIG" \
  > "$V2_LOG" 2>&1 < /dev/null 9>&- &
V2_PID=$!
printf '%s\n' "$V2_PID" | tee "$V2_PID_FILE"
V2_START=
for _ in $(seq 1 20); do
  V2_START=$(process_start_time "$V2_PID" 2>/dev/null || true)
  test -n "$V2_START" && break
  kill -0 "$V2_PID" 2>/dev/null || break
  sleep 0.05
done
test -n "$V2_START" || finish_v2_child

sleep 3
CURRENT_START=$(process_start_time "$V2_PID" 2>/dev/null || true)
CURRENT_STATE=$(process_state "$V2_PID" 2>/dev/null || true)
if ! kill -0 "$V2_PID" 2>/dev/null || test "$CURRENT_STATE" = "Z"; then
  finish_v2_child
fi
if test "$CURRENT_START" != "$V2_START"; then
  rm -f "$V2_PID_FILE"
  echo "ABORT: v2 PID was reused during startup" >&2
  exit 1
fi
if ! is_plain_tessera_job "$V2_PID"; then
  CURRENT_START=$(process_start_time "$V2_PID" 2>/dev/null || true)
  CURRENT_STATE=$(process_state "$V2_PID" 2>/dev/null || true)
  if ! kill -0 "$V2_PID" 2>/dev/null || test "$CURRENT_STATE" = "Z"; then
    finish_v2_child
  fi
  rm -f "$V2_PID_FILE"
  if test "$CURRENT_START" != "$V2_START"; then
    echo "ABORT: v2 PID was reused during identity verification" >&2
  else
    echo "ABORT: live v2 child has unexpected process identity; it was not signalled" >&2
  fi
  exit 1
fi

echo "v2 started as PID $V2_PID"
echo "log: $V2_LOG"
echo "output: $OUTPUT_DIR"
