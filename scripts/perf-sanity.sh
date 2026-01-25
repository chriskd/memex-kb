#!/usr/bin/env bash
set -euo pipefail

ROOT="${PERF_SANITY_ROOT:-}"
KEEP="${PERF_SANITY_KEEP:-0}"
SCOPE="${PERF_SANITY_SCOPE:-project}"
QUERY="${PERF_SANITY_QUERY:-memex}"
MODE="${PERF_SANITY_MODE:-hybrid}"
LIMIT="${PERF_SANITY_LIMIT:-5}"

if ! command -v mx >/dev/null 2>&1; then
  echo "Error: mx not found in PATH" >&2
  exit 1
fi

if [[ -z "$ROOT" ]]; then
  ROOT="$(mktemp -d -t memex-perf-XXXXXX)"
  CLEANUP=1
else
  mkdir -p "$ROOT"
  CLEANUP=0
fi

export MEMEX_INDEX_ROOT="$ROOT/indices"
PUBLISH_OUT="$ROOT/publish"
TIMINGS_CSV="$ROOT/timings.csv"

now_ms() {
  if date +%s%3N >/dev/null 2>&1; then
    date +%s%3N
  else
    python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
  fi
}

run_timed() {
  local label="$1"
  shift
  local start end elapsed
  start=$(now_ms)
  "$@" >/dev/null
  end=$(now_ms)
  elapsed=$((end - start))
  printf "%-10s %6sms\n" "$label" "$elapsed"
  echo "$label,$elapsed" >> "$TIMINGS_CSV"
}

trap 'if [[ "$KEEP" == "1" ]]; then echo "Perf artifacts kept at $ROOT"; else if [[ "$CLEANUP" == "1" ]]; then rm -rf "$ROOT"; fi; fi' EXIT

{
  echo "label,ms"
} > "$TIMINGS_CSV"

cat <<INFO
Perf sanity root: $ROOT
KB scope: $SCOPE
Search: query="$QUERY" mode="$MODE" limit=$LIMIT
Index root: $MEMEX_INDEX_ROOT
Publish output: $PUBLISH_OUT
INFO

run_timed "reindex" mx reindex --scope="$SCOPE"
run_timed "search" mx search "$QUERY" --mode="$MODE" --limit="$LIMIT" --scope="$SCOPE"
run_timed "publish" mx publish --scope="$SCOPE" -o "$PUBLISH_OUT"

echo "Timings written to $TIMINGS_CSV"
