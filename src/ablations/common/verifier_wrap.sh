# Sourced from inside a generated job script. Spawns the verifier watchdog
# in the background and registers an EXIT trap that:
#   1. runs one synchronous final checkpoint (so the 100% milestone is logged
#      even when generation finishes between polls), and
#   2. kills the watchdog cleanly.
#
# Required env vars (set by emit_row_job):
#   ROW_OUT             — row output directory
#   NUM_SAMPLES_TOTAL   — expected final image count
#   COMMON_DIR          — path to src/ablations/common
#
# Optional env vars:
#   VERIFIER_POLL       — seconds between polls (default 60)
#   VERIFIER_STALL_MIN  — minutes of zero progress before STALLED (default 20)
#   VERIFIER_DISABLE=1  — opt out entirely

if [[ "${VERIFIER_DISABLE:-0}" == "1" || -z "${ROW_OUT:-}" ]]; then
    return 0
fi

mkdir -p "$ROW_OUT"
python -u "$COMMON_DIR/verify_progress.py" \
    --row "$ROW_OUT" \
    --total "${NUM_SAMPLES_TOTAL:-1000}" \
    --poll-seconds "${VERIFIER_POLL:-60}" \
    --stall-minutes "${VERIFIER_STALL_MIN:-20}" &
__VERIFIER_PID=$!

_verifier_finalize() {
    # Synchronous final checkpoint — captures whatever was generated even if
    # the polling watchdog hadn't seen it yet.
    python -u "$COMMON_DIR/verify_progress.py" \
        --row "$ROW_OUT" \
        --total "${NUM_SAMPLES_TOTAL:-1000}" \
        --once 2>&1 || true
    if [[ -n "${__VERIFIER_PID:-}" ]]; then
        kill "$__VERIFIER_PID" 2>/dev/null || true
        wait "$__VERIFIER_PID" 2>/dev/null || true
    fi
}
trap _verifier_finalize EXIT
echo "[verifier_wrap] watchdog pid=$__VERIFIER_PID for $ROW_OUT"
