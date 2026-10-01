# GPU dispatcher — keeps N GPUs continuously busy via a FIFO semaphore.
#
# Usage:
#     source common/dispatch.sh
#     dispatch_init 4                       # 4 in-flight slots
#     emit_job <name> <<'EOF'               # write job script via heredoc
#         set -euo pipefail
#         python ...                        # ${VARS} expanded at submit time
#     EOF
#     ...more emit_job calls...
#     dispatch_wait                         # block until all done
#
# Each in-flight job runs with CUDA_VISIBLE_DEVICES set to a free GPU id.
# When it exits, its GPU id is returned to the pool and the next queued job
# starts immediately. With N=4 every GPU stays hot until the queue drains.
#
# Multi-node note: this dispatcher targets one host with N visible GPUs (the
# common A100 server). For 4 separate nodes use slurm/sbatch or ssh-fan-out
# instead — replace `bash -c "$cmd"` with `ssh node$gpu bash -c "$cmd"`
# below and adjust paths. The job-script files are self-contained so they
# can be rsynced to other hosts unchanged.

DISPATCH_INITIALIZED="${DISPATCH_INITIALIZED:-0}"
DISPATCH_PIDS=()
DISPATCH_NAMES=()

dispatch_init() {
    if [[ "$DISPATCH_INITIALIZED" == "1" ]]; then
        return 0
    fi

    # GPU id pool. Two ways to specify:
    #   GPU_IDS="4,5,6,7" bash run_all.sh all       (explicit list, any ids)
    #   NUM_GPUS=4        bash run_all.sh all       (uses ids 0..N-1)
    #   bash run_all.sh all                          (defaults to 0..3)
    local -a gpu_ids
    if [[ -n "${GPU_IDS:-}" ]]; then
        IFS=',' read -ra gpu_ids <<< "$GPU_IDS"
    else
        local n="${1:-${NUM_GPUS:-${DISPATCH_GPUS:-4}}}"
        gpu_ids=()
        local i
        for ((i = 0; i < n; i++)); do gpu_ids+=("$i"); done
    fi
    DISPATCH_GPUS=${#gpu_ids[@]}
    DISPATCH_SEQ_FAILS=()

    # Sequential mode: run jobs one-at-a-time in the foreground, no FIFO
    # semaphore. Auto-enabled for a single GPU (where the parallel dispatcher
    # gives no benefit and its FIFO race caused a single-slot stall), or forced
    # via DISPATCH_SEQUENTIAL=1. Parallel (FIFO) mode is used for >1 GPU.
    if [[ "${DISPATCH_SEQUENTIAL:-0}" == "1" || "$DISPATCH_GPUS" -le 1 ]]; then
        DISPATCH_MODE="sequential"
        DISPATCH_SEQ_GPU="${gpu_ids[0]}"
    else
        # Robust slot-based parallel launcher (replaces the old FIFO semaphore,
        # which intermittently stalled and leaked EMPTY GPU tokens → jobs started
        # with no CUDA_VISIBLE_DEVICES). Each slot i is pinned to gpu_ids[i]; a
        # job runs there until it exits, then the slot is reused. No shared FD.
        DISPATCH_MODE="parallel"
        DISPATCH_GPU_IDS=("${gpu_ids[@]}")
        DISPATCH_SLOT_PID=()
        DISPATCH_SLOT_NAME=()
        DISPATCH_PENDING=()
        local i
        for i in "${!gpu_ids[@]}"; do DISPATCH_SLOT_PID[$i]=""; done
    fi

    DISPATCH_LOG_DIR="${OUT_ROOT}/_logs/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$DISPATCH_LOG_DIR/jobs"
    DISPATCH_INITIALIZED=1
    echo "[dispatch] initialized: mode=$DISPATCH_MODE, $DISPATCH_GPUS slot(s), GPU ids: ${gpu_ids[*]}"
    echo "[dispatch] logs in $DISPATCH_LOG_DIR"
}

# emit_row_job <name> <row_out>
# Wraps emit_job with auto-injected verifier watchdog. Use this for any
# generation job that produces images into <row_out>; the watchdog writes
# progress.json with per-checkpoint pixel stats and a DIRECTION verdict.
emit_row_job() {
    local name="$1" row="$2"
    local body
    body="$(cat)"
    emit_job "$name" <<EOF
ROW_OUT="$row"
NUM_SAMPLES_TOTAL="${NUM_SAMPLES:-1000}"
COCO_CAPTIONS="${COCO_CAPTIONS:-}"
COMMON_DIR="$COMMON_DIR"
source "$COMMON_DIR/verifier_wrap.sh"
$body
EOF
}

# emit_job <name>          # body read from stdin (heredoc), variables expanded.
# Writes a self-contained bash script to $DISPATCH_LOG_DIR/jobs/<name>.sh,
# then submits it through the GPU semaphore.
emit_job() {
    local name="$1"
    local script="$DISPATCH_LOG_DIR/jobs/${name}.sh"
    {
        echo "#!/usr/bin/env bash"
        echo "set -euo pipefail"
        # Activate conda + load shared paths inside the job's own subshell, so
        # the job works even if the launcher process didn't activate anything.
        echo "source \"$COMMON_DIR/paths.sh\""
        cat
    } > "$script"
    chmod +x "$script"
    _dispatch_submit "$name" "$script"
}

_dispatch_submit() {
    local name="$1" script="$2"
    local log="$DISPATCH_LOG_DIR/${name}.log"

    # ── Sequential mode: run the job in the foreground on the single GPU. ──
    if [[ "${DISPATCH_MODE:-parallel}" == "sequential" ]]; then
        echo "[dispatch] run   $name → GPU $DISPATCH_SEQ_GPU (sequential) @ $(date +%H:%M:%S)" | tee -a "$log"
        local rc=0
        CUDA_VISIBLE_DEVICES="$DISPATCH_SEQ_GPU" \
        PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
            bash "$script" >>"$log" 2>&1 || rc=$?
        if (( rc == 0 )); then
            echo "[dispatch] end   $name rc=0 @ $(date +%H:%M:%S)" | tee -a "$log"
        else
            echo "[dispatch] end   $name rc=$rc (see $log) @ $(date +%H:%M:%S)" | tee -a "$log"
            DISPATCH_SEQ_FAILS+=("$name")
        fi
        return 0
    fi

    # ── Parallel mode: queue the job, then fill any free GPU slots. ──
    DISPATCH_PENDING+=("${name}|${script}")
    _dispatch_fill_slots
}

# Launch pending jobs into any free slots (each slot pinned to gpu_ids[i]).
_dispatch_fill_slots() {
    local i entry nm sc gpu log
    for i in "${!DISPATCH_GPU_IDS[@]}"; do
        (( ${#DISPATCH_PENDING[@]} == 0 )) && break
        [[ -n "${DISPATCH_SLOT_PID[$i]:-}" ]] && continue   # slot busy
        entry="${DISPATCH_PENDING[0]}"
        DISPATCH_PENDING=("${DISPATCH_PENDING[@]:1}")
        nm="${entry%%|*}"; sc="${entry#*|}"; gpu="${DISPATCH_GPU_IDS[$i]}"
        log="$DISPATCH_LOG_DIR/${nm}.log"
        echo "[dispatch] start $nm → GPU $gpu (slot $i) @ $(date +%H:%M:%S)" | tee -a "$log"
        ( CUDA_VISIBLE_DEVICES="$gpu" \
          PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
          bash "$sc" >>"$log" 2>&1 ) &
        DISPATCH_SLOT_PID[$i]=$!
        DISPATCH_SLOT_NAME[$i]="$nm"
    done
}

dispatch_wait() {
    # Sequential mode already ran every job inline; just report + reset.
    if [[ "${DISPATCH_MODE:-parallel}" == "sequential" ]]; then
        DISPATCH_INITIALIZED=0
        if (( ${#DISPATCH_SEQ_FAILS[@]} > 0 )); then
            echo "[dispatch] FAILED jobs: ${DISPATCH_SEQ_FAILS[*]}" >&2
            echo "[dispatch] inspect logs in $DISPATCH_LOG_DIR/" >&2
            return 1
        fi
        echo "[dispatch] all jobs completed (sequential)"
        return 0
    fi

    # ── Parallel mode: drain slots — wait for any job, reap it, free its slot,
    # backfill from the pending queue, until nothing is running or queued. ──
    local fails=() i pid rc
    while :; do
        local any_busy=0
        for i in "${!DISPATCH_GPU_IDS[@]}"; do
            [[ -n "${DISPATCH_SLOT_PID[$i]:-}" ]] && any_busy=1
        done
        if (( any_busy == 0 && ${#DISPATCH_PENDING[@]} == 0 )); then break; fi

        wait -n 2>/dev/null || true          # block until *some* child exits
        # Reap every slot whose pid is no longer alive.
        for i in "${!DISPATCH_GPU_IDS[@]}"; do
            pid="${DISPATCH_SLOT_PID[$i]:-}"
            [[ -z "$pid" ]] && continue
            if ! kill -0 "$pid" 2>/dev/null; then
                rc=0; wait "$pid" 2>/dev/null || rc=$?
                local nm="${DISPATCH_SLOT_NAME[$i]}"
                echo "[dispatch] end   $nm → GPU ${DISPATCH_GPU_IDS[$i]} rc=$rc @ $(date +%H:%M:%S)" \
                    | tee -a "$DISPATCH_LOG_DIR/${nm}.log"
                (( rc != 0 )) && fails+=("$nm")
                DISPATCH_SLOT_PID[$i]=""
            fi
        done
        _dispatch_fill_slots                 # backfill freed slots
    done

    DISPATCH_INITIALIZED=0
    if (( ${#fails[@]} > 0 )); then
        echo "[dispatch] FAILED jobs: ${fails[*]}" >&2
        echo "[dispatch] inspect logs in $DISPATCH_LOG_DIR/" >&2
        return 1
    fi
    echo "[dispatch] all jobs completed"
}

# Skip-if-already-scored helper. Returns 0 if row should be submitted, 1 if it
# already has a result.json (idempotent reruns).
needs_run() {
    local row_out="$1"
    local base; base="$(basename "$row_out")"
    # Optional allowlist: ABL_ONLY="arm1 arm2 ..." restricts a run to just those
    # arm basenames (used for targeted reruns of failed/fixed arms without
    # re-touching arms that are already running or scored in a parallel run).
    if [[ -n "${ABL_ONLY:-}" && " $ABL_ONLY " != *" $base "* ]]; then
        echo "[skip] $base (not in ABL_ONLY)"
        return 1
    fi
    if [[ -f "$row_out/result.json" ]]; then
        echo "[skip] $base (result.json exists)"
        return 1
    fi
    mkdir -p "$row_out"
    return 0
}
