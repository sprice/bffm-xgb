#!/usr/bin/env bash
set -euo pipefail

TUNE_N_JOBS="${TUNE_N_JOBS:-}"
TRAIN_N_JOBS="${TRAIN_N_JOBS:-}"
PARALLEL_TRIALS="${PARALLEL_TRIALS:-}"
PARALLEL_DOMAINS="${PARALLEL_DOMAINS:-}"
TRAIN_PARALLEL="${TRAIN_PARALLEL:-}"

# ---------------------------------------------------------------------------
# Stage range flags
# ---------------------------------------------------------------------------
START_STAGE=""
END_STAGE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-stage) START_STAGE="$2"; shift 2 ;;
        --end-stage)   END_STAGE="$2";   shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

STAGE_ORDER=(
    download load norms norms-check prepare correlations
    tune train research-eval export-all notes figures
)

# Validate stage names
for _s in "$START_STAGE" "$END_STAGE"; do
    if [[ -n "$_s" ]]; then
        _found=0
        for _valid in "${STAGE_ORDER[@]}"; do
            [[ "$_s" == "$_valid" ]] && _found=1 && break
        done
        if [[ "$_found" -eq 0 ]]; then
            echo "ERROR: Unknown stage '$_s'. Valid stages: ${STAGE_ORDER[*]}" >&2
            exit 1
        fi
    fi
done

TIMING_LOG="pipeline-timing.log"
PIPELINE_LOG="pipeline.log"

# ---------------------------------------------------------------------------
# Track the current step so the EXIT trap can record failures
# ---------------------------------------------------------------------------
_CURRENT_LABEL=""
_STEP_START=0

cleanup() {
    local exit_code=$?
    if [ "$exit_code" -ne 0 ] && [ -n "$_CURRENT_LABEL" ]; then
        local elapsed=$(( $(date +%s) - _STEP_START ))
        echo "$_CURRENT_LABEL | FAILED (exit $exit_code) | $(fmt_duration "$elapsed")" >> "$TIMING_LOG"
        echo "--- $_CURRENT_LABEL FAILED (exit $exit_code, $(fmt_duration "$elapsed"))" | tee -a "$PIPELINE_LOG"
    fi
    if [ -n "${PIPELINE_START:-}" ]; then
        local total=$(( $(date +%s) - PIPELINE_START ))
        if [ "$exit_code" -ne 0 ]; then
            echo "total (FAILED) | $(fmt_duration "$total")" >> "$TIMING_LOG"
            echo "=== Pipeline FAILED after $(fmt_duration "$total")" | tee -a "$PIPELINE_LOG"
        fi
    fi
}
trap cleanup EXIT

fmt_duration() {
    local secs="$1"
    printf "%02dh%02dm%02ds" $((secs / 3600)) $(((secs % 3600) / 60)) $((secs % 60))
}

_STAGE_ACTIVE=0
[[ -z "$START_STAGE" ]] && _STAGE_ACTIVE=1

run_step() {
    local label="$1"
    shift

    # Skip stages before --start-stage
    if [[ "$_STAGE_ACTIVE" -eq 0 ]]; then
        if [[ "$label" == "$START_STAGE" ]]; then
            _STAGE_ACTIVE=1
        else
            echo "--- $label SKIPPED (before start stage '$START_STAGE')" | tee -a "$PIPELINE_LOG"
            return 0
        fi
    fi

    local start_time
    start_time=$(date -u +"%Y-%m-%d %H:%M:%S UTC")

    _CURRENT_LABEL="$label"
    _STEP_START=$(date +%s)

    echo "--- $label (started $start_time)" | tee -a "$PIPELINE_LOG"
    "$@" 2>&1 | tee -a "$PIPELINE_LOG"

    local elapsed=$(( $(date +%s) - _STEP_START ))
    echo "$label | started $start_time | $(fmt_duration "$elapsed")" >> "$TIMING_LOG"
    echo "--- $label done ($(fmt_duration "$elapsed"))" | tee -a "$PIPELINE_LOG"

    _CURRENT_LABEL=""

    # Stop after --end-stage
    if [[ -n "$END_STAGE" && "$label" == "$END_STAGE" ]]; then
        echo "--- Stopping after end stage '$END_STAGE'" | tee -a "$PIPELINE_LOG"
        local total=$(( $(date +%s) - PIPELINE_START ))
        echo "total | $(fmt_duration "$total")" >> "$TIMING_LOG"
        echo "=== Pipeline complete ($(fmt_duration "$total"))" | tee -a "$PIPELINE_LOG"
        exit 0
    fi
}

# Reset logs
> "$TIMING_LOG"
> "$PIPELINE_LOG"
PIPELINE_START=$(date +%s)

run_step "download"     make download
run_step "load"         make load
run_step "norms"        make norms
run_step "norms-check"  make norms-check
run_step "prepare"      make prepare
run_step "correlations" make correlations
run_step "tune"         make tune N_JOBS="$TUNE_N_JOBS" PARALLEL_TRIALS="$PARALLEL_TRIALS"
run_step "train"        make train N_JOBS="$TRAIN_N_JOBS" PARALLEL_DOMAINS="$PARALLEL_DOMAINS" TRAIN_PARALLEL="$TRAIN_PARALLEL"
run_step "research-eval" make research-eval
run_step "export-all"   make export-all
run_step "notes"        make notes
run_step "figures"      make figures

TOTAL=$(( $(date +%s) - PIPELINE_START ))
echo "total | $(fmt_duration "$TOTAL")" >> "$TIMING_LOG"
echo "=== Pipeline complete ($(fmt_duration "$TOTAL"))" | tee -a "$PIPELINE_LOG"
