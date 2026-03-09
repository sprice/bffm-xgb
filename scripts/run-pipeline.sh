#!/usr/bin/env bash
set -euo pipefail

TUNE_N_JOBS="${TUNE_N_JOBS:-}"
TRAIN_N_JOBS="${TRAIN_N_JOBS:-}"
PARALLEL_TRIALS="${PARALLEL_TRIALS:-}"
PARALLEL_DOMAINS="${PARALLEL_DOMAINS:-}"
CV_PARALLEL_FOLDS="${CV_PARALLEL_FOLDS:-}"
TRAIN_PARALLEL="${TRAIN_PARALLEL:-}"
RESEARCH_EVAL_PARALLEL="${RESEARCH_EVAL_PARALLEL:-}"
GPU="${GPU:-}"
REFERENCE_ONLY="${REFERENCE_ONLY:-}"

# ---------------------------------------------------------------------------
# Stage range flags
# ---------------------------------------------------------------------------
START_STAGE=""
END_STAGE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-stage) START_STAGE="$2"; shift 2 ;;
        --end-stage)   END_STAGE="$2";   shift 2 ;;
        --gpu)         GPU=1;            shift   ;;
        --reference-only) REFERENCE_ONLY=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

canonicalize_stage() {
    case "$1" in
        export-all|export-reference|export) echo "export" ;;
        research-eval-reference|research-eval) echo "research-eval" ;;
        train-1|train) echo "train" ;;
        prepare-default|prepare) echo "prepare" ;;
        correlations-default|correlations) echo "correlations" ;;
        *) echo "$1" ;;
    esac
}

START_STAGE="$(canonicalize_stage "$START_STAGE")"
END_STAGE="$(canonicalize_stage "$END_STAGE")"

STAGE_ORDER=(
    download load norms norms-check prepare correlations
    tune train research-eval export notes figures
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
CHECKPOINT_DIR=".pipeline-checkpoints"
CHECKPOINT_STAGES=(norms prepare correlations tune train research-eval figures)

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

is_checkpoint_stage() {
    local stage="$1"
    local checkpoint
    for checkpoint in "${CHECKPOINT_STAGES[@]}"; do
        if [[ "$stage" == "$checkpoint" ]]; then
            return 0
        fi
    done
    return 1
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
    if is_checkpoint_stage "$label"; then
        : > "$CHECKPOINT_DIR/$label.done"
    fi

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

skip_step() {
    local label="$1"
    local reason="$2"

    if [[ "$_STAGE_ACTIVE" -eq 0 ]]; then
        if [[ "$label" == "$START_STAGE" ]]; then
            _STAGE_ACTIVE=1
        else
            echo "--- $label SKIPPED (before start stage '$START_STAGE')" | tee -a "$PIPELINE_LOG"
            return 0
        fi
    fi

    echo "--- $label SKIPPED ($reason)" | tee -a "$PIPELINE_LOG"
    echo "$label | SKIPPED | $reason" >> "$TIMING_LOG"

    if [[ -n "$END_STAGE" && "$label" == "$END_STAGE" ]]; then
        echo "--- Stopping after end stage '$END_STAGE'" | tee -a "$PIPELINE_LOG"
        local total=$(( $(date +%s) - PIPELINE_START ))
        echo "total | $(fmt_duration "$total")" >> "$TIMING_LOG"
        echo "=== Pipeline complete ($(fmt_duration "$total"))" | tee -a "$PIPELINE_LOG"
        exit 0
    fi
}

_GPU_FLAG=""
if [[ -n "$GPU" ]]; then
    _GPU_FLAG="GPU=1"
fi

# Reset logs
> "$TIMING_LOG"
> "$PIPELINE_LOG"
rm -rf "$CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR"
PIPELINE_START=$(date +%s)

run_step "download" make download
run_step "load" make load
run_step "norms" make norms
run_step "norms-check" make norms-check
if [[ -n "$REFERENCE_ONLY" ]]; then
    run_step "prepare" make prepare-default
    run_step "correlations" make correlations-default
else
    run_step "prepare" make prepare
    run_step "correlations" make correlations
fi

tune_cmd=(make tune)
[[ -n "$TUNE_N_JOBS" ]] && tune_cmd+=("N_JOBS=$TUNE_N_JOBS")
[[ -n "$PARALLEL_TRIALS" ]] && tune_cmd+=("PARALLEL_TRIALS=$PARALLEL_TRIALS")
[[ -n "$_GPU_FLAG" ]] && tune_cmd+=("$_GPU_FLAG")
run_step "tune" "${tune_cmd[@]}"

train_cmd=(make train)
[[ -n "$REFERENCE_ONLY" ]] && train_cmd+=("1")
[[ -n "$TRAIN_N_JOBS" ]] && train_cmd+=("N_JOBS=$TRAIN_N_JOBS")
[[ -n "$PARALLEL_DOMAINS" ]] && train_cmd+=("PARALLEL_DOMAINS=$PARALLEL_DOMAINS")
[[ -n "$CV_PARALLEL_FOLDS" ]] && train_cmd+=("CV_PARALLEL_FOLDS=$CV_PARALLEL_FOLDS")
[[ -n "$TRAIN_PARALLEL" ]] && train_cmd+=("TRAIN_PARALLEL=$TRAIN_PARALLEL")
[[ -n "$_GPU_FLAG" ]] && train_cmd+=("$_GPU_FLAG")
run_step "train" "${train_cmd[@]}"

if [[ -n "$REFERENCE_ONLY" ]]; then
    research_eval_cmd=(make research-eval-reference)
else
    research_eval_cmd=(make research-eval)
fi
[[ -n "$RESEARCH_EVAL_PARALLEL" ]] && research_eval_cmd+=("RESEARCH_EVAL_PARALLEL=$RESEARCH_EVAL_PARALLEL")
run_step "research-eval" "${research_eval_cmd[@]}"

if [[ -n "$REFERENCE_ONLY" ]]; then
    run_step "export" make export-reference export-repo-readme
    skip_step "notes" "reference-only mode requires all four variants"
else
    run_step "export" make export-all
    run_step "notes" make notes
fi
run_step "figures" make figures

TOTAL=$(( $(date +%s) - PIPELINE_START ))
echo "total | $(fmt_duration "$TOTAL")" >> "$TIMING_LOG"
echo "=== Pipeline complete ($(fmt_duration "$TOTAL"))" | tee -a "$PIPELINE_LOG"
