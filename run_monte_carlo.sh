#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_SCRIPT="${SCRIPT_DIR}/monte_carlo_sim.py"

# ── Defaults ──────────────────────────────────────────────────────────────────
N_ENVS=1
POSITION_RANGE=""
ANGLE_RANGE=""
INCLINATION_RANGE=""
RADIUS_RANGE=""
MIN_DISK_MB=1024
DEVICE="cpu"
SEARCH_PATTERN="sinusoidal"
EXTRA_ARGS=""

TARGET_OBJECTS=("cylinder" "h_bar")

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run monte_carlo_sim.py for both target objects (cylinder, h_bar) across all
specified sweep types, checking disk space between each invocation.

At least one range must be provided.

Options:
  -n, --n-envs N              Number of parallel trials per sweep point (default: $N_ENVS)
  -p, --position-range "MIN MAX STEP"
                              Position offset range (X), e.g. "-0.2 0.2 0.05"
  -a, --angle-range "MIN MAX STEP"
                              Yaw angle range (deg), e.g. "-30 30 10"
  -i, --inclination-range "MIN MAX STEP"
                              Inclination range (deg), e.g. "-15 15 5"
  -r, --radius-range "MIN MAX"
                              Radius range (step fixed at 0.005), e.g. "0.02 0.06"
  -d, --device DEVICE         Compute device (default: $DEVICE)
  -s, --search-pattern PAT    Search pattern: linear|sinusoidal|square|spiral (default: $SEARCH_PATTERN)
  -m, --min-disk-mb MB        Minimum free disk space in MB before aborting (default: $MIN_DISK_MB)
  -e, --extra-args "ARGS"     Extra arguments forwarded to monte_carlo_sim.py
  -h, --help                  Show this help message

Examples:
  $(basename "$0") -n 5 -p "-0.2 0.2 0.05" -a "-30 30 10"
  $(basename "$0") -n 10 -p "-0.1 0.1 0.02" -a "-45 45 15" -i "-10 10 5" -r "0.02 0.06"
  $(basename "$0") -n 3 -r "0.01 0.08" --device cuda:0 --min-disk-mb 2048
EOF
    exit 0
}

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--n-envs)            N_ENVS="$2";            shift 2 ;;
        -p|--position-range)    POSITION_RANGE="$2";    shift 2 ;;
        -a|--angle-range)       ANGLE_RANGE="$2";       shift 2 ;;
        -i|--inclination-range) INCLINATION_RANGE="$2"; shift 2 ;;
        -r|--radius-range)      RADIUS_RANGE="$2";      shift 2 ;;
        -d|--device)            DEVICE="$2";            shift 2 ;;
        -s|--search-pattern)    SEARCH_PATTERN="$2";    shift 2 ;;
        -m|--min-disk-mb)       MIN_DISK_MB="$2";       shift 2 ;;
        -e|--extra-args)        EXTRA_ARGS="$2";        shift 2 ;;
        -h|--help)              usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ -z "$POSITION_RANGE" && -z "$ANGLE_RANGE" && -z "$INCLINATION_RANGE" && -z "$RADIUS_RANGE" ]]; then
    echo "ERROR: At least one range (--position-range, --angle-range, --inclination-range, --radius-range) must be specified."
    exit 1
fi

if [[ ! -f "$SIM_SCRIPT" ]]; then
    echo "ERROR: Cannot find simulation script at $SIM_SCRIPT"
    exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
check_disk_space() {
    local available_mb
    available_mb=$(df --output=avail -BM "$(dirname "$SIM_SCRIPT")" | tail -1 | tr -d ' M')
    if (( available_mb < MIN_DISK_MB )); then
        echo "ABORT: Only ${available_mb} MB free disk space (minimum: ${MIN_DISK_MB} MB)."
        return 1
    fi
    echo "  Disk OK: ${available_mb} MB available (minimum: ${MIN_DISK_MB} MB)"
    return 0
}

RUN_COUNT=0
FAIL_COUNT=0

run_sim() {
    local target_object="$1"
    local range_flag="$2"
    local range_value="$3"

    echo "────────────────────────────────────────────────────────────"
    echo "  Target: $target_object | Sweep: $range_flag \"$range_value\""
    echo "────────────────────────────────────────────────────────────"

    check_disk_space || exit 1

    RUN_COUNT=$((RUN_COUNT + 1))

    set +e
    python "$SIM_SCRIPT" \
        --target_object "$target_object" \
        "$range_flag" "$range_value" \
        --n_envs "$N_ENVS" \
        --device "$DEVICE" \
        --search_pattern "$SEARCH_PATTERN" \
        --save \
        $EXTRA_ARGS
    local rc=$?
    set -e

    if [[ $rc -ne 0 ]]; then
        echo "WARNING: Simulation exited with code $rc (target=$target_object, $range_flag=\"$range_value\")"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    echo ""
}

# ── Build sweep list ──────────────────────────────────────────────────────────
declare -a SWEEP_FLAGS=()
declare -a SWEEP_VALUES=()

if [[ -n "$POSITION_RANGE" ]]; then
    SWEEP_FLAGS+=("--position_range")
    SWEEP_VALUES+=("$POSITION_RANGE")
fi
if [[ -n "$ANGLE_RANGE" ]]; then
    SWEEP_FLAGS+=("--angle_range")
    SWEEP_VALUES+=("$ANGLE_RANGE")
fi
if [[ -n "$INCLINATION_RANGE" ]]; then
    SWEEP_FLAGS+=("--inclination_range")
    SWEEP_VALUES+=("$INCLINATION_RANGE")
fi
if [[ -n "$RADIUS_RANGE" ]]; then
    SWEEP_FLAGS+=("--radius_range")
    SWEEP_VALUES+=("$RADIUS_RANGE")
fi

# ── Main loop ─────────────────────────────────────────────────────────────────
TOTAL=$((${#TARGET_OBJECTS[@]} * ${#SWEEP_FLAGS[@]}))
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Monte Carlo Sweep                                         ║"
echo "║  Objects : ${TARGET_OBJECTS[*]}"
echo "║  Sweeps  : ${#SWEEP_FLAGS[@]} range(s) × ${#TARGET_OBJECTS[@]} objects = $TOTAL run(s)"
echo "║  Envs    : $N_ENVS per sweep point"
echo "║  Device  : $DEVICE"
echo "║  Pattern : $SEARCH_PATTERN"
echo "║  Min disk: ${MIN_DISK_MB} MB"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

START_TIME=$SECONDS

for obj in "${TARGET_OBJECTS[@]}"; do
    for idx in "${!SWEEP_FLAGS[@]}"; do
        run_sim "$obj" "${SWEEP_FLAGS[$idx]}" "${SWEEP_VALUES[$idx]}"
    done
done

ELAPSED=$(( SECONDS - START_TIME ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Done.  $RUN_COUNT run(s) completed, $FAIL_COUNT failure(s)"
echo "║  Elapsed: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "╚══════════════════════════════════════════════════════════════╝"

if [[ $FAIL_COUNT -gt 0 ]]; then
    exit 1
fi
