#!/usr/bin/env bash
# stockGPT at a horizon of H trading days (stockai/gpt/features.py): from the close of day d, rank the stocks by their
# return over the next H days. Trains the walk-forward (one model per test year and seed) and its sign-flip null and
# writes their scores files; it grades nothing.
#   stockGPT walk-forward -> sign-flip null
#
#   scripts/gpt_horizon.sh --horizon H [--quick] [--steps LIST] [--runs DIR] [--force]
#
#   --horizon  trading days ahead: 21 (a month), 5 (a week); 1 is stockGPT v1's next-day target, run as in poc.sh.
#   --quick    test years 2022-2023, seed 0, the null on 2022: a smoke run (about 4 minutes at H = 1 on a free GPU).
#              Writes runs/gpt-quick-<H>d/.
#   (default)  test years 2010-2023, seeds 0 1 2 and their ensemble, the null (seed 0) on all 14 years. Writes
#              runs/gpt-<H>d/.
#   --steps    a comma list of: gpt,null (default: both, in that order).
#   --force    let a step replace a previous run in the runs directory.
#
# In RUNS: gpt/ and gpt_null/ (scores_seed<k>.npy, scores_ensemble.npy, train_log.jsonl with each year's validation
# metrics, run.json, checkpoints/), each step's full stderr log, and gpt_horizon_log.txt with the GPU's state and the
# steps' timings. Grade the scores like LightGBM's, for example
#   uv run python -m stockai.h30.evaluate --horizon H --scores gpt=RUNS/gpt/scores_ensemble.npy \
#     gpt_null=RUNS/gpt_null/scores_seed0.npy --out DIR
#
# Nothing here grades, trains on or reads the locked year (2024-06-28 .. 2025-06-27): stockai.gpt cuts it off before
# building anything, splits.py refuses it, and every scores file is checked to have no finite value from the development
# cutoff on.
set -euo pipefail

# Everything runs inside main(), so bash has parsed the whole file before the first step starts.
main() {
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

H=""
QUICK=0
STEPS="gpt,null"
RUNS=""
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --horizon) H="$2"; shift ;;
    --quick) QUICK=1 ;;
    --steps) STEPS="$2"; shift ;;
    --runs) RUNS="$2"; shift ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;  # the header above
    *) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
[[ "$H" =~ ^[1-9][0-9]*$ ]] || { echo "--horizon H (a positive number of trading days) is required" >&2; exit 2; }

if [[ $QUICK == 1 ]]; then
  YEARS="2022-2023"; SEEDS="0"; NULL_YEARS="2022"
  RUNS="${RUNS:-runs/gpt-quick-${H}d}"
else
  YEARS="2010-2023"; SEEDS="0 1 2"; NULL_YEARS="2010-2023"
  RUNS="${RUNS:-runs/gpt-${H}d}"
fi
DATA_DIR="${C2_DATA_DIR:-$HOME/.local/share/correlation2}"
EXPORT="$DATA_DIR/export/full"
mkdir -p "$RUNS"
LOG="$RUNS/gpt_horizon_log.txt"
echo "== scripts/gpt_horizon.sh --horizon $H $([[ $QUICK == 1 ]] && echo --quick) $(date -Is): years $YEARS, seeds $SEEDS, null $NULL_YEARS; runs $RUNS" | tee -a "$LOG"

want() { [[ ",$STEPS," == *",$1,"* ]]; }
T_ALL=$(date +%s)
step_start() { T_STEP=$(date +%s); echo "== $1 ($(date -Is))" | tee -a "$LOG"; }
step_end() { echo "   $1 took $(( $(date +%s) - T_STEP )) s" | tee -a "$LOG"; }
py() { uv run python "$@"; }

if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/   GPU: /' | tee -a "$LOG"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    | awk -F', ' '{split($2, a, " "); n = split(a[1], p, "/"); print "   GPU process: " $1 " " p[n] ", " $NF}' | tee -a "$LOG" || true
fi

OVERWRITE=()
[[ $FORCE == 1 ]] && OVERWRITE=(--overwrite)

if want gpt; then
  step_start "stockGPT walk-forward, horizon $H: years $YEARS, seeds $SEEDS"
  # shellcheck disable=SC2086
  py -m stockai.gpt.walkforward --export "$EXPORT" --out "$RUNS/gpt" --horizon "$H" --years "$YEARS" --seeds $SEEDS \
    "${OVERWRITE[@]}" 2> >(tee "$RUNS/gpt.stderr.log" | grep --line-buffered -E '^seed|done:|wrote' | tee -a "$LOG" >&2)
  step_end "stockGPT walk-forward"
fi

if want null; then
  step_start "stockGPT sign-flip null, horizon $H: years $NULL_YEARS, seed 0"
  py -m stockai.gpt.walkforward --export "$EXPORT" --out "$RUNS/gpt_null" --horizon "$H" --years "$NULL_YEARS" --seeds 0 \
    --null sign-flip "${OVERWRITE[@]}" \
    2> >(tee "$RUNS/gpt_null.stderr.log" | grep --line-buffered -E '^seed|done:|wrote' | tee -a "$LOG" >&2)
  step_end "sign-flip null"
fi
echo "== done in $(( $(date +%s) - T_ALL )) s; scores in $RUNS/gpt and $RUNS/gpt_null" | tee -a "$LOG"
}

main "$@"; exit
