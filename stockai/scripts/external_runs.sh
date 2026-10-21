#!/usr/bin/env bash
# Does correlation2's outside data improve LightGBM's picks? (test-results-external.md) At each horizon:
#   lgbm_ext          stock features plus the 69 time-only outside features (stockai.h30.external)
#   lgbm_ext_null     the same with the sign-flipped target (negative control)
#   lgbm_ext_shifted  the outside series rolled together by 1,000 trading days first (the control for a false gain)
# then grades them with scripts/30d.sh beside the stock-only lgbm, its null and the rules (from runs/<tag>).
#
#   scripts/external_runs.sh [--horizons "5 21 42"] [--models "lgbm_ext lgbm_ext_null lgbm_ext_shifted"]
#                            [--exclude CATS] [--suffix S] [--steps train,grade]
#
#   --exclude  outside categories to leave out ('revised' = macro, prosperity, migration, population); with --suffix
#              the models are written as <model><suffix> (e.g. --exclude revised --suffix _norev).
#
# Needs ~/.local/share/stockai/external/ (`stockai-judge export-external`) and the stock-only runs (runs/1w, runs/30d,
# runs/2m from scripts/30d.sh). Writes runs/ext-<tag>/ and results/ext-<tag>/. The locked year is never read.
set -euo pipefail

main() {
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
HORIZONS="5 21 42"
MODELS="lgbm_ext lgbm_ext_null lgbm_ext_shifted"
EXCLUDE=""
SUFFIX=""
STEPS="train,grade"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --horizons) HORIZONS="$2"; shift ;;
    --models) MODELS="$2"; shift ;;
    --exclude) EXCLUDE="$2"; shift ;;
    --suffix) SUFFIX="$2"; shift ;;
    --steps) STEPS="$2"; shift ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
[[ -f "$HOME/.local/share/stockai/external/meta.json" ]] || { echo "run stockai-judge export-external first" >&2; exit 1; }
want() { [[ ",$STEPS," == *",$1,"* ]]; }
EXPORT="${C2_DATA_DIR:-$HOME/.local/share/correlation2}/export/full"
for h in $HORIZONS; do
  case "$h" in 21) tag=30d ;; 5) tag=1w ;; 42) tag=2m ;; *) tag="h$h" ;; esac
  RUNS="runs/ext-$tag"
  mkdir -p "$RUNS"
  LOG="$RUNS/external_log.txt"
  EXTRA=()
  for m in $MODELS; do
    name="$m$SUFFIX"
    args=(--external default)
    [[ -n "$EXCLUDE" ]] && args+=(--external-exclude "$EXCLUDE")
    case "$m" in
      lgbm_ext) ;;
      lgbm_ext_null) args+=(--null sign-flip) ;;
      lgbm_ext_shifted) args+=(--external-shift 1000) ;;
      *) echo "unknown model $m" >&2; exit 2 ;;
    esac
    if want train; then
      echo "== $(date -Is) horizon $h: $name (${args[*]})" | tee -a "$LOG"
      T0=$(date +%s)
      uv run python -m stockai.h30.model --export "$EXPORT" --out "$RUNS/$name" --years 2010-2023 --horizon "$h" \
        --threads 8 "${args[@]}" 2>&1 | tee -a "$LOG"
      echo "   $name took $(( $(date +%s) - T0 )) s" | tee -a "$LOG"
    fi
    EXTRA+=(--extra "$name=$RUNS/$name/scores.npy")
  done
  if want grade; then
    echo "== $(date -Is) horizon $h: grading" | tee -a "$LOG"
    scripts/30d.sh --horizon "$h" --steps judge,jitter,evaluate,subsets --runs "runs/$tag" --results "results/ext-$tag" \
      "${EXTRA[@]}" 2>&1 | tail -n 3 | tee -a "$LOG"
  fi
done
}

main "$@"
