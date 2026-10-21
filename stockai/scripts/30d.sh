#!/usr/bin/env bash
# stockai-30d end to end (test-results-30d.md, test-results-1w.md): can a model pick the period's 10 biggest gainers?
#   rules -> LightGBM on the next period's returns -> its sign-flip null -> the grader's $10,000 accounts -> the
#   same accounts with tied scores broken at random -> grades -> grades inside two survivorship subsets
#
#   scripts/30d.sh [--horizon H] [--quick] [--steps LIST] [--results DIR] [--runs DIR] [--price-test traded|adjusted]
#
#   --horizon     trading days per period: 21 (a calendar month, the default), 5 (a week) or 42 (two months).
#   --quick       test years 2022-2023 with a small LightGBM: a smoke run (a few minutes). Writes results/30d-quick/
#                 and runs/30d-quick/ (results/1w-quick/ and runs/1w-quick/ at --horizon 5).
#   (default)     test years 2010-2023. Writes results/30d/ and runs/30d/ (results/1w/, runs/1w/ at --horizon 5;
#                 results/2m/, runs/2m/ at --horizon 42).
#   --steps       a comma list of: rules,model,null,judge,jitter,evaluate,subsets (default: all, in that order).
#   --price-test  closes for the $5 tests, passed to the model, the grader and the evaluation. Default: as traded when
#                 the export has them (Sharadar), else adjusted (Stooq).
#   --extra NAME=PATH  another scores file to grade alongside (repeatable), e.g. stockGPT's: judged against uniform
#                 and (unless NAME ends in _null) volatility-matched random accounts, evaluated, and in the subsets.
#                 A NAME starting with lgbm also gets the random tie-breaks (step jitter).
#
# Data: $C2_DATA_DIR (default ~/.local/share/correlation2, the Stooq data; docs/sharadar.md for Sharadar's), whose
# export/full the Python side reads and whose store the grader reads. Nothing here grades, trains on or reads the
# locked year (2024-06-28 .. 2025-06-27): the models and rules write nothing from the development cutoff on, the
# evaluation never grades a period that reaches it, and the grader refuses it without --locked (never passed).
set -euo pipefail

main() {
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

QUICK=0
H=21
STEPS="rules,model,null,judge,jitter,evaluate,subsets"
RESULTS=""
RUNS=""
PRICE_TEST=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --extra) EXTRA+=("$2"); shift ;;
    --horizon) H="$2"; shift ;;
    --quick) QUICK=1 ;;
    --steps) STEPS="$2"; shift ;;
    --results) RESULTS="$2"; shift ;;
    --runs) RUNS="$2"; shift ;;
    --price-test) PRICE_TEST="$2"; shift ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

case "$H" in
  21) TAG=30d; PER=month ;;
  5) TAG=1w; PER=week ;;
  42) TAG=2m; PER="2-month period" ;;
  *) TAG="h$H"; PER="$H-day period" ;;
esac
if [[ $QUICK == 1 ]]; then
  YEARS="2022-2023"; QFLAG=(--quick)
  RESULTS="${RESULTS:-results/$TAG-quick}"; RUNS="${RUNS:-runs/$TAG-quick}"
else
  YEARS="2010-2023"; QFLAG=()
  RESULTS="${RESULTS:-results/$TAG}"; RUNS="${RUNS:-runs/$TAG}"
fi
PT=(); [[ -n "$PRICE_TEST" ]] && PT=(--price-test "$PRICE_TEST")
DATA_DIR="${C2_DATA_DIR:-$HOME/.local/share/correlation2}"
EXPORT="$DATA_DIR/export/full"
JUDGE="$ROOT/judge/target/release/stockai-judge"
THREADS=8                     # fixed, so reruns reproduce the LightGBM scores
mkdir -p "$RESULTS/judge" "$RUNS"
LOG="$RESULTS/${TAG}_log.txt"
echo "== scripts/30d.sh $([[ $QUICK == 1 ]] && echo --quick) $(date -Is): years $YEARS, horizon $H; data $DATA_DIR${PRICE_TEST:+, price test $PRICE_TEST}; results $RESULTS, runs $RUNS" | tee -a "$LOG"

want() { [[ ",$STEPS," == *",$1,"* ]]; }
T_ALL=$(date +%s)
step_start() { T_STEP=$(date +%s); echo "== $1" | tee -a "$LOG"; }
step_end() { echo "   $1 took $(( $(date +%s) - T_STEP )) s" | tee -a "$LOG"; }
py() { uv run python "$@"; }

if want rules; then
  step_start "rules: momentum, reversal, volatility, the oracle canary"
  py -m stockai.h30.rules --export "$EXPORT" --out "$RUNS/rules" --horizon "$H" 2>&1 | tee -a "$LOG"
  step_end rules
fi
if want model; then
  step_start "LightGBM on the next $PER's returns"
  py -m stockai.h30.model --export "$EXPORT" --out "$RUNS/lgbm" --years "$YEARS" --horizon "$H" --threads $THREADS \
    "${PT[@]}" "${QFLAG[@]}" 2>&1 | tee -a "$LOG"
  step_end model
fi
if want null; then
  step_start "LightGBM on a sign-flipped target (negative control)"
  py -m stockai.h30.model --export "$EXPORT" --out "$RUNS/lgbm_null" --years "$YEARS" --horizon "$H" --threads $THREADS \
    --null sign-flip "${PT[@]}" "${QFLAG[@]}" 2>&1 | tee -a "$LOG"
  step_end null
fi

NAMES=(lgbm lgbm_null momentum reversal high_vol low_vol oracle)
declare -A EXTRA_PATH=()
for e in "${EXTRA[@]}"; do
  [[ "$e" == *=* ]] || { echo "--extra wants NAME=PATH, got $e" >&2; exit 2; }
  NAMES+=("${e%%=*}"); EXTRA_PATH["${e%%=*}"]="${e#*=}"
done
path_of() {
  if [[ -n "${EXTRA_PATH[$1]:-}" ]]; then echo "${EXTRA_PATH[$1]}"; return; fi
  case "$1" in
    lgbm) echo "$RUNS/lgbm/scores.npy" ;;
    lgbm_null) echo "$RUNS/lgbm_null/scores.npy" ;;
    *) echo "$RUNS/rules/$1.npy" ;;
  esac
}
# Volatility-matched random accounts too for the models (not their nulls) and the two directional rules.
vol_matched() { [[ $1 == lgbm || $1 == momentum || $1 == reversal || ( -n "${EXTRA_PATH[$1]:-}" && $1 != *_null ) ]]; }
# The grader's walk-forward account for one scores file, rebalanced every H days: judge_run SCORES STEM MATCH
judge_run() {
  "$JUDGE" holdout --family scores --scores "$1" --export "$EXPORT" --picks 10 --rebalance "$H" \
    --walk-forward --start 2023-06-28 --costs 0,5,20,50 --random 1000 --random-match "$3" "${PT[@]}" \
    --json "$RESULTS/judge/$2.json" > "$RESULTS/judge/$2.txt"
}
if want judge; then
  step_start "the grader: top 10 positive scores, rebalanced every $H trading days, 2010-2023"
  if [[ ! -x "$JUDGE" ]]; then (cd judge && cargo build --release); fi
  for name in "${NAMES[@]}"; do
    for match in uniform volatility; do
      [[ $match == volatility ]] && ! vol_matched "$name" && continue
      judge_run "$(path_of "$name")" "${name}_${match}" $match
      echo "   graded $name ($match random accounts)" | tee -a "$LOG"
    done
  done
  step_end judge
fi
# The grader breaks tied scores by ticker order. LightGBM's scores tie heavily in the years where validation stopped it
# after a tree or two, so its account and its null's are graded again with ties broken at random, once per seed.
SEEDS=(1 2 3 4 5)
JITTERED=(lgbm lgbm_null)
for e in "${!EXTRA_PATH[@]}"; do [[ $e == lgbm* ]] && JITTERED+=("$e"); done  # extra LightGBM scores tie the same way
if want jitter; then
  step_start "the grader again, tied scores broken at random (seeds ${SEEDS[*]})"
  if [[ ! -x "$JUDGE" ]]; then (cd judge && cargo build --release); fi
  mkdir -p "$RUNS/jitter"
  for name in "${JITTERED[@]}"; do
    for seed in "${SEEDS[@]}"; do
      py -m stockai.h30.jitter --export "$EXPORT" --scores "$(path_of "$name")" --out "$RUNS/jitter/${name}_s$seed.npy" \
        --seed "$seed" 2>&1 | tee -a "$LOG"
      for match in uniform volatility; do
        [[ $match == volatility && $name != lgbm ]] && continue
        judge_run "$RUNS/jitter/${name}_s$seed.npy" "jitter_${name}_s${seed}_${match}" $match
        echo "   graded $name, seed $seed ($match random accounts)" | tee -a "$LOG"
      done
    done
  done
  step_end jitter
fi
if want evaluate; then
  step_start "grades: top-10 hits on the $PER's biggest gainers"
  SC=(); JU=()
  for name in "${NAMES[@]}"; do
    SC+=("$name=$(path_of "$name")")
    [[ -f "$RESULTS/judge/${name}_uniform.json" ]] && JU+=("$name=$RESULTS/judge/${name}_uniform.json")
    [[ -f "$RESULTS/judge/${name}_volatility.json" ]] && JU+=("$name (volatility-matched)=$RESULTS/judge/${name}_volatility.json")
    for seed in "${SEEDS[@]}"; do
      J="$RESULTS/judge/jitter_${name}_s${seed}"
      [[ -f "${J}_uniform.json" ]] && JU+=("$name, ties at random, seed $seed=${J}_uniform.json")
      [[ -f "${J}_volatility.json" ]] && JU+=("$name, ties at random, seed $seed (volatility-matched)=${J}_volatility.json")
    done
  done
  py -m stockai.h30.evaluate --export "$EXPORT" --years "$YEARS" --horizon "$H" --scores "${SC[@]}" --judge "${JU[@]}" \
    "${PT[@]}" --out "$RESULTS" 2>&1 | tee -a "$LOG"
  step_end evaluate
fi
# Survivorship checks (test-results-30d.md): the same grades with every day restricted to the 500 largest stocks, or to
# stocks not down more than 50% over the past year. No grader accounts: its random accounts can't be restricted.
if want subsets; then
  step_start "grades inside two subsets known at each decision day: large500, no_collapse"
  SC=()
  for name in "${NAMES[@]}"; do SC+=("$name=$(path_of "$name")"); done
  for sub in large500 no_collapse; do
    py -m stockai.h30.evaluate --export "$EXPORT" --years "$YEARS" --horizon "$H" --scores "${SC[@]}" --subset $sub \
      "${PT[@]}" --out "$RESULTS/$sub" 2>&1 | tee -a "$LOG"
  done
  step_end subsets
fi
echo "== done in $(( $(date +%s) - T_ALL )) s; see $RESULTS/summary.md" | tee -a "$LOG"
}

main "$@"
