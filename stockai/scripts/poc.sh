#!/usr/bin/env bash
# The stockai proof of concept, end to end (test-results.md reports it):
#   export -> G0 pipeline check -> stockGPT walk-forward -> sign-flip null -> baselines -> grading -> controls -> summary
#
#   scripts/poc.sh [--quick] [--steps LIST] [--results DIR] [--runs DIR] [--force]
#
#   --quick    test years 2022-2023, seed 0, the null on 2022 only: about 7 minutes (401 s measured, free GPU).
#              Writes results/poc-quick/ and runs/poc-quick/.
#   (default)  test years 2010-2023, seeds 0 1 2, the null on 2012, 2017 and 2022: about an hour (61 minutes
#              measured with the GPU partly shared, before LightGBM on returns). Writes results/poc/ and runs/poc/.
#   --steps    a comma list of: export,g0,gpt,null,baselines,grade,controls,summary (default: all, in that order).
#   --force    let the stockGPT steps replace a previous run in the runs directory.
#
# Environment: C2_DATA_DIR (export at $C2_DATA_DIR/export/full, default ~/.local/share/correlation2),
# C2_REF=<a c2 binary built at correlation2 9335126> adds G0a (its reversal table against the judge's).
#
# Nothing here grades, trains on or reads the locked year (2024-06-28 .. 2025-06-27): the judge refuses it without
# --locked (never passed), and every model writes NaN from the locked year's first decision day on.
set -euo pipefail

# Everything runs inside main(), so bash has parsed the whole file before the first step starts (editing the
# script during an hour-long run cannot then change what the run executes).
main() {
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

QUICK=0
STEPS="export,g0,gpt,null,baselines,grade,controls,summary"
RESULTS=""
RUNS=""
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1 ;;
    --steps) STEPS="$2"; shift ;;
    --results) RESULTS="$2"; shift ;;
    --runs) RUNS="$2"; shift ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

if [[ $QUICK == 1 ]]; then
  YEARS="2022-2023"; SEEDS="0"; NULL_YEARS="2022"; STALE_YEARS="2023"
  RESULTS="${RESULTS:-results/poc-quick}"; RUNS="${RUNS:-runs/poc-quick}"
else
  YEARS="2010-2023"; SEEDS="0 1 2"; NULL_YEARS="2012,2017,2022"; STALE_YEARS="2011-2023"
  RESULTS="${RESULTS:-results/poc}"; RUNS="${RUNS:-runs/poc}"
fi
DATA_DIR="${C2_DATA_DIR:-$HOME/.local/share/correlation2}"
EXPORT="$DATA_DIR/export/full"
JUDGE="$ROOT/judge/target/release/stockai-judge"
START="2023-06-28"            # the walk-forward's last test period: 2023-06-28 .. 2024-06-27
GRADE=(--walk-forward --start "$START" --costs 0,5,20 --random 1000)
LGBM_THREADS=8                # fixed, so reruns reproduce the LightGBM scores
mkdir -p "$RESULTS/tables" "$RUNS/baselines" "$RUNS/g0" "$RUNS/controls"
LOG="$RESULTS/poc_log.txt"
echo "== scripts/poc.sh $([[ $QUICK == 1 ]] && echo --quick) $(date -Is): years $YEARS, seeds $SEEDS, null $NULL_YEARS; results $RESULTS, runs $RUNS" | tee -a "$LOG"

want() { [[ ",$STEPS," == *",$1,"* ]]; }
T_ALL=$(date +%s)
step_start() { T_STEP=$(date +%s); echo "== $1" | tee -a "$LOG"; }
step_end() { echo "   $1 took $(( $(date +%s) - T_STEP )) s" | tee -a "$LOG"; }
py() { uv run python "$@"; }

if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/   GPU: /' | tee -a "$LOG"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    | awk -F', ' '{split($2, a, " "); n = split(a[1], p, "/"); print "   GPU process: " $1 " " p[n] ", " $NF}' | tee -a "$LOG" || true
fi
if [[ ! -x "$JUDGE" ]]; then
  echo "== building the judge (cargo build --release in judge/)" | tee -a "$LOG"
  (cd judge && cargo build --release)
fi

# One walk-forward grading: grade NAME SCORES REBALANCE DELAY MATCH
grade() {
  local name=$1 scores=$2 r=$3 d=$4 m=$5
  local out="$RESULTS/${name}__r${r}_d${d}_${m}"
  "$JUDGE" holdout --family scores --scores "$scores" --export "$EXPORT" --rebalance "$r" --delay "$d" \
    --random-match "$m" "${GRADE[@]}" --json "$out.json" > "$RESULTS/tables/$(basename "$out").txt" 2> "$RESULTS/tables/$(basename "$out").err"
  echo "   graded $(basename "$out"): $(grep -m1 'Rank IC in actual' "$RESULTS/tables/$(basename "$out").txt")" | tee -a "$LOG"
}

if want export; then
  step_start "export"
  if [[ ! -f "$EXPORT/meta.json" ]]; then
    "$JUDGE" export --out "$EXPORT" | tee -a "$LOG"
  fi
  py -c "import stockai.data as d; e = d.load('$EXPORT'); print(f'   export {e.path}: {e.n} stocks x {e.t} days, {e.date(0)} .. {e.date(e.t - 1)}, fingerprint {e.meta[\"fingerprint\"]}')" | tee -a "$LOG"
  step_end export
fi

if want g0; then
  step_start "G0: Python reversal scores graded through --family scores must reproduce --family reversal exactly"
  rm -f "$RESULTS/g0.json"
  py -m stockai.baselines.reversal --export "$EXPORT" --lookback 5 --out "$RUNS/baselines/reversal5.npy" | tee -a "$LOG"
  "$JUDGE" holdout --family reversal --lookback 5 --rebalance 5 "${GRADE[@]}" --json "$RESULTS/g0_reversal_family.json" \
    --dump-scores "$RUNS/g0/judge_rev5_dump.npy" > "$RESULTS/tables/g0_reversal_family.txt" 2> "$RESULTS/tables/g0_reversal_family.err"
  "$JUDGE" holdout --family scores --scores "$RUNS/baselines/reversal5.npy" --export "$EXPORT" --rebalance 5 "${GRADE[@]}" \
    --json "$RESULTS/g0_scores_family.json" > "$RESULTS/tables/g0_scores_family.txt" 2> "$RESULTS/tables/g0_scores_family.err"
  py -m stockai.poc.g0 scores "$RUNS/baselines/reversal5.npy" "$RUNS/g0/judge_rev5_dump.npy" --record "$RESULTS/g0.json" --name python_vs_dump | tee -a "$LOG"
  py -m stockai.poc.g0 json "$RESULTS/g0_reversal_family.json" "$RESULTS/g0_scores_family.json" --record "$RESULTS/g0.json" --name scores_vs_reversal_family_json | tee -a "$LOG"
  if [[ -n "${C2_REF:-}" ]]; then
    "$C2_REF" census holdout --family reversal --lookback 5 --rebalance 5 --walk-forward --start "$START" > "$RUNS/g0/ref_c2_rev5.txt" 2>/dev/null
    "$JUDGE" holdout --family reversal --lookback 5 --rebalance 5 --walk-forward --start "$START" > "$RUNS/g0/judge_rev5_default.txt" 2>/dev/null
    py -m stockai.poc.g0 tables "$RUNS/g0/ref_c2_rev5.txt" "$RUNS/g0/judge_rev5_default.txt" --record "$RESULTS/g0.json" --name ref_c2_vs_judge_reversal | tee -a "$LOG"
  fi
  step_end G0
fi

OVERWRITE=()
[[ $FORCE == 1 ]] && OVERWRITE=(--overwrite)

if want gpt; then
  step_start "stockGPT walk-forward: years $YEARS, seeds $SEEDS"
  # shellcheck disable=SC2086
  py -m stockai.gpt.walkforward --export "$EXPORT" --out "$RUNS/gpt" --years "$YEARS" --seeds $SEEDS "${OVERWRITE[@]}" \
    2> >(tee "$RUNS/gpt.stderr.log" | grep --line-buffered -E '^seed|done:|wrote' | tee -a "$LOG" >&2)
  step_end "stockGPT walk-forward"
fi

if want null; then
  step_start "stockGPT sign-flip null: years $NULL_YEARS, seed 0"
  py -m stockai.gpt.walkforward --export "$EXPORT" --out "$RUNS/gpt_null" --years "$NULL_YEARS" --seeds 0 --null sign-flip "${OVERWRITE[@]}" \
    2> >(tee "$RUNS/gpt_null.stderr.log" | grep --line-buffered -E '^seed|done:|wrote' | tee -a "$LOG" >&2)
  step_end "sign-flip null"
fi

if want baselines; then
  step_start "baselines: LightGBM ($YEARS; targets v and actual returns), 12-1 momentum, 5-day reversal, 1-day census reversal"
  py -m stockai.baselines.lgbm --export "$EXPORT" --out "$RUNS/baselines/lgbm" --years "$YEARS" --threads "$LGBM_THREADS" | tee -a "$LOG"
  # Added after the POC run: the same LightGBM trained on the next-day actual return (a daily rank-gauss score)
  # instead of v[d+1], whose trailing-mean part it otherwise learns (test-results.md).
  py -m stockai.baselines.lgbm --export "$EXPORT" --out "$RUNS/baselines/lgbm_ret" --years "$YEARS" --threads "$LGBM_THREADS" \
    --target ret | tee -a "$LOG"
  py -m stockai.baselines.momentum --export "$EXPORT" --out "$RUNS/baselines/momentum_12_1.npy" | tee -a "$LOG"
  [[ -f "$RUNS/baselines/reversal5.npy" ]] || py -m stockai.baselines.reversal --export "$EXPORT" --lookback 5 --out "$RUNS/baselines/reversal5.npy" | tee -a "$LOG"
  py -m stockai.baselines.vreversal --export "$EXPORT" --out "$RUNS/baselines/vreversal1.npy" | tee -a "$LOG"
  step_end baselines
fi

if want grade; then
  step_start "grading: stockai-judge walk-forward, --start $START, costs 0/5/20 bps, 1,000 random accounts"
  for r in 1 5; do for d in 0 1; do for m in uniform volatility; do
    grade gpt_ensemble "$RUNS/gpt/scores_ensemble.npy" $r $d $m
    grade lgbm "$RUNS/baselines/lgbm/scores.npy" $r $d $m
    grade lgbm_ret "$RUNS/baselines/lgbm_ret/scores.npy" $r $d $m  # extra, added after the POC run
  done; done; done
  for s in $SEEDS; do grade "gpt_seed$s" "$RUNS/gpt/scores_seed$s.npy" 1 0 uniform; done
  grade gpt_null "$RUNS/gpt_null/scores_seed0.npy" 1 0 uniform
  grade momentum_12_1 "$RUNS/baselines/momentum_12_1.npy" 5 0 uniform
  grade momentum_12_1 "$RUNS/baselines/momentum_12_1.npy" 21 0 uniform
  grade reversal5 "$RUNS/baselines/reversal5.npy" 5 0 uniform
  # Extra beyond the plan's list: the ensemble's scores as each day's percentile ranks, so the account always holds
  # the model's top 20. The judge buys only positive scores, and stockGPT's expected values were negative for every
  # candidate in some (seed, year)s, which left those accounts in cash. The order is the same, so are the ICs.
  py -m stockai.poc.controls percentile --export "$EXPORT" --scores "$RUNS/gpt/scores_ensemble.npy" \
    --out "$RUNS/controls/gpt_ensemble_pct.npy" | tee -a "$LOG"
  for v in "1 0 uniform" "1 0 volatility" "1 1 uniform" "5 0 uniform"; do
    # shellcheck disable=SC2086
    grade gpt_ensemble_pct "$RUNS/controls/gpt_ensemble_pct.npy" $v
  done
  # Extras beyond the plan's list: the reversal rules at the GPT's daily horizon and with volatility matching.
  grade reversal5 "$RUNS/baselines/reversal5.npy" 5 0 volatility
  grade reversal5 "$RUNS/baselines/reversal5.npy" 1 0 uniform
  grade vreversal1 "$RUNS/baselines/vreversal1.npy" 1 0 uniform
  grade vreversal1 "$RUNS/baselines/vreversal1.npy" 1 0 volatility
  step_end grading
fi

if want controls; then
  step_start "controls: score canaries and the large-cap tripwire"
  py -m stockai.poc.controls shift --export "$EXPORT" --scores "$RUNS/gpt/scores_ensemble.npy" --days 20 --years "$YEARS" \
    --out "$RUNS/controls/gpt_ensemble_shift20.npy" | tee -a "$LOG"
  py -m stockai.poc.controls permute --export "$EXPORT" --scores "$RUNS/gpt/scores_ensemble.npy" --seed 0 --years "$YEARS" \
    --out "$RUNS/controls/gpt_ensemble_permuted.npy" | tee -a "$LOG"
  grade canary_shift20 "$RUNS/controls/gpt_ensemble_shift20.npy" 1 0 uniform
  grade canary_permuted "$RUNS/controls/gpt_ensemble_permuted.npy" 1 0 uniform
  # The same scores a year (250 trading days) stale, on the years that have a year of earlier scores: whatever IC
  # survives that is a characteristic of the stock (such as its volatility), not timing.
  py -m stockai.poc.controls shift --export "$EXPORT" --scores "$RUNS/gpt/scores_ensemble.npy" --days 250 \
    --years "$STALE_YEARS" --out "$RUNS/controls/gpt_ensemble_shift250.npy" | tee -a "$LOG"
  grade canary_shift250 "$RUNS/controls/gpt_ensemble_shift250.npy" 1 0 uniform
  # References for reading the canaries: the same canaries on the 1-day census reversal (a pure timing signal), and
  # a static point-in-time tilt (-log volatility), which a within-year date permutation keeps.
  py -m stockai.poc.controls shift --export "$EXPORT" --scores "$RUNS/baselines/vreversal1.npy" --days 20 --years "$YEARS" \
    --out "$RUNS/controls/vreversal1_shift20.npy" | tee -a "$LOG"
  py -m stockai.poc.controls permute --export "$EXPORT" --scores "$RUNS/baselines/vreversal1.npy" --seed 0 --years "$YEARS" \
    --out "$RUNS/controls/vreversal1_permuted.npy" | tee -a "$LOG"
  py -m stockai.poc.controls lowvol --export "$EXPORT" --years "$YEARS" --out "$RUNS/controls/lowvol.npy" | tee -a "$LOG"
  grade canary_vreversal1_shift20 "$RUNS/controls/vreversal1_shift20.npy" 1 0 uniform
  grade canary_vreversal1_permuted "$RUNS/controls/vreversal1_permuted.npy" 1 0 uniform
  grade lowvol "$RUNS/controls/lowvol.npy" 1 0 uniform
  # Signal decay: the GPT's scores used one day late (the judge's IC columns are decision-day measures at any --delay).
  py -m stockai.poc.controls shift --export "$EXPORT" --scores "$RUNS/gpt/scores_ensemble.npy" --days 1 --years "$YEARS" \
    --out "$RUNS/controls/gpt_ensemble_shift1.npy" | tee -a "$LOG"
  grade decay_shift1 "$RUNS/controls/gpt_ensemble_shift1.npy" 1 0 uniform
  for x in "gpt_ensemble:$RUNS/gpt/scores_ensemble.npy" "gpt_null:$RUNS/gpt_null/scores_seed0.npy" \
           "lgbm:$RUNS/baselines/lgbm/scores.npy" "lgbm_ret:$RUNS/baselines/lgbm_ret/scores.npy" \
           "reversal5:$RUNS/baselines/reversal5.npy" "vreversal1:$RUNS/baselines/vreversal1.npy"; do
    name="${x%%:*}"; file="${x#*:}"
    yrs="$YEARS"; [[ $name == gpt_null ]] && yrs="$NULL_YEARS"
    py -m stockai.poc.controls tripwire --export "$EXPORT" --scores "$file" --years "$yrs" --json "$RESULTS/tripwire_$name.json" \
      > "$RESULTS/tables/tripwire_$name.txt"
    sed -n 1p "$RESULTS/tables/tripwire_$name.txt" | tee -a "$LOG"
  done
  # What the GPT's and LightGBM's scores resemble (inputs only): today's move, volatility, the baselines, each seed.
  # 12-1 momentum is there because the census variable subtracts each stock's trailing 252-day mean residual, so a
  # score that leans against the past year's performance earns census-unit IC that is not in actual returns.
  vs=("lgbm=$RUNS/baselines/lgbm/scores.npy" "reversal5=$RUNS/baselines/reversal5.npy" "vreversal1=$RUNS/baselines/vreversal1.npy"
      "momentum_12_1=$RUNS/baselines/momentum_12_1.npy")
  for s in $SEEDS; do vs+=("seed$s=$RUNS/gpt/scores_seed$s.npy"); done
  py -m stockai.poc.diagnostics --export "$EXPORT" --scores "$RUNS/gpt/scores_ensemble.npy" --vs "${vs[@]}" --years "$YEARS" \
    --json "$RESULTS/diagnostics_gpt_ensemble.json" | tee -a "$LOG"
  py -m stockai.poc.diagnostics --export "$EXPORT" --scores "$RUNS/baselines/lgbm/scores.npy" \
    --vs "momentum_12_1=$RUNS/baselines/momentum_12_1.npy" "reversal5=$RUNS/baselines/reversal5.npy" \
         "vreversal1=$RUNS/baselines/vreversal1.npy" --years "$YEARS" --json "$RESULTS/diagnostics_lgbm.json" | tee -a "$LOG"
  # The actual-return LightGBM (extra): the same references, plus the GPT ensemble and the v-target LightGBM.
  py -m stockai.poc.diagnostics --export "$EXPORT" --scores "$RUNS/baselines/lgbm_ret/scores.npy" \
    --vs "momentum_12_1=$RUNS/baselines/momentum_12_1.npy" "reversal5=$RUNS/baselines/reversal5.npy" \
         "vreversal1=$RUNS/baselines/vreversal1.npy" "gpt_ensemble=$RUNS/gpt/scores_ensemble.npy" \
         "lgbm=$RUNS/baselines/lgbm/scores.npy" --years "$YEARS" --json "$RESULTS/diagnostics_lgbm_ret.json" | tee -a "$LOG"
  step_end controls
fi

if want summary; then
  step_start "summary"
  py -m stockai.poc.summarize --results "$RESULTS" --runs "$RUNS" --years "$YEARS" | tee -a "$LOG"
  step_end summary
fi
echo "== done in $(( $(date +%s) - T_ALL )) s; tables in $RESULTS/summary.md" | tee -a "$LOG"
}

main "$@"; exit
