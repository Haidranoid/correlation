#!/usr/bin/env bash
# Steps 4 and 5 of test-results-30d.md ("Next steps") on Sharadar data, monthly and weekly (docs/sharadar.md,
# "The runs"), each the unchanged scripts/30d.sh pipeline on another data root:
#   survivors   the survivors-only store, $5 test on adjusted closes: the Stooq-like control, which should roughly
#               reproduce results/30d and results/1w;
#   full        the full store, $5 test on as-traded closes, delisting rule main (bounds reported beside it): the test;
#   adjusted    the full store, $5 test on adjusted closes: separates the $5 rule's effect from the delistings'.
#
#   scripts/sharadar_runs.sh [--horizons "21 5"] [--runs "survivors full adjusted"] [--steps LIST]
#
# Needs the two stores and their exports (README, "Survivorship-free rerun"). Writes results/sharadar-<run>-<30d|1w>/
# and runs/sharadar-<run>-<30d|1w>/ (about 5 GB each, mostly the tie-broken scores). The locked year is never read.
set -euo pipefail

main() {
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
HORIZONS="21 5"
RUNS_WANTED="survivors full adjusted"
STEPS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --horizons) HORIZONS="$2"; shift ;;
    --runs) RUNS_WANTED="$2"; shift ;;
    --steps) STEPS=(--steps "$2"); shift ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
FULL="$HOME/.local/share/stockai/sharadar"
SURV="$HOME/.local/share/stockai/sharadar-survivors"
for root in "$FULL" "$SURV"; do
  [[ -f "$root/export/full/meta.json" && -f "$root/store/source.json" ]] || {
    echo "missing $root/store or $root/export/full: import and export the Sharadar data first (README)" >&2; exit 1; }
done
for h in $HORIZONS; do
  case "$h" in 21) tag=30d ;; 5) tag=1w ;; *) tag="h$h" ;; esac
  for run in $RUNS_WANTED; do
    case "$run" in
      survivors) root="$SURV"; test=adjusted ;;
      full) root="$FULL"; test=traded ;;
      adjusted) root="$FULL"; test=adjusted ;;
      *) echo "unknown run $run" >&2; exit 2 ;;
    esac
    echo "== sharadar $run, horizon $h: C2_DATA_DIR=$root, price test $test"
    C2_DATA_DIR="$root" scripts/30d.sh --horizon "$h" --price-test "$test" "${STEPS[@]}" \
      --results "results/sharadar-$run-$tag" --runs "runs/sharadar-$run-$tag"
  done
done
echo "== done: results/sharadar-*/summary.md; compare with results/30d and results/1w (Stooq)"
}

main "$@"
