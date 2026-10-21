#!/usr/bin/env bash
# Day-one checks of a Sharadar download (docs/sharadar.md; test-results-30d.md, "Next steps"): the files' SHA-256
# against SHA256SUMS, then `stockai-judge check-sharadar` on them:
#   - SIVB, BBBY, FRC and TWTR in stocks and tickers, their last price dates, actions and delisting kinds
#     (TWTR should be a buyout; SIVB, FRC and BBBY a failure or unknown);
#   - delisted common stocks by year and kind, the action codes with counts;
#   - daily returns from closeadj against the Stooq store through 2024-06-27: AAPL, IBM, KO (a check), then every
#     common stock in both (a diagnostic: differing days by year, missing dividends, the biggest gaps).
#
#   scripts/sharadar_check.sh [RAW_DIR] [check-sharadar options, e.g. --stooq DIR]
#
# RAW_DIR defaults to the newest ~/.local/share/stockai/sharadar/raw/<date>/. The report is also written to
# RAW_DIR/check.txt. Exit status 1 if a check fails. JUDGE=path picks the binary (default
# judge/target/release/stockai-judge).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
die() { echo "sharadar_check: $*" >&2; exit 1; }

RAW=""
if [[ $# -gt 0 && "$1" != -* ]]; then
  RAW="$1"
  shift
fi
case "${1:-}" in -h|--help) sed -n '2,14p' "$0"; exit 0 ;; esac
if [[ -z "$RAW" ]]; then
  RAW="$(find "$HOME/.local/share/stockai/sharadar/raw" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1)"
  [[ -n "$RAW" ]] || die "no download in ~/.local/share/stockai/sharadar/raw/ (run scripts/sharadar_download.sh)"
fi
[[ -d "$RAW" ]] || die "$RAW is not a directory"

JUDGE="${JUDGE:-$ROOT/judge/target/release/stockai-judge}"
[[ -x "$JUDGE" ]] || die "$JUDGE is missing: (cd judge && cargo build --release)"
"$JUDGE" check-sharadar --help >/dev/null 2>&1 || die "$JUDGE has no check-sharadar: rebuild it (cd judge && cargo build --release)"

if [[ -f "$RAW/SHA256SUMS" ]]; then
  echo "== SHA-256 of the files in $RAW"
  (cd "$RAW" && sha256sum -c SHA256SUMS) || die "a file does not match SHA256SUMS (damaged or replaced since the download)"
else
  echo "== no SHA256SUMS in $RAW; files not verified"
fi
echo
"$JUDGE" check-sharadar --raw "$RAW" "$@" | tee "$RAW/check.txt"
