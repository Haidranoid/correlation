#!/usr/bin/env bash
# Downloads Sharadar "Prices" tables for stockai (docs/sharadar.md): stocks, funds, actions and tickers.
#
#   scripts/sharadar_download.sh [--out DIR] [--tables LIST] [--force]
#       Bulk: each table's whole history (years=full; the API answers with a redirect to a time-limited compressed
#       file, which is followed). Writes into DIR (default ~/.local/share/stockai/sharadar/raw/<today>/):
#       <table>.zip, <table>.csv.gz or <table>.csv as served, <table>.status.json (the API's description of the
#       bulk file, fetched first), SHA256SUMS (check with `sha256sum -c SHA256SUMS` inside DIR) and manifest.tsv.
#   scripts/sharadar_download.sh --sample --out DIR [--tickers LIST] [--funds LIST] [--from DATE] [--to DATE]
#                                [--limit N] [--tables LIST] [--force]
#       A small download through the query API, paged with skip, into DIR/<table>.csv: stocks and funds rows dated
#       --from .. --to (default 2021-10-20 .. 2024-06-27), actions in the same window, every tickers row. For tests.
#       The public key test-api-key serves prices of AAPL, MSFT, IBM, KO, JNJ, PG, MMM, NVDA, AMZN and SPY from about
#       five years back only, and actions only for AAPL (--tables actions --tickers AAPL --funds ""); no bulk files.
#
#   --tables   comma list of stocks,funds,actions,tickers (default: all four)
#   --force    download again over files that are already there
#
# The key: SHARADAR_API_KEY, else the first line of ~/.config/sharadar/api_key. It is never printed: curl reads the
# URL from a config on stdin (-K -), so the key isn't in the process list either, and error bodies are printed with
# it masked. The bulk redirect's signed URL is neither printed nor saved. SHARADAR_API replaces the API's base URL
# (tests against a local server).
set -euo pipefail

API="${SHARADAR_API:-https://api.sharadar.com/v1.0/data}"
OUT=""
TABLES="stocks,funds,actions,tickers"
SAMPLE=0
FORCE=0
TICKERS="AAPL,MSFT,IBM,KO,JNJ,PG,MMM,NVDA,AMZN"
FUNDS="SPY"
FROM="2021-10-20"
TO="2024-06-27"
LIMIT=10000            # the API's documented default page size
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift ;;
    --tables) TABLES="$2"; shift ;;
    --sample) SAMPLE=1 ;;
    --force) FORCE=1 ;;
    --tickers) TICKERS="$2"; shift ;;
    --funds) FUNDS="$2"; shift ;;
    --from) FROM="$2"; shift ;;
    --to) TO="$2"; shift ;;
    --limit) LIMIT="$2"; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;  # the header above
    *) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

die() { echo "sharadar_download: $*" >&2; exit 1; }

KEY="${SHARADAR_API_KEY:-}"
KEYFILE="$HOME/.config/sharadar/api_key"
if [[ -z "$KEY" && -r "$KEYFILE" ]]; then
  KEY="$(head -n 1 "$KEYFILE" | tr -d '[:space:]')"
fi
[[ -n "$KEY" ]] || die "no API key: put it on the first line of $KEYFILE (chmod 600) or set SHARADAR_API_KEY"
[[ "$KEY" =~ ^[A-Za-z0-9._-]+$ ]] || die "the API key has unexpected characters (expected letters, digits, '.', '_' or '-')"

if [[ -z "$OUT" ]]; then
  [[ $SAMPLE == 0 ]] || die "--sample needs --out DIR"
  OUT="$HOME/.local/share/stockai/sharadar/raw/$(date +%F)"
fi
mkdir -p "$OUT"
for t in ${TABLES//,/ }; do
  case "$t" in stocks|funds|actions|tickers) ;; *) die "unknown table $t (stocks, funds, actions, tickers)" ;; esac
done
command -v curl >/dev/null || die "curl is not installed"
command -v sha256sum >/dev/null || die "sha256sum is not installed"

# Prints a file with the key masked (error bodies may echo the request).
show() { local s; s="$(head -c 2000 "$1" 2>/dev/null || true)"; printf '%s\n' "${s//$KEY/***}" >&2; }

# get URL OUTFILE [curl options...]: GET URL&api_key=KEY into OUTFILE; prints "HTTP_CODE REDIRECTS" (code 000: no
# answer). The URL goes to curl through a config on stdin, never on the command line.
get() {
  local url="$1" out="$2"; shift 2
  printf 'url = "%s&api_key=%s"\n' "$url" "$KEY" | curl -K - -sS -o "$out" -w '%{http_code} %{num_redirects}' "$@" || true
}

# refused TABLE CODE BODYFILE: a clear message for an HTTP error, then exit.
refused() {
  local table="$1" code="$2" body="$3"
  echo "sharadar_download: $table: HTTP $code from the API:" >&2
  show "$body"
  case "$code" in
    401|403) die "the key was refused for $table. Check $KEYFILE (or SHARADAR_API_KEY), that the subscription is active and that it covers this table and the full history. (The public test key gets 401 for bulk downloads and 403 for most tickers.)" ;;
    429) die "rate limited (HTTP 429); wait and run again" ;;
    000) die "no answer from the API for $table (network error; curl's message is above)" ;;
    *) die "unexpected HTTP $code for $table" ;;
  esac
}

rows_of() { awk 'END { print (NR > 0 ? NR - 1 : 0) }' "$1"; }  # data lines (counts a last line without newline)

record() {  # record TABLE FILE NOTE: SHA256SUMS and manifest.tsv lines
  local table="$1" file="$2" note="$3" bytes sum
  bytes=$(stat -c %s "$OUT/$file")
  sum=$(sha256sum "$OUT/$file" | cut -d' ' -f1)
  [[ -s "$OUT/manifest.tsv" ]] || printf 'table\tfile\tbytes\tsha256\tdownloaded_utc\tnote\n' > "$OUT/manifest.tsv"
  awk -F'\t' -v f="$file" 'NR == 1 || $2 != f' "$OUT/manifest.tsv" > "$OUT/manifest.tsv.tmp"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$table" "$file" "$bytes" "$sum" "$(date -u +%FT%TZ)" "$note" >> "$OUT/manifest.tsv.tmp"
  mv "$OUT/manifest.tsv.tmp" "$OUT/manifest.tsv"
  touch "$OUT/SHA256SUMS"
  grep -v "  ${file//./\\.}\$" "$OUT/SHA256SUMS" > "$OUT/SHA256SUMS.tmp" || true
  printf '%s  %s\n' "$sum" "$file" >> "$OUT/SHA256SUMS.tmp"
  mv "$OUT/SHA256SUMS.tmp" "$OUT/SHA256SUMS"
  echo "  $file: $bytes bytes, sha256 $sum${note:+, $note}"
}

existing() {  # existing TABLE: names of files already downloaded for TABLE
  local f
  for f in "$OUT/$1.zip" "$OUT/$1.csv.gz" "$OUT/$1.csv"; do [[ -e "$f" ]] && basename "$f"; done
  return 0
}

bulk_table() {
  local table="$1" tmp code redirects magic ext size_api bytes
  if [[ -n "$(existing "$table")" && $FORCE == 0 ]]; then
    echo "$table: $(existing "$table" | tr '\n' ' ')already in $OUT; skipped (--force downloads again)"
    return
  fi
  tmp="$OUT/.$table.status.part"
  read -r code redirects <<< "$(get "$API/$table?years=full&status=True" "$tmp")"
  [[ "$code" == 200 ]] || refused "$table" "$code" "$tmp"
  mv "$tmp" "$OUT/$table.status.json"
  echo "$table: bulk file status: $(show "$OUT/$table.status.json" 2>&1)"
  # The size the API announces, if its JSON has one ("size" or any key ending in "size").
  size_api=""
  if command -v python3 >/dev/null; then
    size_api=$(python3 -I -c '
import json, sys
def find(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k.lower().endswith("size") and isinstance(v, int):
                return v
        for v in o.values():
            r = find(v)
            if r is not None:
                return r
    if isinstance(o, list):
        for v in o:
            r = find(v)
            if r is not None:
                return r
    return None
try:
    r = find(json.load(open(sys.argv[1])))
except Exception:
    r = None
print("" if r is None else r)' "$OUT/$table.status.json")
  fi

  tmp="$OUT/.$table.part"
  echo "$table: downloading the bulk file (years=full)..."
  read -r code redirects <<< "$(get "$API/$table?years=full" "$tmp" -L --retry 3 --retry-delay 10)"
  if [[ "$code" != 200 ]]; then
    if [[ "$redirects" == 0 || "$code" == 000 ]]; then refused "$table" "$code" "$tmp"; fi
    show "$tmp"
    die "$table: the bulk file's link answered HTTP $code after the API's redirect (expired link?); run again"
  fi
  magic=$(head -c 4 "$tmp" | od -An -tx1 | tr -d ' \n')
  case "$magic" in
    504b0304) ext="zip" ;;
    1f8b*) ext="csv.gz" ;;
    *) ext="csv" ;;
  esac
  case "$ext" in
    zip) if command -v unzip >/dev/null; then unzip -tqq "$tmp" >/dev/null || die "$table: the zip is damaged ($tmp)";
         elif command -v python3 >/dev/null; then python3 -I -m zipfile -t "$tmp" >/dev/null || die "$table: the zip is damaged ($tmp)"; fi ;;
    csv.gz) gzip -t "$tmp" || die "$table: the gzip file is damaged ($tmp)" ;;
    csv) head -c 200 "$tmp" | head -n 1 | grep -q ',' || { show "$tmp"; die "$table: the download is neither a zip, a gzip nor a CSV"; } ;;
  esac
  bytes=$(stat -c %s "$tmp")
  if [[ -n "$size_api" && "$size_api" != "$bytes" ]]; then
    echo "  WARNING: $table: the status said $size_api bytes, the download has $bytes" >&2
  fi
  rm -f "$OUT/$table.zip" "$OUT/$table.csv.gz" "$OUT/$table.csv"
  mv "$tmp" "$OUT/$table.$ext"
  record "$table" "$table.$ext" "bulk years=full${size_api:+; status size $size_api}"
}

sample_table() {
  local table="$1" query code redirects skip rows total page
  if [[ -n "$(existing "$table")" && $FORCE == 0 ]]; then
    echo "$table: $(existing "$table" | tr '\n' ' ')already in $OUT; skipped (--force downloads again)"
    return
  fi
  case "$table" in
    stocks) query="ticker=$TICKERS&from=$FROM&to=$TO" ;;
    funds) [[ -n "$FUNDS" ]] || { echo "funds: no --funds tickers; skipped"; return; }
           query="ticker=$FUNDS&from=$FROM&to=$TO" ;;
    actions) query="ticker=$ALL&from=$FROM&to=$TO" ;;
    tickers) query="ticker=$ALL" ;;
  esac
  page="$OUT/.$table.page"
  : > "$OUT/.$table.part"
  skip=0
  total=0
  while :; do
    read -r code redirects <<< "$(get "$API/$table?format=csv&$query&limit=$LIMIT&skip=$skip" "$page")"
    [[ "$code" == 200 ]] || refused "$table" "$code" "$page"
    rows=$(rows_of "$page")
    if [[ $skip == 0 ]]; then
      cat "$page" > "$OUT/.$table.part"
    elif [[ $rows -gt 0 ]]; then
      [[ -z "$(tail -c 1 "$OUT/.$table.part")" ]] || echo >> "$OUT/.$table.part"
      tail -n +2 "$page" >> "$OUT/.$table.part"
    fi
    total=$((total + rows))
    # A full page (the requested limit, or the documented default of 10,000 if the API caps it) means more may follow.
    if [[ $rows -gt 0 && ( $rows -eq $LIMIT || $rows -eq 10000 ) ]]; then
      skip=$((skip + rows))
    else
      break
    fi
  done
  rm -f "$page"
  [[ -z "$(tail -c 1 "$OUT/.$table.part")" ]] || echo >> "$OUT/.$table.part"
  mv "$OUT/.$table.part" "$OUT/$table.csv"
  record "$table" "$table.csv" "sample: $total rows; $query"
}

ALL="$(printf '%s,%s' "$TICKERS" "$FUNDS" | tr -s ',' | sed 's/^,//; s/,$//')"  # stocks and funds, for actions and tickers
echo "== sharadar_download $([[ $SAMPLE == 1 ]] && echo "--sample ")into $OUT: ${TABLES//,/, }"
for t in ${TABLES//,/ }; do
  if [[ $SAMPLE == 1 ]]; then sample_table "$t"; else bulk_table "$t"; fi
done
echo "Done. Check: (cd \"$OUT\" && sha256sum -c SHA256SUMS)"
