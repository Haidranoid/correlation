#!/bin/bash
# Run one gravity script under the memory cap with 4 threads; output to the work dir's logs/.
#   research/gravity/run.sh <script.py> [args...]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
BIG="$HOME/.local/share/correlation2/experiments/gravity"
mkdir -p "$BIG/logs" "$BIG/numba_cache"
name="$(basename "$1" .py)$(printf '_%s' "${@:2}")"
exec systemd-run --user --scope --quiet -p MemoryMax=5G env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  NUMBA_NUM_THREADS=4 NUMBA_CACHE_DIR="$BIG/numba_cache" "$HERE/.venv/bin/python" -I -B "$HERE/$1" "${@:2}" > "$BIG/logs/$name.log" 2>&1
