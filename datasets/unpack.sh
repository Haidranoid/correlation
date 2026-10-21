#!/usr/bin/env bash
# Rebuilds the price stores from their compressed parts and links the data where correlation2 looks for it.
#
#   datasets/unpack.sh            (needs zstd; safe to run again)
#
# Data goes to $C2_DATA_DIR, default ~/.local/share/correlation2. Existing files there are never overwritten: a store or
# series folder that already exists is left alone and reported.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${C2_DATA_DIR:-$HOME/.local/share/correlation2}"
command -v zstd >/dev/null || { echo "zstd is needed (e.g. apt install zstd)" >&2; exit 1; }

latest=""
for dir in "$HERE"/stocks/*/; do
  dir="${dir%/}"; name="$(basename "$dir")"; latest="$name"
  if [[ -f "$dir/prices.bin" ]] && (cd "$dir" && sha256sum --quiet -c SHA256SUMS 2>/dev/null); then
    echo "$name: prices.bin present and verified"
  else
    echo "$name: rebuilding prices.bin from $(ls "$dir"/prices.bin.zst.* | wc -l) parts"
    cat "$dir"/prices.bin.zst.* | zstd -q -d --long=27 -o "$dir/prices.bin.tmp"
    mv "$dir/prices.bin.tmp" "$dir/prices.bin"
    (cd "$dir" && sha256sum --quiet -c SHA256SUMS) && echo "$name: verified"
  fi
  mkdir -p "$DATA/stores"
  if [[ -e "$DATA/stores/$name" ]]; then echo "  $DATA/stores/$name exists; left alone"
  else ln -s "$dir" "$DATA/stores/$name" && echo "  linked $DATA/stores/$name"; fi
done
if [[ -n "$latest" && ! -f "$DATA/stores/CURRENT" ]]; then
  echo "$latest" > "$DATA/stores/CURRENT" && echo "default store: $latest"
fi

mkdir -p "$DATA/external"
for item in "$HERE"/external/*; do
  name="$(basename "$item")"
  [[ "$name" == scripts* ]] && continue   # code, not data
  if [[ -e "$DATA/external/$name" ]]; then continue; fi
  ln -s "$item" "$DATA/external/$name"
done
echo "outside data linked into $DATA/external"
