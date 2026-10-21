"""Write a copy of a correlation2 price store that keeps only rows dated before a cut day.

Why: `c2 census holdout --walk-forward` grades its one-year holdouts at 0 bps only, and a single `c2 census holdout`
runs to the store's last day. On a store cut at the end of one walk-forward year, the single holdout from that
year's start covers exactly that year, at every cost level. The 0-bps result must equal the walk-forward row
(checked by the caller).

Usage: python3 -I -B truncate_store.py SRC_STORE_DIR OUT_DIR YYYY-MM-DD   (keeps dates < the given day)
Format (crates/c2-data/src/store.rs): "C2PRICE1", u32 version, u32 n, u64 rows, (n+1) u64 offsets, then 6 columns
(date i32, open/high/low/close/volume f32), each starting on a 64-byte boundary; symbols.tsv with first/last/rows.
"""
import sys
from pathlib import Path

import numpy as np

src, out, cut = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
cut_day = int((np.datetime64(cut) - np.datetime64('1970-01-01')).astype(int))
raw = np.memmap(src / 'prices.bin', dtype=np.uint8, mode='r')
assert bytes(raw[:8]) == b'C2PRICE1'
version = int(np.frombuffer(raw[8:12], '<u4')[0])
n = int(np.frombuffer(raw[12:16], '<u4')[0])
rows = int(np.frombuffer(raw[16:24], '<u8')[0])
offs = np.frombuffer(raw[24:24 + 8 * (n + 1)], '<u8').astype(np.int64)
pos = 24 + 8 * (n + 1)
cols = []
for dt in ('<i4', '<f4', '<f4', '<f4', '<f4', '<f4'):
    pos += (64 - pos % 64) % 64
    cols.append(np.frombuffer(raw[pos:pos + 4 * rows], dtype=dt))
    pos += 4 * rows
date = cols[0]
keep = date < cut_day
# Rows of each symbol are sorted by date, so the kept rows are a prefix of each symbol's block.
new_counts = np.add.reduceat(keep.astype(np.int64), offs[:-1]) if rows else np.zeros(n, np.int64)
empty = offs[1:] == offs[:-1]
new_counts[empty] = 0  # reduceat on an empty block returns the next element, not 0
new_offs = np.zeros(n + 1, np.int64)
new_offs[1:] = np.cumsum(new_counts)
new_rows = int(new_offs[-1])
assert new_rows == int(keep.sum())

out.mkdir(parents=True, exist_ok=True)
with open(out / 'prices.bin', 'wb') as f:
    f.write(b'C2PRICE1')
    f.write(np.array([version, n], '<u4').tobytes())
    f.write(np.array([new_rows], '<u8').tobytes())
    f.write(new_offs.astype('<u8').tobytes())
    p = 24 + 8 * (n + 1)
    for c in cols:
        pad = (64 - p % 64) % 64
        f.write(b'\0' * pad)
        p += pad
        b = np.ascontiguousarray(c[keep]).tobytes()
        f.write(b)
        p += len(b)

lines = (src / 'symbols.tsv').read_text().splitlines()
iso = lambda d: str(np.datetime64('1970-01-01') + np.timedelta64(int(d), 'D'))  # noqa: E731
outl = [lines[0]]
for i, line in enumerate(lines[1:]):
    f = line.split('\t')
    assert int(f[0]) == i
    k = int(new_counts[i])
    if k == 0:
        f[4] = f[5] = iso(0)
    else:
        a = int(offs[i])
        f[4] = iso(date[a])
        f[5] = iso(date[a + k - 1])
    f[6] = str(k)
    outl.append('\t'.join(f))
(out / 'symbols.tsv').write_text('\n'.join(outl) + '\n')
print(f'{src.name} cut before {cut}: {new_rows:,} of {rows:,} rows, {int((new_counts > 0).sum()):,} of {n:,} symbols with data')
