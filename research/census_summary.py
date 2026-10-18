#!/usr/bin/python3
"""Summarize a census run's per-matrix summary lines: real vs chance tail counts by horizon
pair and lag, and the share of matrices with a false-discovery cut-off.

    /usr/bin/python3 research/census_summary.py [run name, default main]
"""
import csv
import glob
import math
import sys
from collections import defaultdict
from pathlib import Path

run = sys.argv[1] if len(sys.argv) > 1 else 'main'
d = Path.home() / '.local/share/correlation2/census' / run
rows = []
for f in sorted(glob.glob(str(d / 'tasks/t*.csv'))):
    with open(f) as fh:
        rows.extend(csv.DictReader(fh))
print(f'{run}: {len(rows):,} matrices')

def analytic(row, z):
    n = int(row['stocks'])
    return n * (n - 1) * math.erfc(z / math.sqrt(2))

groups = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0, 0, 0, 0]))
for r in rows:
    g = groups[(int(r['h1']), int(r['h2']))][int(r['lag'])]
    g[0] += int(r['real_z4'])
    g[1] += float(r['null_z4'])
    g[2] += analytic(r, 4.0)
    g[3] += 1
    g[4] += r['fdr_r'] != ''
    g[5] += r.get('placebo_fdr_r', '') not in ('', None)

for (h1, h2), lags in sorted(groups.items()):
    print(f'\nleader horizon {h1}, target horizon {h2}: entries with |z| >= 4, summed over windows')
    print(f'{"lag":>5} {"real":>12} {"chance (null)":>13} {"normal theory":>14} {"real/null":>10} {"FDR cut-off":>12} {"placebo cut-off":>16}')
    keys = sorted(lags)
    show = keys if len(keys) <= 14 else keys[:7] + keys[7::max(1, len(keys) // 8)]
    for k in show:
        real, null, an, n, f, pl = lags[k]
        print(f'{k:>5} {real:>12,} {null:>13,.0f} {an:>14,.0f} {real / max(null, 1):>10.2f} {100 * f / n:>11.1f}% {100 * pl / n:>15.1f}%')
