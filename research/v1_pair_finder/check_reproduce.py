"""Does this re-implementation give v1's own saved numbers? Recompute every saved MSFT/AAPL row from the price store
v1 imported (Stooq, 2025-06-27) and compare r and n. Also checks the matrix form against the per-pair form."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np

import v1lib
import c2eval

rows = [r for r in csv.DictReader(open(HERE / 'v1_saved_relationships.tsv'), delimiter='\t') if not r['symbol1'].startswith('TEST')]
st = c2eval.load_store(c2eval.STORE_OLD)
by = {}
for i, s in enumerate(st['syms']):
    by.setdefault(s['ticker'], []).append(i)


def series(tk):
    cands = [i for i in by.get(tk, []) if st['syms'][i]['kind'] in ('stock', 'unknown', 'etf') or True]
    i = max(cands, key=lambda i: st['syms'][i]['rows'])
    a, b = st['offs'][i], st['offs'][i + 1]
    return np.asarray(st['date'][a:b]), np.asarray(st['close'][a:b], dtype=np.float64)


data = {tk: series(tk) for tk in ('MSFT', 'AAPL')}
cache = {}
diffs, ndiff, out = [], 0, []
for r in rows:
    a, b, w1, w2, k = r['symbol1'], r['symbol2'], int(r['window1_days']), int(r['window2_days']), int(r['offset_days'])
    for tk, w in ((a, w1), (b, w2)):
        if (tk, w) not in cache:
            cache[(tk, w)] = v1lib.v1_changes(*data[tk], w)
    rr, n = v1lib.v1_corr(cache[(a, w1)], cache[(b, w2)], k)
    saved = float(r['correlation_coefficient'])
    diffs.append(abs(rr - saved))
    ndiff += n != int(r['data_points'])
    out.append((a, b, w1, k, saved, rr, int(r['data_points']), n))
d = np.array(diffs)
print(f'{len(rows)} saved rows recomputed: |r - saved| median {np.median(d):.2e}, 95th pct {np.quantile(d, .95):.2e}, max {d.max():.2e}; '
      f'n differs in {ndiff} rows')
for o in sorted(out, key=lambda o: -abs(o[5] - o[4]))[:5]:
    print('  worst', o)
mx = v1lib.v1_max_offset(data['MSFT'][0], data['AAPL'][0], 30, 30)
print('v1 max offset MSFT/AAPL w=30:', mx, ' offsets tested per pair and window:', mx + 1)

# matrix form vs per-pair form on a small common-calendar block
ex = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close', 'dollar_volume'))
s, e = c2eval.day_index(ex['days'], '2015-06-29'), c2eval.day_index(ex['days'], '2022-06-28')
u = v1lib.select_universe(ex, s, e, size=12)
cal = ex['days'][s:e].astype(np.int64)
C = np.asarray(ex['close'][u, s:e], dtype=np.float64)
for w in (7, 30, 180):
    ch = v1lib.pct_changes(C, w)
    valid = np.arange(len(cal)) >= w
    worst = 0.0
    for k in (0, 1, 2, 7, 45, 400, 2000):
        i, j = v1lib.offset_pairs(cal, valid, valid, k)
        R = v1lib.corr_block(ch[:, i], ch[:, j])
        for p in range(3):
            for q in range(3):
                chA = dict(zip(cal[w:].tolist(), ch[p, w:].tolist()))
                chB = dict(zip(cal[w:].tolist(), ch[q, w:].tolist()))
                rr, n = v1lib.v1_corr(chA, chB, k)
                assert n == len(i), (n, len(i))
                worst = max(worst, abs(rr - R[p, q]))
    print(f'matrix vs per-pair, w={w}: max |dr| = {worst:.1e}')
