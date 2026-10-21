"""Diagnostic, added after seeing the unseen months' judge result (reported only): which stocks drove the judge's
account. Approximates the judge's account: decision days h0-1 + 5k from 2025-06-30, the top 20 positive composites
among the universe at h0-1, equal weight, held 5 days. Sums each stock's contribution (1/20 x its 5-day return).

    python3 -I diag_unseen.py    -> diag_unseen.txt
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from collections import defaultdict
from common import c2eval, np, EXPORTS, wdir, HERE, ffill

tag = 'new'
E = c2eval.load_export(EXPORTS[tag], arrays=('close',))
days = E['days']
T = len(days)
P = ffill(E['close'])
S = np.load(wdir(tag) / 'composite.npy', mmap_mode='r')
U = np.load(wdir(tag) / 'universe.npy', mmap_mode='r')
h0 = c2eval.day_index(days, '2025-06-30')
cand = np.asarray(U[h0 - 1]).copy()
contrib = defaultdict(float)
held = defaultdict(int)
value = 1.0
weekly = []
for d in range(h0 - 1, T - 1, 5):
    e = min(d + 5, T - 1)
    s = np.asarray(S[:, d], np.float64)
    ok = np.nonzero(cand & np.isfinite(s) & (s > 0))[0]
    top = ok[np.argsort(-s[ok], kind='mergesort')[:20]]
    r = P[top, e] / P[top, d] - 1
    for j, x in zip(top, r):
        contrib[int(j)] += x / 20
        held[int(j)] += 1
    weekly.append(r.mean())
    value *= 1 + r.mean()
tot = sum(contrib.values())
order = sorted(contrib, key=lambda j: -contrib[j])
lines = [f'approximate account: compounded {value - 1:+.1%}; sum of weekly returns {sum(weekly):+.1%} over {len(weekly)} weeks; '
         f'{len(contrib)} distinct stocks']
for j in order[:10]:
    lines.append(f'  {E["tickers"][j]:8s} contribution {contrib[j]:+.1%} (held {held[j]} weeks)')
top5 = sum(contrib[j] for j in order[:5])
lines.append(f'top 5 contribute {top5:+.1%} of {tot:+.1%} (sum of weekly returns); without them {tot - top5:+.1%}')
lines.append('worst: ' + ', '.join(f'{E["tickers"][j]} {contrib[j]:+.1%}' for j in order[-5:]))
txt = '\n'.join(lines)
print(txt)
(HERE / 'diag_unseen.txt').write_text(txt + '\n')
