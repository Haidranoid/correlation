"""Diagnostic, added after seeing results (reported only): why the composite's higher IC gives a smaller long-short
than 5-day reversal's. Over the 2008-2019 weeks: mean weekly return of each tenth of the ranked stocks by the composite
and by each component, and the mean 252-day volatility of the top and bottom tenths relative to the week's median.
Also QURE's largest weekly moves on the new export (the unseen months' biggest contributor).

    python3 -I diag_deciles.py    -> diag_deciles.txt
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import c2eval, np, EXPORTS, wdir, HERE, COMPONENTS, LABEL, ffill, in_period

lines = []
tag = 'old'
E = c2eval.load_export(EXPORTS[tag], arrays=('close',))
days = E['days']
z = np.load(wdir(tag) / 'weekly.npz')
W, R, comp = z['W'], z['R'], z['comp']
P = ffill(E['close'])
vol = np.load(wdir(tag) / 'vol.npy', mmap_mode='r')
mk = in_period(days, W, '2008-01-01', '2019-12-31')
ws = np.nonzero(mk)[0]
G = (P[:, W[ws] + 5] / P[:, W[ws]]).T - 1
V = np.asarray(vol[:, W[ws]]).T
del P
scores = {'composite': comp[ws]}
for c in COMPONENTS:
    scores[c] = R[ws, COMPONENTS.index(c)]
lines.append('Mean weekly return (bps) by tenth, lowest score first, 2008-2019; then top-tenth and bottom-tenth volatility '
             '÷ the week\'s median, and top minus bottom')
for k, S in scores.items():
    dec = np.zeros(10); cnt = np.zeros(10); vt = []; vb = []
    for i in range(len(ws)):
        m = np.nonzero(np.isfinite(S[i]) & np.isfinite(G[i]))[0]
        if len(m) < 100:
            continue
        o = m[np.argsort(S[i, m], kind='mergesort')]
        b = np.floor(10 * np.arange(len(o)) / len(o)).astype(int)
        for q in range(10):
            dec[q] += G[i, o[b == q]].mean(); cnt[q] += 1
        med = np.nanmedian(V[i, m])
        vt.append(np.nanmean(V[i, o[b == 9]]) / med); vb.append(np.nanmean(V[i, o[b == 0]]) / med)
    d = dec / cnt * 1e4
    lines.append(f'{LABEL.get(k, k):28s} ' + ' '.join(f'{x:+6.1f}' for x in d)
                 + f' | vol top {np.mean(vt):.2f}, bottom {np.mean(vb):.2f} | top-bottom {d[9] - d[0]:+.1f}')
# QURE on the new export
E2 = c2eval.load_export(EXPORTS['new'], arrays=('close',))
j = int(np.nonzero(E2['tickers'] == 'QURE')[0][0])
c = np.asarray(E2['close'][j], np.float64)
h0 = c2eval.day_index(E2['days'], '2025-06-30')
r5 = c[h0 + 5:] / c[h0:-5] - 1
top = np.argsort(-np.nan_to_num(r5))[:3]
lines.append('QURE largest 5-day moves after 2025-06-30: ' + ', '.join(
    f'{c2eval.iso(E2["days"][h0 + t])}..{c2eval.iso(E2["days"][h0 + t + 5])} {r5[t]:+.0%}' for t in top))
r1 = c[h0 + 1:] / c[h0:-1] - 1
t1 = int(np.nanargmax(r1))
lines.append(f'QURE largest day: {c2eval.iso(E2["days"][h0 + t1 + 1])} {r1[t1]:+.0%}')
txt = '\n'.join(lines)
print(txt)
(HERE / 'diag_deciles.txt').write_text(txt + '\n')
