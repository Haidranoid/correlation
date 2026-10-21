"""Checks of the portfolio code (not results): an oracle signal (the next 21 days' return, look-ahead on purpose)
must score hugely on time and less one day late; one hedged leg recomputed by hand; legs really are the deciles.

    OMP_NUM_THREADS=8 python3 -I checks.py      # writes out/checks.txt
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402

E, tw = C.load(c2eval.EXPORT_OLD)
P = C.Panel(E, tw, verbose=False)
lines = []
oracle = np.full((P.N, P.T), np.nan)
oracle[:, :-C.H] = np.log(P.cff[:, C.H:] / P.cff[:, :-C.H])
masks = C.ranked_sets(P, [P.s_twin, P.s_plain, P.s_mkt])
ks = list(range(37, len(P.rebal) - 2))
for delay in (0, 1):
    ms = C.run_strategy(P, ks, oracle, 'twin', delay, masks)
    m, t, n = C.tstat([x['gross'] for x in ms])
    lines.append(f'oracle (next 21 days from close d), twin hedge, delay {delay}: {100 * m:+.2f}%/month t {t:+.1f} n {n}')
# hand check of one leg
k = 120
b = C.month_book(P, k, P.s_twin, 'twin', 0, masks[k])
t0 = int(P.rebal[k])
j = int(b['L'][0])
f = int(tw['fund_row'][k, j])
beta = float(tw['beta'][k, j])
c = np.asarray(E['close'], np.float64)
fc = np.asarray(tw['F']['close'], np.float64)
hand = (c[j, t0 + 21] / c[j, t0] - 1) - beta * (fc[f, t0 + 21] / fc[f, t0] - 1)
lines.append(f'hand check {E["tickers"][j]} vs {tw["F"]["tickers"][f]} beta {beta:.3f} formed {c2eval.iso(P.days[t0])}: '
             f'code {b["legs"][0]:+.6f} hand {hand:+.6f}')
s = P.s_twin[np.nonzero(masks[k])[0], t0]
lines.append(f'deciles: n {b["n"]} legs {len(b["L"])}/{len(b["S"])}; min s in long {P.s_twin[b["L"], t0].min():.3f} '
             f'= 90th pct {np.sort(s)[-len(b["L"])]:.3f}; max s in short {P.s_twin[b["S"], t0].max():.3f} = '
             f'10th pct {np.sort(s)[len(b["S"]) - 1]:.3f}')
lines.append(f'hedge net fund exposure: twin book sum of fund weights {sum(b["fund_start"].values()):+.3f}; '
             f'long-leg mean beta {np.mean(tw["beta"][k, b["L"]]):.2f}, short-leg {np.mean(tw["beta"][k, b["S"]]):.2f}')
print('\n'.join(lines))
(C.OUT / 'checks.txt').write_text('\n'.join(lines) + '\n')
