"""Diagnostics added after the first results (analysis only, no rule changes): which window day the timing rule
picks, by portfolio and side; how often the traded stock is already in the favourable third on day 0; and the
largest single contributions to P2's portfolio effect (to rule out bad prints).

    python3 -I research/free_timing/diagnostics.py > research/free_timing/diagnostics_old.txt
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report as R  # noqa: E402

B = R.Book(os.path.join(R.HERE, 'trades_old.npz'))
z = B.z
print('## Window day chosen by the timing rule (% of timable trades, days 0..4)')
for s in B.sigs:
    ks, tim = B.kstar(s)
    for p in (1, 2):
        for sd in (1, -1):
            m = tim & (B.port == p) & (B.side == sd)
            h = np.bincount(ks[m], minlength=5)
            fav = np.where(sd > 0, z[f'{s}_buy'][:, 0], z[f'{s}_sell'][:, 0])[m]
            opp = np.where(sd > 0, z[f'{s}_sell'][:, 0], z[f'{s}_buy'][:, 0])[m]
            print(f"{s} P{p} {'buys ' if sd > 0 else 'sells'} n {m.sum():5d}  days % {np.round(100 * h / h.sum()).astype(int).tolist()}  "
                  f"mean day {ks[m].mean():.2f}  day-0 favourable {fav.mean():.2f}, opposite third {opp.mean():.2f}")
print('\n## Test 3 sensitivity (S1, next-open, raw, P2)')
r = R.portfolio(B, 'S1', 'open', 2)
full = {y: v for y, v in r['by_year'].items() if y != 2025}
print(f"ahead {r['ahead']}/{r['years']} with 2025's half year; {sum(v > 0 for v in full.values())}/{len(full)} without it")
r = R.portfolio(B, 'S1', 'close', 2)
print(f"same-close: ahead {r['ahead']}/{r['years']}")
print('\n## Largest single contributions to P2, S1, next-open, raw (points of the year they fall in)')
d, ok = B.delta('S1', 'open', False, 'timed')
ks, _ = B.kstar('S1')
c = np.where(ok & (B.port == 2), B.size * d, 0)
for i in np.argsort(-np.abs(c))[:12]:
    print(f"{z['tickers'][i]:6s} {R.iso(z['day'][i])} {'buy ' if B.side[i] > 0 else 'sell'} day {ks[i]} size {B.size[i]:.3f} "
          f"contribution {100 * c[i]:+.2f}  opens {np.round(z['px_open'][i], 2).tolist()} END {z['END'][i]:.2f}")
print(f"sum of all P2 contributions {100 * c.sum():+.2f} points over {len(B.dates)} rebalances; "
      f"largest ten {100 * c[np.argsort(-np.abs(c))[:10]].sum():+.2f}")
