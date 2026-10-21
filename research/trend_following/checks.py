"""Checks that read no strategy outcome, run before the fidelity check (writes checks.txt):
  1. the 12-month excess-return sign against a second, naive computation from dates (simple 12-month total return
     minus the compounded T-bill return, the month-end found by calendar arithmetic);
  2. EWMA volatility of SPY and IEF on a few dates;
  3. weights zero before 12 months of history; active-fund count and gross exposure by year;
  4. turnover in months with no sign change is small (vol drift only).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import common as C  # noqa: E402

P = C.build(C.END_MAIN)
out = []
pr = lambda *a: out.append(' '.join(str(x) for x in a))  # noqa: E731

dates, Px, rf = P['dates'], P['P'], P['rf']
F = len(C.TICKERS)

# 1. Sign cross-check.
mism = 0; n = 0
for k in range(C.LOOKBACK, len(P['me'])):
    d_end = P['me_date'][k]
    ym = d_end.astype('datetime64[M]')
    ym0 = ym - 12
    sel0 = np.nonzero(dates.astype('datetime64[M]') == ym0)[0]
    if len(sel0) == 0:
        continue
    a, b = sel0[-1], P['me'][k]
    for f in range(F):
        if not (np.isfinite(Px[a, f]) and np.isfinite(Px[b, f])):
            assert P['sign_at'][k, f] == 0
            continue
        simple = Px[b, f] / Px[a, f] - 1.0
        bill = np.prod(1.0 + rf[a + 1:b + 1]) - 1.0
        s = np.sign(simple - bill)
        n += 1
        if P['sign_at'][k, f] != 0 and s != P['sign_at'][k, f]:
            mism += 1
pr(f'1. sign cross-check: {n} fund-month-ends compared, {mism} mismatches')

# 2. EWMA volatility.
for t in ('SPY', 'IEF', 'TLT', 'GLD', 'USO'):
    f = C.TICKERS.index(t)
    row = []
    for day in ('2006-12-29', '2008-06-30', '2008-12-31', '2009-06-30', '2017-12-29', '2020-03-31', '2022-12-30'):
        i = int(np.searchsorted(dates, np.datetime64(day)))
        row.append(f'{day} {P["sig_daily"][i, f]:.1%}')
    pr(f'2. EWMA vol {t}: ' + '; '.join(row))

# 3. Activity and gross exposure.
first_active = {}
for f, t in enumerate(C.TICKERS):
    ks = np.nonzero(P['ACT'][:, f])[0]
    first_active[t] = str(P['label'][ks[0]]) if len(ks) else 'never'
    first_bar = str(dates[np.argmax(np.isfinite(Px[:, f]))])
    pr(f'3. {t}: first bar {first_bar}, first return month held {first_active[t]}')
pr('3. by year: return months, mean active funds, mean / max gross exposure, mean monthly turnover')
years = P['label'].astype('datetime64[Y]')
for y in np.unique(years):
    m = (years == y) & (P['n_active'] > 0)
    if not m.any():
        continue
    pr(f'   {y}: {m.sum():2d} months, active {P["n_active"][m].mean():5.1f}, gross {P["gross_exposure"][m].mean():.2f} / '
       f'{P["gross_exposure"][m].max():.2f}, turnover {P["turnover"][m].mean():.3f}')
assert (P['W'][~P['ACT']] == 0).all()

# 4. Turnover in months with no sign change among funds active in both months.
S, ACT = P['S'], P['ACT']
same = np.zeros(len(S), bool)
for k in range(1, len(S)):
    both = ACT[k] & ACT[k - 1]
    same[k] = both.any() and (ACT[k] == ACT[k - 1]).all() and (S[k][both] == S[k - 1][both]).all()
pr(f'4. months with no sign change and no entry: {same.sum()}; their turnover median {np.median(P["turnover"][same]):.4f}, '
   f'max {P["turnover"][same].max():.4f}; other months median {np.median(P["turnover"][~same & (P["n_active"] > 0)]):.4f}')

txt = '\n'.join(out)
print(txt)
(C.HERE / 'checks.txt').write_text(txt + '\n')
