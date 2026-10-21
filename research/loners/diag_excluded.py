"""Check (no returns used): loners that look lonely only because their partner is excluded. For each long-leg stock
at each test-1 formation, the highest market-relative correlation with a stock the rule excludes (fund-like flag or
not_common; e.g. a second share class flagged as a duplicate). Lists the long-leg stocks whose c would be >= 0.6 with
excluded stocks counted, and how many leg-months that is.

    python3 -I diag_excluded.py old    -> excluded_partners_old.txt
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from collections import Counter, defaultdict
from common import c2eval, np, WORK, HERE, EXPORTS, log_returns, standardize, market_relative, day_num, WIN, MIN_FRAC
from book import loner_legs

tag = _sys.argv[1]
E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'market'))
tw = c2eval.twins(E)
exc = c2eval.excluded(E, tw)
z = np.load(WORK / f'connect_{tag}.npz')
cd = np.load(WORK / f'c_daily_{tag}.npy', mmap_mode='r')
cal = z['cal']
C = np.asarray(cd[:, cal]).T
U, meddv, vol = z['U'], z['meddv'], z['vol']
days = E['days']
tk = np.array([str(t) for t in E['tickers']])
sr = log_returns(np.asarray(E['close'], np.float32)).astype(np.float32)
mkt = np.asarray(E['market'], np.float64)
lo, hi = (day_num('2006-01-01'), day_num('2019-12-31')) if tag == 'old' else (day_num('2025-06-01'), day_num('2026-12-31'))
hits = Counter()
partner = defaultdict(Counter)
n_leg = 0
for k, t0 in enumerate(cal):
    if not (lo <= days[t0] <= hi):
        continue
    m = np.nonzero(U[k] & np.isfinite(C[k]) & np.isfinite(vol[k]) & np.isfinite(meddv[k]))[0]
    if len(m) < 81:
        continue
    L = np.concatenate([S for (S, s, _, _) in loner_legs(m, C[k], meddv[k], vol[k]) if s > 0])
    n_leg += len(L)
    a, b = t0 - WIN + 1, t0 + 1
    X = np.nonzero(exc[k] & (np.isfinite(sr[:, a:b]).mean(1) >= MIN_FRAC))[0]
    if len(X) == 0:
        continue
    Zl = standardize(market_relative(sr[L, a:b].astype(np.float64), mkt[a:b])[0])
    Zx = standardize(market_relative(sr[X, a:b].astype(np.float64), mkt[a:b])[0])
    Cx = Zl @ Zx.T
    j = Cx.argmax(1)
    mx = Cx[np.arange(len(L)), j]
    for i in np.nonzero(mx >= 0.6)[0]:
        hits[tk[L[i]]] += 1
        partner[tk[L[i]]][tk[X[j[i]]]] += 1
lines = [f'{tag}: long-leg stock-months {n_leg}; with an excluded stock at market-relative correlation >= 0.6: '
         f'{sum(hits.values())} ({100 * sum(hits.values()) / max(n_leg, 1):.2f}%) in {len(hits)} stocks']
for t, n in hits.most_common():
    lines.append(f'  {t:8s} {n:3d} months; partner(s): ' + ', '.join(f'{p} {c}' for p, c in partner[t].most_common(3)))
print('\n'.join(lines))
(HERE / f'excluded_partners_{tag}.txt').write_text('\n'.join(lines) + '\n')
