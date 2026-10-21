"""Before any results: how many funds pass c2eval.twins' fund screens on a few calendar days, and which of them
c2eval.not_common would flag (to decide how 'excluded applies to funds' is read). No future returns are used."""
import sys
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import numpy as np
import c2eval

E = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close', 'dollar_volume'))
tw = c2eval.twins(E)
F = tw['F']
print('funds in store (kind etf):', len(F['ids']))
nc_f = c2eval.not_common(F['tickers'])
print('not_common flags among all store ETFs:', int(nc_f.sum()))
print(' e.g.', list(F['tickers'][nc_f][:80]))
fr = c2eval._log_returns(F['close'], max_abs=0.5)
fdv = np.asarray(F['dollar_volume'], np.float64)
for iso in ('2006-01-03', '2010-01-04', '2015-01-02', '2020-01-02', '2025-06-02'):
    t = c2eval.day_index(E['days'], iso)
    k = c2eval.rebal_index(tw, t)
    t0 = int(tw['rebal'][k])
    a, b = t0 - 251, t0 + 1
    Fw = fr[:, a:b]
    with np.errstate(all='ignore'):
        ok = ((np.isfinite(Fw).mean(1) >= 0.9) & (np.nanmedian(fdv[:, a:b], 1) >= 5e6)
              & (np.nanstd(Fw, 1) * np.sqrt(252) <= 0.75))
    print(c2eval.iso(E['days'][t0]), 'funds passing liquidity/vol screens:', int(ok.sum()),
          '; of them not_common:', list(F['tickers'][ok & nc_f]))
