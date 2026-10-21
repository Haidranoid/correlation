"""Find funds that Stooq files as stocks: export 'stocks' whose daily returns track some ETF in the store
(|correlation| >= 0.95 over >= 250 shared days). Prints the candidates; the drop list is fixed in common.py.
Static property of the instrument, no outcome data involved.

    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python3 -I funds.py [old|new]
"""
import sys
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import numpy as np
import c2eval

which = sys.argv[1] if len(sys.argv) > 1 else 'old'
EXP = c2eval.EXPORT_OLD if which == 'old' else c2eval.EXPORT_NEW
STO = c2eval.STORE_OLD if which == 'old' else c2eval.STORE_NEW
E = c2eval.load_export(EXP, arrays=('close', 'dollar_volume'))
S = c2eval.load_store(STO)
days = E['days']
etf_ids = [s['id'] for s in S['syms'] if s['kind'] == 'etf']
ec = c2eval.store_panel(S, etf_ids, days, 'close')
c = np.asarray(E['close'], dtype=np.float32)
dv = np.asarray(E['dollar_volume'], dtype=np.float32)


def rets(p):
    r = np.full_like(p, np.nan)
    r[:, 1:] = p[:, 1:] / p[:, :-1] - 1
    r[~np.isfinite(r)] = np.nan
    r = np.clip(r, -0.5, 0.5)
    return r


X = rets(c); Y = rets(ec)
# liquid at some point: 60-day median dollar volume >= 5M (rough, rolling every 20 days)
liq = np.zeros(X.shape[0], bool)
for d in range(60, X.shape[1], 20):
    m = np.nanmedian(dv[:, d - 60:d], axis=1)
    liq |= (m >= 5e6) & (c[:, d - 1] >= 5)
Mx = np.isfinite(X).astype(np.float32); X0 = np.nan_to_num(X)
My = np.isfinite(Y).astype(np.float32); Y0 = np.nan_to_num(Y)
idx = np.where(liq)[0]
Mx, X0 = Mx[idx], X0[idx]
n = Mx @ My.T
sxy = X0 @ Y0.T
sx = X0 @ My.T; sx2 = (X0 * X0) @ My.T
sy = Mx @ Y0.T; sy2 = Mx @ (Y0 * Y0).T
with np.errstate(invalid='ignore', divide='ignore'):
    cov = sxy - sx * sy / n
    vx = sx2 - sx * sx / n; vy = sy2 - sy * sy / n
    r = cov / np.sqrt(vx * vy)
r[n < 250] = 0
r = np.nan_to_num(r)
ar = np.abs(r)
best = ar.max(1)
cnt = (ar >= 0.95).sum(1)
order = np.argsort(-best)
etk = [S['syms'][i]['ticker'] for i in etf_ids]
print('liquid-ever stocks:', len(idx))
for k in order:
    if best[k] < 0.9:
        break
    j = idx[k]
    top = np.argsort(-ar[k])[:4]
    print(f"{E['tickers'][j]:8s} best {best[k]:.3f} n_etf>=.95 {cnt[k]:3d}  " + ' '.join(f'{etk[t]}({r[k, t]:+.2f})' for t in top))
