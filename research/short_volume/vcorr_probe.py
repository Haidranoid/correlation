"""Short volume: compare two volume-match measures (no returns used): the plain 63-day correlation of log FINRA total
volume with log export volume, and the same after removing each day's cross-sectional median of both (the market-wide
volume factor, e.g. index-rebalance days when exchange closing auctions swell). Prints shares below thresholds among
universe stock-days, by period.

    python3 -I -B vcorr_probe.py new
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import mapcheck as M  # noqa: E402
np, c2eval = C.np, C.c2eval

tag = sys.argv[1]
E = c2eval.load_export(C.EXPORTS[tag], arrays=('close', 'dollar_volume'))
days = E['days']
iso = [c2eval.iso(d) for d in days]
N, T = E['close'].shape
total = np.load(C.WORK / tag / 'total.npy')
with np.errstate(all='ignore'):
    vol = np.asarray(E['dollar_volume'], np.float32) / np.asarray(E['close'], np.float32)
    lx = np.where(total > 0, np.log(total), np.nan).astype(np.float32)
    ly = np.where(vol > 0, np.log(vol), np.nan).astype(np.float32)
del total, vol
both = np.isfinite(lx) & np.isfinite(ly)
mx = np.array([np.median(lx[both[:, d], d]) if both[:, d].sum() > 50 else np.nan for d in range(T)])
my = np.array([np.median(ly[both[:, d], d]) if both[:, d].sum() > 50 else np.nan for d in range(T)])
vc_plain = np.load(C.WORK / tag / 'vcorr.npy')
vc_dm = np.full((N, T), np.nan, np.float32)
for a in range(0, N, 1000):
    b = min(N, a + 1000)
    vc_dm[a:b] = M.rolling_corr((lx[a:b] - mx[None, :]).astype(np.float64), (ly[a:b] - my[None, :]).astype(np.float64), M.VC_W, M.VC_MIN)
np.save(C.WORK / tag / 'vcorr_dm.npy', vc_dm)
z = np.load(C.WORK / tag / 'universe.npz')
uni, rebal = z['uni'], z['rebal']
for lab, a, b in (('2009-2015', '2009-08-03', '2015-12-31'), ('2016-2019', '2016-01-01', '2019-12-31'),
                  ('2020-2025-06', '2020-01-01', '2025-06-27'), ('2025-07..2026-02', '2025-06-30', '2026-02-27'),
                  ('2026-03..05', '2026-03-01', '2026-05-31'), ('2026-06..10', '2026-06-01', '2026-10-19')):
    ds = [d for d in range(T) if a <= iso[d] <= b]
    p, q = [], []
    for d in ds:
        k = C.kidx(rebal, d)
        u = np.nonzero(uni[k])[0]
        p.append(vc_plain[u, d]); q.append(vc_dm[u, d])
    p, q = np.concatenate(p), np.concatenate(q)
    fp, fq = p[np.isfinite(p)], q[np.isfinite(q)]
    print(f'{lab:18s} plain: median {np.median(fp):.3f}, <0.2 {np.mean(fp < 0.2):.2%}, <0.3 {np.mean(fp < 0.3):.2%} | '
          f'demeaned: median {np.median(fq):.3f}, <0.2 {np.mean(fq < 0.2):.2%}, <0.3 {np.mean(fq < 0.3):.2%}; both <0.3 {np.mean((p < 0.3) & (q < 0.3)):.2%}')
# the known wrong matches on rebalance days: plain vs demeaned
tk = list(E['tickers'])
for t, dte in (('B', '2009-11-02'), ('JCI', '2010-05-05'), ('WTW', '2012-01-03'), ('CB', '2010-08-04'), ('SAN', '2011-07-05'),
               ('ELV', '2010-05-05'), ('WOLF', '2011-02-02'), ('BRK-B', '2010-03-05'), ('AAPL', '2026-04-01'), ('XOM', '2026-04-01')):
    if t in tk and dte in iso:
        j, d = tk.index(t), iso.index(dte)
        print(t, dte, 'plain', round(float(vc_plain[j, d]), 2), 'demeaned', round(float(vc_dm[j, d]), 2))
