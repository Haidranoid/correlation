"""Robustness and cross-checks: winsorized look-ahead partials, Stooq-vs-Coinbase crypto alignment,
Fear & Greed labelling vs BTC, Stooq US 10y vs FRED DGS10."""
import csv, os, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import dmlib
from probe import load, to_days, moves, partial, corr, LO, HI

def wins(a, q=0.01):
    m = np.isfinite(a)
    lo, hi = np.quantile(a[m], [q, 1 - q])
    return np.clip(a, lo, hi)

cat = {r['id']: r for r in csv.DictReader(open(dmlib.CATALOG))}
sd, sv = load('etf_spy'); days = [d for d in sd if LO <= d <= HI]
spy = wins(moves(to_days(sd, sv, days), 'logdiff'))
gd, gv = load('fred_dgs10'); g10 = wins(moves(to_days(gd, gv, days), 'diff'))
print('winsorized (1%/99%) look-ahead partials: id  spy_pc1  dgs10_c0 dgs10_cm1 dgs10_pc1')
for sid in ['stooq_de10y','stooq_uk10y','stooq_fr10y','stooq_it10y','stooq_es10y','stooq_it2y','stooq_de2y','stooq_uk2y','stooq_jp10y','stooq_ca10y','stooq_mx10y','stooq_tsx','stooq_bubor3m','stooq_set','stooq_shibor3m','etf_ief','etf_iwm','etf_xly','etf_ura','stooq_crb','fred_t10yie','fred_dfii10']:
    d, v = load(sid); x = wins(moves(to_days(d, v, days), cat[sid]['transform']))
    print(f"  {sid:<18} {partial(x[:-1], spy[1:], spy[:-1]):+.3f}   {corr(x, g10)[0]:+.2f}  {corr(x[1:], g10[:-1])[0]:+.2f}  {partial(x[:-1], g10[1:], g10[:-1]):+.3f}")
# TSX by sub-period
d, v = load('stooq_tsx'); x = moves(to_days(d, v, days), 'logdiff'); s_raw = moves(to_days(sd, sv, days), 'logdiff')
dd = np.array(days[1:])
for a, b in [('2005', '2010'), ('2010', '2015'), ('2015', '2020'), ('2020', '2026')]:
    m = (dd >= a) & (dd < b)
    xi, si = x[m], s_raw[m]
    print(f"  tsx {a}-{b}: c0={corr(xi, si)[0]:+.2f} pc1={partial(xi[:-1], si[1:], si[:-1]):+.3f}")

# crypto: Stooq (UTC-day bars) vs Coinbase on FRED ('as of 5 PM PST')
print('\nStooq vs Coinbase (daily log moves on common dates; lag k = Stooq shifted by k days):')
for st, cb in [('stooq_btc', 'fred_cbbtcusd'), ('stooq_eth', 'fred_cbethusd'), ('stooq_ltc', 'fred_cbltcusd'), ('stooq_bch', 'fred_cbbchusd')]:
    d1, v1 = load(st); d2, v2 = load(cb)
    a = dict(zip(d1, v1)); b = dict(zip(d2, v2))
    common = sorted(set(a) & set(b))
    common = [c for c in common if c >= '2017-01-01']
    res = []
    import datetime as dt
    for k in (-1, 0, 1):
        xs, ys = [], []
        for c in common:
            c0 = dt.date.fromisoformat(c)
            p0, p1 = (c0 - dt.timedelta(days=1)).isoformat(), c0.isoformat()
            q0, q1 = (c0 + dt.timedelta(days=k - 1)).isoformat(), (c0 + dt.timedelta(days=k)).isoformat()
            if p0 in b and p1 in b and q0 in a and q1 in a:
                ys.append(np.log(b[p1] / b[p0])); xs.append(np.log(a[q1] / a[q0]))
        res.append((k, np.corrcoef(xs, ys)[0, 1], len(xs)))
    ratio = np.array([a[c] / b[c] for c in common])
    print(f"  {st} vs {cb}: " + ", ".join(f"lag{k:+d} r={r:.3f}" for k, r, n in res) + f"; n={res[1][2]}; median |price ratio-1| same date = {np.median(np.abs(ratio - 1)):.4f}")

# Fear & Greed labelling: F&G(d) vs BTC UTC-day returns of day d and d-1
d1, v1 = load('stooq_btc'); a = dict(zip(d1, v1))
d2, v2 = load('altme_crypto_fear_greed'); b = dict(zip(d2, v2))
import datetime as dt
for k in (-2, -1, 0, 1):
    xs, ys = [], []
    for c in d2:
        c0 = dt.date.fromisoformat(c)
        p = (c0 - dt.timedelta(days=1)).isoformat()
        q0, q1 = (c0 + dt.timedelta(days=k - 1)).isoformat(), (c0 + dt.timedelta(days=k)).isoformat()
        if p in b and q0 in a and q1 in a:
            ys.append(b[c] - b[p]); xs.append(np.log(a[q1] / a[q0]))
    print(f"F&G change(d) vs BTC return of UTC day d{k:+d}: r={np.corrcoef(xs, ys)[0,1]:+.3f} n={len(xs)}")

# Stooq US 10y vs FRED DGS10 levels
z = zipfile.ZipFile('/run/media/bhoult/My Book/backups/popos-cosmic-2025/bhoult/programming/correlation/d_world_txt.zip')
L = z.read('data/daily/world/bonds/10yusy.b.txt').decode().splitlines()[1:]
s = {dmlib.ymd(l.split(',')[2]): float(l.split(',')[7]) for l in L}
gd, gv = load('fred_dgs10'); g = dict(zip(gd, gv))
com = [c for c in sorted(set(s) & set(g)) if '2005-01-01' <= c <= '2025-06-27']
diff = np.array([s[c] - g[c] for c in com])
print(f"\nStooq 10yusy - DGS10 on {len(com)} common dates: exact (|d|<0.005) {np.mean(np.abs(diff)<0.005):.1%}, median |d| {np.median(np.abs(diff)):.3f}, mean d {diff.mean():+.4f}")
gm = {c: g[c] for c in com}
