"""Plumbing checks (no outcomes): the rolling Dimson betas against a direct OLS on random stock-days; point in time
(the betas at t are unchanged when every price after t is replaced); the legs add up to the close-to-close return;
the betas' distribution in the universe.

    python3 -I checks.py old
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np = C.np
c2eval = C.c2eval

tag = sys.argv[1] if len(sys.argv) > 1 else 'old'
P = C.load(tag)
E = c2eval.load_export(C.EXPORTS[tag], arrays=('close',))
close = E['close']
m = P['m']
N, T = close.shape
rng = np.random.default_rng(7)
# 1. direct OLS
errs = []
picked = 0
while picked < 200:
    j, t = int(rng.integers(N)), int(rng.integers(600, T))
    if not np.isfinite(P['blag'][j, t]):
        continue
    c = np.asarray(close[j, t - C.W - 1:t + 1], np.float64)
    r = np.log(c[1:] / c[:-1])                      # r at tau = t-W .. t
    mm = m[t - C.W:t + 1]
    m1 = m[t - C.W - 1:t]
    r, mm, m1 = r[1:], mm[1:], m1[1:]                # tau = t-W+1 .. t
    ok = np.isfinite(r) & (np.abs(r) <= C.MAX_LOG) & np.isfinite(mm) & np.isfinite(m1)
    X = np.column_stack([np.ones(ok.sum()), mm[ok], m1[ok]])
    b = np.linalg.lstsq(X, r[ok], rcond=None)[0]
    errs.append(max(abs(b[1] - P['b0'][j, t]), abs(b[2] - P['blag'][j, t])))
    picked += 1
print('1. direct OLS vs rolling: max abs diff over 200 stock-days', max(errs))
# 2. point in time: recompute on a truncated panel (prices after t0 replaced by NaN, m after t0 NaN)
t0 = c2eval.day_index(E['days'], '2012-06-29')
rows = rng.choice(N, 300, replace=False)
sub = np.asarray(close[rows], np.float64).copy()
sub[:, t0 + 1:] = np.nan
mt = m.copy(); mt[t0 + 1:] = np.nan
b0t, blt = C.dimson(sub, mt)
d = np.nanmax(np.abs(blt[:, :t0 + 1] - np.asarray(P['blag'][rows, :t0 + 1])))
same_nan = np.array_equal(np.isfinite(blt[:, :t0 + 1]), np.isfinite(np.asarray(P['blag'][rows, :t0 + 1])))
print('2. point in time: max |b_lag diff| through t0', d, 'same NaN pattern', same_nan)
# 3. legs add up to the store's close-to-close
S = c2eval.load_store(C.STORES[tag])
cs = c2eval.store_panel(S, E['ids'][rows], E['days'], 'close').astype(np.float64)
cc = np.full(cs.shape, np.nan); cc[:, 1:] = np.log(cs[:, 1:] / cs[:, :-1])
tot = np.asarray(P['on'][rows], np.float64) + np.asarray(P['intra'][rows], np.float64)
f = np.isfinite(tot)
print('3. legs sum vs close-to-close: max abs diff', float(np.nanmax(np.abs(tot[f] - cc[f]))))
# 4. distribution of betas in the universe, by year (2008+)
days = P['days']
yrs = np.array([int(c2eval.iso(x)[:4]) for x in days])
for y in range(2006, 2026):
    ts = np.nonzero(yrs == y)[0][::21]
    b0s, bls = [], []
    for t in ts:
        k = C.kidx(P['rebal'], t)
        if k < 0:
            continue
        u = P['uni'][k] & np.isfinite(np.asarray(P['blag'][:, t]))
        b0s.append(np.asarray(P['b0'][u, t])); bls.append(np.asarray(P['blag'][u, t]))
    if not b0s:
        continue
    b0s, bls = np.concatenate(b0s), np.concatenate(bls)
    print(f'4. {y}: universe stock-days {len(b0s)}, b0 median {np.median(b0s):.3f}, b_lag median {np.median(bls):+.3f} '
          f'(p10 {np.quantile(bls, .1):+.3f}, p90 {np.quantile(bls, .9):+.3f}), corr(b0, b_lag) {np.corrcoef(b0s, bls)[0, 1]:+.2f}')
uni_n = P['uni'].sum(1)
big_n = P['big'].sum(1)
for y in (2008, 2012, 2016, 2020, 2024):
    k = C.kidx(P['rebal'], c2eval.day_index(days, f'{y}-01-02'))
    print(f'5. universe on {c2eval.iso(days[P["rebal"][k]])}: {uni_n[k]} stocks, 500 largest {big_n[k]}')
