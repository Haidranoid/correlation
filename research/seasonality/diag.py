"""Diagnostics (not tests): Heston & Sadka's lag profile and a plumbing check of the book arithmetic.

    python3 -I -B diag.py old      -> diag_old.txt

1. Lag profile: for each holding month of 2007-05..2019-12 (and 2020-01..2025-06), the cross-sectional OLS slope of
   the universe stocks' holding log return g[:, i] on their monthly log return r[:, i - l], l = 1..120 (univariate,
   Fama-MacBeth). Mean slope x 100 and t, for annual lags (multiples of 12) and the other lags, by lag-year.
2. Plumbing: the same decile-book arithmetic on 1-month reversal (sort on -r[:, i-1]) and on 12-2 momentum
   (mean of r[:, i-12..i-2], lags 2..12), gross, in the universe.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import np, WORK, HERE, tstat, month_label  # noqa: E402

tag = sys.argv[1]
z = np.load(WORK / f'panel_{tag}.npz')
months = z['months']
r = z['r'].astype(np.float64)
g = z['g'].astype(np.float64)
uni_f = z['uni_f']
M = len(months)
lines = []


def P(s=''):
    print(s, flush=True)
    lines.append(s)


def idx(a, b):
    ya, ma = map(int, a.split('-'))
    yb, mb = map(int, b.split('-'))
    return [i for i in range(1, M) if ya * 12 + ma - 1 <= months[i] <= yb * 12 + mb - 1]


def slope(y, x):
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 50:
        return np.nan
    y, x = y[ok], x[ok]
    # winsorize both at 1% / 99% so a few extreme months do not dominate a single slope
    lo, hi = np.percentile(x, [1, 99])
    x = np.clip(x, lo, hi)
    lo, hi = np.percentile(y, [1, 99])
    y = np.clip(y, lo, hi)
    xc = x - x.mean()
    return float((xc @ (y - y.mean())) / (xc @ xc))


for lab, a, b in (('2007-05..2019-12', '2007-05', '2019-12'), ('2020-01..2025-06', '2020-01', '2025-06')):
    I = idx(a, b)
    S = np.full((len(I), 121), np.nan)
    for n, i in enumerate(I):
        m = np.nonzero(uni_f[i])[0]
        for l in range(1, 121):
            if i - l >= 1:
                S[n, l] = slope(g[m, i], r[m, i - l])
    P(f'== Lag profile, {lab}: mean Fama-MacBeth slope x 100 (t), univariate, universe stocks (winsorized 1/99) ==')
    for k in range(1, 11):
        ann = S[:, 12 * k]
        oth = np.nanmean(S[:, 12 * (k - 1) + 1:12 * k], 1)
        ma, ta, na = tstat(ann)
        mo, to, no = tstat(oth)
        P(f'  lag-year {k:2d}: annual lag {12 * k:3d}: {100 * ma:+.2f} ({ta:+.2f}), n {na};   mean of the other 11 lags: '
          f'{100 * mo:+.2f} ({to:+.2f})')
    ann_all = np.nanmean(S[:, [12 * k for k in range(1, 11)]], 1)
    ann25 = np.nanmean(S[:, [12 * k for k in range(2, 6)]], 1)
    oth_all = np.nanmean(S[:, [l for l in range(13, 121) if l % 12]], 1)
    for nm, v in (('mean of annual lags 12..120', ann_all), ('mean of annual lags 24..60 (years 2-5)', ann25),
                  ('mean of non-annual lags 13..119 (years 2-10)', oth_all)):
        m_, t_, n_ = tstat(v)
        P(f'  {nm}: {100 * m_:+.3f} ({t_:+.2f}), n {n_}')
    m1, t1, _ = tstat(S[:, 1])
    P(f'  lag 1: {100 * m1:+.2f} ({t1:+.2f})')
    P()


def book_gross(sigfun, a, b):
    out = []
    for i in idx(a, b):
        s = sigfun(i)
        m = np.nonzero(uni_f[i] & np.isfinite(s))[0]
        if len(m) < 100:
            continue
        n = len(m)
        rk = np.argsort(np.argsort(s[m], kind='mergesort'), kind='mergesort')
        Sx, Lx = m[rk < n * 0.1], m[rk >= n * 0.9]
        out.append(np.expm1(g[Lx, i]).mean() - np.expm1(g[Sx, i]).mean())
    return np.array(out)


def mom(i):
    with np.errstate(all='ignore'):
        x = np.nanmean(r[:, i - 12:i - 1], 1)
        n = np.isfinite(r[:, i - 12:i - 1]).sum(1)
    x[n < 8] = np.nan
    return x


P('== Plumbing: the same decile arithmetic on known signals, gross %/month (t) ==')
for lab, a, b in (('2007-05..2019-12', '2007-05', '2019-12'), ('2020-01..2025-06', '2020-01', '2025-06')):
    rev = book_gross(lambda i: -r[:, i - 1], a, b)
    mo = book_gross(mom, a, b)
    P(f'  {lab}: 1-month reversal {100 * rev.mean():+.3f} ({tstat(rev)[1]:+.2f}), n {len(rev)};  '
      f'12-2 momentum {100 * mo.mean():+.3f} ({tstat(mo)[1]:+.2f}), n {len(mo)}')
(HERE / f'diag_{tag}.txt').write_text('\n'.join(lines) + '\n')
