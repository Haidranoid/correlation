"""Diagnostic added after the fidelity check (checks_old.txt) and before any out-of-sample result: why the in-sample
slope on 1993-2007 is near zero when BTZ's VIX-with-daily-RV variant gave t 2.34 on 1990-2007. Reported only; no
rule changes.

    python3 -I -B fidelity_diag.py        -> fidelity_diag.txt

Starts from BTZ's own convention as closely as daily data allow (A), then changes one thing at a time toward this
plan's tradeable convention (D):
  A. calendar month-end origins; RV = 10,000 x the sum of squared daily log returns inside calendar month t;
     IV = VIX^2 / 12 on the month's last trading day; the target is the log excess return from the close of the
     origin over the next k calendar months (no one-day delay);
  B. as A, with the plan's one-day delay: the target runs from the close of t + 1 to the close one day after the
     k-th month-end;
  C. as B, with the plan's RV (the 21 trading days ending at t) instead of the calendar month's;
  D. the plan's grid: origins every 21 trading days, the plan's RV and delay, k periods of 21 days (= checks_old.txt).
Hodrick (1992) 1B t-statistics throughout; slope in BTZ's units (annualized % per monthly %^2).
Returns: French's market (Mkt-RF + RF, CRSP value-weighted, from 1990) and Yahoo's SPY (from 1993-02).
"""
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/variance_risk_premium')
import numpy as np  # noqa: E402

import vrp_common as V  # noqa: E402

L = []
p = L.append


def series_market():
    fd, ff = V.pubdata._read(V.FF3_CSV)
    vd, vv = V._read_fred(V.FRED['vix'])
    m = (fd >= np.datetime64('1990-01-02')) & (fd <= np.datetime64('2008-06-30'))
    ax = fd[m]
    r = np.log1p(ff['Mkt-RF'][m] + ff['RF'][m])
    vix, _ = V.fill_back(V._align(vd, vv, ax), ax, vd[-1])
    return ax.astype(str), r, ff['RF'][m], vix


def series_spy():
    D = V.load('old', verbose=False)
    m = D['dates'] <= '2008-06-30'
    return D['dates'][m], V.logret(D['spy'])[m], D['rf'][m], D['vix'][m]


def run(lab, dates, r, rf, vix, lo, hi, conv):
    T = len(dates)
    ex = r - np.log1p(rf)
    t_idx = np.arange(T)
    rv21 = 1e4 * V.window_sum(r ** 2, t_idx - 20, t_idx)
    if conv == 'D':
        a = int(np.searchsorted(dates, V.CAL_ANCHOR))
        assert dates[a] == V.CAL_ANCHOR
        orig = np.arange(a % 21, T, 21)
    else:
        ym = np.array([d[:7] for d in dates])
        orig = np.nonzero(ym[:-1] != ym[1:])[0]                 # last trading day of each month
    orig = orig[(dates[orig] >= lo) & (dates[orig] <= hi)]
    M = len(orig)
    x = np.full(M, np.nan)
    r1 = np.full(M, np.nan)
    shift = 0 if conv == 'A' else 1
    for m_, t in enumerate(orig):
        if conv == 'A' or conv == 'B':
            prev = orig[m_ - 1] if m_ > 0 else None
            if prev is None:
                ym0 = dates[t][:7]
                k = np.nonzero(np.array([d[:7] for d in dates]) == ym0)[0]
                lo_i = k[0]
            else:
                lo_i = prev + 1
            rvm = 1e4 * np.sum(r[lo_i:t + 1] ** 2) if np.isfinite(r[lo_i:t + 1]).all() and lo_i > 0 else np.nan
        else:
            rvm = rv21[t]
        x[m_] = vix[t] ** 2 / 12 - rvm
        nxt = orig[m_ + 1] if m_ + 1 < M else (t + 21 if conv == 'D' else None)
        if nxt is None or nxt + shift >= T:
            continue
        if dates[nxt + shift] > hi:                              # targets end inside the sample
            continue
        r1[m_] = ex[t + 1 + shift:nxt + 1 + shift].sum()
    res = []
    for k in (1, 3):
        hk = V.hodrick_1b(x, r1, k)
        res.append(f"k={k}: {hk['beta'] * 100 * 12 / k:.2f} (t {hk['t']:.2f}), {100 * hk['adj_r2']:.2f}%")
    ok = np.isfinite(x)
    p(f"| {lab} | {conv} | {lo}..{hi} | {M} | {np.nanmean(x):.2f} | {V.corr(x[ok][1:], x[ok][:-1]):.2f} | {res[0]} | {res[1]} |")


p('| Returns | Convention | Origins from..to | n | Mean VRP | AR(1) VRP | 1 period: slope (t), adj R2 | 3 periods: slope (t), adj R2 |')
p('|---|---|---|---|---|---|---|---|')
mk = series_market()
sp = series_spy()
for conv in ('A', 'B', 'C', 'D'):
    if conv != 'D':
        run("French's market", *mk, '1990-01-01', '2007-12-31', conv)
    run("French's market", *mk, '1993-03-01', '2007-12-31', conv)
    run('Yahoo SPY', *sp, '1993-03-01', '2007-12-31', conv)
p("\nBTZ, 1990-2007: 5-minute RV, 1 month 0.39 (1.76) 1.07%, 3 months 0.47 (2.86) 6.82%; VIX with daily RV, 3 months t 2.34, 4.45%. "
  "Table 1: IV 33.23, RV 14.93, VRP 18.30; AR(1) VRP 0.49. "
  "A: BTZ's convention; B: + the one-day delay; C: + the plan's trailing 21-day RV; D: + the plan's 21-day grid (the plan).")
txt = '\n'.join(L)
print(txt)
(V.HERE / 'fidelity_diag.txt').write_text(txt + '\n')
