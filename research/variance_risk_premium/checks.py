"""Data checks and fidelity to BTZ, run before any out-of-sample result (plan, "Code checks and fidelity").

    python3 -I -B checks.py old        -> checks_old.txt, checks_old.json

1. Yahoo's adjusted SPY against the store's SPY: daily log returns 2005-02-28..2025-06-27, their correlation, the
   largest gaps, and ex-dividend days before and after the store stopped adjusting dividends (late 2022).
2. The filled VIX/VXN/RVX days, the risk-free fills, the calendar's dates around the first fit.
3. Fidelity, in sample, on BTZ's period: the VRP regression on -3.2's calendar origins 1993-03..2007-12 whose
   targets end by 2007-12-31, at 21 and 63 trading days, with Hodrick (1992) 1B t-statistics, slope in BTZ's units
   (annualized percent excess return per monthly percent squared). Means of IV, RV and VRP, and the VRP's
   first-order autocorrelation on that grid, against BTZ's Table 1. Not pass/fail.
"""
import json
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/variance_risk_premium')
import numpy as np  # noqa: E402

import vrp_common as V  # noqa: E402

tag = sys.argv[1]
assert tag == 'old'
D = V.load(tag)
dates, T = D['dates'], D['T']
res = {'axis': [dates[0], dates[-1], T], 'export_days': D.get('export_days'), 'fills': D['fills'],
       'store_fills': D['store_fills'], 'rf_fill_inside_french': D['rf_fill_inside_french'],
       'rf_fill_after_french': D['rf_fill_after_french']}
L = []
p = L.append
p(f"Checks ({tag}): day axis {dates[0]}..{dates[-1]}, {T} Yahoo SPY days; the export's {D.get('export_days')} days equal the axis over its span "
  f"(asserted), and export day 252 is {V.CAL_ANCHOR}.")
p(f"Calendar: {len(D['cal'])} dates on the axis; around the first fit: "
  + ', '.join(dates[c] for c in D['cal'] if '2007-10-01' <= dates[c] <= '2008-02-28'))
p(f"Filled from the previous day (<= 3 back): {D['fills']}")
p(f"Store closes forward-filled: {D['store_fills']}")
p(f"Risk-free: French's RF through {D['rf_french_last']}; days inside French's span without a value (DGS3MO used): "
  f"{D['rf_fill_inside_french'] or 'none'}; after it: {D['rf_fill_after_french']}")

# ---- 1. Yahoo vs store SPY ----------------------------------------------------------------------------------------
ry, rs = V.logret(D['spy']), V.logret(D['spy_store'])
m = (dates >= '2005-02-28') & np.isfinite(ry) & np.isfinite(rs)
fixed = set(D['store_fills']['SPY'])
m2 = m & ~np.isin(dates, list(fixed)) & ~np.isin(np.roll(dates, 1), list(fixed))
dif = rs - ry
dd, cc = V.pubdata._read(V.ETF_DIR / 'SPY.csv')
exd = set(dd[cc['dividends'] > 0].astype(str))
isx = np.isin(dates, list(exd))
q = {'days': int(m.sum()), 'corr': V.corr(ry[m], rs[m]), 'corr_without_filled_day': V.corr(ry[m2], rs[m2]),
     'mean_abs_diff_bp': float(1e4 * np.mean(np.abs(dif[m2]))),
     'largest': [(dates[i], round(1e4 * dif[i], 1)) for i in np.argsort(-np.abs(np.where(m2, dif, 0)))[:8]]}
for lab, lo, hi in (('ex-div days 2005..2022-06', '2005-02-28', '2022-06-30'), ('ex-div days 2022-07..2022-12-19', '2022-07-01', '2022-12-19'),
                    ('ex-div days 2022-12-20..', '2022-12-20', '2099')):
    mm = m2 & isx & (dates >= lo) & (dates <= hi)
    q[lab] = {'n': int(mm.sum()), 'mean_store_minus_yahoo_bp': float(1e4 * dif[mm].mean()) if mm.any() else float('nan')}
mm = m2 & ~isx
q['other days: mean |diff| bp'] = float(1e4 * np.mean(np.abs(dif[mm])))
yrs = {}
for y in range(2005, 2026):
    k = m2 & (np.char.startswith(dates.astype(str), str(y)))
    yrs[str(y)] = round(100 * (np.exp(ry[k].sum()) - np.exp(rs[k].sum())), 2)
q['yearly total return, Yahoo minus store, points'] = yrs
res['spy_yahoo_vs_store'] = q
p(f"\nYahoo vs store SPY, daily log returns {dates[m][0]}..{dates[m][-1]} ({q['days']} days): correlation {q['corr']:.6f} "
  f"({q['corr_without_filled_day']:.6f} without the store's filled day and the day after); mean |difference| {q['mean_abs_diff_bp']:.2f} bp")
p(f"  largest differences (store - Yahoo, bp): {q['largest']}")
for lab in ('ex-div days 2005..2022-06', 'ex-div days 2022-07..2022-12-19', 'ex-div days 2022-12-20..'):
    p(f"  {lab}: {q[lab]['n']} days, mean store - Yahoo {q[lab]['mean_store_minus_yahoo_bp']:+.1f} bp")
p(f"  other days: mean |difference| {q['other days: mean |diff| bp']:.2f} bp")
p(f"  yearly total return, Yahoo minus store (points): {yrs}")

# ---- 3. Fidelity --------------------------------------------------------------------------------------------------
S = V.build(D)
cal = D['cal']
y21 = V.target(S['r'], D['rf'], 21)
gm = (dates[cal] >= V.TRAIN_FROM)
c = cal[gm]
x = S['vrp'][c]
r1 = y21[c].copy()
# only targets ending by 2007-12-31
end_ok = np.array([dates[t + 22] <= '2007-12-31' if t + 22 < T else False for t in c])
r1[~end_ok] = np.nan
fid = {}
for k, h in ((1, 21), (3, 63)):
    xx = x.copy()
    xx[np.array([not (m_ + k - 1 < len(c) and end_ok[m_ + k - 1]) for m_ in range(len(c))])] = np.nan
    hk = V.hodrick_1b(xx, r1, k)
    scale = 100 * 252 / h                                     # annualized percent
    fid[h] = {'beta_btz_units': hk['beta'] * scale, 't_hodrick': hk['t'], 'adj_r2': hk['adj_r2'], 'n': hk['n']}
grid = gm.copy()
inper = (dates[cal] >= V.TRAIN_FROM) & (dates[cal] <= '2007-12-31')
cc_ = cal[inper]
iv, rvv, vrp = S['iv'][cc_], S['rv'][cc_], S['vrp'][cc_]
okv = np.isfinite(vrp)
ac = V.corr(vrp[okv][1:], vrp[okv][:-1])
fid['means'] = {'iv': float(np.nanmean(iv)), 'rv': float(np.nanmean(rvv)), 'vrp': float(np.nanmean(vrp)), 'n': int(okv.sum()),
                'vrp_ar1_calendar': ac, 'share_vrp_negative': float(np.mean(vrp[okv] < 0)),
                'iv_sd': float(np.nanstd(iv)), 'rv_sd': float(np.nanstd(rvv)), 'vrp_sd': float(np.nanstd(vrp))}
dmask = (dates >= V.TRAIN_FROM) & (dates <= '2007-12-31') & np.isfinite(S['vrp'])
v = S['vrp']
fid['vrp_daily_ar1_1993_2007'] = V.corr(v[1:][dmask[1:] & dmask[:-1]], v[:-1][dmask[1:] & dmask[:-1]])
res['fidelity'] = fid
p('\nFidelity, in sample, on -3.2\'s calendar origins (every 21 trading days) from 1993-03 with targets ending by 2007-12-31; '
  'slope in BTZ\'s units, Hodrick (1992) 1B t:')
p('| Horizon | Origins | Slope | Hodrick t | Adj. R2 | BTZ, 5-minute RV (Table 2) | BTZ, VIX with daily RV (4.2.1) |')
p('|---|---|---|---|---|---|---|')
p(f"| 21 days | {fid[21]['n']} | {fid[21]['beta_btz_units']:.2f} | {fid[21]['t_hodrick']:.2f} | {100 * fid[21]['adj_r2']:.2f}% | 0.39 (1.76), 1.07% | not reported |")
p(f"| 63 days | {fid[63]['n']} | {fid[63]['beta_btz_units']:.2f} | {fid[63]['t_hodrick']:.2f} | {100 * fid[63]['adj_r2']:.2f}% | 0.47 (2.86), 6.82% | t 2.34, 4.45% |")
mm_ = fid['means']
p(f"Means 1993-03..2007-12 on the calendar ({mm_['n']} dates): IV {mm_['iv']:.2f} (sd {mm_['iv_sd']:.2f}), RV {mm_['rv']:.2f} (sd {mm_['rv_sd']:.2f}), "
  f"VRP {mm_['vrp']:.2f} (sd {mm_['vrp_sd']:.2f}); BTZ 1990-2007: 33.23 (23.73), 14.93 (15.25), 18.30 (15.13). "
  f"VRP first-order autocorrelation on the 21-day grid {mm_['vrp_ar1_calendar']:.2f} (BTZ monthly 0.49); share negative {mm_['share_vrp_negative']:.3f}")
p(f"VRP daily first-order autocorrelation 1993-03-02..2007-12-31 (the noise calibration's phi): {fid['vrp_daily_ar1_1993_2007']:.4f}")
txt = '\n'.join(L)
print(txt)
(V.HERE / f'checks_{tag}.txt').write_text(txt + '\n')
with open(V.HERE / f'checks_{tag}.json', 'w') as f:
    json.dump(res, f, indent=1, default=str)
