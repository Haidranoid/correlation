"""The plan's tests and reported numbers on data through 2025-06-30 (writes results_old.txt and results_old.json).

  Test 1  2008-01..2019-12: mean monthly excess return net of 2 bps; pass t >= 2 (iid t; Newey-West 6 lags reported).
  Test 2  calendar 2008 and 2022: compounded net excess return positive in both.
  Test 3  2020-01..2025-06: mean positive (the unseen months are in unseen.py, once, at the end).
  Number  CAGR of the account (T-bill + overlay) at 0 bps minus the median CAGR of 1,000 random-sign portfolios,
          2008-01..2025-06 (seed common.SEED).
Also: Sharpe, max drawdown, correlation with SPY and 60/40 SPY/IEF, contribution by kind, gross exposure, years,
FF5 + UMD alphas, AQR's TSMOM over the same windows, costs at 0/2/5/20 bps.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import common as C  # noqa: E402

P = C.build(C.END_MAIN)
K, F = P['W'].shape
gross = P['overlay_gross']
netx, cf_net = C.net(P, C.COST_BPS)
rfm = np.nan_to_num(P['rfm'])
lab = P['label']
ad, av = C.pubdata.aqr_tsmom_monthly()
aqr = C.calendar_month_join(lab, ad, av['TSMOM'])
iS, iI = C.TICKERS.index('SPY'), C.TICKERS.index('IEF')
spy_x = P['XR'][:, iS]
sixty = 0.6 * P['R'][:, iS] + 0.4 * P['R'][:, iI] - P['rfm']
fd, fv = C.pubdata.ff5_monthly()
ud, uv = C.pubdata.umd_monthly()
fac = np.stack([C.calendar_month_join(lab, fd, fv[c]) for c in ('Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA')]
               + [C.calendar_month_join(lab, ud, uv['UMD'])], 1)

out = []; J = {}
pr = lambda s='': out.append(s)  # noqa: E731

PERIODS = [('2006-2019', '2006-01', '2019-12'), ('2008-2019', '2008-01', '2019-12'), ('2009-2019 (no 2008)', '2009-01', '2019-12'),
           ('2020-2025-06', '2020-01', '2025-06'), ('2008-2025-06', '2008-01', '2025-06'), ('2009-2025-06 (no 2008)', '2009-01', '2025-06')]


def stats(x, m):
    v = x[m]
    return {'months': int(m.sum()), 'mean_pct_month': 100 * v.mean(), 'ann_pct': 1200 * v.mean(), 't': C.tstat(v),
            'nw_t': C.nw_t(v), 'sharpe': C.sharpe(v), 'vol_pct': 100 * v.std(ddof=1) * np.sqrt(12),
            'max_dd_pct': 100 * C.max_drawdown(v), 'compound_pct': 100 * (np.prod(1 + v) - 1)}


def alpha(x, m):
    ok = m & np.isfinite(fac).all(1)
    X = np.c_[np.ones(ok.sum()), fac[ok]]
    y = x[ok]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    s2 = e @ e / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return {'months': int(ok.sum()), 'alpha_pct_month': 100 * b[0], 'alpha_ann_pct': 1200 * b[0], 't': b[0] / se[0],
            'betas': dict(zip(('Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD'), [round(float(z), 3) for z in b[1:]]))}


# --- Test 1 -----------------------------------------------------------------------------------------------------
pr('== Test 1: 2008-01..2019-12, mean monthly excess return net of 2 bps a side; pass t >= 2')
m1 = C.window(P, '2008-01', '2019-12')
t1 = stats(netx, m1); J['test1'] = t1
pr(f"   mean {t1['mean_pct_month']:+.3f}%/month ({t1['ann_pct']:+.2f}%/yr), t {t1['t']:+.2f} (Newey-West 6 lags {t1['nw_t']:+.2f}), "
   f"Sharpe {t1['sharpe']:.2f}, vol {t1['vol_pct']:.1f}%/yr, max drawdown {t1['max_dd_pct']:.1f}%")
pass1 = t1['t'] >= 2
pr(f"   -> {'PASS' if pass1 else 'FAIL'}")
m1b = C.window(P, '2009-01', '2019-12')
t1b = stats(netx, m1b); J['test1_no2008'] = t1b
pr(f"   without 2008 (2009-01..2019-12): mean {t1b['mean_pct_month']:+.3f}%/month, t {t1b['t']:+.2f}, Sharpe {t1b['sharpe']:.2f}")

# --- Test 2 -----------------------------------------------------------------------------------------------------
pr('\n== Test 2: crisis years, compounded net excess return (2 bps) positive in 2008 and 2022')
yrs = lab.astype('datetime64[Y]')
J['test2'] = {}
for y in ('2008', '2022'):
    m = yrs == np.datetime64(y, 'Y')
    J['test2'][y] = {'strategy_pct': 100 * (np.prod(1 + netx[m]) - 1), 'sum_pct': 100 * netx[m].sum(),
                     'spy_pct': 100 * (np.prod(1 + spy_x[m]) - 1), 'sixty40_pct': 100 * (np.prod(1 + sixty[m]) - 1),
                     'aqr_pct': 100 * (np.prod(1 + aqr[m]) - 1)}
    d = J['test2'][y]
    pr(f"   {y}: strategy {d['strategy_pct']:+.2f}% (sum of months {d['sum_pct']:+.2f}%); SPY excess {d['spy_pct']:+.1f}%, "
       f"60/40 excess {d['sixty40_pct']:+.1f}%, AQR TSMOM {d['aqr_pct']:+.1f}%")
pass2 = all(J['test2'][y]['strategy_pct'] > 0 for y in ('2008', '2022'))
pr(f"   -> {'PASS' if pass2 else 'FAIL'}")

# --- Test 3 -----------------------------------------------------------------------------------------------------
pr('\n== Test 3: 2020-01..2025-06, same (positive) sign; the unseen months come at the end')
m3 = C.window(P, '2020-01', '2025-06')
t3 = stats(netx, m3); J['test3'] = t3
pr(f"   mean {t3['mean_pct_month']:+.3f}%/month ({t3['ann_pct']:+.2f}%/yr), t {t3['t']:+.2f}, Sharpe {t3['sharpe']:.2f}, "
   f"max drawdown {t3['max_dd_pct']:.1f}%")
pass3 = t3['mean_pct_month'] > 0
pr(f"   -> {'PASS' if pass3 else 'FAIL'} (unseen months pending)")

# --- The number -------------------------------------------------------------------------------------------------
pr('\n== The number: account CAGR (T-bill + overlay, 0 bps) minus the median of 1,000 random-sign portfolios')
rng = np.random.default_rng(C.SEED)
sg = rng.choice(np.array([-1.0, 1.0]), size=(1000, K, F))
XRz = np.where(P['ACT'], np.nan_to_num(P['XR']), 0.0)
rand_overlay = np.einsum('bkf,kf->bk', sg, P['WA'] * XRz)
rand_total = rfm[None, :] + rand_overlay
strat_total = rfm + gross
J['number'] = {'seed': C.SEED}
for name, a, b in (('2008-2025-06', '2008-01', '2025-06'), ('2008-2019', '2008-01', '2019-12'),
                   ('2020-2025-06', '2020-01', '2025-06'), ('2009-2025-06 (no 2008)', '2009-01', '2025-06')):
    m = C.window(P, a, b)
    sc = float(C.cagr(strat_total[m])); rc = C.cagr(rand_total[:, m], axis=1)
    med = float(np.median(rc))
    arith = 1200 * (gross[m].mean() - np.median(rand_overlay[:, m].mean(1)))
    pct = float((rc >= sc).mean())
    J['number'][name] = {'strategy_cagr_pct': 100 * sc, 'random_median_cagr_pct': 100 * med, 'points': 100 * (sc - med),
                         'points_arith': arith, 'p_random_ge': pct, 'random_p5_p95_pct': [100 * float(np.percentile(rc, 5)), 100 * float(np.percentile(rc, 95))]}
    d = J['number'][name]
    pr(f"   {name}: strategy {d['strategy_cagr_pct']:.2f}%/yr, random median {d['random_median_cagr_pct']:.2f}%/yr "
       f"(5-95%: {d['random_p5_p95_pct'][0]:.2f} .. {d['random_p5_p95_pct'][1]:.2f}) -> {d['points']:+.2f} points "
       f"(arithmetic {d['points_arith']:+.2f}); share of random portfolios at or above the strategy {d['p_random_ge']:.3f}")
NUMBER = J['number']['2008-2025-06']['points']
pr(f"   NUMBER FOR THE FILE NAME: {NUMBER:+.1f}")

# Added after seeing results (not a plan test): the same weights always long (MOP's "diversified passive long"),
# to show how much of the gap to random signs is the funds' own drift rather than timing.
pr('\n== Extra (added after results): always-long, same funds and weights, 0 bps')
long_overlay = (P['WA'] * XRz).sum(1)
J['passive_long'] = {}
for name, a, b in (('2008-2025-06', '2008-01', '2025-06'), ('2008-2019', '2008-01', '2019-12'), ('2020-2025-06', '2020-01', '2025-06')):
    m = C.window(P, a, b)
    lc = float(C.cagr((rfm + long_overlay)[m])); sc = float(C.cagr(strat_total[m]))
    rc = C.cagr(rand_total[:, m], axis=1)
    d = gross[m] - long_overlay[m]
    J['passive_long'][name] = {'long_cagr_pct': 100 * lc, 'long_minus_random_median_pts': 100 * (lc - float(np.median(rc))),
                               'p_random_ge_long': float((rc >= lc).mean()), 'strategy_minus_long_pts': 100 * (sc - lc),
                               'diff_t': C.tstat(d), 'long_sharpe': C.sharpe(long_overlay[m]), 'corr_strategy_long': float(np.corrcoef(gross[m], long_overlay[m])[0, 1])}
    e = J['passive_long'][name]
    pr(f"   {name}: always-long {e['long_cagr_pct']:.2f}%/yr ({e['long_minus_random_median_pts']:+.2f} points over the random median, "
       f"share of random at or above {e['p_random_ge_long']:.3f}; Sharpe {e['long_sharpe']:.2f}); strategy minus always-long "
       f"{e['strategy_minus_long_pts']:+.2f} points (monthly difference t {e['diff_t']:+.2f}); correlation {e['corr_strategy_long']:+.2f}")

# --- Reported: by period ----------------------------------------------------------------------------------------
pr('\n== By period (net of 2 bps unless stated); correlations of monthly excess returns')
J['periods'] = {}
for name, a, b in PERIODS:
    m = C.window(P, a, b)
    s = stats(netx, m); g = stats(gross, m)
    am = m & np.isfinite(aqr)
    s.update({'gross_ann_pct': g['ann_pct'], 'gross_t': g['t'], 'gross_sharpe': g['sharpe'],
              'corr_spy': float(np.corrcoef(netx[m], spy_x[m])[0, 1]), 'corr_6040': float(np.corrcoef(netx[m], sixty[m])[0, 1]),
              'corr_aqr': float(np.corrcoef(gross[am], aqr[am])[0, 1]),
              'aqr': stats(aqr, am) if am.sum() > 2 else None,
              'gross_exposure_mean': float(P['gross_exposure'][m].mean()), 'gross_exposure_max': float(P['gross_exposure'][m].max()),
              'turnover_month': float(P['turnover'][m].mean()), 'active_mean': float(P['n_active'][m].mean()),
              'net5_ann_pct': 1200 * C.net(P, 5)[0][m].mean(), 'net5_t': C.tstat(C.net(P, 5)[0][m]),
              'net20_ann_pct': 1200 * C.net(P, 20)[0][m].mean(), 'net20_t': C.tstat(C.net(P, 20)[0][m]),
              'alpha_ff5_umd': alpha(netx, m),
              'spy_sharpe': C.sharpe(spy_x[m]), 'sixty40_sharpe': C.sharpe(sixty[m])})
    J['periods'][name] = s
    a_ = s['aqr']
    pr(f"   {name}: {s['months']} months, {s['active_mean']:.1f} funds; net {s['ann_pct']:+.2f}%/yr t {s['t']:+.2f} "
       f"(gross {s['gross_ann_pct']:+.2f} t {s['gross_t']:+.2f}; 5 bps {s['net5_ann_pct']:+.2f} t {s['net5_t']:+.2f}; "
       f"20 bps {s['net20_ann_pct']:+.2f} t {s['net20_t']:+.2f}); Sharpe {s['sharpe']:.2f}, vol {s['vol_pct']:.1f}%, "
       f"max DD {s['max_dd_pct']:.1f}%; corr SPY {s['corr_spy']:+.2f}, 60/40 {s['corr_6040']:+.2f}, AQR {s['corr_aqr']:+.2f}; "
       f"gross exposure {s['gross_exposure_mean']:.2f} (max {s['gross_exposure_max']:.2f}), turnover {s['turnover_month']:.2f}/month")
    al = s['alpha_ff5_umd']
    pr(f"      FF5+UMD alpha {al['alpha_ann_pct']:+.2f}%/yr t {al['t']:+.2f} ({al['months']} months), betas {al['betas']}")
    if a_:
        pr(f"      AQR TSMOM same months: {a_['ann_pct']:+.2f}%/yr t {a_['t']:+.2f} Sharpe {a_['sharpe']:.2f}, vol {a_['vol_pct']:.1f}%; "
           f"SPY Sharpe {s['spy_sharpe']:.2f}, 60/40 Sharpe {s['sixty40_sharpe']:.2f}")

# --- Contribution by kind -----------------------------------------------------------------------------------------
pr('\n== Contribution by kind of fund (net of each fund\'s own 2 bps costs), %/yr and t')
J['kinds'] = {}
for name, a, b in PERIODS:
    m = C.window(P, a, b)
    row = {}
    for kind, tick in C.KINDS.items():
        idx = [C.TICKERS.index(t) for t in tick]
        x = cf_net[:, idx].sum(1)[m]
        row[kind] = {'ann_pct': 1200 * x.mean(), 't': C.tstat(x)}
    J['kinds'][name] = row
    pr(f"   {name}: " + '; '.join(f"{k} {v['ann_pct']:+.2f} (t {v['t']:+.2f})" for k, v in row.items()))

# --- Per fund, 2008-2025-06 ----------------------------------------------------------------------------------------
pr('\n== Per fund, 2008-01..2025-06: contribution %/yr (net), share of months long')
m = C.window(P, '2008-01', '2025-06')
J['funds'] = {}
for f, t in enumerate(C.TICKERS):
    x = cf_net[m, f]
    J['funds'][t] = {'ann_pct': 1200 * x.mean(), 't': C.tstat(x), 'long_share': float((P['S'][m, f] > 0).sum() / max(P['ACT'][m, f].sum(), 1))}
pr('   ' + '; '.join(f"{t} {v['ann_pct']:+.2f} ({v['long_share']:.0%} long)" for t, v in J['funds'].items()))

# --- Years --------------------------------------------------------------------------------------------------------
pr('\n== Calendar years: compounded excess return, % (strategy net of 2 bps; 2025 = Jan-Jun)')
pr('   year  strategy  gross   SPY    60/40   AQR TSMOM  funds')
J['years'] = {}
for y in range(2006, 2026):
    m = yrs == np.datetime64(str(y), 'Y')
    if not m.any():
        continue
    d = {k: 100 * (np.prod(1 + v[m]) - 1) for k, v in (('strategy', netx), ('gross', gross), ('spy', spy_x), ('sixty40', sixty), ('aqr', aqr))}
    d['funds'] = float(P['n_active'][m].mean())
    J['years'][y] = d
    pr(f"   {y}  {d['strategy']:+7.2f}  {d['gross']:+7.2f}  {d['spy']:+6.1f}  {d['sixty40']:+6.1f}  {d['aqr']:+7.1f}  {d['funds']:5.1f}")
pos = sum(1 for y in range(2008, 2025) if J['years'][y]['strategy'] > 0)
pr(f'   positive years 2008-2024: {pos} of 17')

# --- AQR series over the papers' samples ------------------------------------------------------------------------
pr('\n== AQR TSMOM (current vintage) over other windows')
J['aqr_windows'] = {}
alab = ad.astype('datetime64[M]')
for name, a, b in (('1985-2009 (MOP sample)', '1985-01', '2009-12'), ('1985-2025-06', '1985-01', '2025-06')):
    mm = (alab >= np.datetime64(a, 'M')) & (alab <= np.datetime64(b, 'M'))
    s = stats(av['TSMOM'], mm)
    J['aqr_windows'][name] = s
    pr(f"   {name}: {s['months']} months, {s['ann_pct']:+.2f}%/yr, vol {s['vol_pct']:.1f}%, Sharpe {s['sharpe']:.2f}, t {s['t']:+.2f}")

pr('\n== Verdict so far: test 1 ' + ('PASS' if pass1 else 'FAIL') + ', test 2 ' + ('PASS' if pass2 else 'FAIL')
   + ', test 3 (2020-2025-06) ' + ('PASS' if pass3 else 'FAIL'))
J['pass'] = {'test1': bool(pass1), 'test2': bool(pass2), 'test3_old': bool(pass3)}
J['NUMBER'] = NUMBER

txt = '\n'.join(out)
print(txt)
(C.HERE / 'results_old.txt').write_text(txt + '\n')
(C.HERE / 'results_old.json').write_text(json.dumps(J, indent=1, default=float))
np.savez(C.WORK / 'monthly_old.npz', label=lab.astype(str), gross=gross, net=netx, rfm=P['rfm'], aqr=aqr, spy_x=spy_x, sixty=sixty,
         W=P['W'], ACT=P['ACT'])
