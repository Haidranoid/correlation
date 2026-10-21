"""The unseen months, run once after tests.py was final: 2025-06-30 .. 2026-10-19 (return months 2025-07 .. 2026-09
plus October 2026 through the 19th). Same rule, same code, data through 2026-10-19 (2026-10-20 is not used).
Also: the fidelity correlation over the full AQR overlap (through 2026-05) and AQR's post-publication Sharpe.
Writes results_unseen.txt / .json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import common as C  # noqa: E402

P = C.build(C.END_UNSEEN)
P0 = C.build(C.END_MAIN)
out = []; J = {}
pr = lambda s='': out.append(s)  # noqa: E731

# Point in time: the longer data must not change any month through 2025-06.
n0 = len(P0['label'])
assert (P['label'][:n0] == P0['label']).all()
same = np.allclose(P['overlay_gross'][:n0], P0['overlay_gross'], atol=1e-12, equal_nan=True) and np.allclose(P['W'][:n0], P0['W'], atol=1e-12)
pr(f'point-in-time check: months through 2025-06 identical with data to 2026-10-19: {same}')
assert same

lab = P['label']
gross = P['overlay_gross']
netx, cf_net = C.net(P, C.COST_BPS)
rfm = np.nan_to_num(P['rfm'])
m = C.window(P, '2025-07', '2026-10')
pr(f"unseen months: {m.sum()} return months, {str(lab[m][0])} .. {str(lab[m][-1])} (last one partial, through {str(P['dates'][-1])})")
iS, iI = C.TICKERS.index('SPY'), C.TICKERS.index('IEF')
spy_x = P['XR'][:, iS]
sixty = 0.6 * P['R'][:, iS] + 0.4 * P['R'][:, iI] - P['rfm']
ad, av = C.pubdata.aqr_tsmom_monthly()
aqr = C.calendar_month_join(lab, ad, av['TSMOM'])

v = netx[m]
J['unseen'] = {'months': int(m.sum()), 'compound_pct': 100 * (np.prod(1 + v) - 1), 'mean_pct_month': 100 * v.mean(), 't': C.tstat(v),
               'sharpe': C.sharpe(v), 'max_dd_pct': 100 * C.max_drawdown(v), 'gross_compound_pct': 100 * (np.prod(1 + gross[m]) - 1),
               'net5_compound_pct': 100 * (np.prod(1 + C.net(P, 5)[0][m]) - 1), 'net20_compound_pct': 100 * (np.prod(1 + C.net(P, 20)[0][m]) - 1),
               'spy_pct': 100 * (np.prod(1 + spy_x[m]) - 1), 'sixty40_pct': 100 * (np.prod(1 + sixty[m]) - 1),
               'corr_spy': float(np.corrcoef(v, spy_x[m])[0, 1]), 'corr_6040': float(np.corrcoef(v, sixty[m])[0, 1]),
               'gross_exposure_mean': float(P['gross_exposure'][m].mean())}
u = J['unseen']
pr(f"strategy net of 2 bps: {u['compound_pct']:+.2f}% compounded excess ({u['mean_pct_month']:+.3f}%/month, t {u['t']:+.2f}, "
   f"Sharpe {u['sharpe']:.2f}, max DD {u['max_dd_pct']:.1f}%); gross {u['gross_compound_pct']:+.2f}%, 5 bps {u['net5_compound_pct']:+.2f}%, "
   f"20 bps {u['net20_compound_pct']:+.2f}%")
pr(f"SPY excess {u['spy_pct']:+.1f}%, 60/40 excess {u['sixty40_pct']:+.1f}%; correlation with SPY {u['corr_spy']:+.2f}, 60/40 {u['corr_6040']:+.2f}; "
   f"gross exposure {u['gross_exposure_mean']:.2f}")
pass_u = u['compound_pct'] > 0
pr(f"Test 3, unseen months same (positive) sign: {'PASS' if pass_u else 'FAIL'}")
J['pass_unseen'] = bool(pass_u)

# Random signs over the unseen months (seed SEED, drawn for this run's full month grid).
K, F = P['W'].shape
rng = np.random.default_rng(C.SEED)
sg = rng.choice(np.array([-1.0, 1.0]), size=(1000, K, F))
XRz = np.where(P['ACT'], np.nan_to_num(P['XR']), 0.0)
rand_overlay = np.einsum('bkf,kf->bk', sg[:, m], (P['WA'] * XRz)[m])
rc = np.prod(1 + rfm[m][None, :] + rand_overlay, 1) - 1
sc = np.prod(1 + rfm[m] + gross[m]) - 1
long_c = np.prod(1 + rfm[m] + (P['WA'] * XRz)[m].sum(1)) - 1
J['unseen_random'] = {'strategy_total_pct': 100 * sc, 'random_median_total_pct': 100 * float(np.median(rc)),
                      'points_total': 100 * (sc - float(np.median(rc))), 'p_random_ge': float((rc >= sc).mean()),
                      'always_long_total_pct': 100 * long_c}
d = J['unseen_random']
pr(f"vs 1,000 random-sign portfolios (seed {C.SEED}), 0 bps, account total return over the unseen months: strategy {d['strategy_total_pct']:+.2f}%, "
   f"random median {d['random_median_total_pct']:+.2f}% -> {d['points_total']:+.2f} points (not annualized); share at or above {d['p_random_ge']:.3f}; "
   f"always-long {d['always_long_total_pct']:+.2f}%")

pr('\nby kind, unseen months, compounded contribution (sum of monthly contributions), %:')
J['unseen_kinds'] = {}
for kind, tick in C.KINDS.items():
    idx = [C.TICKERS.index(t) for t in tick]
    J['unseen_kinds'][kind] = 100 * cf_net[m][:, idx].sum()
pr('   ' + '; '.join(f'{k} {x:+.2f}' for k, x in J['unseen_kinds'].items()))

pr('\nmonth     net%    SPY%   AQR%   long funds')
J['unseen_months'] = []
for k in np.nonzero(m)[0]:
    nl = int((P['S'][k] > 0).sum())
    J['unseen_months'].append({'month': str(lab[k]), 'net_pct': 100 * netx[k], 'spy_pct': 100 * spy_x[k], 'aqr_pct': 100 * aqr[k] if np.isfinite(aqr[k]) else None, 'long': nl})
    pr(f"{str(lab[k])}  {100 * netx[k]:+6.2f}  {100 * spy_x[k]:+6.2f}  {(100 * aqr[k]) if np.isfinite(aqr[k]) else float('nan'):+6.2f}   {nl}/23")

# Fidelity over the full overlap, and AQR's post-publication record.
pr('\nfidelity (gross, 0 bps) with AQR TSMOM:')
J['fidelity_full'] = {}
for name, a, b in (('2008-01..2026-05', '2008-01', '2026-05'), ('unseen months with AQR data, 2025-07..2026-05', '2025-07', '2026-05'),
                   ('1994-02..2026-05', '1994-02', '2026-05')):
    mm = C.window(P, a, b) & np.isfinite(aqr) & (P['n_active'] > 0)
    r = float(np.corrcoef(gross[mm], aqr[mm])[0, 1])
    J['fidelity_full'][name] = {'months': int(mm.sum()), 'corr': r}
    pr(f'   {name}: {mm.sum()} months, correlation {r:+.3f}')

pr("\nAQR TSMOM (current vintage; AQR rebuilds full history on each update, so this is a post-publication reconstruction, not a fund's live record):")
alab = ad.astype('datetime64[M]')
x = av['TSMOM']
J['aqr_post'] = {}
for name, a, b in (('2010-01..2026-05', '2010-01', '2026-05'), ('2013-01..2026-05 (after the JFE issue, May 2012)', '2013-01', '2026-05'),
                   ('2025-07..2026-05', '2025-07', '2026-05'), ('1985-01..2009-12', '1985-01', '2009-12')):
    mm = (alab >= np.datetime64(a, 'M')) & (alab <= np.datetime64(b, 'M'))
    v = x[mm]
    J['aqr_post'][name] = {'months': int(mm.sum()), 'ann_pct': 1200 * v.mean(), 'vol_pct': 100 * v.std(ddof=1) * np.sqrt(12),
                           'sharpe': C.sharpe(v), 't': C.tstat(v), 'compound_pct': 100 * (np.prod(1 + v) - 1)}
    e = J['aqr_post'][name]
    pr(f"   {name}: {e['months']} months, {e['ann_pct']:+.2f}%/yr, vol {e['vol_pct']:.1f}%, Sharpe {e['sharpe']:.2f}, t {e['t']:+.2f}")

txt = '\n'.join(out)
print(txt)
(C.HERE / 'results_unseen.txt').write_text(txt + '\n')
(C.HERE / 'results_unseen.json').write_text(json.dumps(J, indent=1, default=float))
