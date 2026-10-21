"""group_c files whose number is not the judge's: each strategy's own account or book at 5 bps a side, compounded by
year. Long-only accounts subtract SPY's Yahoo total return over the same span; long-short books don't.
Writes own_books.json and own_books.txt next to this script. Read-only on every experiment's outputs."""
import json
import sys

import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/after_costs/group_c')
import ac_common as A  # noqa: E402

c2eval = A.c2eval
iso = c2eval.iso
W = A.WORK
Eo = c2eval.load_export(c2eval.EXPORT_OLD, arrays=())
DAYS = Eo['days']
OUT, LINES = {}, []


def say(s=''):
    LINES.append(s)
    print(s, flush=True)


def dnum(s):
    return int((np.datetime64(s, 'D') - np.datetime64('1970-01-01', 'D')).astype(int))


def yrs_of(days):
    return np.array([int(iso(d)[:4]) for d in days])


def report(name, rows, long_only, extra=None):
    a = np.array([r['ret'] for r in rows])
    res = {'rows': rows, 'n': len(rows), 'ret_mean': float(a.mean())}
    if long_only:
        s = np.array([r['spy'] for r in rows])
        res['spy_mean'] = float(s.mean())
        res['after_cost'] = float((a - s).mean())
    else:
        res['after_cost'] = float(a.mean())
    if extra:
        res.update(extra)
    OUT[name] = res
    say(f'\n== {name}')
    for r in rows:
        say(f"   {r['period']:28s} ret {r['ret']:+9.2f}" + (f"  SPY {r['spy']:+7.2f}  diff {r['ret'] - r['spy']:+8.2f}" if long_only else '')
            + (f"  ret0 {r['ret0']:+8.2f}" if 'ret0' in r else ''))
    say(f"   mean ret {res['ret_mean']:+.2f}" + (f", SPY {res['spy_mean']:+.2f}" if long_only else '')
        + f" -> after-cost {res['after_cost']:+.2f} ({A.fmt(res['after_cost'])})")
    for k, v in (extra or {}).items():
        say(f'   {k}: {v}')


# ---------------------------------------------------------------------------------------------------------------
# +54.1 overnight auctions: the 20-stock overnight account (long only); bought at the close, sold at the next open,
# every day, so 5 bps a side = 10 bps a day on the whole account. Calendar years by exit day (money.py).
D = np.load(W / 'overnight_auctions/money_daily_old.npz')
d, acc = D['d'], D['acc']
yr = yrs_of(DAYS[d])
rows = []
for y in np.unique(yr):
    m = yr == y
    first_exit = iso(DAYS[d[m][0]])
    last_exit = iso(DAYS[d[m][-1]])
    r5 = 100 * (np.prod(1.0 + acc[m] - 2 * 5e-4) - 1.0)
    r0 = 100 * (np.prod(1.0 + acc[m]) - 1.0)
    # SPY: close before the year's first exit day (the first entry close) to the close of its last exit day
    s = 100 * A.spy_calendar_year(int(y), first=first_exit, last=last_exit)
    rows.append({'period': f'{y} ({first_exit}..{last_exit})', 'ret': float(r5), 'ret0': float(r0), 'spy': float(s)})
report('+15.7-overnight-auctions.md', rows, True,
       {'check_0bps_mean': float(np.mean([r['ret0'] for r in rows])),
        'note': 'account pays 10 bps a day (5 bps buying at the close + 5 bps selling at the open)'})

# ---------------------------------------------------------------------------------------------------------------
# -0.4 correlation portfolios: MV-LW (primary), long only, daily net at 5 bps a side (build.py), calendar years
# 2006..2025-06.
S = np.load(W / 'correlation_portfolios/series_old.npz')
assert list(S['ports']).index('MV-LW') == 2
sd = S['days']
assert np.array_equal(sd, DAYS)
net, gross = S['net'][2], S['gross'][2]
yr = yrs_of(sd)
rows = []
for y in range(2006, 2026):
    m = (yr == y) & np.isfinite(net)
    if not m.any():
        continue
    a, b = iso(sd[m][0]), iso(sd[m][-1])
    rows.append({'period': f'{y} ({a}..{b})', 'ret': float(100 * (np.prod(1 + net[m]) - 1)),
                 'ret0': float(100 * (np.prod(1 + gross[m]) - 1)), 'spy': float(100 * A.spy_calendar_year(y, first=a, last=b))})
report('-4.6-correlation-portfolios.md', rows, True, {'portfolio': 'MV-LW', 'first_day': iso(sd[np.isfinite(net)][0])})

# ---------------------------------------------------------------------------------------------------------------
# +1.4 trend following: the fund overlay (excess return over T-bills, self-financing), 5 bps a side on its turnover,
# compounded by calendar year, 2008..2025-06 (2025 = Jan-Jun). turnover = (gross - net at 2 bps) / 2 bps.
T = np.load(W / 'trend_following/monthly_old.npz')
lab, g, n2, rfm = T['label'], T['gross'], T['net'], np.nan_to_num(T['rfm'])
turn = (g - n2) / 2e-4
rows, rows1 = [], []
for y in range(2008, 2026):
    m = np.array([l.startswith(f'{y}-') for l in lab])
    m &= np.array([l <= '2025-06' for l in lab])
    n5 = g[m] - 5e-4 * turn[m]
    n1 = g[m] - 1e-4 * turn[m]
    rows.append({'period': f'{y} ({lab[m][0]}..{lab[m][-1]})', 'ret': float(100 * (np.prod(1 + n5) - 1)),
                 'ret0': float(100 * (np.prod(1 + g[m]) - 1)),
                 'with_tbill': float(100 * (np.prod(1 + rfm[m] + n5) - 1))})
    rows1.append(float(100 * (np.prod(1 + n1) - 1)))
mm = np.array([('2008-01' <= l <= '2025-06') for l in lab])
n5all = g[mm] - 5e-4 * turn[mm]
report('+1.2-trend-following.md', rows, False,
       {'mean_turnover_per_month': float(turn[mm].mean()),
        'at_1bp_mean': float(np.mean(rows1)),
        'overlay_plus_tbill_mean': float(np.mean([r['with_tbill'] for r in rows])),
        'cagr_overlay_5bps_2008_2025_06': float(100 * (np.exp(np.log1p(n5all).sum() * 12 / mm.sum()) - 1)),
        'arith_overlay_5bps_2008_2025_06': float(1200 * n5all.mean())})

# ---------------------------------------------------------------------------------------------------------------
# +5.4 risk-managed momentum: RMOM (the scaled book), dividend-corrected, monthly net at 5 bps a side including scale
# changes (build.py), compounded by calendar year 2006..2025-06.
R = np.load(W / 'risk_managed_momentum/series_old_divfix.npz')
months = R['months']
rn, rg = R['all__rmom__net_m'], R['all__rmom__gross_m']
rows = []
for y in range(2006, 2026):
    m = np.array([s.startswith(f'{y}-') for s in months]) & np.isfinite(rn)
    if not m.any():
        continue
    rows.append({'period': f'{y} ({months[m][0]}..{months[m][-1]})', 'ret': float(100 * (np.prod(1 + rn[m]) - 1)),
                 'ret0': float(100 * (np.prod(1 + rg[m]) - 1))})
Rp = np.load(W / 'risk_managed_momentum/series_old_plain.npz')
rows_p = []
for y in range(2006, 2026):
    m = np.array([s.startswith(f'{y}-') for s in Rp['months']]) & np.isfinite(Rp['all__rmom__net_m'])
    if m.any():
        rows_p.append(float(100 * (np.prod(1 + Rp['all__rmom__net_m'][m]) - 1)))
report('+3.3-risk-managed-momentum.md', rows, False, {'plain_prices_mean': float(np.mean(rows_p)),
                                                      'months_used': int(np.isfinite(rn).sum())})

# ---------------------------------------------------------------------------------------------------------------
# +0.3 pairs trading: committed-capital book (export closes, dividend-corrected, all stocks), 21-day blocks net of
# 5 bps a side (the plan's own cost), compounded by the block's last day's calendar year, 2006..2025-06.
P = np.load(W / 'pairs_trading/run_old_export_div_all.npz')
NPAIRS, NPORT, BLOCK = 20, 6, 21
rebal, TT = P['rebal'], int(P['T'])
pdays = P['days']
K = len(rebal)
cg, cc = np.full(K, np.nan), np.full(K, np.nan)
rg_med = np.full(K, np.nan)
last = np.minimum(rebal + BLOCK, TT - 1)
for b in range(K):
    if rebal[b] + 1 > TT - 1:
        continue
    ks = np.array([k for k in range(b - NPORT + 1, b + 1) if k >= 0 and P['valid'][k]], int)
    if len(ks) == 0:
        continue
    m_ = b - ks
    cg[b] = (P['pnl'][ks, :, m_].sum(1) / NPAIRS).mean()
    cc[b] = (P['cost'][ks, :, m_].sum(1) / NPAIRS).mean()
lastd = pdays[last]
rows = []
for y in range(2006, 2026):
    m = np.isfinite(cg) & (lastd >= dnum(f'{y}-01-01')) & (lastd <= dnum(f'{y}-12-31')) & (lastd <= dnum('2025-06-27'))
    if not m.any():
        continue
    rows.append({'period': f'{y} ({m.sum()} blocks)', 'ret': float(100 * (np.prod(1 + cg[m] - cc[m]) - 1)),
                 'ret0': float(100 * (np.prod(1 + cg[m]) - 1)), 'arith12_net': float(1200 * (cg[m] - cc[m]).mean())})
m_all = np.isfinite(cg) & (lastd >= dnum('2006-01-01')) & (lastd <= dnum('2025-06-27'))
report('+0.8-pairs-trading.md', rows, False,
       {'check_gross_12x_mean_2006_2025_06': float(1200 * cg[m_all].mean()),
        'net_12x_mean_2006_2025_06': float(1200 * (cg[m_all] - cc[m_all]).mean())})

# ---------------------------------------------------------------------------------------------------------------
# +0.1 turn of the month: SPY on TOM days (close before to close), T-bill otherwise, 5 bps a side on each position
# change (the plan used 1 bp); calendar years 2006..2025-06 on the old export's days (analyze.py's construction).
sys.path.insert(0, '/home/bhoult/programming/correlation/research/turn_of_month')
import common as TC  # noqa: E402
En = c2eval.load_export(c2eval.EXPORT_NEW, arrays=())
do, dn = DAYS, En['days']
span = TC.span_mask(do, *TC.SPAN)
days = do[span]
labt = TC.tom_label(do, dn)[span]
r_all, _, _ = TC.spy_total(do)
r = r_all[span]
rf = A.pubdata.rf_daily(do)[span]
ind = labt != 0
prev = np.r_[False, ind[:-1]]
nxt = np.r_[ind[1:], False]
trades = (ind & ~prev).astype(int) + (ind & ~nxt).astype(int)
lr, lrf = np.log1p(r), np.log1p(rf)
yr = yrs_of(days)


def tom_rows(cost):
    gg = np.where(ind, lr, lrf) + np.log1p(-cost) * trades
    out = []
    for y in np.unique(yr):
        m = yr == y
        a, b = iso(days[m][0]), iso(days[m][-1])
        out.append({'period': f'{y} ({a}..{b})', 'ret': float(100 * np.expm1(gg[m].sum())),
                    'spy': float(100 * A.spy_calendar_year(int(y), first=a, last=b))})
    return out


rows = tom_rows(5e-4)
r0 = tom_rows(0.0)
r1 = tom_rows(1e-4)
for x, z in zip(rows, r0):
    x['ret0'] = z['ret']
m_ = np.ones(len(days), bool)
cagr0 = float(100 * (np.exp(np.where(ind, lr, lrf)[m_].sum() * 252 / m_.sum()) - 1))
report('-9.6-turn-of-month.md', rows, True,
       {'at_1bp_after_cost': float(np.mean([a['ret'] - a['spy'] for a in r1])),
        'trades_per_year': float(trades.sum() / (len(days) / 252)), 'check_cagr_0bps': cagr0,
        'span': f'{iso(days[0])}..{iso(days[-1])}'})

# ---------------------------------------------------------------------------------------------------------------
# +7.8 etf stat arb, ALTERNATIVE (its file-name number is the judge's): the s-score book, export closes,
# dividend-corrected, all stocks, same-close execution, daily net (5 bps a side stocks, 1 bp netted fund legs: the
# plan's own costs), compounded by calendar year 2006..2025-06, and by the judge's July-June years.
B = np.load(W / 'etf_stat_arb/books_old_export_fix.npz')
bd = B['days']
bn, bg = np.nan_to_num(B['all_same_net']), np.nan_to_num(B['all_same_gross'])
yr = yrs_of(bd)
rows = []
for y in range(2006, 2026):
    m = yr == y
    rows.append({'period': f'{y}', 'ret': float(100 * (np.prod(1 + bn[m]) - 1)), 'ret0': float(100 * (np.prod(1 + bg[m]) - 1))})
js, jr = A.judge_rows(W / 'etf_stat_arb/money_old__r1_d0_uniform.json')
jy = []
for r_ in jr:
    m = (bd >= dnum(r_['first_day'])) & (bd <= dnum(r_['last_day']))
    jy.append(float(100 * (np.prod(1 + bn[m]) - 1)))
report('-6.8-etf-stat-arb.md (book, alternative)', rows, False,
       {'judge_years_mean': float(np.mean(jy)), 'judge_years': f"{jr[0]['first_day']}..{jr[-1]['last_day']}"})

# ---------------------------------------------------------------------------------------------------------------
# +23.6 intraday lead-lag: the 5-minute long-short (test 3), test half 2026-09-09..10-20, 2,280 bars. Only the
# summary is saved: mean gross per bar and mean turnover (4 = whole book). Net per bar at 5 bps a side =
# gross - 5 bp x turnover; compounded over the bars (the per-bar spread around the mean adds a negligible drag).
J5 = json.loads((A.OUT.parent.parent / 'intraday_lead_lag/out/results_5m.json').read_text())
t3 = J5.get('test3') or J5
gb, tv, nb = t3['gross_bp'], t3['turnover'], t3['bars']
net_bp = gb - 5.0 * tv
period = float(np.expm1(nb * np.log1p(net_bp / 1e4)))
bars_per_day = nb / 30.0
yearly = float(np.expm1(252 * bars_per_day * np.log1p(net_bp / 1e4)))
OUT['-100.0-intraday-lead-lag.md'] = {'gross_bp_per_bar': gb, 'turnover_per_bar': tv, 'bars': nb,
                                     'net_bp_per_bar_5bps': net_bp, 'compounded_test_half_pct': 100 * period,
                                     'annualized_pct': 100 * yearly, 'after_cost': 100 * yearly}
say('\n== -100.0-intraday-lead-lag.md')
say(f'   gross {gb:+.3f} bp/bar, turnover {tv:.3f}, net at 5 bps {net_bp:+.2f} bp/bar over {nb} bars: compounded '
    f'{100 * period:+.2f}% over the 30-day test half; a 252-day year at {bars_per_day:.0f} bars/day: {100 * yearly:+.4f}%')

(A.OUT / 'own_books.json').write_text(json.dumps(OUT, indent=1))
(A.OUT / 'own_books.txt').write_text('\n'.join(LINES) + '\n')
