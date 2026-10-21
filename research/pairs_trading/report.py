"""Markdown tables for the .md from out/tests_old.json (and out/tests_new.json if present). Writes out/tables.md.

    python3 -I -B report.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / 'out'
r = json.loads((OUT / 'tests_old.json').read_text())
rn = json.loads((OUT / 'tests_new.json').read_text()) if (OUT / 'tests_new.json').exists() else None
L = []


def f(x, nd=2, sign=True):
    if x is None or x != x:
        return 'n/a'
    return f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'


def per(key, p):
    return r[key]['periods'][p]


PRIMARY = 'export_div_all' if 'export_div_all' in r else 'export_nodiv_all'
L.append(f'Primary configuration: {PRIMARY}\n')

# Tests
q = per(PRIMARY, '2006-2019')
q2 = per(PRIMARY, '2020-2025-06')
L.append('### Tests (committed capital, net of 5 bps a side)\n')
L.append('| Test | Number | Pass needs | Result |')
L.append('|---|---|---|---|')
cn = q['committed_net5']
L.append(f"| 1. 2006–2019 mean monthly net return | {f(cn['mean_pct'], 3)}% a month ({f(cn['yearly_pct'])}% a year), Newey-West t {f(cn['nw_t'])} (plain t {f(cn['t'])}), {cn['n']} months | t ≥ 2 | {'Pass' if cn['nw_t'] >= 2 else 'Fail'} |")
rn5 = q['random_net5']
L.append(f"| 2. Beats random pairs, 2006–2019 | book {f(rn5['book_yearly_pct'])}% a year; random books median {f(rn5['rand_median_yearly_pct'])}, 95th percentile {f(rn5['rand_p95_yearly_pct'])}; {rn5['share_random_ge_book']:.0%} of random books did as well | above the 95th percentile | {'Pass' if rn5['above_p95'] else 'Fail'} |")
c2 = q2['committed_net5']
L.append(f"| 3a. 2020–2025-06, same sign | {f(c2['mean_pct'], 3)}% a month ({f(c2['yearly_pct'])}% a year), t {f(c2['nw_t'])} | same sign as test 1 | {'Pass' if (c2['mean_pct'] > 0) == (cn['mean_pct'] > 0) else 'Fail'} |")
if rn:
    u = rn[PRIMARY.replace('old', 'new')] if PRIMARY.replace('old', 'new') in rn else rn[PRIMARY]
    uc = u['committed_net5']
    L.append(f"| 3b. The 16 unseen months, same sign | {f(uc['mean_pct'], 3)}% a month ({f(uc['yearly_pct'])}% a year), t {f(uc['nw_t'])}, {uc['n']} months | same sign as test 1 | {'Pass' if (uc['mean_pct'] > 0) == (cn['mean_pct'] > 0) else 'Fail'} |")
L.append('')

# Main table by period, all configurations
L.append('### Book returns by period (% a month; Newey-West t, 6 lags)\n')
L.append('| Prices | Dividends | Stocks | Period | Committed gross | t | Committed net 5 bps | t | Employed gross | Employed net 5 bps | t | FF5 + UMD alpha, net, % a year | t |')
L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for key in r:
    src, div, var = key.split('_')
    for p in ('2006-2019', '2020-2025-06', '2006-2025-06'):
        q = per(key, p)
        cg, cn, eg, en, a = q['committed_gross'], q['committed_net5'], q['employed_gross'], q['employed_net5'], q['alpha_net5']
        L.append(f"| {src} | {'fixed' if div == 'div' else 'none'} | {'500 largest' if var == 'big' else 'all'} | {p} | {f(cg['mean_pct'], 3)} | {f(cg['nw_t'])} | "
                 f"{f(cn['mean_pct'], 3)} | {f(cn['nw_t'])} | {f(eg['mean_pct'], 3)} | {f(en['mean_pct'], 3)} | {f(en['nw_t'])} | "
                 f"{f(a['alpha_yearly_pct'])} | {f(a['alpha_t'])} |")
L.append('')

# Random books
L.append('### Against 1,000 random-pair books (committed capital, % a year)\n')
L.append('| Stocks | Period | Costs | Book | Random median | Random 5th–95th percentile | Book − median | Random books ≥ book |')
L.append('|---|---|---|---|---|---|---|---|')
for var in ('all', 'big'):
    key = PRIMARY.replace('_all', f'_{var}')
    for p in ('2006-2019', '2020-2025-06', '2006-2025-06'):
        for lab, nm in (('random_gross', '0 bps'), ('random_net5', '5 bps')):
            x = per(key, p)[lab]
            L.append(f"| {'500 largest' if var == 'big' else 'all'} | {p} | {nm} | {f(x['book_yearly_pct'])} | {f(x['rand_median_yearly_pct'])} | "
                     f"{f(x['rand_p5_yearly_pct'])} to {f(x['rand_p95_yearly_pct'])} | {f(x['book_minus_median_pts'])} | {x['share_random_ge_book']:.1%} |")
L.append('')

# Sub-periods
L.append('### Sub-periods, committed capital (% a year)\n')
L.append('| Period | Gross, dividends fixed | Gross, no fix | Net 5 bps, fixed | t | Random median, gross |')
L.append('|---|---|---|---|---|---|')
alt = PRIMARY.replace('_div_', '_nodiv_')
for p in ('2006-2009', '2010-2014', '2015-2019', '2020-2022-06', '2022-07-2025-06'):
    q = per(PRIMARY, p)
    qa = per(alt, p) if alt in r else q
    L.append(f"| {p} | {f(q['committed_gross']['yearly_pct'])} | {f(qa['committed_gross']['yearly_pct'])} | {f(q['committed_net5']['yearly_pct'])} | "
             f"{f(q['committed_net5']['nw_t'])} | {f(q['random_gross']['rand_median_yearly_pct'])} |")
L.append('')

# By year
L.append('### By calendar year, committed capital, gross (% a year)\n')
L.append('| Year | Book | Random median | Book − median |')
L.append('|---|---|---|---|')
for y, v in r[PRIMARY]['years'].items():
    L.append(f"| {y}{' (to 06-27)' if y == '2025' else ''} | {f(v['committed_gross_yearly_pct'])} | {f(v['random_median_gross_yearly_pct'])} | "
             f"{f(v['committed_gross_yearly_pct'] - v['random_median_gross_yearly_pct'])} |")
L.append('')

# Trading statistics
L.append('### Trading statistics (per six-month portfolio of 20 pairs)\n')
L.append('| | GGR 1963–2002, top 20, wait one day | Ours, formations of 2006–2019 | Ours, 2020–2025-06 | 500 largest, 2006–2019 |')
L.append('|---|---|---|---|---|')
t1, t2 = r[PRIMARY]['trades_2006-2019'], r[PRIMARY]['trades_2020-2025-06']
tb = r[PRIMARY.replace('_all', '_big')]['trades_2006-2019']
rows = [('Universe (median stocks)', '', 'universe_median', 0), ('Pairs that open (of 20)', '19.30', 'pairs_opened_of_20', 2),
        ('Round trips per pair', '1.96', 'round_trips_per_pair', 2), ('Months open per pair', '3.76', 'months_open_per_pair', 2),
        ('2σ trigger (normalized price gap)', '0.053', 'trigger_2sigma', 3),
        ('Pairs that never open', '', 'share_pairs_never_open', 3),
        ('Trades closed at the period end, not by converging', '', 'share_trades_not_converging', 3),
        ('Opened pairs that never converge', '', 'share_opened_pairs_never_converging', 3),
        ('Random pairs: share that open', '', 'random_pairs_open_rate', 3),
        ('Random pairs: round trips per pair', '', 'random_round_trips_per_pair', 2),
        ('Random pairs: median 2σ trigger', '', 'random_trigger_2sigma_median', 3),
        ('Random pairs: trades not converging', '', 'random_share_trades_not_converging', 3)]
for nm, g, k, nd in rows:
    L.append(f"| {nm} | {g} | {f(t1[k], nd, False)} | {f(t2[k], nd, False)} | {f(tb[k], nd, False)} |")
L.append('')

# Data check
L.append('### Data check: export closes vs the store\'s raw closes (test 1\'s statistic)\n')
L.append('| Dividends | Stocks | Period | Export: mean, t | Store: mean, t | Same sign | |Δt| | Within 1 |')
L.append('|---|---|---|---|---|---|---|---|')
for div in ('div', 'nodiv'):
    for var in ('all', 'big'):
        ke, ks = f'export_{div}_{var}', f'store_{div}_{var}'
        if ke not in r or ks not in r:
            continue
        for p in ('2006-2019', '2020-2025-06'):
            a, b = per(ke, p)['committed_net5'], per(ks, p)['committed_net5']
            L.append(f"| {'fixed' if div == 'div' else 'none'} | {'500 largest' if var == 'big' else 'all'} | {p} | {f(a['mean_pct'], 4)}, {f(a['nw_t'])} | "
                     f"{f(b['mean_pct'], 4)}, {f(b['nw_t'])} | {'yes' if (a['mean_pct'] > 0) == (b['mean_pct'] > 0) else 'NO'} | "
                     f"{abs(a['nw_t'] - b['nw_t']):.3f} | {'yes' if abs(a['nw_t'] - b['nw_t']) <= 1 else 'NO'} |")
L.append('')

# File-name number
x = per(PRIMARY, '2006-2025-06')['random_gross']
L.append(f"File-name number ({PRIMARY}, 2006–2025-06, 0 bps): book {f(x['book_yearly_pct'])}% a year − random median "
         f"{f(x['rand_median_yearly_pct'])} = {f(x['book_minus_median_pts'], 1)} points\n")

if rn:
    L.append('### The 16 unseen months (new export and store; blocks from 2025-06-30)\n')
    L.append('| Prices | Dividends | Stocks | Months | Committed gross, % a month | Committed net 5 bps | t | Employed net 5 bps | Random median gross, % a year | Book − median, gross, pts | Random books ≥ book | Alpha net, % a year | t |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for key, u in rn.items():
        src, div, var = key.split('_')
        rg = u['random_gross']
        L.append(f"| {src} | {'fixed' if div == 'div' else 'none'} | {'500 largest' if var == 'big' else 'all'} | {u['blocks']} ({u['first']}..{u['last']}) | "
                 f"{f(u['committed_gross']['mean_pct'], 3)} | {f(u['committed_net5']['mean_pct'], 3)} | {f(u['committed_net5']['nw_t'])} | "
                 f"{f(u['employed_net5']['mean_pct'], 3)} | {f(rg['rand_median_yearly_pct'])} | {f(rg['book_minus_median_pts'])} | {rg['share_random_ge_book']:.1%} | "
                 f"{f(u['alpha_net5']['alpha_yearly_pct'])} | {f(u['alpha_net5']['alpha_t'])} |")
    L.append('')
    for key, u in rn.items():
        L.append(f"{key}: monthly committed net (%): {u['monthly_committed_net5_pct']}; new export 2006–2019 committed net "
                 f"{f(u['new_export_2006_2019_committed_net5']['mean_pct'], 3)}% a month, t {f(u['new_export_2006_2019_committed_net5']['nw_t'])}")
    L.append('')

txt = '\n'.join(L)
(OUT / 'tables.md').write_text(txt + '\n')
print(txt)
