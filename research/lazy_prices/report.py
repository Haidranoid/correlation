"""Markdown tables for the write-up, from tests_out.json and money_out.json.

    python3 -I report.py       # writes report.txt here
"""
import json
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import HERE, tstat, np  # noqa: E402


def row(name, s):
    if s is None:
        return f'| {name} | – | | | | | |'
    ae, asp = s['alpha_ew'], s['alpha_spy']
    a = lambda x: '–' if not np.isfinite(x[0]) else f'{100 * x[0]:+.2f} ({x[1]:+.2f})'  # noqa: E731
    return (f"| {name} | {s['months']} | {100 * s['gross']:+.2f} ({s['t_gross']:+.2f}) | {100 * s['net']:+.2f} "
            f"({s['t_net']:+.2f}) | {a(ae)} | {a(asp)} | {100 * s['long_vs_all'][0]:+.2f} ({s['long_vs_all'][1]:+.2f}) | "
            f"{100 * s['short_vs_all'][0]:+.2f} ({s['short_vs_all'][1]:+.2f}) |")


HEAD = ('| | Months | Gross, % a month (t) | Net, % a month (t) | Alpha vs EW market, % a month (t) | '
        'Alpha vs SPY (t) | Long − universe (t) | Short − universe (t) |\n|---|---|---|---|---|---|---|---|')


def main():
    R = json.loads((HERE / 'tests_out.json').read_text())
    L = []
    L.append('## Tests 1-3 (sim)\n' + HEAD)
    L.append(row('Test 1: 2007-2021', R['sim/test1']))
    L.append(row('Test 2: 500 largest, 2007-2021', R['sim/test2_large']))
    L.append(row('Test 3: 2022 (Jan-May)', R['sim/test3_2022']))
    L.append('\n## By period\n' + HEAD)
    for p in ('2007-2009', '2010-2014', '2015-2019', '2020-2021'):
        L.append(row(f'All, {p}', R[f'sim/test1/{p}']))
    for p in ('2007-2009', '2010-2014', '2015-2019', '2020-2021'):
        L.append(row(f'500 largest, {p}', R[f'sim/test2_large/{p}']))
    L.append('\n## Sections and sensitivities (test 1 window)\n' + HEAD)
    for m, name in (('sim', 'Items 1+1A+7 (main)'), ('sim_1', 'Item 1 alone'), ('sim_1A', 'Item 1A alone'),
                    ('sim_7', 'Item 7 alone'), ('sim_all', 'All non-empty sections, as is'),
                    ('sim_april', 'April 1 dates')):
        L.append(row(name, R[f'{m}/test1']))
    L.append('\n500 largest:\n' + HEAD)
    for m, name in (('sim_1', 'Item 1 alone'), ('sim_1A', 'Item 1A alone'), ('sim_7', 'Item 7 alone'),
                    ('sim_all', 'All non-empty sections, as is'), ('sim_april', 'April 1 dates')):
        L.append(row(name, R[f'{m}/test2_large']))
    L.append('\n2022:\n' + HEAD)
    for m, name in (('sim_1', 'Item 1 alone'), ('sim_1A', 'Item 1A alone'), ('sim_7', 'Item 7 alone'),
                    ('sim_all', 'All non-empty sections, as is'), ('sim_april', 'April 1 dates')):
        L.append(row(name, R[f'{m}/test3_2022']))
    # by formation year
    L.append('\n## By formation year (main, all stocks)\n| Year | Months | Gross, % a month | Net | Months > 0 | Long − universe | Short − universe | Book |\n|---|---|---|---|---|---|---|---|')
    mo = R['sim/test1/monthly']
    for y in sorted({m[0][:4] for m in mo}):
        s = [m for m in mo if m[0][:4] == y]
        g = np.array([m[1] for m in s]); n = np.array([m[2] for m in s])
        L.append(f'| {y} | {len(s)} | {100 * g.mean():+.2f} | {100 * n.mean():+.2f} | {(n > 0).sum()} | '
                 f'{100 * np.mean([m[3] - m[5] for m in s]):+.2f} | {100 * np.mean([m[4] - m[5] for m in s]):+.2f} | '
                 f'{np.mean([m[6] for m in s]):.0f} |')
    L.append('\n2022 months: ' + '; '.join(f'{m[0]}: net {100 * m[2]:+.2f}%' for m in R['sim/test3_2022/monthly']))
    n = np.array([m[2] for m in mo])
    L.append(f'\nTest 1 net: {len(n)} months, {(n > 0).sum()} positive; annualized mean {1200 * n.mean():+.2f}%, '
             f'sd {100 * n.std(ddof=1) * np.sqrt(12):.2f}%, Sharpe {n.mean() / n.std(ddof=1) * np.sqrt(12):.2f}')
    L.append(f"turnover (sum of |weight change|, both legs, per month): {R['sim/test1']['turnover']:.2f}; "
             f"book {R['sim/test1']['n_book']:.0f}, legs {R['sim/test1']['n_leg']:.0f}; mean similarity long "
             f"{R['sim/test1']['sim_L']:.4f}, short {R['sim/test1']['sim_S']:.4f}")
    L.append(f"pairs used: {R['used_pairs']}")
    # money
    M = json.loads((HERE / 'money_out.json').read_text())
    L.append('\n## Money test\n| | Uniform | Volatility-matched |\n|---|---|---|')
    u, v = M['uniform'], M['volatility']
    L.append(f"| Change over random, 0 bps | {u['change'][0]['change']:+.1f} (se {u['change'][0]['se']:.1f}) | "
             f"{v['change'][0]['change']:+.1f} (se {v['change'][0]['se']:.1f}) |")
    L.append(f"| Years ahead | {u['change'][0]['ahead']} of {u['change'][0]['years']} | {v['change'][0]['ahead']} of {v['change'][0]['years']} |")
    bu = sum(y['costs'][0]['p'] < 0.1 for y in u['years']); bv = sum(y['costs'][0]['p'] < 0.1 for y in v['years'])
    L.append(f'| Years beating random at p < 0.1 | {bu} | {bv} |')
    for key, nm in (('final_pct', 'Picks, mean yearly return'), ('random_median_pct', 'Median random account')):
        L.append(f"| {nm} | {np.mean([y['costs'][0][key] for y in u['years']]):+.1f}% | {np.mean([y['costs'][0][key] for y in v['years']]):+.1f}% |")
    L.append(f"| Change at 5 / 20 bps | {u['change'][1]['change']:+.1f} / {u['change'][2]['change']:+.1f} | "
             f"{v['change'][1]['change']:+.1f} / {v['change'][2]['change']:+.1f} |")
    L.append(f"\nuniform: {u['summary']}\nvolatility: {v['summary']}\nskipped: {u['skipped']}")
    L.append('\n| Year | Candidates | Picks | Median random | p | Median vol-matched | p | SPY |\n|---|---|---|---|---|---|---|---|')
    for yu, yv in zip(u['years'], v['years']):
        cu, cv = yu['costs'][0], yv['costs'][0]
        L.append(f"| {yu['start'][:4]}-{yu['end'][2:4]} | {yu['candidates']} | {cu['final_pct']:+.1f}% | "
                 f"{cu['random_median_pct']:+.1f}% | {cu['p']:.2f} | {cv['random_median_pct']:+.1f}% | {cv['p']:.2f} | {yu['spy']:+.1f}% |")
    (HERE / 'report.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
