"""Prints the .md's tables from out/*.json (no computation). python3 -I -B report.py > out/tables.txt"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'out'


def j(name):
    return json.load(open(OUT / name))


def money(x):
    return f'${x:,.0f}'


def pct(x, nd=1, sign=True):
    return f'{100 * x:+.{nd}f}%' if sign else f'{100 * x:.{nd}f}%'


def settings_table(sel, test, per='week'):
    srows = {r['label']: r for r in sel['rows']}
    print(f"| Setting | Names traded per {per} (sel; test) | Selection: 0 bps; 20 bps | Test: 0 bps; 5 bps; 20 bps | "
          f"Test CAGR at 20 bps | Test t vs average stock (20 bps) | Test 20 bps without div_fix |")
    print('|---|---|---|---|---|---|---|')
    for r in test['rows']:
        s = srows[r['label']]
        mark = ' **(selected)**' if r['label'] == test['selected'] else (' (baseline)' if r['label'] == 'buffer K=10' else '')
        print(f"| {r['label']}{mark} | {s['names_per_week']:.2f}; {r['names_per_week']:.2f} | {money(s['final']['0'])}; "
              f"{money(s['final']['20'])} | {money(r['final']['0'])}; {money(r['final']['5'])}; {money(r['final']['20'])} | "
              f"{pct(r['cagr']['20'])} | {r['weekly_t_20']:+.2f} | {money(r['no_div_fix']['final']['20'])} |")
    r0 = test['rows'][0]
    s0 = sel['rows'][0]
    print(f"| SPY (adjusted) | | {money(s0['spy_final'])} | {money(r0['spy_final'])} | {pct(r0['spy_cagr'])} | | |")


def null_table(test):
    print('| Setting | Account at 20 bps | Median random at 20 bps (5th–95th pct) | p (20 bps) | '
          'CAGR minus median random, 0 bps | same, 20 bps | Without div_fix: p (20 bps); 0-bps difference |')
    print('|---|---|---|---|---|---|---|')
    for r in test['rows']:
        if 'null' not in r:
            continue
        n = r['null']
        nd = r['no_div_fix']['null']
        print(f"| {r['label']} | {money(r['final']['20'])} | {money(n['20']['median_final'])} ({money(n['20']['p05_final'])}–"
              f"{money(n['20']['p95_final'])}) | {n['20']['p']:.3f} | {n['0']['strategy_minus_median_cagr_pts']:+.1f} pts | "
              f"{n['20']['strategy_minus_median_cagr_pts']:+.1f} pts | {nd['20']['p']:.3f}; "
              f"{nd['0']['strategy_minus_median_cagr_pts']:+.1f} pts |")
        if 'null_shadow' in r:
            n = r['null_shadow']
            print(f"| {r['label']}, slot-shadowing null (robustness) | {money(r['final']['20'])} | "
                  f"{money(n['20']['median_final'])} ({money(n['20']['p05_final'])}–{money(n['20']['p95_final'])}) | "
                  f"{n['20']['p']:.3f} | {n['0']['strategy_minus_median_cagr_pts']:+.1f} pts | "
                  f"{n['20']['strategy_minus_median_cagr_pts']:+.1f} pts | |")


def by_year(test, labels):
    rows = {r['label']: r for r in test['rows']}
    head = '| Test year | SPY | Average scored stock | ' + ' | '.join(f'{l}: 0 bps; 20 bps (median random 20 bps)' for l in labels) + ' |'
    print(head)
    print('|' + '---|' * (3 + len(labels)))
    ys = rows[labels[0]]['by_year']
    for i, y in enumerate(ys):
        cells = []
        for l in labels:
            r = rows[l]['by_year'][i]
            nb = rows[l]['null']['by_year'][i]['median_ret_20'] if 'null' in rows[l] else float('nan')
            cells.append(f"{pct(r['ret_0'])}; {pct(r['ret_20'])} ({pct(nb)})")
        print(f"| {y['year']} | {pct(y['spy'])} | {pct(y['avg_stock'])} | " + ' | '.join(cells) + ' |')


def main():
    sel = j('selected.json')
    for name in ['lgbm_s4', 'lgbm_s4_large500', 'reversal', 'lgbm_daily']:
        base = name.replace('_large500', '')
        S, T = j(f'selection_{name}.json'), j(f'test_{name}.json')
        print(f'\n### {name}\n')
        if 'diagnostic' in S:
            print('Diagnostic (selection years): ' + '; '.join(f"window {d['k']}: IC {d['ic']:+.4f} (t {d['t']:.1f})"
                                                              for d in S['diagnostic']))
            print()
        settings_table(S, T, per='day' if base == 'lgbm_daily' else 'week')
        print()
        null_table(T)
        if 'tests' in T:
            print('\nTests:', T['tests'])
        labels = [sel[base]['selected']['label'], 'buffer K=10']
        print()
        by_year(T, labels)
        # after-costs number
        srows = {r['label']: r for r in S['rows']}
        print('\nAfter costs (mean of yearly return at 5 bps minus SPY): ')
        want = ['buffer K=10', sel[base]['selected']['label']] + [v['label'] for v in sel[base]['family_best'].values()]
        seen = set()
        for r in T['rows']:
            if r['label'] in want and r['label'] not in seen:
                seen.add(r['label'])
                print(f"   {r['label']:20s} test {pct(r['after_costs_5_minus_spy'])} (no div_fix "
                      f"{pct(r['no_div_fix']['after_costs_5_minus_spy'])}); selection {pct(srows[r['label']]['after_costs_5_minus_spy'])};"
                      f" test mean yearly 0/5/20 {pct(r['mean_yearly']['0'])}/{pct(r['mean_yearly']['5'])}/{pct(r['mean_yearly']['20'])}"
                      f" SPY mean {pct(r['spy_mean_yearly'])}; $turnover/yr {r['dollar_turnover_per_year']:.1f}")
        for r in T['rows']:
            if 'ff5_umd' in r:
                print('FF5+UMD:', r['label'], r['ff5_umd'])
    print('\n### tie-breaks\n')
    for r in j('test_lgbm_s4_tiebreaks.json'):
        print(f"{r['scores']:8s} {r['label']:16s} 0 {money(r['final']['0'])} 5 {money(r['final']['5'])} "
              f"20 {money(r['final']['20'])} SPY {money(r['spy_final'])} t {r['weekly_t_20']:+.2f} "
              f"after costs {pct(r['after_costs_5_minus_spy'])}")
    print('\n### bridge\n')
    for w in ('sel', 'test'):
        print(w, j(f'bridge_{w}.json'))


if __name__ == '__main__':
    main()
