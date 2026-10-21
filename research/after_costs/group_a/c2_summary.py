"""Summarize the recomputed c2 walk-forward years (runs/<family>_<year>.txt from c2_walkforward_costs.py and
c2_walkforward_clusters.py): check each year's 0-bps result against the walk-forward row in research/*.txt, then
the after-cost number = mean over the years of (picks' return at 5 bps - SPY's total return, Yahoo adjusted close).
Writes c2_summary.json. Run: python3 -I -B c2_summary.py
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path('/home/bhoult/programming/correlation/research/after_costs/group_a')
sys.path.insert(0, str(HERE))
import c2_walkforward_costs as base  # noqa: E402
import spy  # noqa: E402

RESEARCH = HERE.parent.parent
FAMS = {'census': ('-11.4-census-picks.md', 'census_main_walkforward.txt'),
        'peers': ('-5.2-peer-momentum.md', 'peer_momentum_walkforward.txt'),
        'reversal5': ('+5.4-reversal-control.md', 'walkforward_reversal_5.txt'),
        'clusters': ('-4.5-cluster-lead-lag.md', 'cluster_leadlag_walkforward.txt')}
G0 = Path('/home/bhoult/programming/correlation/stockai/results/poc/g0_reversal_family.json')


def parse_run(text):
    m = re.search(r'Holdout (\S+) \.\. (\S+) \((\d+) trading days\)', text)
    out = {'first_day': m.group(1), 'last_day': m.group(2), 'costs': {}}
    for label, val, pct, med, p in re.findall(r'Picks, (no trading costs|[\d.]+ bps per trade)\s+(\d+)\s+(-?[\d.]+)%\s+random guessers: median (\d+), [\d.]+% did at least as well \(p = ([\d.]+)\)', text):
        c = 0.0 if label.startswith('no') else float(label.split()[0])
        out['costs'][c] = {'final': int(val), 'printed_pct': float(pct), 'pct': 100 * (int(val) / 1e4 - 1),
                           'random_median_pct': 100 * (int(med) / 1e4 - 1), 'p': float(p)}
    return out


def main():
    g0 = {y['first_day']: {c['cost_bps']: c['final_pct'] for c in y['costs']} for y in json.loads(G0.read_text())['years']}
    out = {}
    for fam, (fname, wf_file) in FAMS.items():
        wf = base.parse_walkforward(RESEARCH / wf_file)
        rows = []
        for year, w in sorted(wf.items()):
            f = HERE / 'runs' / f'{fam}_{year}.txt'
            r = parse_run(f.read_text())
            c0, c5 = r['costs'][0.0], r['costs'][5.0]
            ok = (r['first_day'] == w['first_day'] and r['last_day'] == w['last_day'] and c0['printed_pct'] == w['picks0']
                  and abs(c0['random_median_pct'] - w['random0']) < 0.06 and abs(c0['p'] - w['p0']) < 1e-9)
            row = {'year': year, 'first_day': r['first_day'], 'last_day': r['last_day'], 'match_walkforward_0bps': ok,
                   'acct0': c0['pct'], 'rand0': c0['random_median_pct'], 'acct5': c5['pct'],
                   'spy_yahoo': spy.spy_pct(r['first_day'], r['last_day'])}
            if fam == 'reversal5' and r['first_day'] in g0:
                row['g0_judge_acct5'] = g0[r['first_day']][5.0]
            rows.append(row)
        m = lambda k: st.mean(x[k] for x in rows)  # noqa: E731
        out[fname] = {'family': fam, 'years': len(rows), 'span': f"{rows[0]['first_day']}..{rows[-1]['last_day']}",
                      'all_years_match_walkforward_0bps': all(x['match_walkforward_0bps'] for x in rows),
                      'acct0_mean': m('acct0'), 'rand0_mean': m('rand0'), 'change_over_random_0bps': m('acct0') - m('rand0'),
                      'acct5_mean': m('acct5'), 'spy_yahoo_mean': m('spy_yahoo'), 'after_cost': m('acct5') - m('spy_yahoo'),
                      'per_year': rows}
        d = out[fname]
        print(f"{fname:30s} {fam:9s} yrs {d['years']} {d['span']} match {d['all_years_match_walkforward_0bps']}  0bps {d['acct0_mean']:+.1f} "
              f"rand {d['rand0_mean']:+.1f} cor {d['change_over_random_0bps']:+.2f} | 5bps {d['acct5_mean']:+.1f} SPY {d['spy_yahoo_mean']:+.1f} -> {d['after_cost']:+.2f}")
        bad = [x['year'] for x in rows if not x['match_walkforward_0bps']]
        if bad:
            print('   years not matching:', bad)
        if fam == 'reversal5':
            diffs = [abs(x['acct5'] - x['g0_judge_acct5']) for x in rows if 'g0_judge_acct5' in x]
            print(f'   5-bps check against stockai judge G0 JSON: {len(diffs)} years, max |diff| {max(diffs):.3f} pts')
    (HERE / 'c2_summary.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
