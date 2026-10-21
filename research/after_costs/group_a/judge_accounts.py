"""After-cost numbers for group_a's files graded by stockai's judge (JSONs with yearly accounts at 0/5/20/50 bps).

For each JSON: per year, the account's return at 5 bps (cost index of 5.0) minus SPY's total return over the same
judge year (Yahoo adjusted close, spy.py), then the mean over the file's test years.

Run: python3 -I -B research/after_costs/group_a/judge_accounts.py  (writes judge_accounts.json next to it)
"""
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path('/home/bhoult/programming/correlation/research/after_costs/group_a')
sys.path.insert(0, str(HERE))
import spy  # noqa: E402

SA = Path('/home/bhoult/programming/correlation/stockai/results')
WORK = Path.home() / '.local/share/correlation2/experiments'


def years_of(path, first_start=None, last_start=None):
    js = json.loads(Path(path).read_text())
    out = []
    for y in js['years']:
        if first_start and y['start'] < first_start:
            continue
        if last_start and y['start'] > last_start:
            continue
        cost = {c['cost_bps']: c for c in y['costs']}
        out.append({
            'start': y['start'], 'end': y['end'], 'first_day': y['first_day'], 'last_day': y['last_day'],
            'acct0': cost[0.0]['final_pct'], 'acct5': cost[5.0]['final_pct'],
            'rand0': cost[0.0]['random_median_pct'],
            'spy_store': y['spy_pct'], 'spy_yahoo': spy.spy_pct(y['first_day'], y['last_day']),
        })
    return out


def summarize(rows):
    m = lambda k: st.mean(r[k] for r in rows)  # noqa: E731
    return {
        'years': len(rows), 'span': f"{rows[0]['first_day']}..{rows[-1]['last_day']}",
        'acct0_mean': m('acct0'), 'rand0_mean': m('rand0'), 'change_over_random_0bps': m('acct0') - m('rand0'),
        'acct5_mean': m('acct5'), 'spy_yahoo_mean': m('spy_yahoo'), 'spy_store_mean': m('spy_store'),
        'after_cost': m('acct5') - m('spy_yahoo'),
        'per_year': [(r['start'], round(r['acct5'], 2), round(r['spy_yahoo'], 2), round(r['acct5'] - r['spy_yahoo'], 2)) for r in rows],
    }


def seeds(tag, prefix, n=5):
    """Median over the 5 random tie-break seeds of the after-cost number (the files quote the median of 5)."""
    per = {}
    for s in range(1, n + 1):
        per[s] = summarize(years_of(SA / tag / 'judge' / f'jitter_{prefix}_s{s}_uniform.json'))
    vals = sorted(per.values(), key=lambda d: d['after_cost'])
    med = vals[n // 2]
    return {'median_seed_after_cost': med['after_cost'], 'median_seed_acct5_mean': med['acct5_mean'],
            'median_seed_spy_yahoo_mean': med['spy_yahoo_mean'],
            'by_seed': {s: {k: d[k] for k in ('after_cost', 'acct5_mean', 'acct0_mean', 'change_over_random_0bps')} for s, d in per.items()},
            'spy_yahoo_mean': med['spy_yahoo_mean'], 'years': med['years'], 'span': med['span'], 'per_year_median_seed': med['per_year']}


def main():
    out = {}
    # The poc (next-day / weekly / monthly rule) accounts: 14 judge years 2010-06-28..2024-06-28.
    poc = SA / 'poc'
    out['-19.8-stockgpt-next-day.md'] = summarize(years_of(poc / 'gpt_ensemble__r1_d0_uniform.json'))
    out['-4.9-lightgbm-next-day.md'] = summarize(years_of(poc / 'lgbm__r1_d0_uniform.json'))
    out['+5.8-lightgbm-next-day-actual-returns.md'] = summarize(years_of(poc / 'lgbm_ret__r1_d0_uniform.json'))
    out['+2.1-momentum-12-1.md'] = summarize(years_of(poc / 'momentum_12_1__r21_d0_uniform.json', first_start='2010-06-28'))
    out['+2.1-momentum-12-1.md (all 17 judge years, for reference)'] = summarize(years_of(poc / 'momentum_12_1__r21_d0_uniform.json'))

    # Top-10 accounts (the grader's account; the files quote the median of 5 random tie-breaks).
    for name, tag, prefix in [('+18.6-monthly-top10-picks.md', '30d', 'lgbm'),
                              ('+7.4-weekly-top10-picks.md', '1w', 'lgbm'),
                              ('+3.6-two-month-top10-picks.md', '2m', 'lgbm'),
                              ('+20.9-outside-data-inputs-1-week.md', 'ext-1w', 'lgbm_ext'),
                              ('+10.9-outside-data-inputs-1-month.md', 'ext-30d', 'lgbm_ext'),
                              ('-6.6-outside-data-inputs-2-months.md', 'ext-2m', 'lgbm_ext')]:
        d = seeds(tag, prefix)
        d['ticker_order_ties'] = summarize(years_of(SA / tag / 'judge' / f'{prefix}_uniform.json'))
        out[name] = d
    # stockGPT on the top-10 tests: one account (continuous scores, no tie-break runs).
    out['-5.7-stockgpt-1-week.md'] = summarize(years_of(SA / '1w' / 'judge' / 'gpt_uniform.json'))
    out['-7.2-stockgpt-1-month.md'] = summarize(years_of(SA / '30d' / 'judge' / 'gpt_uniform.json'))
    out['-3.0-stockgpt-2-months.md'] = summarize(years_of(SA / '2m' / 'judge' / 'gpt_uniform.json'))

    # v1 pair finder: the judge via c2eval, w = 30, rebalance 21, walk-forward 2007-08..2024-25 (18 graded years).
    out['-3.6-v1-pair-finder.md'] = summarize(years_of(WORK / 'v1_pair_finder' / 'aligned_w30__r21_d0_uniform.json'))

    (HERE / 'judge_accounts.json').write_text(json.dumps(out, indent=1))
    for k, d in out.items():
        if 'after_cost' in d:
            print(f"{k:58s} yrs {d['years']:2d} {d['span']}  0bps {d['acct0_mean']:+6.1f} rand {d['rand0_mean']:+6.1f} "
                  f"cor {d['change_over_random_0bps']:+6.1f} | 5bps {d['acct5_mean']:+6.1f} SPY(yahoo) {d['spy_yahoo_mean']:+5.1f} "
                  f"SPY(store) {d['spy_store_mean']:+5.1f} -> {d['after_cost']:+.1f}")
        else:
            bs = d['by_seed']
            t = d['ticker_order_ties']
            print(f"{k:58s} yrs {d['years']:2d} {d['span']}  median-seed after-cost {d['median_seed_after_cost']:+.1f} "
                  f"(5bps {d['median_seed_acct5_mean']:+.1f}, SPY {d['spy_yahoo_mean']:+.1f}); seeds "
                  + ' '.join(f"{bs[s]['after_cost']:+.1f}" for s in bs)
                  + f"; ticker-order ties {t['after_cost']:+.1f} (0bps {t['acct0_mean']:+.1f}, cor {t['change_over_random_0bps']:+.1f})")


if __name__ == '__main__':
    main()
