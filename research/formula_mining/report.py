"""Print the results tables (markdown) from selection_old.json, shuffled_runs.json, money_old.json, money_new.json and
the probe of 5-day reversal on the same test years.

    python3 -I report.py > report.txt
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/formula_mining')
from common import BIG, HERE, c2eval  # noqa: E402

S = json.load(open(HERE / 'selection_old.json'))
R = json.load(open(HERE / 'shuffled_runs.json'))
MO = json.load(open(HERE / 'money_old.json'))
MN = json.load(open(HERE / 'money_new.json'))
REV = {rm: json.load(open(BIG / f'probe_reversal5_from2015__r5_d0_{rm}.json')) for rm in ('uniform', 'volatility')}


def f(x, d=2, sign=True):
    return f'{x:+.{d}f}'.replace('-', '−') if sign else f'{x:.{d}f}'.replace('-', '−')


def name(formula, sign):
    s = formula.replace('*', ' × ').replace('-', ' − ')
    return s if sign > 0 else f'−({s})' if (' × ' in s or ' − ' in s) else f'−{s}'


b = np.array([r['best_abs_t'] for r in R])
h = np.array([r['holdout_ic_mean'] for r in R])
ht = np.array([r['holdout_ic_t'] for r in R])
p95 = np.percentile(b, 95)
print('### Test 1: the training-period best t against the shuffled runs\n')
print('| | Best training \\|t\\| of the 676 | Its formula | 95th percentile of the 676 \\|t\\| |')
print('|---|---|---|---|')
bs = S['selected'][0]
print(f"| Real outcomes | **{S['best_abs_t']:.2f}** | {name(bs['formula'], bs['sign'])} | {S['p95_abs_t_of_676']:.2f} |")
p95s = np.array([r['p95_abs_t_of_676'] for r in R])
print(f'| 20 shuffled runs, median | {np.median(b):.2f} | | {np.median(p95s):.2f} |')
print(f'| 20 shuffled runs, min–max | {b.min():.2f}–{b.max():.2f} | | {p95s.min():.2f}–{p95s.max():.2f} |')
print(f'| **The bar:** 95th percentile of the shuffled runs\' best | **{p95:.2f}** | | |')
print(f"\nShuffled runs at or above the real best: {(b >= S['best_abs_t']).sum()} of 20. "
      f"All 20 shuffled bests, sorted: {', '.join(f'{x:.2f}' for x in np.sort(b))}.\n")
print('**Each run\'s composite, later (its holdout IC, 2015–2025-06, real outcomes):**\n')
print('| Selection | Mean weekly IC | t |')
print('|---|---|---|')
ch = S['composite_holdout']
print(f"| Real outcomes | {f(ch['mean'], 4)} | {f(ch['t'])} |")
print(f'| Shuffled runs, median | {f(np.median(h), 4)} | {f(np.median(ht))} |')
print(f'| Shuffled runs, min–max | {f(h.min(), 4)} to {f(h.max(), 4)} | {f(ht.min())} to {f(ht.max())} |')
print(f"\nShuffled composites at or above the real composite's holdout IC: {(h >= ch['mean']).sum()} of 20; with t ≥ 3: {(ht >= 3).sum()}.")
signs = [s for r in R for s in r['sign']]
slow = ('ret126', 'ret252', 'mom12_1', 'dvol21', 'twres63')
print(f"Of the 400 formulas the shuffled runs selected, {np.mean([any(x in g for x in slow) for r in R for g in r['selected']]):.0%} "
      f"involve a slow feature (126- or 252-day return, 12-1 momentum, dollar volume, 63-day twin residual) and "
      f"{np.mean(np.array(signs) > 0):.0%} are oriented positive.\n")

print('### The 20 selected formulas\n')
tr, ho = S['weeks']['train'], S['weeks']['holdout']
print(f'Oriented by the training sign, so a positive t is the selected direction working. Training: {tr[2]} weeks '
      f'{tr[0]}..{tr[1]}; holdout: {ho[2]} weeks {ho[0]}..{ho[1]}.\n')
print('| # | Formula (as used) | Training IC | Training t | Holdout IC | Holdout t |')
print('|---|---|---|---|---|---|')
for i, s in enumerate(S['selected'], 1):
    print(f"| {i} | {name(s['formula'], s['sign'])} | {f(s['sign'] * s['train_ic'], 4)} | {f(abs(s['train_t']))} | "
          f"{f(s['holdout_ic_oriented'], 4)} | {f(s['holdout_t_oriented'])} |")
hts = np.array([s['holdout_t_oriented'] for s in S['selected']])
print(f'\nHoldout t ≥ 2: {(hts >= 2).sum()} of 20; ≥ 3: {(hts >= 3).sum()}; negative: {(hts < 0).sum()}.\n')

print('### Test 2: the composite\'s holdout IC against 5-day reversal\n')
print('| Weekly rank IC with the next 5-day market-relative return | Mean | t | Weeks |')
print('|---|---|---|---|')
for lab, k in (('Composite, training 2005–2014 (in sample)', 'composite_train'), ('**Composite, holdout 2015–2025-06**', 'composite_holdout'),
               ('5-day reversal, holdout', 'reversal5_holdout'), ('Composite minus reversal, holdout (paired)', 'composite_minus_reversal_holdout')):
    v = S[k]
    print(f"| {lab} | {f(v['mean'], 4)} | {f(v['t'])} | {v['weeks']} |")
print(f"\nMean weekly rank correlation of the composite with 5-day reversal (holdout): {S['composite_vs_reversal_rank_corr_mean']:.2f}.\n")
print('| Year | Composite IC | 5-day reversal IC |')
print('|---|---|---|')
for y, (c, r) in S['composite_holdout_by_year'].items():
    lab = f'{y} (to June)' if y == '2025' else y
    print(f'| {lab} | {f(c, 4)} | {f(r, 4)} |')

print('\n### The money test (the judge), 2015-16..2024-25\n')
print('Scores = the composite every trading day from 2015-06-29, NaN before; `rebalance=5`; the judge skips the nine '
      'earlier years as having no scores. Change over random in points a year (picks minus the median random account), mean of the 10 years.\n')
print('| | 0 bps | 5 bps | 20 bps | Ahead | Years beating random at p < 0.1 (0 / 5 / 20 bps) |')
print('|---|---|---|---|---|---|')
for lab, js_rm in (('**Composite, uniform random**', ('c', 'uniform')), ('Composite, volatility-matched', ('c', 'volatility')),
                   ('5-day reversal, uniform (same years and rule)', ('r', 'uniform')), ('5-day reversal, volatility-matched', ('r', 'volatility'))):
    if js_rm[0] == 'c':
        m = MO[js_rm[1]]
        ch0, se, ah, c5, c20, br = m['change'], m['se'], m['ahead'], m['change_5bps'], m['change_20bps'], m['beat_random_p_lt_0_1']
        brs = ' / '.join(str(br[k]) for k in ('0', '5', '20'))
    else:
        js = REV[js_rm[1]]
        c0 = c2eval.change_over_random(js)
        ch0, se, ah = c0['change'], c0['se'], c0['ahead']
        c5, c20 = c2eval.change_over_random(js, 1)['change'], c2eval.change_over_random(js, 2)['change']
        brs = ' / '.join(str(b_['years']) for b_ in js['summary']['beat_random_p_lt_0_1'])
    print(f'| {lab} | {f(ch0, 1)} (se {se:.1f}) | {f(c5, 1)} | {f(c20, 1)} | {ah} of 10 | {brs} of 10 |')
m = MO['uniform']
rs = REV['uniform']['summary']
print(f"\n- **Rank IC in actual returns** (the judge's): z = {f(m['ic_ret_z'])} (5-day reversal: {f(rs['ic_ret_z'])}).")
print(f"- **Long-short, top minus bottom 10%:** {m['ls_bps_per_day']:.2f} bps a day gross, z = {f(m['ls_z'])} "
      f"(reversal: {rs['ls_bps_per_day_mean']:.2f}, z = {f(rs['ls_z'])}). Turnover is {m['ls_turnover']:.0%} of the book per rebalance, "
      f"so c bps a side costs about {m['ls_cost_per_bps']:.2f} × c bps a day: about "
      f"{m['ls_bps_per_day'] - 5 * m['ls_cost_per_bps']:.1f} bps a day left at 5 bps, "
      f"{m['ls_bps_per_day'] - 20 * m['ls_cost_per_bps']:.1f} at 20 bps.\n")
print('| Year from June 28 | Picks, 0 bps | Median random (uniform) | p | Median random (volatility) | p |')
print('|---|---|---|---|---|---|')
mv = MO['volatility']
for i, y in enumerate(m['year_starts']):
    print(f"| {y[:4]}-{int(y[2:4]) + 1:02d} | {f(m['final_pct_0bps'][i], 1)}% | {f(m['random_median_pct_0bps'][i], 1)}% | {m['p_0bps'][i]:.3f} | "
          f"{f(mv['random_median_pct_0bps'][i], 1)}% | {mv['p_0bps'][i]:.3f} |")

print('\n### The unseen months, 2025-06-30..2026-10-19 (new export and store, once)\n')
u = MN['unseen_ic']
print('| | Value |')
print('|---|---|')
print(f"| **Composite weekly IC** ({u['composite']['weeks']} decision days {u['weeks_from']}..{u['weeks_to']}) | "
      f"**{f(u['composite']['mean'], 4)}**, t = {f(u['composite']['t'])} |")
print(f"| 5-day reversal weekly IC, same weeks | {f(u['reversal5']['mean'], 4)}, t = {f(u['reversal5']['t'])} |")
su, sv = MN['uniform'], MN['volatility']
for k, c in enumerate(su['strategy']):
    print(f"| Picks at {int(c['cost_bps'])} bps | {f(c['final_pct'], 1)}%; median random {f(c['random_median_pct'], 1)}% uniform (p = {c['p']:.3f}), "
          f"{f(sv['strategy'][k]['random_median_pct'], 1)}% volatility-matched (p = {sv['strategy'][k]['p']:.3f}) |")
print(f"| **Change over random, 0 bps** | **{f(su['change'], 1)}** uniform; {f(sv['change'], 1)} volatility-matched |")
bm = {x['ticker'] or 'all': x['return_pct'] for x in su['benchmarks']}
print(f"| SPY / QQQ / all {su['candidates']:,} candidates | {f(bm['SPY'], 1)}% / {f(bm['QQQ'], 1)}% / {f(bm['all'], 1)}% |")
print(f"| Judge's rank IC in actual returns | {f(su['ic_ret_mean'], 4)}, t = {f(su['ic_ret_t'])} |")
print(f"| Long-short | {f(su['ls_bps_per_day'])} bps a day, t = {f(su['ls_t'])}; turnover {su['ls_turnover']:.0%} |")
