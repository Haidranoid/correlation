"""Prints the markdown tables for the .md from out/reproduce.json, out/part1.json, out/part2.json, out/part1_stocks.tsv."""
import csv
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'out'
rep = json.loads((OUT / 'reproduce.json').read_text())
p1 = json.loads((OUT / 'part1.json').read_text())
p2 = json.loads((OUT / 'part2.json').read_text())


def pc(x, nd=2, sign=True):
    return f"{100 * x:+.{nd}f}" if sign else f"{100 * x:.{nd}f}"


def usd(x):
    return f"${x:,.0f}"


w = sys.stdout.write
r = rep['raw_lgbm']
b = p2['main']['base']
w('#### Check first: the headline numbers before any haircut\n\n| | The file | Reproduced here |\n|---|---|---|\n')
w(f"| Picks' month return vs the average stock (raw scores, tied picks share) | +3.21% vs +1.03% (t 2.8) | "
  f"{pc(r['pick'])}% vs {pc(r['all'])}% (t {r['excess_t']:.2f}) |\n")
w(f"| Extra over comparable stocks | +2.05 points (t 3.6) | {pc(r['comparable'])} (t {r['comparable_t']:.2f}) |\n")
w(f"| $10,000 after 14 years, tie-break seed 2 (the median run) | $230,343 at 20 bps; $100,417 at 50 bps | "
  f"{usd(rep['account_end']['20'])} at 20 bps; {usd(rep['account_end']['50'])} at 50 bps; "
  f"{usd(rep['account_end']['5'])} at 5 bps (every yearly return within {rep['account_max_abs_diff_pct']:.0e} points of the grader's) |\n")
w(f"| SPY over the same 14 years | $65,036 | {usd(rep['spy_store_end'])} (Stooq store); "
  f"{usd(rep['spy_tr_end'])} total return (pubdata) |\n")
w(f"| Seed 2's own picks, 162 months (the grid starts here) | | {pc(b['pick'])}% vs {pc(b['all'])}% "
  f"(t {b['excess_t']:.2f}); {pc(b['comparable'])} over comparable stocks (t {b['comparable_t']:.2f}) |\n\n")

d = p1['diagnostics']
w('#### Part 1: raw disappearance rates (2,264 stocks alive on 2025-06-27)\n\n')
w('| Stocks on 2025-06-27 | Stocks | Disappeared | Over the window | Per month |\n|---|---|---|---|---|\n')
labels = [('all', 'All'), ('below -80%', 'Past 12 months below −80%'), ('-80% to -50%', '−80% to −50%'),
          ('-50% to 0', '−50% to 0'), ('above 0', 'Above 0'), ('price below $5', 'Price below $5'),
          ('price $5 or more', 'Price $5 or more')]
for k, lab in labels:
    x = p1['raw_rates'][k]
    w(f"| {lab} | {x['n']:,} | {x['gone']} | {100 * x['rate']:.1f}% | {100 * x['monthly']:.2f}% |\n")
w('\n')
w('#### Part 1: the hazard model\n\n| Regressor at 2025-06-27 | Coefficient | SE | z | Odds ratio |\n|---|---|---|---|---|\n')
names = {'intercept': 'Intercept', 'ret12': '12-month return (per sd)', 'down50': 'Down more than 50% (flag)',
         'vol63': '63-day volatility (per sd)', 'logp': 'Log price (per sd)', 'logdv': 'Log median dollar volume, 63 days (per sd)'}
for c in p1['coefficients']:
    w(f"| {names[c['name']]} | {c['beta']:+.3f} | {c['se']:.3f} | {c['z']:+.2f} | {c['odds_ratio']:.2f} |\n")
mo = p1['model']
w(f"\nAUC {mo['auc']:.2f}, McFadden R² {mo['mcfadden_r2']:.3f}; m = {mo['m_days']} / 21 = {mo['m']:.2f}.\n\n")
w('Calibration by tenth of predicted probability (predicted / actual over the window): ' +
  '; '.join(f"{100 * c['pred']:.1f} / {100 * c['actual']:.1f}" for c in p1['calibration']) + '.\n\n')
hz = p2['main']['hazard_month']
w(f"Monthly hazards the model gives in 2010-2023: picks {100 * hz['picks_mean']:.2f}% (median {100 * hz['picks_median']:.2f}%, "
  f"90th percentile {100 * hz['picks_p90']:.2f}%); their comparable stocks {100 * hz['comparables_mean']:.2f}%; the average "
  f"eligible stock {100 * hz['eligible_mean']:.2f}%; collapsed picks ({100 * hz['collapsed_pick_share']:.1f}% of picks) "
  f"{100 * hz['collapsed_picks_mean']:.2f}%.\n\n")

rows = list(csv.DictReader(open(OUT / 'part1_stocks.tsv'), delimiter='\t'))
w('#### Part 1: stocks that ended early in the newer store\n\n| Ticker | Last close | Return from 2025-06-27 to the last close | Last 21 trading days | 12-month return at 2025-06-27 |\n|---|---|---|---|---|\n')
for x in rows:
    if x['status'] == 'ended early':
        w(f"| {x['ticker']} | {x['new_last']} | {100 * float(x['ret_to_last']):+.1f}% | {100 * float(x['ret_last21']):+.1f}% | "
          f"{100 * float(x['ret12']):+.0f}% |\n")
w('\n')

spy = p2['spy_tr_end']


def grid(g, title):
    w(f"#### {title}\n\n| L | k | Extra over comparable stocks, points a month (t) | Extra over the average stock (t) | "
      f"$10,000 at 5 bps | at 20 bps | Rule passes |\n|---|---|---|---|---|---|---|\n")
    w(f"| none | 0 | {pc(b['comparable'])} ({b['comparable_t']:.2f}) | {pc(b['excess'])} ({b['excess_t']:.2f}) | "
      f"{usd(b['acct5'])} | {usd(b['acct20'])} | yes |\n")
    for c in g:
        w(f"| {100 * c['L']:.0f}% | {c['k']} | {pc(c['comparable'])} ({c['comparable_t']:.2f}) | {pc(c['excess'])} "
          f"({c['excess_t']:.2f}) | {usd(c['acct5'])} | {usd(c['acct20'])} | {'yes' if c['pass'] else '**no**'}"
          f"{'' if c['pass'] == c['pass_store_spy'] else ' (differs against the Stooq SPY)'} |\n")
    w(f"\nSPY: {usd(spy)} total return (the bar); {usd(p2['spy_store_end'])} in the Stooq store. Benchmarks use k = 1.\n\n")


grid(p2['main']['grids']['none'], 'Part 2: the haircut grid (seed 2 picks, 162 months; accounts over 14 years)')
w('#### Break-even k (base hazards)\n\n| L | Comparable-stock t falls to 2 at k = | 20 bps account falls to SPY at k = | Rule passes up to k = |\n|---|---|---|---|\n')
for L in ('-0.3', '-0.55', '-1.0'):
    be = p2['main']['breakeven'][L]
    f = lambda v: 'n/a' if v is None else ('below 0' if v == 'below' else f"{v:.2f}")  # noqa: E731
    lim = min(x for x in (be['comparable_t2'], be['account_spy']) if isinstance(x, float))
    w(f"| {100 * float(L):.0f}% | {f(be['comparable_t2'])} | {f(be['account_spy'])} | {lim:.2f} |\n")
w('\n')

gd = {(c['L'], c['k']): c for c in p2['main']['grids']['drawdown']}
ga = {(c['L'], c['k']): c for c in p2['main']['grids']['all']}
w('#### Crisis sensitivity\n\n| L | k | Hazards doubled in months with SPY 20% below its 1-year high: comparable (t); 20 bps account | '
  'Hazards doubled in every month: comparable (t); 20 bps account |\n|---|---|---|---|\n')
for c in p2['main']['grids']['none']:
    x, y = gd[(c['L'], c['k'])], ga[(c['L'], c['k'])]
    w(f"| {100 * c['L']:.0f}% | {c['k']} | {pc(x['comparable'])} ({x['comparable_t']:.2f}); {usd(x['acct20'])}"
      f"{'' if x['pass'] else ' **fails**'} | {pc(y['comparable'])} ({y['comparable_t']:.2f}); {usd(y['acct20'])}"
      f"{'' if y['pass'] else ' **fails**'} |\n")
w('\n')
for L in ('-0.3', '-0.55', '-1.0'):
    be = p2['main']['breakeven'][L]
    w(f"Break-even k for comparable t = 2 at L = {100 * float(L):.0f}%: {be['comparable_t2_drawdown']:.2f} (drawdown months doubled), "
      f"{be['comparable_t2_all_doubled']:.2f} (all doubled).\n")
w('\n')
gn = {(c['L'], c['k']): c for c in p2['no_logp']['grids']['none']}
for key in ((-0.3, 1), (-0.55, 2)):
    c = gn[key]
    w(f"Without log price (L = {100 * key[0]:.0f}%, k = {key[1]}): comparable {pc(c['comparable'])} (t {c['comparable_t']:.2f}), "
      f"account {usd(c['acct20'])} at 20 bps; passes: {c['pass']}.\n")
w(f"\nVerdict: {p2['verdict']}; against the Stooq SPY: {p2['verdict_store_spy']}; drawdown crisis: "
  f"{p2['verdict_crisis_drawdown']}; all hazards doubled: {p2['verdict_crisis_all']}; without log price: {p2['verdict_no_logp']}.\n")
