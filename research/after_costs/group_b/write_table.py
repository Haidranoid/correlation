"""Write group_b's after_costs.tsv and a checking table from the saved computations."""
import json, sys
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
sys.path.insert(0, '/home/bhoult/programming/correlation/research/after_costs/group_b')
A = '/home/bhoult/.local/share/correlation2/experiments/after_costs/'
OUT = '/home/bhoult/programming/correlation/research/after_costs/group_b/'
W = '~/.local/share/correlation2/experiments/'
SPYSRC = 'research/published_data/pubdata.py etf_adjclose(SPY)'

def r1(x):
    return f"{Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP):+}".replace('+-', '-')

J = json.load(open(A + 'group_b_judge.json'))
U = json.load(open(A + 'group_b_uml.json'))
T = json.load(open(A + 'group_b_twin_gap_fade.json'))
M = json.load(open(A + 'group_b_market_timing.json'))
ft = json.load(open('/home/bhoult/programming/correlation/research/free_timing/report_old.json'))['test3']

JUDGE_DESC = {
    '-7.5-bystanders.md': ('bystanders/bystanders_mismatch_old__r1_d0_uniform.json', '-mismatch scores held 5 days, daily rebalance'),
    '-9.1-phantom-portfolios.md': ('phantom_portfolios/scores_old__r1_d0_uniform.json', 'unwind/squeeze scores, daily rebalance; the 14 years the judge graded'),
    '+0.1-imposters.md': ('imposters/imposters_money__r5_d0_uniform.json', 'imposter scores, weekly rebalance (r5, the file-name run)'),
    '-5.5-betting-against-correlation.md': ('betting_against_correlation/bac_money_old__r21_d0_uniform.json', 'BAC scores, rebalance 21'),
    '+1.1-twin-reversal.md': ('twin_reversal/twin_reversal_old__r21_d0_uniform.json', 'twin-reversal scores s, rebalance 21'),
    '-7.7-volume-clock.md': ('volume_clock/volume_clock_old__r1_d0_uniform.json', 'volume-clock scores, daily rebalance'),
    '-11.5-analog-days.md': ('analog_days/analog_proxy_old__r5_d0_uniform.json', 'the stock proxy (20 stocks most correlated with the top-forecast sector), rebalance 5; not the sector-fund long-short'),
    '-8.2-silent-peer-drift.md': ('silent_peer_drift/silent_scores_old__r1_d0_uniform.json', 'silent-event scores, daily rebalance'),
    '-6.0-night-and-day.md': ('night_and_day/s_old__r1_d0_uniform.json', 'scores s, daily rebalance, close to close'),
    '+2.5-close-location.md': ('close_location/score_old__r5_d0_uniform.json', 'scores -P, rebalance 5'),
    '-1.3-lazy-prices.md': ('lazy_prices/scores/lazy_prices_old__r21_d0_uniform.json', '20 least-changed filings, rebalance 21; the 15 years with text'),
    '-3.2-signal-ensemble.md': ('signal_ensemble/old/composite__r5_d0_uniform.json', 'composite scores, rebalance 5'),
    '-4.3-loners.md': ('loners/loners_money_old__r21_d0_uniform.json', 'loner scores, rebalance 21'),
    '-0.1-twin-momentum.md': ('twin_momentum/twin_momentum_old__r21_d0_uniform.json', 'twin-momentum scores, rebalance 21'),
    '-2.4-formula-mining.md': ('formula_mining/composite_old__r5_d0_uniform.json', 'mined composite, rebalance 5; the 10 years 2015-16..2024-25'),
    '+2.6-breadth-thrust.md': ('breadth_thrust/scores_old__r1_d0_uniform.json', 'breadth-thrust scores, daily rebalance'),
}

def jyears(r):
    a, b = r['rows'][0][0], r['rows'][-1][1]
    return f"{r['n']} judge July-June years {a[:4]}-{int(a[:4]) + 1 - 2000:02d}..{int(b[:4]) - 1}-{b[2:4]} ({a}..{b})"

ORDER = ['-7.5-bystanders.md', '-1.9-unequal-move-lengths.md', '-9.1-phantom-portfolios.md', '+0.1-imposters.md',
         '+0.7-free-timing.md', '-5.5-betting-against-correlation.md', '+1.1-twin-reversal.md', '-7.7-volume-clock.md',
         '-11.5-analog-days.md', '-8.2-silent-peer-drift.md', '-6.0-night-and-day.md', '-16.2-twin-gap-fade.md',
         '+2.5-close-location.md', '-1.3-lazy-prices.md', '-3.2-market-timing.md', '-3.2-signal-ensemble.md',
         '-4.3-loners.md', '-0.1-twin-momentum.md', '-2.4-formula-mining.md', '+2.6-breadth-thrust.md']
rows, check = [], []
for f in ORDER:
    if f in JUDGE_DESC:
        p, desc = JUDGE_DESC[f]
        r = J[f]
        yl = jyears(r)
        if f == '-9.1-phantom-portfolios.md':
            yl = ("14 of the 18 judge July-June years 2007-08..2024-25 (2007-06-28..2025-06-27); the judge skipped "
                  "2010-11, 2013-14, 2014-15 and 2017-18 (no event); 2008-09 and 2023-24 held nothing (0% vs SPY)")
        rows.append((f, r1(r['after_cost']), 'long-only vs SPY', yl,
                     f"Mean over the graded years of the judge's long-only top-20 account at 5 bps a side ({desc}; final_pct at cost_bps 5) minus SPY's Yahoo adjusted-close total return from the close before each year's first_day to its last_day.",
                     f"{W}{p}; {SPYSRC}; research/lib/c2eval.py"))
        check.append((f, r['strat5'], r['spy'], r['after_cost'], r['cor0']))
    elif f == '-1.9-unequal-move-lengths.md':
        vals = list(U.values())
        ac_ = float(np.mean([v['after_cost'] for v in vals])); s5 = float(np.mean([v['strat5'] for v in vals])); sp = float(np.mean([v['spy'] for v in vals]))
        wm = [v for k, v in U.items() if 'money_walk_07_A5_B21' in k][0]
        rows.append((f, r1(ac_), 'long-only vs SPY', jyears(vals[0]),
                     f"The file-name number is the mean over the 20 move-length combinations, so this is the mean over the same 20 judge accounts (top 20, rebalance = B's move length) of [mean over the 18 years of (final_pct at 5 bps minus SPY's Yahoo adjusted total return)]; week->month alone is {r1(wm['after_cost'])}.",
                     f"{W}unequal_move_lengths/money_walk_01..20_*__r*_d0_uniform.json; {SPYSRC}"))
        check.append((f, s5, sp, ac_, float(np.mean([v['cor0'] for v in vals]))))
    elif f == '+0.7-free-timing.md':
        by = ft['by_year']; v = float(np.mean(list(by.values())))
        rows.append((f, r1(v), 'long-short net', '18 calendar years 2008..2025 (2025 = 2025-01..06, 6 rebalances), as the file',
                     "Taken as a self-financing overlay (P2's 12-1 momentum trades executed on the S1-timed day minus the same trades on a random window day, next open, raw): the same trades and sizes in both arms, so 5 bps a side costs the same in each and the overlay's net equals its gross; mean of the saved yearly effects.",
                     "research/free_timing/report_old.json (test3.by_year); experiments/+0.7-free-timing.md"))
        check.append((f, v, float('nan'), v, ft['points']))
    elif f == '-16.2-twin-gap-fade.md':
        m = T['main_5bps_every_trade_20bps_day_fund1bp']
        rows.append((f, r1(m['mean_yearly_compounded_pct']), 'long-short net', '18 calendar years 2008..2025 (2008-01-02..2025-06-27; 2025 is a half year), as the file',
                     "The daily all-stocks $1-long/$1-short book (in at the open, out at the close) at 5 bps on each stock trade = 4 x 5 = 20 bps a day (the same stock cost as the file's 10 bps entry penalty) plus 1 bp a side on the netted twin legs (half the saved 2-bp twin cost), from the saved daily gross; compounded within each calendar year, then averaged.",
                     "research/twin_gap_fade/out/daily_old.tsv (gross, bound, head); research/twin_gap_fade/books.py (cost definitions)"))
        check.append((f, m['mean_yearly_compounded_pct'], float('nan'), m['mean_yearly_compounded_pct'], 148.8))
    elif f == '-3.2-market-timing.md':
        rows.append((f, r1(M['after_cost']), 'long-only vs SPY', '21 calendar years 2005..2025 by trading day (2005 from the close of 2005-04-04, 2025 to the close of 2025-06-13), the file\'s 242-month span',
                     "The rule rebuilt from the saved monthly weights (100% or 50% SPY, rest cash at 0, bought at the close of t0 and held 21 days), holding Yahoo's adjusted-close SPY, 5 bps on every SPY trade (initial buy, switches, monthly trims of drift), daily returns compounded per calendar year, minus SPY's adjusted total return over the same days, averaged.",
                     f"research/market_timing/months_old.tsv; research/market_timing/engine.py, common.py; {SPYSRC}"))
        check.append((f, M['mean_rule5'], M['mean_spy'], M['after_cost'], -0.3))
with open(OUT + 'after_costs.tsv', 'w') as fh:
    fh.write('file\tafter_cost\tkind\tyears\thow\tsources\n')
    for r in rows:
        fh.write('\t'.join(r) + '\n')
with open(OUT + 'check_means.tsv', 'w') as fh:
    fh.write('file\tstrategy_or_book_yearly_at_5bps_mean_pct\tSPY_yearly_mean_pct\tafter_cost_unrounded\tfile_name_number_recomputed\n')
    for c in check:
        fh.write(f"{c[0]}\t{c[1]:+.2f}\t{c[2]:+.2f}\t{c[3]:+.3f}\t{c[4]:+.2f}\n")
print(open(OUT + 'check_means.tsv').read())
for r in rows:
    print(r[0], r[1], '|', r[2], '|', r[3])
