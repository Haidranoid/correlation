"""Build group_a's after_costs.tsv (and after_costs_check.tsv) from judge_accounts.json, c2_summary.json and the two
single-period results. Run after judge_accounts.py and c2_summary.py: python3 -I -B build_table.py
"""
import json
import sys
from pathlib import Path

HERE = Path('/home/bhoult/programming/correlation/research/after_costs/group_a')
sys.path.insert(0, str(HERE))
import spy  # noqa: E402

J = json.loads((HERE / 'judge_accounts.json').read_text())
C = json.loads((HERE / 'c2_summary.json').read_text())
SA = 'stockai/results'
W = '~/.local/share/correlation2/experiments'
SPYSRC = 'research/published_data/pubdata.py etf_adjclose SPY (spy.py)'
POC_YEARS = '2010-06-28..2024-06-27: 14 stockai June-June test years (2010-2023), the judge\'s June 28 to June 27 years'


def fmt(x):
    return f'{x:+.1f}'


rows, checks = [], []


def add(file, after, kind, years, how, sources, acct5=None, spy5=None, extra=''):
    rows.append([file, after, kind, years, how, sources])
    checks.append([file, after, '' if acct5 is None else f'{acct5:+.2f}', '' if spy5 is None else f'{spy5:+.2f}', extra])


def judge_row(file, json_path, what, extra=''):
    d = J[file]
    add(file, fmt(d['after_cost']), 'long-only vs SPY', POC_YEARS if d['years'] == 14 else d['span'],
        f"Mean over the {d['years']} judge years of ({what} yearly return at 5 bps per side, read from the judge JSON) minus SPY's total return over the same judge year.",
        f'{json_path}; {SPYSRC}', d['acct5_mean'], d['spy_yahoo_mean'],
        f"0-bps mean {d['acct0_mean']:+.2f}, random median {d['rand0_mean']:+.2f} (change over random {d['change_over_random_0bps']:+.2f}). {extra}")


def seed_row(file, tag, prefix, what):
    d = J[file]
    t = d['ticker_order_ties']
    seeds = ', '.join(f"s{s} {v['after_cost']:+.1f}" for s, v in d['by_seed'].items())
    add(file, fmt(d['median_seed_after_cost']), 'long-only vs SPY', POC_YEARS,
        f"Median over the 5 random tie-break runs (as the file quotes its accounts) of the mean over the 14 judge years of ({what} top-10 account's yearly return at 5 bps per side minus SPY's total return that year).",
        f'{SA}/{tag}/judge/jitter_{prefix}_s1..s5_uniform.json, {SA}/{tag}/summary.md; {SPYSRC}',
        d['median_seed_acct5_mean'], d['median_seed_spy_yahoo_mean'],
        f"seeds {seeds}; the median seed is also the one whose 20-bps final matches the file's dollar figure; ties broken by ticker order ({prefix}_uniform.json) give {t['after_cost']:+.1f}.")


def c2_row(file, what, fam_note, src):
    d = C[file]
    add(file, fmt(d['after_cost']), 'long-only vs SPY',
        f"{d['span']}: {d['years']} c2 walk-forward years (June 28 to June 27)",
        f"Mean over the walk-forward years of ({what} yearly return at 5 bps per side minus SPY's total return); the 5-bps returns were recomputed by running c2's single holdout for each year on a copy of the price store cut at that year's end, and every year's 0-bps result equals the walk-forward row.",
        f'{src}; research/after_costs/group_a/runs/; {SPYSRC}', d['acct5_mean'], d['spy_yahoo_mean'],
        f"recomputed (exact mean {d['after_cost']:+.3f}); 0-bps mean {d['acct0_mean']:+.2f}, random median {d['rand0_mean']:+.2f} (change over random {d['change_over_random_0bps']:+.2f}, as the file).{fam_note}")


def none_row(file, how, sources):
    add(file, 'none', 'no trading', 'n/a', how, sources)


# 1. v1 pair finder
d = J['-3.6-v1-pair-finder.md']
add('-3.6-v1-pair-finder.md', fmt(d['after_cost']), 'long-only vs SPY', f"{d['span']}: 18 judge walk-forward years 2007-08..2024-25 (June 28 to June 27)",
    "Mean over the 18 graded years of (the headline w = 30, rebalance-21 picks account's yearly return at 5 bps per side, from the judge JSON) minus SPY's total return that year.",
    f'experiments/-3.6-v1-pair-finder.md; {W}/v1_pair_finder/aligned_w30__r21_d0_uniform.json; research/v1_pair_finder/money_results.txt; {SPYSRC}',
    d['acct5_mean'], d['spy_yahoo_mean'], f"0-bps mean {d['acct0_mean']:+.2f}, random {d['rand0_mean']:+.2f} (change over random {d['change_over_random_0bps']:+.2f})")

# 2. pair lead-lag holdout: one year, 5 bps already in the output
s69 = spy.spy_pct('2024-06-28', '2025-06-27')
a69 = 100 * (9199 / 10000 - 1)
add('-22.0-pair-lead-lag-holdout.md', fmt(a69 - s69), 'long-only vs SPY', '2024-06-28..2025-06-27: the one holdout year',
    "The correlation picks' return at 5 bps per side over the holdout year ($9,199 from $10,000, -8.0%) minus SPY's total return over the same year.",
    f'experiments/-22.0-pair-lead-lag-holdout.md; research/results.txt; research/leadlag_holdout.py; {SPYSRC}', a69, s69,
    'SPY from the close of 2024-06-27 to 2025-06-27, as the script does; store SPY gave 13.6%')

none_row('+0.0-lag-spectrum.md',
         "Not a trading result (similarity of lagged correlation matrices between periods) and the file has no strategy; the money tests of lead-lag pairs are -22.0-pair-lead-lag-holdout.md and the census's -11.4-census-picks.md.",
         'experiments/+0.0-lag-spectrum.md; research/lag_spectrum.txt')
none_row('+2.0-correlation-census.md',
         "Not a trading result (strong lagged correlations against the sign-flip null); its money test is -11.4-census-picks.md.",
         'experiments/+2.0-correlation-census.md')

c2_row('-11.4-census-picks.md', "the census picks' daily-target top-20 account's", '',
       'experiments/-11.4-census-picks.md; research/census_main_walkforward.txt, census_main_holdout.txt; c2 census holdout --name main on stooq-2025-06-27')
c2_row('+5.4-reversal-control.md', "the 5-day reversal rule's weekly top-20 account's",
       " For 17 of the 18 years the 5-bps returns also equal stockai's judge reproduction (stockai/results/poc/g0_reversal_family.json) within 0.005 points.",
       'experiments/+5.4-reversal-control.md; research/walkforward_reversal_5.txt; stockai/results/poc/g0_reversal_family.json; c2 census holdout --family reversal --lookback 5 on stooq-2025-06-27')
c2_row('-5.2-peer-momentum.md', "the peer-momentum monthly top-20 account's", '',
       'experiments/-5.2-peer-momentum.md; research/peer_momentum_walkforward.txt, peer_momentum_holdout.txt; c2 census holdout --family peers on stooq-2025-06-27')

none_row('+0.8-forecast-from-relationships.md',
         "Not a trading result (direction hit rate of forecasts) and the file trades nothing; the same census relationships are traded in -11.4-census-picks.md and, on the unseen months, -19.9-out-of-sample-16-months.md.",
         'experiments/+0.8-forecast-from-relationships.md; research/oos_2026-10-19.txt')
none_row('+4.0-outside-data-leaders.md',
         "Not a trading result (strong correlations against chance) and the file has no strategy; the outside data's money tests are +20.9-outside-data-inputs-1-week.md, +10.9-outside-data-inputs-1-month.md and -6.6-outside-data-inputs-2-months.md.",
         'experiments/+4.0-outside-data-leaders.md; research/external_test_results.txt')

judge_row('-19.8-stockgpt-next-day.md', f'{SA}/poc/gpt_ensemble__r1_d0_uniform.json', "the stockGPT ensemble's daily top-20 account's")
judge_row('-4.9-lightgbm-next-day.md', f'{SA}/poc/lgbm__r1_d0_uniform.json', "the LightGBM (census target) daily top-20 account's")
judge_row('+5.8-lightgbm-next-day-actual-returns.md', f'{SA}/poc/lgbm_ret__r1_d0_uniform.json', "the LightGBM (actual-return target) daily top-20 account's")
dm = J['+2.1-momentum-12-1.md (all 17 judge years, for reference)']
judge_row('+2.1-momentum-12-1.md', f'{SA}/poc/momentum_12_1__r21_d0_uniform.json (its 14 years from 2010-06-28)', "the 12-1 momentum monthly top-20 account's",
          f"The JSON also holds 2007-08..2009-10; over all 17 years the number would be {dm['after_cost']:+.1f}.")

seed_row('+18.6-monthly-top10-picks.md', '30d', 'lgbm', "the LightGBM monthly")
seed_row('+7.4-weekly-top10-picks.md', '1w', 'lgbm', "the LightGBM weekly")
seed_row('+3.6-two-month-top10-picks.md', '2m', 'lgbm', "the LightGBM two-month")
judge_row('-5.7-stockgpt-1-week.md', f'{SA}/1w/judge/gpt_uniform.json', "the stockGPT (3-seed) weekly top-10 account's")
judge_row('-7.2-stockgpt-1-month.md', f'{SA}/30d/judge/gpt_uniform.json', "the stockGPT (3-seed) monthly top-10 account's")
judge_row('-3.0-stockgpt-2-months.md', f'{SA}/2m/judge/gpt_uniform.json', "the stockGPT (3-seed) two-month top-10 account's")
seed_row('+20.9-outside-data-inputs-1-week.md', 'ext-1w', 'lgbm_ext', "the LightGBM-with-outside-data weekly")
seed_row('+10.9-outside-data-inputs-1-month.md', 'ext-30d', 'lgbm_ext', "the LightGBM-with-outside-data monthly")
seed_row('-6.6-outside-data-inputs-2-months.md', 'ext-2m', 'lgbm_ext', "the LightGBM-with-outside-data two-month")

# 23. the unseen months: census picks, 5 bps already in the output; scaled to a year as the file scales its +0.8 to +0.6
s06 = spy.spy_pct('2025-06-30', '2026-10-19')
a06 = 100 * (10042 / 10000 - 1)
diff = a06 - s06
add('-19.9-out-of-sample-16-months.md', fmt(diff * 252 / 329), 'long-only vs SPY', '2025-06-30..2026-10-19: the 16 unseen months (329 trading days), scaled to a year x 252/329 as the file does',
    "The census picks' return at 5 bps per side over the 16 unseen months ($10,042, +0.4%) minus SPY's total return over the same days, times 252/329 (the file's own scaling of +0.8 to +0.6).",
    f'experiments/-19.9-out-of-sample-16-months.md; research/oos_2026-10-19.txt (section 1); {SPYSRC}', a06, s06,
    f"16-month difference {diff:+.2f} points; compound-annualized difference {100 * ((1 + a06 / 100) ** (252 / 329) - 1) - 100 * ((1 + s06 / 100) ** (252 / 329) - 1):+.2f}; store SPY gave 24.6%")

c2_row('-4.5-cluster-lead-lag.md', "the cluster lead-lag (paper settings) daily top-20 account's",
       ' The walk-forward was made on stooq-2026-10-19 (its 2024-25 row is reproduced on that store, not on stooq-2025-06-27), so that store was cut.',
       'experiments/-4.5-cluster-lead-lag.md; research/cluster_leadlag_walkforward.txt; c2 census holdout --family clusters --name main-2026-10-19 on stooq-2026-10-19')

none_row('+5.1-co-movement-tools.md',
         "A risk tool: its number is a next-year volatility reduction, and the file reports no return or trading result for any of the four tools.",
         'experiments/+5.1-co-movement-tools.md')

order = """-3.6-v1-pair-finder.md
-22.0-pair-lead-lag-holdout.md
+0.0-lag-spectrum.md
+2.0-correlation-census.md
-11.4-census-picks.md
+5.4-reversal-control.md
-5.2-peer-momentum.md
+0.8-forecast-from-relationships.md
+4.0-outside-data-leaders.md
-19.8-stockgpt-next-day.md
-4.9-lightgbm-next-day.md
+5.8-lightgbm-next-day-actual-returns.md
+2.1-momentum-12-1.md
+18.6-monthly-top10-picks.md
+7.4-weekly-top10-picks.md
+3.6-two-month-top10-picks.md
-5.7-stockgpt-1-week.md
-7.2-stockgpt-1-month.md
-3.0-stockgpt-2-months.md
+20.9-outside-data-inputs-1-week.md
+10.9-outside-data-inputs-1-month.md
-6.6-outside-data-inputs-2-months.md
-19.9-out-of-sample-16-months.md
-4.5-cluster-lead-lag.md
+5.1-co-movement-tools.md""".split('\n')
by = {r[0]: r for r in rows}
cb = {r[0]: r for r in checks}
assert sorted(by) == sorted(order), set(by) ^ set(order)
for r in rows:
    assert all('\t' not in str(x) and '\n' not in str(x) for x in r), r
(HERE / 'after_costs.tsv').write_text('file\tafter_cost\tkind\tyears\thow\tsources\n' + ''.join('\t'.join(by[f]) + '\n' for f in order))
(HERE / 'after_costs_check.tsv').write_text('file\tafter_cost\tmean_yearly_return_at_5bps\tmean_spy_total_return\tnotes\n'
                                            + ''.join('\t'.join(cb[f]) + '\n' for f in order))
for f in order:
    print(f"{f:44s} {cb[f][1]:>6s}  5bps {cb[f][2]:>7s}  SPY {cb[f][3]:>7s}")
