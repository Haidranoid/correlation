"""Writes after_costs.tsv from judge_based.json and own_books.json (run judge_based.py and own_books.py first)."""
import json
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/after_costs/group_c')
import ac_common as A  # noqa: E402

J = json.loads((A.OUT / 'judge_based.json').read_text())
O = json.loads((A.OUT / 'own_books.json').read_text())
X = '~/.local/share/correlation2/experiments/'
SPY = 'research/published_data/pubdata.py etf_adjclose SPY'
LO, LS = 'long-only vs SPY', 'long-short net'


def jyears(f):
    r = J[f]['rows']
    return f"{r[0]['first_day']}..{r[-1]['last_day']}, {len(r)} judge years (July-June)"


def judge_row(f, src, extra=''):
    how = ("Mean over the judge's years of the judge account's 5-bps yearly return (final_pct at cost_bps 5) minus SPY's "
           "Yahoo total return over the judge's benchmark window (close before first_day to close of last_day)" + extra + '.')
    return [f, A.fmt(J[f]['after_cost']), LO, jyears(f), how, f'{X}{src}; {SPY}']


R = []
R.append(['+15.7-overnight-auctions.md', A.fmt(O['+15.7-overnight-auctions.md']['after_cost']), LO,
          'calendar years 2008 (first exit 2008-01-28)..2025 (last exit 06-27), 18 years by exit day',
          "Mean over calendar years of the plan's 20-stock overnight-only account (bought at the close, sold at the next open, "
          "long only) compounded daily at 10 bps a day (5 bps a side on both legs, every day), minus SPY's Yahoo total return "
          "from the close before the year's first exit day to the close of its last exit day.",
          f'{X}overnight_auctions/money_daily_old.npz; research/overnight_auctions/money.py; {SPY}'])
R.append(judge_row('-1.6-overnight-peers.md', 'overnight_peers/scores_on_old__r21_d0_uniform.json'))
R.append(judge_row('+4.6-hidden-text-peers.md', 'hidden_text_peers/scores/hidden_text_peers_old__r21_d0_uniform.json',
                   ', over the 15 years with text'))
R.append(judge_row('-3.8-factor-momentum.md', 'factor_momentum/book_old__r21_d0_uniform.json'))
R.append(['-4.6-correlation-portfolios.md', A.fmt(O['-4.6-correlation-portfolios.md']['after_cost']), LO,
          'calendar years 2006..2025-06 (20 years; 2025 = 01-02..06-27)',
          "Mean over calendar years of the primary MV-LW portfolio's daily net return at 5 bps a side on rebalance turnover "
          "(build.py's saved net series) compounded within the year, minus SPY's Yahoo total return over the same days; the "
          "file-name number is a Sharpe difference, so the portfolio it reports is used.",
          f'{X}correlation_portfolios/series_old.npz; research/correlation_portfolios/build.py; {SPY}'])
il = O['-100.0-intraday-lead-lag.md']
R.append(['-100.0-intraday-lead-lag.md', A.fmt(il['after_cost']), LS,
          '2026-09-09..2026-10-20, the 5-minute test half (30 days, 2,280 bars), annualized to a 252-day year',
          f"The file-name number counts pairs, so its tradable book (test 3: long top tenth, short bottom tenth of followers, "
          f"held one 5-minute bar) is used: gross {il['gross_bp_per_bar']:+.3f} bp a bar minus 5 bp x turnover "
          f"{il['turnover_per_bar']:.3f} = {il['net_bp_per_bar_5bps']:+.1f} bp a bar, compounded over the 2,280 bars is "
          f"{il['compounded_test_half_pct']:+.1f}% in 30 days and {il['annualized_pct']:+.1f}% over a 252-day year (bounded "
          f"at -100%); computed from the saved per-bar means, as no per-bar series is on disk.",
          'research/intraday_lead_lag/out/results_5m.json'])
R.append(judge_row('+0.0-short-volume.md', 'short_volume/scores_G_old__r5_d0_uniform.json'))
R.append(judge_row('-3.5-theme-joiners.md', 'theme_joiners/scores_old__r21_d0_uniform.json'))
R.append(judge_row('-10.2-decoupling.md', 'decoupling/scores_old__r5_d0_uniform.json'))
R.append(judge_row('-0.6-announcement-beta.md', 'announcement_beta/scores/beta_timing_old__r1_d0_uniform.json'))
R.append(['+1.2-trend-following.md', A.fmt(O['+1.2-trend-following.md']['after_cost']), LS,
          'calendar years 2008..2025-06 (18 years of return months; 2025 = Jan-Jun)',
          "Mean over calendar years of the 23-fund TSMOM overlay's monthly excess return (self-financing; T-bill not added) "
          "net of 5 bps a side on its turnover, compounded within the year; turnover recovered from the saved series as "
          "(gross - 2-bps net) / 2 bps.",
          f'{X}trend_following/monthly_old.npz; research/trend_following/common.py; research/trend_following/results_old.txt'])
R.append(judge_row('-0.2-seasonality.md', 'seasonality/scores/season_old_raw__r1_d0_uniform.json',
                   ', with the raw-price scores (their 0-bps change, +1.87, is the file number)'))
R.append(judge_row('-4.4-betting-against-beta.md', 'betting_against_beta/bab_money_old__r21_d0_uniform.json'))
R.append(['+3.3-risk-managed-momentum.md', A.fmt(O['+3.3-risk-managed-momentum.md']['after_cost']), LS,
          'calendar years 2006..2025-06 (20 years; 2025 = Jan-Jun)',
          "The file-name number is a Sharpe difference, so its RMOM book (12-1 WML scaled to 12% volatility, all stocks, "
          "dividend-corrected) is used: mean over calendar years of its monthly net return at 5 bps a side on turnover "
          "including scale changes (build.py's saved net_m), compounded within the year.",
          f'{X}risk_managed_momentum/series_old_divfix.npz; research/risk_managed_momentum/build.py'])
R.append(judge_row('-7.4-52-week-high.md', 'week52_high/pth_money_old__r21_d0_uniform.json'))
R.append(['+0.8-pairs-trading.md', A.fmt(O['+0.8-pairs-trading.md']['after_cost']), LS,
          "calendar years 2006..2025-06 (20 years, by each 21-day block's last day; 2025 = 7 blocks to 06-27)",
          "Mean over calendar years of the committed-capital top-20 pairs book (export closes, dividend-corrected, all "
          "stocks), its 21-day block returns net of 5 bps a side (run.py's saved pnl and cost, aggregated as tests.py's "
          "series()), compounded within the year.",
          f'{X}pairs_trading/run_old_export_div_all.npz; research/pairs_trading/tests.py'])
esa = O['-6.8-etf-stat-arb.md (book, alternative)']
R.append(judge_row('-6.8-etf-stat-arb.md', 'etf_stat_arb/money_old__r1_d0_uniform.json',
                   f" (score -s, top 20, daily rebalance; the file names the judge as its number's source). Its own s-score "
                   f"book (export closes, dividend-corrected, same close, 5 bps a side on stocks and 1 bp on funds) would give "
                   f"{A.fmt(esa['judge_years_mean'])} over the same judge years ({A.fmt(esa['after_cost'])} by calendar year)"))
R[-1][5] += f'; {X}etf_stat_arb/books_old_export_fix.npz'
R.append(judge_row('-3.6-max-lottery.md', 'max_lottery/max_money_old__r21_d0_uniform.json'))
tom = O['-9.6-turn-of-month.md']
R.append(['-9.6-turn-of-month.md', A.fmt(tom['after_cost']), LO, 'calendar years 2006..2025-06 (20 years; 2025 = 01-02..06-27)',
          "Mean over calendar years of the TOM-only rule (SPY's Yahoo total return on the last and first three trading days "
          "of each month, the T-bill otherwise, built as analyze.py does) compounded daily with 5 bps a side on each position "
          f"change (the plan used 1 bp, which gives {A.fmt(tom['at_1bp_after_cost'])}), minus SPY's Yahoo total return over "
          "the same days.",
          f'research/turn_of_month/common.py; research/turn_of_month/analyze.py; {SPY}; pubdata.rf_daily'])
R.append(judge_row('+2.0-frog-in-the-pan.md', 'frog_in_the_pan/score_old_divfix__r21_d0_uniform.json',
                   ', with the dividend-corrected scores (+3.55 at 0 bps; the plain scores, +3.49, give '
                   + A.fmt(J['+2.0-frog-in-the-pan.md (plain)']['after_cost']) + ' too)'))
R[-1][5] += f'; {X}frog_in_the_pan/score_old_plain__r21_d0_uniform.json'

assert len(R) == 20 and len({r[0] for r in R}) == 20
head = ['file', 'after_cost', 'kind', 'years', 'how', 'sources']
for r in R:
    assert len(r) == 6 and not any('\t' in c or '\n' in c for c in r)
(A.OUT / 'after_costs.tsv').write_text('\n'.join('\t'.join(r) for r in [head] + R) + '\n')
print((A.OUT / 'after_costs.tsv').read_text())
