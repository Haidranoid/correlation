"""After-cost numbers for group_b's judge-based files (judge's long-only account at 5 bps a side minus SPY)."""
import sys, json
sys.path.insert(0, '/home/bhoult/programming/correlation/research/after_costs/group_b')
import ac
W = '/home/bhoult/.local/share/correlation2/experiments/'
FILES = [
    ('-7.5-bystanders.md', 'bystanders/bystanders_mismatch_old__r1_d0_uniform.json'),
    ('-9.1-phantom-portfolios.md', 'phantom_portfolios/scores_old__r1_d0_uniform.json'),
    ('+0.1-imposters.md', 'imposters/imposters_money__r5_d0_uniform.json'),
    ('-5.5-betting-against-correlation.md', 'betting_against_correlation/bac_money_old__r21_d0_uniform.json'),
    ('+1.1-twin-reversal.md', 'twin_reversal/twin_reversal_old__r21_d0_uniform.json'),
    ('-7.7-volume-clock.md', 'volume_clock/volume_clock_old__r1_d0_uniform.json'),
    ('-11.5-analog-days.md', 'analog_days/analog_proxy_old__r5_d0_uniform.json'),
    ('-8.2-silent-peer-drift.md', 'silent_peer_drift/silent_scores_old__r1_d0_uniform.json'),
    ('-6.0-night-and-day.md', 'night_and_day/s_old__r1_d0_uniform.json'),
    ('+2.5-close-location.md', 'close_location/score_old__r5_d0_uniform.json'),
    ('-1.3-lazy-prices.md', 'lazy_prices/scores/lazy_prices_old__r21_d0_uniform.json'),
    ('-3.2-signal-ensemble.md', 'signal_ensemble/old/composite__r5_d0_uniform.json'),
    ('-4.3-loners.md', 'loners/loners_money_old__r21_d0_uniform.json'),
    ('-0.1-twin-momentum.md', 'twin_momentum/twin_momentum_old__r21_d0_uniform.json'),
    ('-2.4-formula-mining.md', 'formula_mining/composite_old__r5_d0_uniform.json'),
    ('+2.6-breadth-thrust.md', 'breadth_thrust/scores_old__r1_d0_uniform.json'),
]
out = {}
for f, p in FILES:
    r = ac.judge_after_cost(W + p)
    # the volatility-matched JSON must hold the same strategy account
    pv = (W + p).replace('_uniform.json', '_volatility.json')
    try:
        rv = ac.judge_after_cost(pv)
        same = abs(rv['strat5'] - r['strat5']) < 1e-9
    except FileNotFoundError:
        same = None
    print(f"{f:40s} cor0 {r['cor0']:+6.2f}  n={r['n']:2d} {r['first']}..{r['last']}  strat5 {r['strat5']:+7.2f}  SPYadj {r['spy']:+6.2f} (judge spy {r['judge_spy']:+6.2f})  after {r['after_cost']:+6.2f}  vol-same={same}")
    out[f] = r
json.dump(out, open('/home/bhoult/.local/share/correlation2/experiments/after_costs/group_b_judge.json', 'w'), indent=1)
