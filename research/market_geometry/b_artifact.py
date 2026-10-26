"""Collect artifact_data_B.json (the explainer page's data, at most 1 MB) from B's result files.

    python -I -B b_artifact.py      (after tests.py, money.py, covariance.py, b_extra.py and unseen.py)
"""
import json
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/market_geometry')
import numpy as np  # noqa: E402

import b_common as B  # noqa: E402


def r(x, d=4):
    if x is None:
        return None
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return round(float(x), d)


def main():
    T = json.loads((B.HERE / 'tests_old.json').read_text())
    Mo = json.loads((B.HERE / 'money_old.json').read_text())
    C = json.loads((B.HERE / 'covariance_old.json').read_text())
    X = json.loads((B.HERE / 'extra_old.json').read_text())
    U = json.loads((B.HERE / 'unseen_B.json').read_text())
    b2d = T['b2_spy']['y_dir']['2008-2019']['m2->m3']
    b2r = T['b2_spy']['y_risk']['2008-2019']['m2->m3']
    b5 = C['2006-2019']['euc_minus_le']
    full = Mo['periods']['2005-04..2025-06']
    out = {}
    out['claim'] = ("Luca: treat the market's correlations as a curved space, as general relativity treats gravity; the "
                    "curvature of that space should tell you where aggregates (the market, funds, groups) go next.")
    out['version_tested'] = [
        {'geometry': 'G, the stock graph',
         'what': 'Each month, the 500 most-traded US stocks; edges join the 5% most-correlated pairs over 63 days '
                 '(6,237 edges); edge length sqrt(2(1 - rho)). Ollivier-Ricci curvature (optimal transport between '
                 'neighbourhoods) and Forman curvature (degrees and triangles), averaged over edges.',
         'tests': 'B1 (what it is made of), B2 (does it improve forecasts of SPY next month?), B3 (a timing rule)'},
        {'geometry': 'M, the covariance manifold',
         'what': 'Covariance matrices of the same 500 stocks over 21, 63 and 252 days, shrunk; combined either as '
                 'plain matrices (Euclidean) or in matrix-log coordinates (log-Euclidean); judged by the realized risk '
                 'of the minimum-variance portfolio each builds.',
         'tests': 'B5'},
        {'geometry': 'S, the macro state space',
         'what': '(market return, volatility, average correlation) with a learned metric; do markets follow its geodesics?',
         'tests': 'B4: deferred by the plan, not run'}]
    out['verdict'] = {
        'short': 'No: curvature adds nothing to forecasts of the market\'s direction or its risk, a curvature timing rule '
                 'matched random timing, and the curved-space covariance representation ' +
                 ('did not beat the flat one.' if not C['pass'] else 'beat the flat one (exploratory).'),
        'b2_direction': 'fail', 'b2_risk': 'fail', 'b5': 'pass (exploratory)' if C['pass'] else 'fail',
        'b3_market_rule': 'fail', 'b3_fund_risk_scaling': 'fail'}
    out['key_numbers'] = [
        {'name': 'B2 direction, Clark-West t', 'value': r(b2d['t'], 2), 'bar': 3.0, 'p_one_sided': r(b2d['p'], 4),
         'meaning': 'Adding Ollivier curvature to the forecast of SPY\'s next-month excess return, beyond volatility, '
                    'average correlation, the absorption ratio and plain graph statistics, 2008-2019. Needed t >= 3.'},
        {'name': 'B2 risk, Clark-West t', 'value': r(b2r['t'], 2), 'bar': 3.0, 'p_one_sided': r(b2r['p'], 4),
         'meaning': 'The same for SPY\'s next-month volatility: the forecast got slightly worse with curvature.'},
        {'name': 'Curvature explained by plain graph statistics, R^2', 'value': r(T['b1']['ollivier']['stats_all']['r2'], 3),
         'meaning': 'How much of the monthly Ollivier series four counting statistics (degree concentration, '
                    'triangles, edge correlation, largest component) reproduce. Forman: exactly 1.'},
        {'name': 'B3 timing rule minus random timing, points a year', 'value': Mo['number'],
         'meaning': 'Holding 100% SPY in calm-curvature months and 50% otherwise, 2005-04..2025-06, against 10,000 '
                    'random orders of the same months (p ' + f"{full['random']['p']:.2f}" + '). After 5 bps costs, '
                    + f"{Mo['after_costs']['after_cost']:+.1f}" + ' points a year against SPY.'},
        {'name': 'B5 Euclidean minus log-Euclidean risk, t', 'value': r(b5['t'], 2), 'bar': 3.0, 'p_one_sided': r(b5['p'], 4),
         'meaning': 'Paired monthly difference in the minimum-variance portfolio\'s realized variance, 2006-2019; '
                    'positive would favour the log-Euclidean (matrix-log) representation.'},
        {'name': 'Odd vs even days, curvature correlation', 'value': r(T['checks']['ollivier']['odd_even_corr'], 2),
         'meaning': 'Ollivier curvature computed from a window\'s odd days against its even days, across months. '
                    'The halves agree only moderately, so much of the monthly movement is estimation noise.'}]
    # chart 1: curvature vs drawdowns
    mon = T['monthly']
    dd = T['spy_drawdown_daily']
    zo = [B.trailing_z(np.array([np.nan if v is None else v for v in mon['ollivier']]), k) for k in range(len(mon['date']))]
    out['chart_curvature_vs_drawdown'] = {
        'description': 'Monthly mean Ollivier-Ricci curvature of the 5% graph (and its z-score against the previous 60 '
                       'month-ends), against SPY\'s drawdown from its running peak. The plan reads high Ollivier as '
                       'the crisis side (Sandhu et al.).',
        'months': mon['date'], 'ollivier': [r(v) for v in mon['ollivier']], 'ollivier_z': [r(v, 3) for v in zo],
        'forman': [r(v, 2) for v in mon['forman']], 'avg_corr': [r(v) for v in mon['avg_corr']],
        'spy_drawdown_dates': dd['date'], 'spy_drawdown': dd['dd'],
        'unseen_months': U['b2_monthly']['date'], 'unseen_ollivier': [r(v) for v in U['b2_monthly']['ollivier']],
        'drawdowns_15pct': [{'peak': d['peak'], 'trough': d['trough'], 'depth': r(d['depth'], 3),
                             'ollivier_first_z_above_1_months_from_peak': d['ollivier']['first_beyond_1_months_from_peak']}
                            for d in T['drawdowns']],
        'base_rate_note': f"A z-score above +1 occurs in {100 * X['base_rate_ollivier']['share_months']:.0f}% of months and at "
                          f"least once in {100 * X['base_rate_ollivier']['share_13_month_spans_with_one']:.0f}% of all "
                          "13-month spans, so a spike in the year before a peak is what chance gives."}
    # chart 2: forecast skill with and without curvature
    def cum_gain(y, f2, f3):
        out_, c = [], 0.0
        for a, b, c3 in zip(y, f2, f3):
            if a is None or b is None or c3 is None:
                out_.append(None)
                continue
            c += (a - b) ** 2 - (a - c3) ** 2
            out_.append(round(c, 6))
        return out_
    sel = [i for i, d in enumerate(mon['date']) if '2008-01-01' <= d <= '2025-06-27']
    pick = lambda key: [mon[key][i] for i in sel]  # noqa: E731
    out['chart_forecast_skill'] = {
        'description': 'Cumulative out-of-sample squared-error gain from adding Ollivier curvature (model 3 vs model 2), '
                       'by month; rising means curvature helps. Also the Clark-West t of each comparison (bar 3).',
        'months': [mon['date'][i] for i in sel],
        'risk_cum_gain': cum_gain(pick('y_risk'), pick('f2_risk'), pick('f3_risk')),
        'direction_cum_gain': cum_gain(pick('y_dir'), pick('f2_dir'), pick('f3_dir')),
        'risk_actual': [r(v) for v in pick('y_risk')], 'risk_forecast_without': [r(v) for v in pick('f2_risk')],
        'risk_forecast_with': [r(v) for v in pick('f3_risk')],
        'cw_t': {tgt: {per: {cmp: r(v['t'], 2) for cmp, v in T['b2_spy'][tgt][per].items()}
                       for per in T['b2_spy'][tgt]} for tgt in ('y_dir', 'y_risk', 'y_mdd')},
        'cw_t_unseen': {tgt: {cmp: r(v['t'], 2) for cmp, v in U['b2_spy'][tgt]['unseen'].items()} for tgt in ('y_dir', 'y_risk')},
        'labels': {'m2->m3': 'Ollivier beyond graph statistics (the key test)', 'm1->m4': 'Forman vs baseline',
                   'm1->m2': 'plain graph statistics vs baseline', 'm1->m3': 'statistics + Ollivier vs baseline',
                   'm1->m_samal': "Samal et al.'s Forman vs baseline", 'm1->m_pub_oll': 'published-graph Ollivier vs baseline',
                   'm1->m_pub_for': 'published-graph Forman vs baseline'}}
    # chart 3: B3 rule against SPY and random timing
    hist_counts, hist_edges = full['random_draws_hist']
    out['chart_b3_rule'] = {
        'description': 'Growth of $1 at 5 bps costs, bought at the close after each month-end decision, cash at the '
                       'T-bill; and the distribution of 10,000 random timings with the same months at 100% and 50%.',
        'months': Mo['monthly']['date'], 'weight': Mo['monthly']['w_rule'],
        'value_rule': [r(v) for v in Mo['monthly']['value_rule (Ollivier)']],
        'value_spy': [r(v) for v in Mo['monthly']['value_buy and hold']],
        'value_constant': [r(v) for v in Mo['monthly']['value_constant']],
        'value_vol_threshold': [r(v) for v in Mo['monthly']['value_volatility threshold']],
        'random_hist_counts': hist_counts, 'random_hist_edges_annual': [r(v) for v in hist_edges],
        'rule_annual': r(full['strategies']['rule (Ollivier)']['gross']['annual']),
        'random_median_annual': r(full['random']['median']), 'random_p': r(full['random']['p'], 3),
        'table': {nm: {'annual_net5': r(s['net5']['annual']), 'sharpe_net5': r(s['net5']['sharpe'], 2),
                       'max_drawdown': r(s['net5']['mdd'], 3), 'exposure': r(s['exposure'], 3)}
                  for nm, s in full['strategies'].items()},
        'after_costs_vs_spy_pts': r(Mo['after_costs']['after_cost'], 2), 'file_name_number': Mo['number'],
        'unseen': {'rule_minus_random_pts': U['b3']['random']['number'], 'after_costs_annualized_pts': r(U['b3']['after_costs_annualized'], 1),
                   'weights': U['b3']['weights']}}
    # chart 4: B5 portfolio volatility by geometry
    out['chart_b5_volatility'] = {
        'description': 'Annualized realized volatility of the capped minimum-variance portfolio built from each '
                       'covariance forecast, by period. Euclidean and log-Euclidean share windows, grid, shrinkage and '
                       'weight choice; only the representation differs. LW-252 is the existing MV-LW estimator; the '
                       'affine step is B A^-1 B (reported only).',
        'periods': {p: {k: r(v, 5) for k, v in C[p]['vol'].items()} for p in
                    ('2006-2019', '2020-2025-06', '2006-2009', '2010-2014', '2015-2019')},
        'unseen': {k: r(v, 5) for k, v in U['b5']['vol'].items()},
        'labels': {'e': 'Euclidean combination', 'le': 'log-Euclidean combination', 'lw': 'LW-252 (MV-LW)',
                   'aff': 'affine-invariant step'},
        't_euclid_minus_logeuclid': {p: r(C[p]['euc_minus_le']['t'], 2) for p in ('2006-2019', '2020-2025-06')},
        'monthly': {k: [r(v, 4) for v in C['monthly'][k]] for k in ('vol_e', 'vol_le', 'vol_lw')} | {'date': C['monthly']['date']}}
    out['what_results_can_and_cant_show'] = [
        'B2 direction is the closest test of Luca\'s geometric claim. It failed: curvature did not improve the forecast '
        'of SPY\'s next-month return beyond volatility, average correlation, the absorption ratio and plain graph statistics.',
        'B2 risk failed too: Ollivier curvature added nothing to the forecast of next-month volatility beyond the same '
        f"inputs. Forman curvature (degrees and triangles) improved the risk forecast over the baseline at t "
        f"{T['b2_spy']['y_risk']['2008-2019']['m1->m4']['t']:.1f} in 2008-2019, below the bar of 3, and not in 2020-2025 "
        f"(t {T['b2_spy']['y_risk']['2020-2025-06']['m1->m4']['t']:.1f}).",
        'B3\'s rule is one allocation rule; it matched random timing at the same exposure. Its smaller drawdown than SPY '
        f"({100 * full['strategies']['rule (Ollivier)']['net5']['mdd']:.0f}% against {100 * full['strategies']['buy and hold']['net5']['mdd']:.0f}%) "
        f"is matched by the plain volatility threshold ({100 * full['strategies']['volatility threshold']['net5']['mdd']:.0f}%) and beaten by "
        f"volatility-managed SPY ({100 * full['strategies']['volatility-managed']['net5']['mdd']:.0f}%).",
        'B5 is about representing covariances, not about market curvature: the log-Euclidean metric is flat. ' +
        ('It did not beat the Euclidean combination.' if not C['pass'] else 'Its pass is exploratory until forward data confirm it.'),
        f"Curvature here is mostly simpler statistics renamed: four counting statistics explain "
        f"{100 * T['b1']['ollivier']['stats_all']['r2']:.0f}% of the Ollivier series and 100% of Forman, and rebuilding "
        f"returns from market and sector factors alone gives slightly more Ollivier curvature than the real data "
        f"({T['checks']['ollivier']['factor']['null_mean']:.3f} against {T['checks']['ollivier']['factor']['real_mean']:.3f}).",
        f"Much of a month's curvature is estimation noise: computed from a window's odd days and its even days, the two "
        f"series correlate {T['checks']['ollivier']['odd_even_corr']:.2f} across months.",
        'Twenty years hold only a handful of large drawdowns; curvature spikes before peaks occur at the rate chance gives.',
        'The stock data has only companies still listed, so crisis-period results may be biased.',
        'No result here could show that relativity\'s equations, or any law like them, govern markets; no field equation links money to this curvature.',
        'Forward confirmation on closes after 2026-10-19 is pending; the 16 unseen months were already read by about 65 experiments.']
    s = json.dumps(out, separators=(',', ':'))
    assert len(s) < 1_000_000, len(s)
    (B.HERE / 'artifact_data_B.json').write_text(s)
    print('artifact_data_B.json', len(s), 'bytes')


if __name__ == '__main__':
    main()
