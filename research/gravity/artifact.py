"""Gravity: research/gravity/artifact_data.json, the data for an explainer page addressed to Brandon.

    python -I -B artifact.py

Reads out/tests_old.json, shape_old.json, judge_old.json, checks_old.json, pilot.json and, if present, unseen.json,
tests_new.json, judge_new.json, intraday.json; three worked examples come from the panel and the attractor tables
(the hand-print stock-days). Nothing here computes a test.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import data  # noqa: E402
np = K.np

VARS = K.variants()
KEYS = [v['key'] for v in VARS]


def r(x, n=4):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, n) if math.isfinite(x) else None


def load(name):
    p = K.OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def example(P, PE, F, tk, ds):
    tick = list(P['tickers'])
    b = tick.index(tk)
    t = K.day_of(P['days'], ds)
    refit = np.asarray(P['refit'])
    k = int(np.searchsorted(refit, t, side='right')) - 1
    u = np.asarray(P['uni'][k])
    i = int(np.nonzero(u == b)[0][0])
    lnC, lnH, lnL = (np.asarray(P[x]) for x in ('lnC', 'lnH', 'lnL'))
    win = range(t - 10, min(t + 3, P['T']))
    atts = []
    for s in range(K.K_ATT):
        a, m, be = int(PE['A1'][k, i, s]), float(PE['M1'][k, i, s]), float(PE['B1'][k, i, s])
        if a < 0:
            continue
        xs = lnC[b, t - 63:t] - be * lnC[a, t - 63:t]
        xs = xs[np.isfinite(xs)]
        mu, sd = xs.mean(), xs.std(ddof=1)
        gc = (lnC[b, t] - be * lnC[a, t] - mu) / sd
        gh = (lnH[b, t] - be * lnC[a, t] - mu) / sd
        gl = (lnL[b, t] - be * lnC[a, t] - mu) / sd
        rr = 0.0 if gl <= 0 <= gh else min(abs(gl), abs(gh))
        # where A puts B: the close of B that would make the gap zero, exp(mu + beta ln C_A)
        atts.append({'ticker': tick[a], 'mass_rho': r(m, 3), 'beta': r(be, 3), 'gap_close_sd': r(gc, 3), 'gap_high_sd': r(gh, 3),
                     'gap_low_sd': r(gl, 3), 'closest_approach_sd': r(rr, 3),
                     'attractor_price_for_B': r(math.exp(mu + be * lnC[a, t]), 2),
                     'pull_spring': r(-m * gc, 3), 'pull_p1_eps0.3': r(-m * np.sign(gc) / (rr + 0.3), 3),
                     'pull_p2_eps0.3': r(-m * np.sign(gc) / (rr + 0.3) ** 2, 3),
                     'prices': [r(math.exp(lnC[a, d]), 2) if np.isfinite(lnC[a, d]) else None for d in win]})
    fv = {VARS[j]['name']: r(F[key][b, t], 3) for j, key in enumerate(KEYS) if key.endswith('_real') and VARS[j]['kind'] in ('cand', 'bench')}
    ranks = {}
    for key in ('g1_spring_real', 'g1_5_real', 'g2_14_real', 'g3_4_real'):
        col = np.asarray(F[key][:, t])
        m = np.asarray(P['inU'])[:, t] & np.isfinite(col)
        ranks[key] = r((col[m] < col[b]).mean(), 3)
    nxt = math.exp(np.asarray(P['lnO'])[b, t + 2] - np.asarray(P['lnO'])[b, t + 1]) - 1 if t + 2 < P['T'] else None
    return {'stock': tk, 'day': ds, 'refit': K.iso(P['days'], int(refit[k])),
            'dates': [K.iso(P['days'], d) for d in win],
            'B_close': [r(math.exp(lnC[b, d]), 2) if np.isfinite(lnC[b, d]) else None for d in win],
            'B_high': r(math.exp(lnH[b, t]), 2), 'B_low': r(math.exp(lnL[b, t]), 2),
            'attractors': atts, 'forces': fv,
            'force_percentile_in_universe': {'spring': ranks['g1_spring_real'], 'G1 p=2 eps=0.3': ranks['g1_5_real'],
                                             'G2 N=21 sharp': ranks['g2_14_real'], 'G3 N=5 a=1 p=2': ranks['g3_4_real']},
            'next_open_to_open_return_pct': r(100 * nxt, 3) if nxt is not None else None}


def main():
    T = load('tests_old.json')
    S = load('shape_old.json')
    J = load('judge_old.json')
    C = load('checks_old.json')
    PI = load('pilot.json')
    U = load('unseen.json')
    TN = load('tests_new.json')
    JN = load('judge_new.json')
    IN = load('intraday.json')
    dev = T['periods']['2008-2019']
    per = {p['key']: p for p in dev['per']}
    six = T['six']

    def cand_rows(stage):
        rows = []
        for v in VARS:
            if v['stage'] != stage or v['world'] != 'real' or v['kind'] != 'cand':
                continue
            p = per[v['key']]
            rw = dev['rw'][f'{stage}_real_cand']
            rows.append({'name': v['name'], 'S': r(p['S_oo']['est'], 6), 'S_t': r(p['S_oo']['t'], 2),
                         'S_ci90': [r(x, 6) for x in p['S_oo']['ci90']], 'S_rw_p': r(rw['S_oo']['adj'][v['key']], 4),
                         'M_bps_day': r(1e4 * p['M5']['est'], 3), 'M_t': r(p['M5']['t'], 2),
                         'M_ci90_bps': [r(1e4 * x, 3) for x in p['M5']['ci90']], 'M_rw_p': r(rw['M5']['adj'][v['key']], 4),
                         'gross_bps_day': r(p['gross_bps'], 3), 'turnover': r(p['turnover'], 3),
                         'breakeven_bps': r(p['breakeven_bps'], 2), 'ic': r(p['ic']['est'], 5), 'ic_t': r(p['ic']['t'], 2),
                         'random_S_t': r(per[v['key'].replace('_real', '_random')]['S_oo']['t'], 2),
                         'random_M_bps_day': r(1e4 * per[v['key'].replace('_real', '_random')]['M5']['est'], 3)})
        return rows

    stage_rows = {st: cand_rows(st) for st in ('G1', 'G2', 'G3')}
    # force law
    fl = S['periods']['2008-2019']['force_law_hl']
    flc = S['periods']['2008-2019']['force_law_close']
    law = {'bins': S['bins'],
           'measured_real_minus_random': [r(x['diff']['est'], 5) for x in fl],
           'ci90': [[r(y, 5) for y in x['diff']['ci90']] for x in fl],
           't': [r(x['diff']['t'], 2) for x in fl],
           'real': [r(x['real'], 5) for x in fl], 'random': [r(x['random'], 5) for x in fl],
           'pairs': [x['pairs_real'] for x in fl],
           'closes_only_real_minus_random': [r(x['diff']['est'], 5) for x in flc],
           'closes_only_ci90': [[r(y, 5) for y in x['diff']['ci90']] for x in flc],
           'predicted_shape_raw': {'spring (mean |g_C|)': [r(x['mean_absg'], 4) for x in fl],
                                   **{k: [r(x['pred'][k], 4) for x in fl] for k in fl[0]['pred']}},
           'units': 'mean next open-to-open return toward the attractor, in units of the stock\'s daily sigma, per attractor-day; '
                    'predicted shapes are each law\'s mean pull in the bin, before scaling'}
    kern = {}
    for N, rows in S['periods']['2008-2019']['kernel'].items():
        N = int(N)
        kern[str(N)] = {'tau': [x['tau'] for x in rows],
                        'measured_signed_real_minus_random': [r(x['diff_signed']['est'], 5) for x in rows],
                        'ci90': [[r(y, 5) for y in x['diff_signed']['ci90']] for x in rows],
                        't': [r(x['diff_signed']['t'], 2) for x in rows],
                        'measured_unsigned_real_minus_random': [r(x['diff_unsigned']['est'], 5) for x in rows],
                        'gravity_kernels': {f'a={a:g}': [r(w, 4) for w in K.kernel_weights(N, a)] for a in (0.5, 1, 2, 3)},
                        'sharp_offset': [r(w, 4) for w in K.kernel_weights(N, np.inf)]}
    # judge by year (mean over the 30 candidates)
    by_year = [{'year': y['start'][:4] + '-' + str(int(y['start'][:4]) + 1)[2:], 'over_random_0bps': r(y['over_random0_mean'], 2),
                'account5_minus_spy': r(y['after_cost_mean'], 2), 'spy_pct': r(y['spy'], 2)} for y in J['summary']['by_year']]
    acc = [{'name': a['name'], 'stage': a['stage'], 'kind': a['kind'], 'over_random0': r(a['over_random0'], 2),
            'after_cost': r(a['after_cost'], 2), 'acct5_mean': r(a['acct5_mean'], 2)} for a in J['accounts']]
    # long-short books by period
    books = {}
    for pname, pr in T['periods'].items():
        pp = {p['key']: p for p in pr['per']}
        books[pname] = {st: {'best_S_t': r(pr['rw'][f'{st}_real_cand']['S_oo']['t_best'], 2),
                             'best_M_t': r(pr['rw'][f'{st}_real_cand']['M5']['t_best'], 2),
                             'mean_M_bps_day': r(np.mean([1e4 * pp[v['key']]['M5']['est'] for v in VARS if v['stage'] == st and v['world'] == 'real' and v['kind'] == 'cand']), 3),
                             'mean_gross_bps_day': r(np.mean([pp[v['key']]['gross_bps'] for v in VARS if v['stage'] == st and v['world'] == 'real' and v['kind'] == 'cand']), 3)}
                         for st in ('G1', 'G2', 'G3')}
        books[pname]['reversal_gross_bps_day'] = r(pr['rev']['gross_bps'], 3)
    g = {x['test']: x for x in six}
    fl0 = fl[0]
    kN = {N: [x for x in rows if x['tau'] == int(N)][0] for N, rows in S['periods']['2008-2019']['kernel'].items()}
    spring = next(a for a in J['accounts'] if a['key'] == 'g1_spring_real')
    verdict = ('No edge. The stages were taken in Brandon\'s order and the verdict stops at the first, G1 (price): '
               f"its signal test S (best candidate t {g['G1-S']['t']:+.2f}, Romano-Wolf p {g['G1-S']['p']:.2f}) and money test M "
               f"(t {g['G1-M']['t']:+.2f}, p {g['G1-M']['p']:.2f}) both fail; G2 and G3 fail too (no t above "
               f"{max(x['t'] for x in six):.2f}; the smallest Holm p is {min(x['holm_p'] for x in six):.2f}).")
    key_numbers = [
        {'value': f"t {g['G1-S']['t']:+.2f}", 'label': 'G1-S, the best price-gravity force beyond the spring and reversal',
         'meaning': 'Ranking stocks by how hard their peers pull adds nothing the old spring and plain reversal did not already say (it needed t >= 3).'},
        {'value': f"{1e4 * g['G1-M']['est']:+.2f} bps a day", 'label': 'G1-M, the best price-gravity long-short book after 5 bps a side',
         'meaning': 'The best book breaks even; the steeper laws (p >= 1) lose 3 to 12 bps a day because they replace most of the book every day.'},
        {'value': f"{fl0['diff']['est']:+.4f} sigma (t {fl0['diff']['t']:+.1f})", 'label': 'Measured pull at r = 0, real minus random peers, 2008-2019',
         'meaning': 'When a stock\'s day range touches where its peer puts it, the next day it moves toward that point by about half a percent of a typical day\'s move more than with random peers (under 1 bp). Every other distance shows no difference.'},
        {'value': ', '.join(f"{float(x['diff_signed']['est']):+.4f}" for x in kN.values()), 'label': 'Measured kernel at tau = N (N = 5, 10, 21), real minus random',
         'meaning': 'A peer\'s move N days ago does not pull at all on the offset day: there is no peak at N, in gravity\'s shape or the sharp offset\'s.'},
        {'value': f"{J['summary']['over_random0_mean_30']:+.1f} points a year", 'label': 'Improvement over random (the judge, before costs, mean of 30 candidates, 2008-09..2024-25)',
         'meaning': f"About the same as random accounts that trade as often. The spring alone, the old idea, gives {spring['over_random0']:+.1f}."},
        {'value': f"{J['summary']['after_cost_mean_30']:+.1f} points a year", 'label': 'After costs: the 20-stock account at 5 bps a side minus SPY (the file-name number)',
         'meaning': 'Daily rebalancing on a force that changes every day costs far more than any edge it has.'},
    ]
    can_cant = [
        {'case': 'A stage\'s S passes', 'applies': False, 'text': 'None of the three did. The pull\'s shape carries no information that the spring, the sharp offset and plain reversal do not.'},
        {'case': 'S passes and M fails', 'applies': False, 'text': 'Not this case: there is no gravity signal for costs to eat.'},
        {'case': 'Both pass', 'applies': False, 'text': 'Nothing goes to the forward test as a pass; it can only confirm the failure.'},
        {'case': 'Everything fails', 'applies': True, 'text': 'With these definitions (peers by correlation, mass = correlation, distance in spread standard deviations, a symmetric power-law kernel) and this grid, gravity adds nothing to what the project already found. It cannot rule out other definitions of distance or mass.'},
        {'case': 'The measured force law and kernel', 'applies': True, 'text': 'The raw pull grows with distance for real and random peers alike, which is the stock\'s own reversal in spring shape. The part due to real peers is confined to the touch bin and is tiny. The time kernel is flat at zero, with no peak at the offset.'},
        {'case': 'The unseen months and the 5-minute bars', 'applies': True, 'text': 'Reported, not confirming: all six fail on the 16 unseen months as well, and at 5-minute steps the real-minus-random pull is flat in every bin but the farthest.'},
    ]
    P = data.load_panel('old')
    PE = dict(np.load(K.BIG / 'peers_old.npz'))
    F = {k: np.load(K.BIG / 'forces_old' / f'{k}.npy', mmap_mode='r') for k in KEYS}
    examples = [example(P, PE, F, tk, ds) for tk, ds in (('XOM', '2015-12-08'), ('JPM', '2011-08-08'), ('AAPL', '2016-05-02'))]
    out = {
        'title': 'Gravity: do correlated stocks pull harder as they get closer?',
        'source': 'Brandon\'s own idea, 2026-10-24; experiments/untested-brandon-gravity.md',
        'claim': 'Treat correlations as gravitational attractors rather than fixed offsets: a related stock pulls on another '
                 'harder as it gets closer, in price, in time, or in both, with several scaling factors.',
        'version_tested': 'The 500 most-traded US stocks, 2008-2019 (development), with each stock\'s 10 most-correlated '
                          'stocks as attractors (mass = correlation). G1 price: pull = mass / (closest approach + eps)^p, '
                          'p in {0,1,2,3}, eps in {0.1,0.3,1}, using the day\'s high and low; G2 time: a lag kernel '
                          '(|tau - N| + 1)^-a around the offset N in {5,10,21}, a in {0.5,1,2,3}; G3 both: 8 combinations. '
                          'Each must beat the old version (the spring or the sharp offset) and plain reversal (S), and pay '
                          'after 5 bps a side in a daily long-short (M). 30 candidates, six primaries under Romano-Wolf and Holm.',
        'verdict': verdict,
        'six_primaries': six, 'verdict_order': T['verdict_order'],
        'key_numbers': key_numbers,
        'charts': {'force_law': law, 'time_kernel': kern, 'candidates': stage_rows, 'judge_by_year': by_year,
                   'judge_accounts': acc, 'books_by_period': books},
        'judge_summary': J['summary'],
        'worked_examples': examples,
        'checks': {'truncation_mismatches': sum(x['mismatches'] for x in C['truncation']), 'truncation_values': sum(x['values'] for x in C['truncation']),
                   'slow_mismatches': C['slow']['mismatches'], 'slow_values': C['slow']['values']},
        'pilot': {k: PI[k] for k in ('peers_full_min', 'numba_g3_full_min', 'numpy_g3_full_min', 'decision')},
        'unseen_months': {'six': TN['six'] if TN else None, 'judge': JN['summary'] if JN else None,
                          'coverage_first_day': U['first_day_force_coverage_min'] if U else None},
        'intraday_force_law': IN,
        'what_it_can_and_cant_show': can_cant,
    }
    p = K.HERE / 'artifact_data.json'
    p.write_text(json.dumps(out, separators=(',', ':')))
    print(f'wrote {p} ({p.stat().st_size / 1e3:.0f} kB)')


if __name__ == '__main__':
    main()
