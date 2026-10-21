"""Step 6, the test years 2017-2023 (2017-06-27 close .. 2024-06), with the frozen calibration and the selection in
out/selected.json (written into the .md first). Every setting is reported; the turnover-matched null (1,000 accounts,
seed 20261021) is run for the baseline, each family's best and the selected setting; the three tests are applied to
the selected setting at 20 bps with pubdata.div_fix on stock returns from 2022-07 (and every number again without it).

    python3 -I -B test.py [lgbm_s4|reversal|lgbm_daily ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import json  # noqa: E402
import pickle  # noqa: E402

import numpy as np  # noqa: E402

import common as C  # noqa: E402
import engine as G  # noqa: E402
from selection import SIGNALS, COSTS  # noqa: E402

SEED = 20261021
NR = 1000


def factor_periods(E, panel):
    """FF5 + UMD (and RF) compounded over each holding period [d_k, d_{k+1}): the day-d return is the factor's on
    export day d+1."""
    import pubdata
    d5, f5 = pubdata.ff5_daily()
    du, fu = pubdata.umd_daily()
    days = E['days']
    cols = {k: pubdata.on_days(d5, v, days) for k, v in f5.items()}
    cols['UMD'] = pubdata.on_days(du, fu['UMD'], days)
    ends = np.r_[panel.decisions[1:], panel.end]
    out = {k: [] for k in cols}
    for a, e in zip(panel.decisions, ends):
        for k, v in cols.items():
            x = v[int(a) + 1:int(e) + 1]
            out[k].append(float(np.prod(1 + np.nan_to_num(x)) - 1))
    return {k: np.array(v) for k, v in out.items()}


def alpha(wk, fac, per_year):
    y = wk - fac['RF']
    X = np.column_stack([np.ones(len(y))] + [fac[k] for k in ('Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD')])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    s2 = res @ res / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return {'alpha_per_period': float(b[0]), 't': float(b[0] / se[0]),
            'alpha_per_year': float(b[0] * per_year),
            'betas': dict(zip(('Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD'), map(float, b[1:])))}


def null_stats(Vs, Vr, panel, specs, costs, strat_sum):
    """Median random account, p (share of random finals >= the strategy's) and CAGRs, per cost; by-year medians."""
    ndays = panel.end - int(panel.decisions[0])
    out = {}
    for i, c in enumerate(costs):
        fin = Vr[i, :, -1]
        med = float(np.median(fin))
        out[f'{c:g}'] = {'median_final': med, 'median_cagr': float(G.cagr(1e4, med, ndays)),
                         'p': float((fin >= Vs[i, -1]).mean()),
                         'strategy_minus_median_cagr_pts': 100 * (float(G.cagr(1e4, Vs[i, -1], ndays))
                                                                  - float(G.cagr(1e4, med, ndays))),
                         'p05_final': float(np.percentile(fin, 5)), 'p95_final': float(np.percentile(fin, 95))}
    marks = G.year_marks(panel, specs, C.TEST_YEARS)
    by = []
    for y, a, b in zip(C.TEST_YEARS, marks[:-1], marks[1:]):
        row = {'year': y}
        for i, c in enumerate(costs):
            row[f'median_ret_{c:g}'] = float(np.median(Vr[i, :, b] / Vr[i, :, a] - 1))
        by.append(row)
    out['by_year'] = by
    return out


def run_signal(E, specs, name, cfg, Rm, Rr, spy_r, spy_adj, sel, large500=False, fac=None):
    scores = np.load(C.SCORES[name])
    with open(C.WORK / f'calib_{name}.pkl', 'rb') as f:
        calib = pickle.load(f)
    cal = G.weekly_calendar if cfg['calendar'] == 'weekly' else G.daily_calendar
    dec, end = cal(specs, C.TEST_YEARS)
    tag = name + ('_large500' if large500 else '')
    C.log(tag, 'test decisions', len(dec), C.c2eval.iso(E['days'][dec[0]]), '..', C.c2eval.iso(E['days'][end]))
    panel = G.Panel(E, specs, scores, dec, end, calib=calib, hs=cfg['hs'], large500=large500)
    Pm, Ps = G.period_products(Rm, panel, spy_r)
    Pr, _ = G.period_products(Rr, panel, spy_r)
    avgm, avgr = G.avg_stock_products(panel, Pm), G.avg_stock_products(panel, Pr)
    out = {'signal': name, 'large500': large500, 'decisions': len(dec), 'first_decision': C.c2eval.iso(E['days'][dec[0]]),
           'end': C.c2eval.iso(E['days'][end]), 'seed': SEED, 'random_accounts': NR}
    s = sel[name]
    wanted = {s['selected']['label']} | {v['label'] for v in s['family_best'].values()} | {'buffer K=10'}
    rows = []
    for fam, p in [G.BASELINE] + G.settings(cfg['hs']):
        park = 'spy' if fam == 'confident' else 'cash'
        rule = G.make_rule(fam, p)
        res = G.simulate(panel, rule, Pm, Ps, costs=COSTS, park=park)
        sm = G.summarize(res, panel, avgm, spy_adj, specs, C.TEST_YEARS, COSTS)
        res_r = G.simulate(panel, rule, Pr, Ps, costs=COSTS, park=park)
        sr = G.summarize(res_r, panel, avgr, spy_adj, specs, C.TEST_YEARS, COSTS)
        sm.update(family=fam, params=p, label=G.label(fam, p))
        sm['no_div_fix'] = {k: sr[k] for k in ('final', 'cagr', 'weekly_t_20', 'weekly_excess_20', 'by_year',
                                               'after_costs_5_minus_spy')}
        if sm['label'] in wanted:
            Vr = G.simulate_random(panel, res['events'], Pm, Ps, R=NR, seed=SEED, costs=COSTS, park=park)
            sm['null'] = null_stats(res['V'], Vr, panel, specs, COSTS, sm)
            Vr0 = G.simulate_random(panel, res_r['events'], Pr, Ps, R=NR, seed=SEED, costs=COSTS, park=park)
            sm['no_div_fix']['null'] = null_stats(res_r['V'], Vr0, panel, specs, COSTS, sr)
            if sm['label'] == s['selected']['label']:
                Vsh = G.simulate_shadow(panel, res['slots'], Pm, Ps, R=NR, seed=SEED, costs=COSTS, park=park)
                sm['null_shadow'] = null_stats(res['V'], Vsh, panel, specs, COSTS, sm)
                if fac is not None:
                    wk = res['V'][COSTS.index(20.0), 1:] / res['V'][COSTS.index(20.0), :-1] - 1
                    sm['ff5_umd'] = alpha(wk, fac, len(wk) / (sm['days'] / 252.0))
            C.log(f"   null {sm['label']}: 20bp median ${sm['null']['20']['median_final']:,.0f} p {sm['null']['20']['p']:.3f}; "
                  f"0bp diff {sm['null']['0']['strategy_minus_median_cagr_pts']:+.1f} pts")
        rows.append(sm)
        C.log(f"{tag:18s} {sm['label']:20s} 0bp ${sm['final']['0']:>10,.0f} 20bp ${sm['final']['20']:>10,.0f} "
              f"SPY ${sm['spy_final']:,.0f} names {sm['names_per_week']:.2f} t20 {sm['weekly_t_20']:+.2f}")
    out['rows'] = rows
    out['selected'] = s['selected']['label']
    return out


def tests(row):
    """The plan's three tests for the selected setting, at 20 bps (stop at the first failure; all computed)."""
    t1 = row['final']['20'] > row['spy_final']
    t2 = row['weekly_t_20'] >= 2.0
    t3 = row['final']['20'] > row['null']['20']['median_final'] and row['null']['20']['p'] < 0.10
    verdict = 'pass' if (t1 and t2 and t3) else ('fails test 1 (money)' if not t1 else
                                                   'fails test 2 (significance)' if not t2 else 'fails test 3 (skill)')
    return {'money': t1, 'significance': t2, 'skill': t3, 'verdict': verdict,
            'number_0bp': round(row['null']['0']['strategy_minus_median_cagr_pts'], 1),
            'number_20bp': round(row['null']['20']['strategy_minus_median_cagr_pts'], 1)}


def main():
    names = sys.argv[1:] or ['lgbm_s4', 'reversal', 'lgbm_daily']
    sel = json.load(open(C.OUT / 'selected.json'))
    E = C.load()
    specs = C.year_specs(E)
    lo = [s for s in specs if s['year'] == 2017][0]['h0'] - 1
    end = [s for s in specs if s['year'] == 2023][0]['t'] - 1
    C.log('returns', C.c2eval.iso(E['days'][lo]), '..', C.c2eval.iso(E['days'][end]))
    Rm = C.Returns(E, lo, end, div_fix=True)
    Rr = C.Returns(E, lo, end, div_fix=False)
    spy_r, spy_adj = C.spy_returns(E)
    for name in names:
        cfg = SIGNALS[name]
        fac = None
        if name == 'lgbm_s4':
            dec, e2 = G.weekly_calendar(specs, C.TEST_YEARS)
            fac = factor_periods(E, G.Panel(E, specs, np.load(C.SCORES['lgbm_s4']), dec, e2))
        out = run_signal(E, specs, name, cfg, Rm, Rr, spy_r, spy_adj, sel, fac=fac)
        selrow = [r for r in out['rows'] if r['label'] == out['selected']][0]
        out['tests'] = tests(selrow)
        C.log(name, 'TESTS', out['tests'])
        C.save_json(out, f'test_{name}.json')
        if name == 'lgbm_s4':
            o5 = run_signal(E, specs, name, cfg, Rm, Rr, spy_r, spy_adj, sel, large500=True)
            C.save_json(o5, f'test_{name}_large500.json')
            # the selected rule on the other tie-breaks (frozen seed-4 calibration), no null
            rob = []
            s = sel[name]['selected']
            dec, e2 = G.weekly_calendar(specs, C.TEST_YEARS)
            with open(C.WORK / 'calib_lgbm_s4.pkl', 'rb') as f:
                calib = pickle.load(f)
            for alt in ['lgbm', 'lgbm_s1', 'lgbm_s2', 'lgbm_s3', 'lgbm_s4', 'lgbm_s5']:
                pa = G.Panel(E, specs, np.load(C.SCORES[alt]), dec, e2, calib=calib)
                Pm, Ps = G.period_products(Rm, pa, spy_r)
                avgm = G.avg_stock_products(pa, Pm)
                for fam, p in (G.BASELINE, (s['family'], s['params'])):
                    res = G.simulate(pa, G.make_rule(fam, p), Pm, Ps, costs=COSTS)
                    sm = G.summarize(res, pa, avgm, spy_adj, specs, C.TEST_YEARS, COSTS)
                    rob.append({'scores': alt, 'label': G.label(fam, p), 'final': sm['final'], 'cagr': sm['cagr'],
                                'spy_final': sm['spy_final'], 'weekly_t_20': sm['weekly_t_20'],
                                'after_costs_5_minus_spy': sm['after_costs_5_minus_spy']})
                    C.log('robust', alt, G.label(fam, p), {k: round(v) for k, v in sm['final'].items()})
            C.save_json(rob, 'test_lgbm_s4_tiebreaks.json')


if __name__ == '__main__':
    main()
