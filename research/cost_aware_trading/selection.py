"""Steps 2-4 on the selection years only (stockai's test years 2010-2016; the period ends at the close of the first
2017 decision day). Reads no return after that close: the Returns table stops there and SPY is cut there.

  2. diagnostic: rank IC of the score against the market-relative return in window 1..kmax ahead;
  3. calibration: isotonic g_k, fitted on every selection day whose window ends by the period's end;
  4. the baseline and all 19 settings, at 0 / 5 / 20 bps; the selected setting = highest compound net return at
     20 bps (K = 10 excluded), and each family's best. Written to out/selected.json.

    python3 -I -B selection.py [lgbm_s4|reversal|lgbm_daily ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pickle  # noqa: E402

import numpy as np  # noqa: E402

import common as C  # noqa: E402
import engine as G  # noqa: E402

SIGNALS = {
    'lgbm_s4': dict(calendar='weekly', step=5, kmax=4, hs=(1, 2, 4)),
    'reversal': dict(calendar='weekly', step=5, kmax=4, hs=(1, 2, 4)),
    'lgbm_daily': dict(calendar='daily', step=1, kmax=5, hs=(1, 2, 5)),
}
COSTS = (0.0, 5.0, 20.0)


def run_signal(E, specs, name, cfg, R, spy_r, spy_adj, sel_end, large500=False):
    scores = np.load(C.SCORES[name])
    cal = G.weekly_calendar if cfg['calendar'] == 'weekly' else G.daily_calendar
    dec, end = cal(specs, C.SEL_YEARS)
    assert end == sel_end, (end, sel_end)
    tag = name + ('_large500' if large500 else '')
    out = {'signal': name, 'calendar': cfg['calendar'], 'large500': large500, 'decisions': len(dec),
           'first_decision': C.c2eval.iso(E['days'][dec[0]]), 'end': C.c2eval.iso(E['days'][end])}
    if not large500:
        C.log(tag, 'diagnostic')
        out['diagnostic'] = G.rank_ic(E, specs, scores, R, dec, cfg['step'], cfg['kmax'], sel_end)
        for row in out['diagnostic']:
            C.log('   IC window', row)
        C.log(tag, 'calibration')
        fit_days = np.arange(int(dec[0]), sel_end)
        calib = G.calibrate(E, specs, scores, R, fit_days, cfg['step'], cfg['kmax'], sel_end)
        with open(C.WORK / f'calib_{name}.pkl', 'wb') as f:
            pickle.dump(calib, f)
    else:
        with open(C.WORK / f'calib_{name}.pkl', 'rb') as f:
            calib = pickle.load(f)
    cg = np.cumsum(calib['g'], axis=0)
    out['G_at'] = {f'h={h}': {f'{q:g}': float(cg[h - 1][min(int(q * G.NB), G.NB - 1)])
                              for q in (0.999, 0.995, 0.99, 0.95, 0.9, 0.5, 0.1, 0.01)} for h in cfg['hs']}
    C.log(tag, 'G at percentiles', out['G_at'])
    panel = G.Panel(E, specs, scores, dec, end, calib=calib, hs=cfg['hs'], large500=large500)
    P, Pspy = G.period_products(R, panel, spy_r)
    avgP = G.avg_stock_products(panel, P)
    rows = []
    for fam, p in [G.BASELINE] + G.settings(cfg['hs']):
        park = 'spy' if fam == 'confident' else 'cash'
        res = G.simulate(panel, G.make_rule(fam, p), P, Pspy, costs=COSTS, park=park)
        s = G.summarize(res, panel, avgP, spy_adj, specs, C.SEL_YEARS, COSTS)
        s.update(family=fam, params=p, label=G.label(fam, p))
        rows.append(s)
        C.log(f"{tag:22s} {s['label']:22s} 0bp ${s['final']['0']:>10,.0f}  20bp ${s['final']['20']:>10,.0f}  "
              f"names/wk {s['names_per_week']:.2f}  stocks held {s['mean_stocks_held']:.1f}")
    out['rows'] = rows
    graded = [r for r in rows if not (r['family'] == 'buffer' and r['params']['K'] == 10)]
    best = max(graded, key=lambda r: r['final']['20'])
    out['selected'] = {'family': best['family'], 'params': best['params'], 'label': best['label']}
    out['family_best'] = {}
    for fam in ('buffer', 'switch', 'confident'):
        b = max([r for r in graded if r['family'] == fam], key=lambda r: r['final']['20'])
        out['family_best'][fam] = {'params': b['params'], 'label': b['label']}
    C.log(tag, 'SELECTED', out['selected'], 'family bests', out['family_best'])
    return out


def main():
    names = sys.argv[1:] or list(SIGNALS)
    E = C.load()
    specs = C.year_specs(E)
    h0_2010 = [s for s in specs if s['year'] == 2010][0]['h0']
    sel_end = [s for s in specs if s['year'] == 2017][0]['h0'] - 1   # the first test decision day
    C.log('selection period: decisions from', C.c2eval.iso(E['days'][h0_2010 - 1]), 'to the close of',
          C.c2eval.iso(E['days'][sel_end]))
    R = C.Returns(E, h0_2010 - 1, sel_end, div_fix=False)   # div_fix is 0 before 2022-07
    spy_r, spy_adj = C.spy_returns(E)
    spy_r = spy_r.copy(); spy_adj = spy_adj.copy()
    spy_r[sel_end:] = np.nan; spy_adj[sel_end + 1:] = np.nan  # leak guard
    for name in names:
        out = run_signal(E, specs, name, SIGNALS[name], R, spy_r, spy_adj, sel_end)
        C.save_json(out, f'selection_{name}.json')
        if name == 'lgbm_s4':
            o5 = run_signal(E, specs, name, SIGNALS[name], R, spy_r, spy_adj, sel_end, large500=True)
            C.save_json(o5, f'selection_{name}_large500.json')


if __name__ == '__main__':
    main()
