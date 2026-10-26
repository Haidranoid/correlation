"""B on the 16 unseen months (new export, 2025-06-30..2026-10-19), run once at the end with the frozen code.
Reported, never confirming: about 65 earlier experiments have read these months.

    python -I -B unseen.py
Runs curvature.py new and covariance.py new first (the new export's full history, so every expanding fit and B5's
weight history continue from all earlier months), then scores only months with t0 >= 2025-06-27.
Writes unseen_B.txt, unseen_B.json here.
"""
import json
import pickle
import subprocess
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/market_geometry')
import numpy as np  # noqa: E402

import b_common as B  # noqa: E402

c2eval = B.c2eval
UNSEEN = ('2025-06-27', '2026-10-19')


def step(script):
    if (B.WORK / {'curvature.py': 'curv_new.npz', 'covariance.py': 'cov_new.npz'}[script]).exists():
        B.log('already built:', script, 'new')
        return
    B.log('running', script, 'new')
    r = subprocess.run([sys.executable, '-I', '-B', str(B.HERE / script), 'new'], capture_output=True, text=True)
    (B.WORK / f'{script[:-3]}_new.log').write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f'{script} new failed: {r.stderr[-2000:]}')


def main():
    Eo = c2eval.load_export(B.EXPORTS['old'], arrays=())
    En = c2eval.load_export(B.EXPORTS['new'], arrays=())
    assert np.array_equal(En['days'][:len(Eo['days'])], Eo['days']), 'new export days do not continue the old ones'
    step('curvature.py')
    step('covariance.py')
    T_ = B.load_module('b_tests', B.HERE / 'tests.py')
    Mo_ = B.load_module('b_money', B.HERE / 'money.py')
    C_ = B.load_module('b_cov', B.HERE / 'covariance.py')
    out = {}
    L = []
    p = L.append
    # --- consistency of the new export's series with the old over the old span
    zo, zn = np.load(B.WORK / 'curv_old.npz'), np.load(B.WORK / 'curv_new.npz')
    K = len(zo['rebal'])
    assert np.array_equal(zo['rebal'], zn['rebal'][:K])
    cons = {}
    for s in ('main_ollivier', 'main_forman', 'avg_corr'):
        a, b = zo[s], zn[s][:K]
        ok = np.isfinite(a) & np.isfinite(b)
        cons[s] = {'corr': float(np.corrcoef(a[ok], b[ok])[0, 1]), 'max_abs_diff': float(np.abs(a[ok] - b[ok]).max())}
    out['consistency'] = cons
    p(f"New vs old export over the old span: {cons}")
    # --- B2
    M, groups, days, P, rfd, res, fc, Pn, ffc = T_.run('new', {'unseen': UNSEEN})
    out['b2_spy'] = res['b2_spy']
    out['b2_funds'] = res['b2_funds']
    with open(B.WORK / 'fund_forecasts_new.pkl', 'wb') as f:
        pickle.dump({'k': Pn['k'], 'f': Pn['f'], 'm3': ffc['y_risk']['m3'], 'm2': ffc['y_risk']['m2'],
                     'rv63': Pn['f_rv63'], 'date': M['date'][Pn['k']]}, f)
    um = (M['date'] >= UNSEEN[0]) & (M['date'] <= UNSEEN[1]) & np.isfinite(M['y_risk'])
    out['b2_monthly'] = {'date': M['date'][um].tolist(), 'y_dir': M['y_dir'][um].tolist(), 'y_risk': M['y_risk'][um].tolist(),
                         'f2_dir': fc['y_dir']['m2'][0][um].tolist(), 'f3_dir': fc['y_dir']['m3'][0][um].tolist(),
                         'f2_risk': fc['y_risk']['m2'][0][um].tolist(), 'f3_risk': fc['y_risk']['m3'][0][um].tolist(),
                         'ollivier': M['ollivier'][um].tolist()}
    p('\n## B2 on the unseen months (Clark-West; models fitted on every earlier month of the new export)')
    for tgt in ('y_dir', 'y_risk', 'y_mdd'):
        for key, r in res['b2_spy'][tgt]['unseen'].items():
            p(f"SPY {tgt} {key}" + ('' if r is None else f" {r['label']} [{r['first']}..{r['last']}]") + f": {T_.fmt_cw(r)}")
    for tgt in ('y_dir', 'y_risk', 'y_mdd'):
        for key, r in res['b2_funds'][tgt]['unseen'].items():
            p(f"funds {tgt} {key}: {T_.fmt_cw(r)}")
    # --- B3
    M2, days2, P2, rfd2, var21, W, ks, med_o = Mo_.build('new')
    ks_u = ks[M2['date'][ks] >= UNSEEN[0]]
    ev, mret, _, _, _ = Mo_.evaluate(M2, days2, P2, rfd2, var21, W, ks_u)
    ev.pop('random_draws_hist', None)
    better = max(('avg-correlation rule (-3.2)', 'absorption rule (-3.2)'), key=lambda s: ev['strategies'][s]['net5']['annual'])
    ev['better_32_rule'] = better
    ev['paired'] = {b: Mo_.paired(mret, 'rule (Ollivier)', b) for b in ('constant', 'volatility threshold', better)}
    ev['weights'] = W['ollivier'][ks_u].tolist()
    ev['dates'] = M2['date'][ks_u].tolist()
    ac = Mo_.after_costs(M2, days2, P2, rfd2, W, ks_u)
    # the unseen months are scaled to a year: compound over the span, then annualized
    rel, g, idx = Mo_.paths(M2['t0'][ks_u], P2, rfd2, 1)
    dr, _, _ = Mo_.simulate(W['ollivier'][ks_u], rel, g, 5e-4)
    d_idx = idx[:, 1:].ravel()
    spy = P2[d_idx] / P2[d_idx - 1] - 1.0
    nd = len(d_idx)
    ann_rule = np.prod(1 + dr.ravel()) ** (252 / nd) - 1
    ann_spy = np.prod(1 + spy) ** (252 / nd) - 1
    ev['after_costs_annualized'] = float(100 * (ann_rule - ann_spy))
    ev['after_costs_calendar'] = ac
    out['b3'] = ev
    fs = Mo_.fund_scaling('new', M2, days2, rfd2, UNSEEN[0], UNSEEN[1])
    out['fund_scaling'] = fs
    r = ev['strategies']
    p(f"\n## B3 on the unseen months: {ev['n_months']} months ({ev['first']}..{ev['last']}); rule weights {ev['weights']}")
    for s in ('rule (Ollivier)', 'buy and hold', 'constant', 'volatility threshold', 'volatility-managed',
              'avg-correlation rule (-3.2)', 'absorption rule (-3.2)', 'Forman rule'):
        x = r[s]
        p(f"  {s}: gross {100 * x['gross']['annual']:+.1f}%, 5 bps {100 * x['net5']['annual']:+.1f}%, Sharpe {x['net5']['sharpe']:.2f}, "
          f"max DD {100 * x['net5']['mdd']:.1f}%, exposure {x['exposure']:.0%}")
    p(f"  random timing median {100 * ev['random']['median']:+.1f}%, rule - median {ev['random']['number']:+.1f} pts, p {ev['random']['p']:.3f}")
    for b, x in ev['paired'].items():
        p(f"  rule - {b}: {100 * x['mean']:+.3f}% a month, t {x['t']:+.2f}")
    p(f"  after costs, annualized over the span: {ev['after_costs_annualized']:+.1f} pts vs SPY")
    if fs['months']:
        p(f"  fund risk scaling: {fs['months']} months; vol with curvature {100 * fs['with curvature (model 3)']['vol']:.2f}%, "
          f"without {100 * fs['without curvature (model 2)']['vol']:.2f}%, trailing {100 * fs['trailing 63-day volatility']['vol']:.2f}%; "
          f"RV diff without - with {fs['test_rv_without_minus_with']['mean']:+.3e}, t {fs['test_rv_without_minus_with']['t']:+.2f}")
    else:
        p('  fund risk scaling: no months')
    # --- B5
    z = np.load(B.WORK / 'cov_new.npz')
    arr = {k: z[k] for k in z.files}
    s = C_.series(arr)
    dts = np.array([c2eval.iso(arr['days'][t]) for t in s['t0']])
    m = s['scored'] & (dts >= UNSEEN[0])
    assert m.any(), 'no unseen B5 months'
    cmp_ = C_.compare(s, m)
    cmp_['first'], cmp_['last'] = dts[m][0], dts[m][-1]
    cmp_['weights_e'] = [list(C_.GRID[c]) for c in s['ch_e'][m]]
    cmp_['weights_le'] = [list(C_.GRID[c]) for c in s['ch_le'][m]]
    out['b5'] = cmp_
    d = cmp_['euc_minus_le']
    p(f"\n## B5 on the unseen months: {cmp_['n']} months ({cmp_['first']}..{cmp_['last']}); annualized MV vol Euclidean "
      f"{100 * cmp_['vol']['e']:.2f}%, log-Euclidean {100 * cmp_['vol']['le']:.2f}%, LW-252 {100 * cmp_['vol']['lw']:.2f}%, "
      f"affine {100 * cmp_['vol']['aff']:.2f}%; Euclidean - log-Euclidean {d['mean']:+.3e}, t {d['t']:+.2f}, p {d['p']:.3f}")
    (B.HERE / 'unseen_B.json').write_text(json.dumps(out, indent=1, default=float))
    txt = '\n'.join(L)
    print(txt)
    (B.HERE / 'unseen_B.txt').write_text(txt + '\n')


if __name__ == '__main__':
    main()
