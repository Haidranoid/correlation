"""The unseen months (2025-06-30..2026-10-19), run once after the rules were final: the plan's hedged portfolio and the
baselines on the new export, and the judge's single holdout from 2025-06-30 for the twin scores and plain momentum.

    python3 -I unseen.py      # writes out/unseen.txt and out/unseen.json
"""
import json
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402
from checks import start_index  # noqa: E402
from money import scores_for  # noqa: E402

LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def main():
    E, tw = C.load(c2eval.EXPORT_NEW)          # twins(E_new), cached
    say(f'new export {E["meta"]["first_day"]}..{E["meta"]["last_day"]}, {E["meta"]["n"]} stocks; gap days filled: '
        f'{E["gap_days"]} (stocks), {E["fund_gap_days"]} (funds)')
    P = C.Panel(E, tw, verbose=False, diagnostics=False)
    iso = lambda t: c2eval.iso(P.days[t])  # noqa: E731
    masks = C.ranked_sets(P, [P.s_twin, P.s_plain, P.s_mkt])
    cov = C.coverage(P, masks)
    k0 = start_index(P, cov)
    first = c2eval.day_index(P.days, '2025-06-27')
    assert iso(first) == '2025-06-27'
    warm = c2eval.day_index(P.days, '2025-01-01')   # the book starts earlier so the first unseen hold pays real turnover
    ks = [k for k in range(P.K) if P.rebal[k] >= warm and P.rebal[k] + C.HOLD <= P.T - 1]
    strategies = {'twin': (P.s_twin, 'twin'), 'plain': (P.s_plain, None), 'mkt': (P.s_mkt, 'spy'),
                  'twin signal, unhedged (diag)': (P.s_twin, None),
                  'plain signal, twin-hedged (diag)': (P.s_plain, 'twin')}
    runs = {n: [m for m in C.run_strategy(P, ks, sig, h, masks) if m['t0'] >= first] for n, (sig, h) in strategies.items()}
    tw_ = runs['twin']
    spy = {m['t0']: float(P.fund_growth(np.array([P.spy]), m['t0'], m['x'])[0] - 1) for m in tw_}
    say(f'unseen holds: {len(tw_)}, formed {iso(tw_[0]["t0"])} .. {iso(tw_[-1]["t0"])}, returns through '
        f'{iso(tw_[-1]["x"])}; ranked set {[m["n"] for m in tw_]}; Test 1 start on this export {iso(P.rebal[k0])}')
    cs = [c for c in cov if c['day'] >= '2025-07']
    say(f'twin changed k vs k-1 (mean): {np.mean([c["twin_changed"] for c in cs]):.3f}; median twin corr '
        f'{np.mean([c["median_corr"] for c in cs]):.3f}')
    res = {'holds': len(tw_), 'first': iso(tw_[0]['t0']), 'last': iso(tw_[-1]['t0']), 'last_return_day': iso(tw_[-1]['x']),
           'strategies': {}}
    say(f'\n{"":<40} {"mean %":>8} {"t":>6} {"n":>3} {"sum %":>7} {"sd %":>6} {"SR":>6}')
    for n, ms in runs.items():
        for kind in ('gross', 'net'):
            xs = np.array([m[kind] for m in ms])
            mu, t, k_ = C.tstat(xs)
            sr = float(C.sharpe(xs))
            say(f'{n + " " + kind:<40} {100 * mu:+8.3f} {t:+6.2f} {k_:3d} {100 * xs.sum():+7.2f} '
                f'{100 * xs.std(ddof=1):6.2f} {sr:+6.2f}')
            res['strategies'][f'{n}|{kind}'] = dict(mean=mu, t=t, n=k_, sum=float(xs.sum()), sd=float(xs.std(ddof=1)),
                                                   sharpe=sr, monthly=[float(x) for x in xs])
    for base in ('plain', 'mkt'):
        for kind in ('gross', 'net'):
            dd = [x[kind] - y[kind] for x, y in zip(runs['twin'], runs[base])]
            mu, t, k_ = C.tstat(dd)
            say(f'{"twin - " + base + " " + kind:<40} {100 * mu:+8.3f} {t:+6.2f} {k_:3d}')
            res['strategies'][f'diff|{base}|{kind}'] = dict(mean=mu, t=t, n=k_)
    say('\nby hold, gross % (twin [net] | plain | mkt | SPY):')
    res['by_hold'] = []
    for a, b, c in zip(runs['twin'], runs['plain'], runs['mkt']):
        say(f'  {iso(a["t0"])}  {100 * a["gross"]:+6.2f} [{100 * a["net"]:+6.2f}] | {100 * b["gross"]:+6.2f} | '
            f'{100 * c["gross"]:+6.2f} | {100 * spy[a["t0"]]:+5.1f}')
        res['by_hold'].append((iso(a['t0']), a['gross'], a['net'], b['gross'], c['gross'], spy[a['t0']]))
    res['test3_unseen_pass'] = bool(res['strategies']['twin|gross']['mean'] > 0)
    say(f'\nTest 3, unseen part (twin book gross mean positive): '
        f'{"PASS" if res["test3_unseen_pass"] else "FAIL"}')

    # the judge, single holdout from 2025-06-30
    res['judge'] = {}
    for name, sig in (('twin', P.s_twin), ('plain', P.s_plain)):
        s = scores_for(P, sig, k0)
        p = c2eval.save_scores(s, f'twin_momentum/{name}_momentum_new', export=c2eval.EXPORT_NEW)
        del s
        res['judge'][name] = {}
        for rm in ('uniform', 'volatility'):
            txt, js = c2eval.judge(p, export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30', rebalance=21,
                                   random_match=rm)
            rep = js['report']
            ch = []
            for i in range(3):
                c = rep['strategy'][i]
                ch.append(dict(cost_bps=c['cost_bps'], change=c2eval.change_over_random(js, i)['change'],
                               picks_pct=c['final_pct'], random_median_pct=c['random_median_pct'], p=c['p']))
            res['judge'][name][rm] = dict(change=ch, ls_bps_per_day=rep['ls_bps_per_day'], ls_t=rep['ls_t'],
                                          ic_ret_t=rep['ic_ret_t'], candidates=rep['candidates'],
                                          benchmarks=[(b['ticker'], b['return_pct']) for b in rep['benchmarks']])
            say(f'\n== judge, unseen months, {name}, {rm} random accounts ==')
            say(' | '.join(f'{c["cost_bps"]:.0f} bps: picks {c["picks_pct"]:+.1f}% vs random median '
                           f'{c["random_median_pct"]:+.1f}% = {c["change"]:+.2f} pts, p {c["p"]:.3f}' for c in ch))
            say(f'long-short {rep["ls_bps_per_day"]:.2f} bps/day t {rep["ls_t"]:+.2f}; IC actual returns t '
                f'{rep["ic_ret_t"]:+.2f}; candidates {rep["candidates"]}; '
                + ', '.join(f'{b["ticker"]} {b["return_pct"]:+.1f}%' for b in rep['benchmarks']))
            say(txt)
    (C.OUT / 'unseen.txt').write_text('\n'.join(LINES) + '\n')
    (C.OUT / 'unseen.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
