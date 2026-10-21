"""The 16 unseen months (2025-06-30..2026-10-19), run once after the rules were final: the plan's hedged portfolio and
its baselines on the new export, and the judge's single holdout from 2025-06-30.

    OMP_NUM_THREADS=8 python3 -I unseen.py      # writes out/unseen.txt and out/unseen.json
"""
import json
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402
from money import scores_for  # noqa: E402

LINES = []


def single_change(js, i):
    """Change over random for a single holdout. c2eval.change_over_random reads js['costs'], but the judge's
    single-holdout JSON keeps the cost levels under js['report']['strategy'] (KeyError: 'costs'); worked around here."""
    c = js['report']['strategy'][i]
    return dict(cost_bps=c['cost_bps'], change=c['final_pct'] - c['random_median_pct'], picks_pct=c['final_pct'],
                random_median_pct=c['random_median_pct'], p=c['p'])


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def main():
    E, tw = C.load(c2eval.EXPORT_NEW, cache=False)     # twins recomputed, not cached outside this folder
    say(f'new export {E["meta"]["first_day"]}..{E["meta"]["last_day"]}, {E["meta"]["n"]} stocks; gap days filled: '
        f'{E["gap_days"][-3:]} (stocks), {E["fund_gap_days"][-3:]} (funds)')
    P = C.Panel(E, tw, verbose=False)
    first = c2eval.day_index(P.days, '2025-06-27')
    assert c2eval.iso(P.days[first]) == '2025-06-27'
    warm = c2eval.day_index(P.days, '2025-01-01')      # start the book earlier so the first unseen month pays real turnover
    ks = [k for k in range(len(P.rebal)) if P.rebal[k] >= warm and P.rebal[k] + C.H + 1 <= P.T - 1]
    masks = C.ranked_sets(P, [P.s_twin, P.s_plain, P.s_mkt, P.s_mkt_scaled, P.s_twin_raw])
    strategies = {'twin': (P.s_twin, 'twin'), 'plain': (P.s_plain, None), 'mkt': (P.s_mkt, 'spy'),
                  'mkt_scaled (diag)': (P.s_mkt_scaled, 'spy')}
    runs = {(n, d): [m for m in C.run_strategy(P, ks, sig, h, d, masks) if m['t0'] >= first]
            for n, (sig, h) in strategies.items() for d in (0, 1)}
    tw_ = runs[('twin', 0)]
    say(f'unseen holds: {len(tw_)}, formed {c2eval.iso(P.days[tw_[0]["t0"]])} .. {c2eval.iso(P.days[tw_[-1]["t0"]])}, '
        f'returns through {c2eval.iso(P.days[tw_[-1]["x"]])}; ranked set {[m["n"] for m in tw_]}')
    res = {'holds': len(tw_), 'first': c2eval.iso(P.days[tw_[0]['t0']]), 'last_return_day': c2eval.iso(P.days[tw_[-1]['x']]),
           'strategies': {}}
    say(f'\n{"":<34} {"mean %/mo":>9} {"t":>6} {"n":>3} {"sum %":>7}')
    for (n, d), ms in runs.items():
        for kind in ('gross', 'net'):
            xs = [m[kind] for m in ms]
            mu, t, k_ = C.tstat(xs)
            lab = f'{n}{" (1 day late)" if d else ""} {kind}'
            say(f'{lab:<34} {100 * mu:+9.3f} {t:+6.2f} {k_:3d} {100 * np.sum(xs):+7.2f}')
            res['strategies'][f'{n}|{d}|{kind}'] = dict(mean=mu, t=t, n=k_, sum=float(np.sum(xs)),
                                                       monthly=[float(x) for x in xs])
    for base in ('plain', 'mkt'):
        for kind in ('gross', 'net'):
            dd = [x[kind] - y[kind] for x, y in zip(runs[('twin', 0)], runs[(base, 0)])]
            mu, t, k_ = C.tstat(dd)
            say(f'{"twin - " + base + " " + kind:<34} {100 * mu:+9.3f} {t:+6.2f} {k_:3d}')
            res['strategies'][f'diff|{base}|{kind}'] = dict(mean=mu, t=t, n=k_)
    say('\nby month (twin gross, net; plain gross; mkt gross), %:')
    for a, b, c in zip(runs[('twin', 0)], runs[('plain', 0)], runs[('mkt', 0)]):
        say(f'  {c2eval.iso(P.days[a["t0"]])}  {100 * a["gross"]:+6.2f} {100 * a["net"]:+6.2f}  {100 * b["gross"]:+6.2f}  '
            f'{100 * c["gross"]:+6.2f}')
    # the judge, single holdout from 2025-06-30
    s = scores_for(P, '2006-02-01')
    p = c2eval.save_scores(s, 'twin_reversal/twin_reversal_new', export=c2eval.EXPORT_NEW)
    res['judge'] = {}
    for rm in ('uniform', 'volatility'):
        txt, js = c2eval.judge(p, export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30', rebalance=21,
                               random_match=rm)
        ch = [single_change(js, i) for i in range(3)]
        rep = js['report']
        res['judge'][rm] = dict(change=ch, text=txt, ls_bps_per_day=rep['ls_bps_per_day'], ls_t=rep['ls_t'],
                                ic_ret_t=rep['ic_ret_t'], benchmarks=[(b['ticker'], b['return_pct']) for b in rep['benchmarks']])
        say(f'\n== judge, unseen months, {rm} random accounts ==')
        say(' | '.join(f'{c["cost_bps"]:.0f} bps: picks {c["picks_pct"]:+.1f}% vs random median {c["random_median_pct"]:+.1f}%'
                       f' = {c["change"]:+.2f} pts, p {c["p"]:.3f}' for c in ch))
        say(f'long-short {rep["ls_bps_per_day"]:.2f} bps/day t {rep["ls_t"]:+.2f}; IC actual returns t {rep["ic_ret_t"]:+.2f}; '
            + ', '.join(f'{b["ticker"]} {b["return_pct"]:+.1f}%' for b in rep['benchmarks']))
        say(txt)
    (C.OUT / 'unseen.txt').write_text('\n'.join(LINES) + '\n')
    (C.OUT / 'unseen.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
