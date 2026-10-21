"""The money test: the daily composite graded by the judge with rebalance=5 (no buffer: the judge can't hold one),
uniform and volatility-matched random accounts.

    python3 -I money.py old    -> walk-forward 2006-07..2024-25: money_old.txt / .json
    python3 -I money.py new    -> the unseen months 2025-06-30..2026-10-19, once: money_new.txt / .json
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json
from common import c2eval, np, HERE, WORK


def grade(tag):
    p = WORK / tag / 'composite.npy'
    res, out = {}, []
    for rm in ('uniform', 'volatility'):
        if tag == 'old':
            txt, js = c2eval.judge(p, rebalance=5, random_match=rm)
        else:
            txt, js = c2eval.judge(p, export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30', rebalance=5,
                                   random_match=rm)
        cor = c2eval.change_over_random(js)
        r = {'change_over_random': cor,
             'at_cost': [c2eval.change_over_random(js, cost_index=ci)['change'] for ci in (0, 1, 2)]}
        out.append(f'===== random_match={rm} =====\n{txt}')
        if tag == 'old':
            out.append('SUMMARY ' + c2eval.summary_line(js))
            ys = js['years']
            r['years'] = [{'start': y['start'], 'picks': y['costs'][0]['final_pct'], 'random': y['costs'][0]['random_median_pct'],
                           'spy': y['spy_pct'], 'p': y['costs'][0]['p'], 'candidates': y['candidates']} for y in ys]
            r['picks_mean'] = float(np.mean([y['costs'][0]['final_pct'] for y in ys]))
            r['random_mean'] = float(np.mean([y['costs'][0]['random_median_pct'] for y in ys]))
            r['beat_p01'] = [b for b in js['summary']['beat_random_p_lt_0_1']]
            r['summary'] = {k: js['summary'][k] for k in ('ic_ret_z', 'ls_bps_per_day_mean', 'ls_z', 'ls_turnover_mean', 'years')}
        else:
            rep = js['report']
            r['strategy'] = rep['strategy']
            r['benchmarks'] = rep.get('benchmarks')
            r['report'] = {k: rep[k] for k in ('candidates', 'days', 'first_day', 'last_day', 'ic_ret_mean', 'ic_ret_t',
                                               'ls_bps_per_day', 'ls_t', 'ls_turnover')}
        res[rm] = r
        out.append(f"CHANGE OVER RANDOM ({rm}, 0 bps): {cor['change']:+.2f}; per year {cor['per_year']}; "
                   f"ahead {cor['ahead']}/{cor['years']}; se {cor['se']:.2f}; at 0/5/20 bps {np.round(r['at_cost'], 2).tolist()}")
    (HERE / f'money_{tag}.txt').write_text('\n'.join(out) + '\n')
    (HERE / f'money_{tag}.json').write_text(json.dumps(res, indent=1, default=float))
    print('\n'.join(l for l in out if l.startswith(('SUMMARY', 'CHANGE'))))


if __name__ == '__main__':
    grade(_sys.argv[1])
