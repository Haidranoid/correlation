"""The money test on the old export: scores = the twin signal on every day (the scaled 231-day twin residual ending 21
days earlier), for the universe of the latest rebalance (NaN elsewhere and before Test 1's start, 2006-03-03), graded
by the judge's walk-forward with rebalance=21 against uniform and volatility-matched random accounts. Plain 12-1
momentum is graded the same way on the same universe.

    python3 -I money.py      # writes out/money_old.txt and out/money_old.json
"""
import json
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402
from checks import start_index  # noqa: E402


def scores_for(P, signal, start_k):
    """[N, T] float32: signal on every day for the universe of the latest rebalance at or before that day."""
    blk = P.block_index()
    s = np.full((P.N, P.T), np.nan, np.float32)
    for k in range(start_k, P.K):
        days = np.nonzero(blk == k)[0]
        rows = np.nonzero(P.universe[k])[0]
        s[np.ix_(rows, days)] = signal[np.ix_(rows, days)]
    return s


def grade(path, export=c2eval.EXPORT_OLD, **kw):
    out = {}
    for rm in ('uniform', 'volatility'):
        txt, js = c2eval.judge(path, export=export, rebalance=21, random_match=rm, **kw)
        out[rm] = dict(text=txt, change=[c2eval.change_over_random(js, i) for i in range(3)],
                       summary=c2eval.summary_line(js) if 'summary' in js else None, json=js)
    return out


def main():
    E, tw = C.load(c2eval.EXPORT_OLD)
    P = C.Panel(E, tw, verbose=False, diagnostics=False)
    masks = C.ranked_sets(P, [P.s_twin, P.s_plain, P.s_mkt])
    k0 = start_index(P, C.coverage(P, masks))
    lines = [f'scores from rebalance {k0} = {c2eval.iso(P.days[P.rebal[k0]])}']
    res = {}
    for name, sig in (('twin', P.s_twin), ('plain', P.s_plain)):
        s = scores_for(P, sig, k0)
        p = c2eval.save_scores(s, f'twin_momentum/{name}_momentum_old')
        d0 = int(P.rebal[k0])
        fin = np.isfinite(s[:, d0:])
        lines.append(f'\n#### {name}: {p}; finite per day median {int(np.median(fin.sum(0)))}, positive per day median '
                     f'{int(np.median((s[:, d0:] > 0).sum(0)))}')
        del s, fin
        r = grade(p)
        res[name] = r
        for rm, x in r.items():
            lines.append(f'\n== {name}, {rm} random accounts ==')
            lines.append(x['summary'])
            for i, cb in enumerate((0, 5, 20)):
                ch = x['change'][i]
                lines.append(f'{cb:>2} bps: change over random {ch["change"]:+.2f} pts/yr (se {ch["se"]:.2f}, ahead '
                             f'{ch["ahead"]}/{ch["years"]}); per year {ch["per_year"]}')
            lines.append('test year      SPY    picks  random-median  p')
            for y in x['json']['years']:
                c0 = y['costs'][0]
                lines.append(f'{y["start"][:4]}-{y["end"][2:4]}  {y["spy_pct"]:+6.1f}%  {c0["final_pct"]:+6.1f}%  '
                             f'{c0["random_median_pct"]:+6.1f}%  {c0["p"]:.3f}  (candidates {y["candidates"]}, '
                             f'LS {y["ls_bps_per_day"]:+.2f} bps/day)')
    print('\n'.join(lines))
    (C.OUT / 'money_old.txt').write_text('\n'.join(lines) + '\n')
    (C.OUT / 'money_old.json').write_text(json.dumps(
        {n: {rm: {k: v for k, v in x.items() if k not in ('json', 'text')} |
             {'years': [dict(year=y['start'][:4], spy=y['spy_pct'], picks=y['costs'][0]['final_pct'],
                             random_median=y['costs'][0]['random_median_pct'], p=y['costs'][0]['p'],
                             candidates=y['candidates'], ls_bps_per_day=y['ls_bps_per_day'])
                        for y in x['json']['years']],
              'summary_json': x['json']['summary']}
             for rm, x in r.items()} for n, r in res.items()}, indent=1, default=float))


if __name__ == '__main__':
    main()
