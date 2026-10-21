"""The money test on the old export: scores = s (minus the scaled 21-day twin residual) on every day, for the
universe of the latest rebalance (NaN elsewhere and before 2006-02-01), graded by the judge's walk-forward with
rebalance=21 against uniform and volatility-matched random accounts.

    OMP_NUM_THREADS=8 python3 -I money.py      # writes out/money_old.txt and out/money_old.json
"""
import json
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402

START = '2006-02-01'


def scores_for(P, start_iso):
    blk = np.searchsorted(P.rebal, np.arange(P.T), side='right') - 1
    start_k = int(np.searchsorted(P.rebal, c2eval.day_index(P.days, start_iso)))
    s = np.full((P.N, P.T), np.nan, np.float32)
    for k in range(start_k, len(P.rebal)):
        days = np.nonzero(blk == k)[0]
        rows = np.nonzero(P.universe[k])[0]
        s[np.ix_(rows, days)] = P.s_twin[np.ix_(rows, days)]
    return s


def grade(path, export, **kw):
    out = {}
    for rm in ('uniform', 'volatility'):
        txt, js = c2eval.judge(path, export=export, rebalance=21, random_match=rm, **kw)
        out[rm] = dict(text=txt, change=[c2eval.change_over_random(js, i) for i in range(3)],
                       summary=c2eval.summary_line(js) if 'summary' in js else None, json=js)
    return out


def main():
    E, tw = C.load(c2eval.EXPORT_OLD)
    P = C.Panel(E, tw, verbose=False)
    s = scores_for(P, START)
    p = c2eval.save_scores(s, 'twin_reversal/twin_reversal_old')
    fin = np.isfinite(s)
    lines = [f'scores {p}: finite per day median {int(np.median(fin.sum(0)[c2eval.day_index(P.days, START):]))}, '
             f'positive per day median {int(np.median((s > 0).sum(0)[c2eval.day_index(P.days, START):]))}']
    res = grade(p, c2eval.EXPORT_OLD)
    for rm, r in res.items():
        lines.append(f'\n== {rm} random accounts ==')
        lines.append(r['summary'])
        for i, cb in enumerate((0, 5, 20)):
            ch = r['change'][i]
            lines.append(f'{cb:>2} bps: change over random {ch["change"]:+.2f} pts/yr (se {ch["se"]:.2f}, ahead '
                         f'{ch["ahead"]}/{ch["years"]}); per year {ch["per_year"]}')
        lines.append(r['text'])
    print('\n'.join(lines))
    (C.OUT / 'money_old.txt').write_text('\n'.join(lines) + '\n')
    (C.OUT / 'money_old.json').write_text(json.dumps({rm: {k: v for k, v in r.items() if k != 'json'}
                                                       for rm, r in res.items()}, indent=1))


if __name__ == '__main__':
    main()
