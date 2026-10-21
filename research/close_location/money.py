"""Close location: the money test (the judge) and the wiring check.

    python3 -I money.py sanity   # 5-day reversal, rebalance 5: should give +11.8 pts/yr, 8 of 18 years
    python3 -I money.py old      # walk-forward on the old export, uniform and volatility-matched random accounts
    python3 -I money.py new      # the unseen months 2025-06-30 .. 2026-10-19 (once, at the end)

Scores = sign x P, the sign fixed by Test 1 (out/tests_old.json); P_<tag>.npy from build.py (universe members with an
export close that day, from 2008-01-08; NaN elsewhere). Graded with rebalance=5.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np = C.np
c2eval = C.c2eval

EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}


def sanity():
    e = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
    c = np.asarray(e['close'], dtype=np.float64)
    s = np.full(c.shape, np.nan, np.float32)
    with np.errstate(all='ignore'):
        s[:, 5:] = -np.log(c[:, 5:] / c[:, :-5])
    del c
    p = c2eval.save_scores(s, 'close_location/sanity_reversal5')
    del s
    txt, js = c2eval.judge(p, rebalance=5)
    out = c2eval.summary_line(js) + '\n' + json.dumps(c2eval.change_over_random(js)) + '\n'
    (C.OUT / 'sanity_reversal5.txt').write_text(out)
    print(out)


def main(tag):
    if tag == 'sanity':
        return sanity()
    sign = json.loads((C.OUT / 'tests_old.json').read_text())['sign']
    export = EXPORTS[tag]
    P = np.load(C.BIG / f'P_{tag}.npy')
    sc = (sign * P).astype(np.float32)
    del P
    p = c2eval.save_scores(sc, f'close_location/score_{tag}', export=export)
    del sc
    res, lines = {'sign': sign}, [f'scores = {sign:+d} x P -> {p}']
    for rm in ('uniform', 'volatility'):
        if tag == 'old':
            txt, js = c2eval.judge(p, rebalance=5, random_match=rm)
            line = c2eval.summary_line(js)
        else:
            txt, js = c2eval.judge(p, export=export, walk_forward=False, start='2025-06-30', rebalance=5, random_match=rm)
            line = ''
        cor = c2eval.change_over_random(js)
        (C.OUT / f'judge_{tag}_{rm}.txt').write_text(txt)
        res[rm] = {'change': cor, 'summary_line': line}
        if tag == 'old':
            res[rm]['years'] = [{k: y.get(k) for k in ('start', 'end', 'first_day', 'last_day', 'candidates', 'spy_pct',
                                                       'ls_bps_per_day', 'ls_t', 'ic_ret_t')}
                                | {'costs': [{k: c.get(k) for k in ('cost_bps', 'final_pct', 'random_median_pct', 'p')}
                                             for c in y['costs']]} for y in js['years']]
            res[rm]['summary'] = js['summary']
            res[rm]['skipped'] = js.get('skipped')
            res[rm]['change_by_cost'] = [c2eval.change_over_random(js, i)['change'] for i in range(3)]
        else:
            res[rm]['report'] = js['report']
            res[rm]['change_by_cost'] = [c2eval.change_over_random(js, i)['change'] for i in range(3)]
        lines.append(f'{rm}: change over random {cor["change"]:+.2f} pts (per year {cor["per_year"]}, ahead {cor["ahead"]}/{cor["years"]}, '
                     f'se {cor["se"]:.2f}); by cost 0/5/20 bps {[round(x, 2) for x in res[rm]["change_by_cost"]]}')
        if line:
            lines.append('  ' + line)
        print('\n'.join(lines[-2:]), flush=True)
    (C.OUT / f'money_{tag}.json').write_text(json.dumps(res, indent=1, default=float))
    (C.OUT / f'money_{tag}.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
