"""Night and day: the money test (the judge) for s, and the plumbing check.

    python3 -I money.py sanity   # 5-day reversal, rebalance 5: should give +11.8 pts/yr, 8 of 18 years
    python3 -I money.py old      # walk-forward 2006-07 .. 2024-25, uniform and volatility-matched random accounts
    python3 -I money.py new      # the unseen months 2025-06-30 .. 2026-10-19 (once, at the end)

Scores: s = b_lag x SPY's return at the close of each day, on the universe (the judge-like liquid set minus
c2eval.excluded, point in time on the twins calendar), NaN elsewhere; every day where s exists (from 2006), since the
judge trades close to close and needs no opens. Graded with rebalance=1.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np = C.np
c2eval = C.c2eval

tag = sys.argv[1]


def sanity():
    e = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
    c = np.asarray(e['close'], dtype=np.float64)
    s = np.full(c.shape, np.nan, np.float32)
    with np.errstate(all='ignore'):
        s[:, 5:] = -np.log(c[:, 5:] / c[:, :-5])
    p = c2eval.save_scores(s, 'night_and_day/sanity_reversal5')
    txt, js = c2eval.judge(p, rebalance=5)
    line = c2eval.summary_line(js)
    cor = c2eval.change_over_random(js)
    out = line + '\n' + json.dumps(cor) + '\n'
    (C.OUT / 'sanity_reversal5.txt').write_text(out)
    print(out)


def scores(tag):
    P = C.load(tag, mmap=False)
    s = np.asarray(P['s'])
    N, T = s.shape
    sc = np.full((N, T), np.nan, np.float32)
    for t in range(T):
        k = C.kidx(P['rebal'], t)
        if k < 0:
            continue
        u = P['uni'][k]
        sc[u, t] = s[u, t]
    return sc, P['days']


def describe(js):
    rep = js['report'] if 'report' in js else None
    return rep


def main():
    if tag == 'sanity':
        return sanity()
    sc, days = scores(tag)
    fin = np.isfinite(sc)
    per_day = fin.sum(0)
    first = int(np.argmax(per_day > 0))
    C.log('scores: first day', c2eval.iso(days[first]), 'median names per scored day', int(np.median(per_day[per_day > 0])))
    export = C.EXPORTS[tag]
    p = c2eval.save_scores(sc, f'night_and_day/s_{tag}', export=export)
    del sc
    res = {}
    lines = []
    for rm in ('uniform', 'volatility'):
        C.log('judge', rm)
        if tag == 'old':
            txt, js = c2eval.judge(p, rebalance=1, random_match=rm)
            line = c2eval.summary_line(js)
        else:
            txt, js = c2eval.judge(p, export=export, walk_forward=False, start='2025-06-30', rebalance=1, random_match=rm)
            line = ''
        cor = c2eval.change_over_random(js)
        (C.OUT / f'judge_{tag}_{rm}.txt').write_text(txt)
        res[rm] = {'change': cor, 'summary_line': line}
        if tag == 'old':
            res[rm]['years'] = [{'start': y.get('start'), 'end': y.get('end'),
                                 'costs': [{k: c.get(k) for k in ('cost_bps', 'final_pct', 'random_median_pct', 'p_value')}
                                           for c in y['costs']]} for y in js['years']]
            res[rm]['summary'] = js['summary']
        else:
            res[rm]['report'] = js['report']
        lines.append(f'{rm}: change over random {cor["change"]:+.2f} pts (per year {cor["per_year"]}, ahead {cor["ahead"]}/{cor["years"]}, se {cor["se"]:.2f})')
        if line:
            lines.append('  ' + line)
        print(lines[-1], flush=True)
    (C.OUT / f'money_{tag}.json').write_text(json.dumps(res, indent=1, default=float))
    (C.OUT / f'money_{tag}.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
