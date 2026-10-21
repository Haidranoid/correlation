"""Overnight peers: the money test (the judge) and the plumbing check.

    python3 -I money.py sanity   # 5-day reversal, rebalance 5: should give +11.8 pts/yr, 8 of 18 years
    python3 -I money.py old      # walk-forward, uniform and volatility-matched random accounts, rebalance 21
    python3 -I money.py new      # the unseen months 2025-06-30 .. 2026-10-19 (once, at the end)

Scores every day d: the overnight-peer residual signal at d (peers and universe from the latest calendar day at or
before d, the 21-day windows ending at d, the OLS on own return at d), as a percentile among universe stocks with a
signal that day: rank / (n - 1), highest residual = 1. NaN elsewhere. From 2009-01-02 (old) or the calendar day
2025-06-13 (new).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np = C.np
c2eval = C.c2eval

tag = sys.argv[1]
SUB = 'overnight_peers'


def sanity():
    e = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
    c = np.asarray(e['close'], dtype=np.float64)
    s = np.full(c.shape, np.nan, np.float32)
    with np.errstate(all='ignore'):
        s[:, 5:] = -np.log(c[:, 5:] / c[:, :-5])
    p = c2eval.save_scores(s, f'{SUB}/sanity_reversal5')
    txt, js = c2eval.judge(p, rebalance=5)
    line = c2eval.summary_line(js)
    cor = c2eval.change_over_random(js)
    out = line + '\n' + json.dumps(cor) + '\n'
    (C.OUT / 'sanity_reversal5.txt').write_text(out)
    print(out)


def scores(tag, ty='on'):
    res = np.load(C.WORK / f'resid_{ty}_{tag}.npy')
    N, T = res.shape
    sc = np.full((N, T), np.nan, np.float32)
    for d in range(T):
        x = res[:, d]
        f = np.nonzero(np.isfinite(x))[0]
        if len(f) < C.MIN_NAMES:
            continue
        sc[f, d] = C.avg_rank(x[f].astype(np.float64)) / (len(f) - 1)
    return sc


def main():
    if tag == 'sanity':
        return sanity()
    M = C.meta(tag)
    days = M['days']
    sc = scores(tag)
    per_day = np.isfinite(sc).sum(0)
    first = int(np.argmax(per_day > 0))
    C.log('scores: first day', c2eval.iso(days[first]), 'last', c2eval.iso(days[len(days) - 1 - int(np.argmax(per_day[::-1] > 0))]),
          'median names per scored day', int(np.median(per_day[per_day > 0])))
    export = C.EXPORTS[tag]
    p = c2eval.save_scores(sc, f'{SUB}/scores_on_{tag}', export=export)
    del sc
    res = {}
    lines = []
    for rm in ('uniform', 'volatility'):
        C.log('judge', rm)
        if tag == 'old':
            txt, js = c2eval.judge(p, rebalance=21, random_match=rm)
            line = c2eval.summary_line(js)
        else:
            txt, js = c2eval.judge(p, export=export, walk_forward=False, start='2025-06-30', rebalance=21, random_match=rm)
            line = ''
        cor = c2eval.change_over_random(js)
        (C.OUT / f'judge_{tag}_{rm}.txt').write_text(txt)
        res[rm] = {'change': cor, 'summary_line': line}
        if tag == 'old':
            res[rm]['years'] = [{'start': y.get('start'), 'end': y.get('end'),
                                 'costs': [{k: c.get(k) for k in ('cost_bps', 'final_pct', 'random_median_pct', 'p_value')}
                                           for c in y['costs']]} for y in js['years']]
            res[rm]['summary'] = js['summary']
            res[rm]['skipped'] = js.get('skipped')
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
