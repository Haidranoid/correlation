"""Decoupling: the money test. Score = D x |S3| (= S3) for each event, written on days t..t+62, NaN elsewhere; the
judge buys only positive scores. Graded with rebalance=5 against uniform and volatility-matched random accounts:
walk-forward on the old export, one holdout from 2025-06-30 on the new export (the unseen months, once).

    python3 -I -B money.py old|new      # writes out/money_{tag}.txt and .json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402

LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def main(tag):
    export = C.EXPORTS[tag]
    E = c2eval.load_export(export, arrays=('close',))
    N, T = E['close'].shape
    ev = dict(np.load(C.WORK / f'events_{tag}.npz'))
    s = np.full((N, T), np.nan, np.float32)
    for j, t, v in zip(ev['j'].tolist(), ev['t'].tolist(), ev['S3'].tolist()):
        s[j, t:min(T, t + C.FWD)] = v                      # days t .. t+62
    p = c2eval.save_scores(s, f'decoupling/scores_{tag}', export=export)
    fin = np.isfinite(s)
    pos = s > 0
    d0 = c2eval.day_index(E['days'], '2006-06-28' if tag == 'old' else '2025-06-30')
    say(f'# Decoupling money test, {tag} export; scores {p}')
    say(f'events {len(ev["t"])}; from {c2eval.iso(E["days"][d0])}: scored stocks per day median '
        f'{int(np.median(fin[:, d0:].sum(0)))} (max {int(fin[:, d0:].sum(0).max())}), positive per day median '
        f'{int(np.median(pos[:, d0:].sum(0)))} (max {int(pos[:, d0:].sum(0).max())}); days with no positive score '
        f'{np.mean(pos[:, d0:].sum(0) == 0):.0%}')
    del s, fin, pos
    res = {}
    for rm in ('uniform', 'volatility'):
        if tag == 'old':
            txt, js = c2eval.judge(p, rebalance=5, random_match=rm)
        else:
            txt, js = c2eval.judge(p, export=export, walk_forward=False, start='2025-06-30', rebalance=5,
                                   random_match=rm)
        ch = [c2eval.change_over_random(js, i) for i in range(3)]
        r = dict(change=ch)
        say(f'\n== {rm} random accounts ==')
        if tag == 'old':
            r['summary'] = c2eval.summary_line(js)
            r['skipped'] = js.get('skipped')
            say(r['summary'])
            say(f'skipped: {js.get("skipped")}')
            for i, cb in enumerate((0, 5, 20)):
                say(f'{cb:>2} bps: change over random {ch[i]["change"]:+.2f} pts/yr (se {ch[i]["se"]:.2f}, ahead '
                    f'{ch[i]["ahead"]}/{ch[i]["years"]})')
            say('year       SPY     picks   random-median  p      candidates  LS bps/day (t)')
            r['years'] = []
            for y in js['years']:
                c0 = y['costs'][0]
                say(f'{y["start"][:4]}-{y["end"][2:4]}  {y["spy_pct"]:+6.1f}%  {c0["final_pct"]:+6.1f}%  '
                    f'{c0["random_median_pct"]:+6.1f}%        {c0["p"]:.3f}  {y["candidates"]:>5}       '
                    f'{y["ls_bps_per_day"]:+.1f} ({y["ls_t"]:+.2f})')
                r['years'].append(dict(year=y['start'][:4], spy=y['spy_pct'], picks=c0['final_pct'],
                                       picks5=y['costs'][1]['final_pct'], picks20=y['costs'][2]['final_pct'],
                                       random_median=c0['random_median_pct'], p=c0['p'], candidates=y['candidates'],
                                       ls_bps_per_day=y['ls_bps_per_day'], ls_t=y['ls_t'], ic_ret_t=y['ic_ret_t']))
            r['summary_json'] = js['summary']
        else:
            rep = js['report']
            r['strategy'] = rep['strategy']
            r['benchmarks'] = rep.get('benchmarks')
            r['ls_bps_per_day'], r['ls_t'], r['ic_ret_t'] = rep['ls_bps_per_day'], rep['ls_t'], rep['ic_ret_t']
            say(txt)
            for i, cb in enumerate((0, 5, 20)):
                st = rep['strategy'][i]
                say(f'{cb:>2} bps: picks {st["final_pct"]:+.1f}%, random median {st["random_median_pct"]:+.1f}%, '
                    f'p {st["p"]:.3f}, change {ch[i]["change"]:+.1f}')
            say(f'LS {rep["ls_bps_per_day"]:+.2f} bps/day (t {rep["ls_t"]:+.2f}); IC actual returns t {rep["ic_ret_t"]:+.2f}; '
                f'candidates {rep["candidates"]}')
        res[rm] = r
    (C.OUT / f'money_{tag}.txt').write_text('\n'.join(LINES) + '\n')
    (C.OUT / f'money_{tag}.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main(sys.argv[1])
