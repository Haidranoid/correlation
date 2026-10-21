"""Volume clock: the money test (the judge).

    python3 -I money.py old     # walk-forward 2006-07..2024-25 on the old export
    python3 -I money.py new     # the unseen months on the new export (once, at the end)

Score = the event's ranking value (mean volume ratio on its counted spikes over the past 504 trading days, at entry)
on days c-6..c of each predicted window that passes the universe and fund-like checks at entry; NaN elsewhere.
rebalance=1, so the account holds from close c-6 to close c+1. Uniform and volatility-matched random accounts.
Writes money_<tag>.md / .json next to this file; scores and judge JSON under WORK/volume_clock/.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402

HERE = Path(__file__).resolve().parent
BIG = c2eval.WORK / 'volume_clock'
EXPORT = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
PRE = 6


def scores_for(tag):
    z = np.load(BIG / f'events_{tag}.npz')
    N, T = len(z['ids']), len(z['days'])
    s = np.full((N, T), np.nan, np.float32)
    ok = z['univ'] & ~z['fund_like'] & np.isfinite(z['rank'])
    for j, c, r in zip(z['j'][ok], z['c'][ok], z['rank'][ok]):
        a = c - PRE
        assert np.all(np.isnan(s[j, a:c + 1])), 'windows of one stock overlap'
        s[j, a:c + 1] = r
    return s, z['days'], int(ok.sum())


def book_sizes(dump_path, days, years):
    """Average number of names held per account day (min(20, positive finite judge-candidate scores)), per year."""
    D = np.load(dump_path, mmap_mode='r')
    out = []
    for y in years:
        d0 = c2eval.day_index(days, y['first_day']) - 1
        d1 = c2eval.day_index(days, y['last_day'])  # decision days d0..d1-1 (hold over d+1)
        blk = np.asarray(D[:, d0:d1])
        n = np.minimum((blk > 0).sum(0), 20)
        out.append(float(n.mean()))
    return out


def main(tag):
    s, days, n_ev = scores_for(tag)
    name = f'volume_clock/volume_clock_{tag}'
    p = c2eval.save_scores(s, name)
    fin = np.isfinite(s)
    lines = [f'# Volume clock money test, {tag} export\n',
             f'Scores: {p} ({n_ev} windows; on days with any score, the median number of scored stocks is '
             f'{int(np.median(fin.sum(0)[fin.any(0)]))})\n']
    kw = dict(rebalance=1) if tag == 'old' else dict(export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30', rebalance=1)
    dump = BIG / f'dump_{tag}.npy'
    res = {}
    for rm in ('uniform', 'volatility'):
        extra = ('--dump-scores', str(dump)) if rm == 'uniform' else ()
        txt, js = c2eval.judge(p, random_match=rm, extra=extra, **kw)
        ch = [c2eval.change_over_random(js, i) for i in range(3)]
        res[rm] = dict(change=ch, summary=c2eval.summary_line(js) if js.get('years') else None, text=txt)
        lines.append(f'## {rm} random accounts\n')
        if res[rm]['summary']:
            lines.append(res[rm]['summary'] + '\n')
        for i, cb in enumerate((0, 5, 20)):
            c = ch[i]
            lines.append(f'- {cb} bps: change over random {c["change"]:+.2f} pts/yr (se {c["se"]:.2f}, ahead {c["ahead"]}/{c["years"]})')
        lines.append('')
        if js.get('years'):
            if rm == 'uniform':
                res['book'] = book_sizes(dump, days, js['years'])
            lines.append('| Test year | Candidates | Names held (avg) | Picks 0 bps | Random median | p | Picks 20 bps | Random median 20 bps |')
            lines.append('|---|---|---|---|---|---|---|---|')
            for k, y in enumerate(js['years']):
                c0, c2 = y['costs'][0], y['costs'][2]
                lines.append(f"| {y['first_day']}..{y['last_day']} | {y['candidates']} | {res['book'][k]:.1f} | "
                             f"{c0['final_pct']:+.1f}% | {c0['random_median_pct']:+.1f}% | {c0['p']:.3f} | "
                             f"{c2['final_pct']:+.1f}% | {c2['random_median_pct']:+.1f}% |")
            lines.append('')
            res[rm]['years'] = [dict(first=y['first_day'], last=y['last_day'], candidates=y['candidates'],
                                     costs=y['costs'], random_check=y.get('random_check')) for y in js['years']]
            res[rm]['skipped'] = js.get('skipped')
        else:
            rep = js['report']
            if rm == 'uniform':
                D = np.load(dump, mmap_mode='r')
                d0 = c2eval.day_index(days, '2025-06-30') - 1
                n = np.minimum((np.asarray(D[:, d0:len(days) - 1]) > 0).sum(0), 20)
                res['book'] = [float(n.mean())]
            lines.append(f"Names held (avg) {res['book'][0]:.1f}; candidates {rep.get('candidates')}")
            for c in rep['strategy']:
                lines.append(f"- {c['cost_bps']:.0f} bps: picks {c['final_pct']:+.1f}%, random median {c['random_median_pct']:+.1f}%, p {c['p']:.3f}")
            lines.append('')
            res[rm]['report'] = {k: v for k, v in rep.items() if k not in ('curve',)}
        lines.append('```')
        lines.append(txt.strip())
        lines.append('```\n')
    txt = '\n'.join(lines)
    (HERE / f'money_{tag}.md').write_text(txt + '\n')
    (HERE / f'money_{tag}.json').write_text(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != 'text'} if isinstance(v, dict) else v)
                                                        for k, v in res.items()}, indent=1, default=float))
    print(txt)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'old')
