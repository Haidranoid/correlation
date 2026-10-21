"""Side check with the judge (not the file-name number): scores = the Akbas signal at the close (overnight_d -
intraday_d, high = positive overnight then a falling day), for the universe in force for day d+1, from 2008.
The judge trades close to close, so this is not the intraday book.

    python3 -I money.py old     # walk-forward, rebalance=1, uniform random accounts -> out/money_old.txt / .json
    python3 -I money.py new     # single holdout 2025-06-30..2026-10-19 (once, at the end)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import books as B  # noqa: E402
from books import c2eval, np  # noqa: E402


def main(tag):
    export = c2eval.EXPORT_OLD if tag == 'old' else c2eval.EXPORT_NEW
    p = B.BIG / f'akbas_close_{tag}.npy'
    s = np.load(p, mmap_mode='r')
    lines = [f'scores {p}: finite per day (median over days with any) '
             f'{int(np.median(np.isfinite(s).sum(0)[np.isfinite(s).sum(0) > 0]))}']
    if tag == 'old':
        txt, js = c2eval.judge(p, export=export, rebalance=1, walk_forward=True, random_match='uniform')
    else:
        txt, js = c2eval.judge(p, export=export, rebalance=1, walk_forward=False, start='2025-06-30',
                               random_match='uniform')
    out = {'change': [c2eval.change_over_random(js, i) for i in range(3)]}
    if js.get('years'):
        lines.append(c2eval.summary_line(js))
        # the scores start 2008-01-02, so only test years starting 2008-06-28 or later are clean
        keep = [y for y in js['years'] if str(y.get('start', y.get('first_day', '')))[:10] >= '2008-06-28']
        sub = dict(js, years=keep)
        out['change_from_2008'] = [c2eval.change_over_random(sub, i) for i in range(3)] if len(keep) > 1 else None
        out['years_kept'] = [str(y.get('start', y.get('first_day'))) for y in keep]
    for i, cb in enumerate((0, 5, 20)):
        ch = out['change'][i]
        lines.append(f'{cb:>2} bps: change over random {ch["change"]:+.2f} pts/yr (se {ch["se"]:.2f}, ahead '
                     f'{ch["ahead"]}/{ch["years"]}); per year {ch["per_year"]}')
        if out.get('change_from_2008'):
            c8 = out['change_from_2008'][i]
            lines.append(f'      test years from 2008-06-28 only: {c8["change"]:+.2f} pts/yr (se {c8["se"]:.2f}, ahead '
                         f'{c8["ahead"]}/{c8["years"]})')
    lines.append(txt)
    print('\n'.join(lines))
    (B.OUT / f'money_{tag}.txt').write_text('\n'.join(lines) + '\n')
    (B.OUT / f'money_{tag}.json').write_text(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main(sys.argv[1])
