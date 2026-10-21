"""The money test: scores = the book's weight on each stock at formation, held on every day until the next formation
(NaN for stocks outside the book and before the first formation), graded by the judge with rebalance=21 against
uniform and volatility-matched random accounts. Plain 12-1 momentum on the same universe (daily signal, universe of
the latest formation) is graded the same way for reference.

    python3 -I -B money.py old      # walk-forward on the old export -> out/money_old.txt, out/money_old.json
    python3 -I -B money.py new      # the unseen months (single holdout from 2025-06-30; run once, at the end)
                                    # needs book_new.npz from unseen.py -> out/money_new.txt, out/money_new.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402


def book_scores(D):
    z = np.load(C.WORK / f'book_{D["tag"]}.npz')
    s = np.full((D['N'], D['T']), np.nan, np.float32)
    ts = z['t']
    for i in range(len(ts)):
        a = int(ts[i])
        b = int(ts[i + 1]) if i + 1 < len(ts) else D['T']
        rows = z['rows'][z['offs'][i]:z['offs'][i + 1]]
        s[rows, a:b] = z['W'][z['offs'][i]:z['offs'][i + 1]][:, None]
    return s, int(ts[0])


def plain_scores(D, k0):
    """ln(close[d-21] / close[d-252]) on every day d, for the universe (the 500) of the latest formation."""
    c = np.asarray(D['close'], np.float64)
    m = np.full(c.shape, np.nan, np.float32)
    old, new = c[:, :-252], c[:, 231:-21]
    with np.errstate(all='ignore'):
        v = np.log(new / old)
    v[~((old > 0) & (new > 0)) | ~np.isfinite(v)] = np.nan
    m[:, 252:] = v
    del c, v
    s = np.full(m.shape, np.nan, np.float32)
    rebal = D['rebal']
    for k in range(k0, D['K']):
        a = int(rebal[k])
        b = int(rebal[k + 1]) if k + 1 < D['K'] else D['T']
        rows = np.nonzero(D['big'][k])[0]
        s[np.ix_(rows, np.arange(a, b))] = m[np.ix_(rows, np.arange(a, b))]
    return s


def grade(path, export, walk):
    out = {}
    for rm in ('uniform', 'volatility'):
        kw = dict(export=export, rebalance=21, random_match=rm)
        if not walk:
            kw.update(walk_forward=False, start='2025-06-30')
        txt, js = c2eval.judge(path, **kw)
        out[rm] = dict(text=txt, change=[c2eval.change_over_random(js, i) for i in range(3)],
                       summary=c2eval.summary_line(js) if js.get('summary') else None, json=js)
    return out


def main(tag):
    D = C.load(tag)
    walk = tag == 'old'
    k0 = C.first_k(D)
    lines = []
    res = {}
    sb, d0 = book_scores(D)
    for name in ('book', 'plain'):
        s = sb if name == 'book' else plain_scores(D, k0)
        p = c2eval.save_scores(s, f'factor_momentum/{name}_{tag}', export=C.EXPORTS[tag])
        fin = np.isfinite(s[:, d0:])
        lines.append(f'\n#### {name}: {p}; scores from {C.iso(D, d0)}; finite per day median {int(np.median(fin.sum(0)))}, '
                     f'positive per day median {int(np.median((s[:, d0:] > 0).sum(0)))}')
        del s, fin
        r = grade(p, C.EXPORTS[tag], walk)
        res[name] = r
        for rm, x in r.items():
            lines.append(f'\n== {name}, {rm} random accounts ==')
            if x['summary']:
                lines.append(x['summary'])
            for i, cb in enumerate((0, 5, 20)):
                ch = x['change'][i]
                lines.append(f'{cb:>2} bps: change over random {ch["change"]:+.2f} pts/yr (se {ch["se"]:.2f}, ahead '
                             f'{ch["ahead"]}/{ch["years"]}); per year {ch["per_year"]}')
            if walk:
                lines.append('test year      SPY    picks  random-median  p')
                for y in x['json']['years']:
                    c0 = y['costs'][0]
                    lines.append(f'{y["start"][:4]}-{y["end"][2:4]}  {y["spy_pct"]:+6.1f}%  {c0["final_pct"]:+6.1f}%  '
                                 f'{c0["random_median_pct"]:+6.1f}%  {c0["p"]:.3f}  (candidates {y["candidates"]}, '
                                 f'LS {y["ls_bps_per_day"]:+.2f} bps/day)')
            else:
                lines.append(x['text'])
    print('\n'.join(lines))
    (C.OUT / f'money_{tag}.txt').write_text('\n'.join(lines) + '\n')

    def slim(x):
        d = {k: v for k, v in x.items() if k not in ('json', 'text')}
        js = x['json']
        if js.get('years'):
            d['years'] = [dict(year=y['start'][:4], spy=y['spy_pct'], picks=y['costs'][0]['final_pct'],
                               random_median=y['costs'][0]['random_median_pct'], p=y['costs'][0]['p'],
                               candidates=y['candidates'], ls_bps_per_day=y['ls_bps_per_day']) for y in js['years']]
            d['summary_json'] = js['summary']
        else:
            d['report'] = js.get('report')
        return d
    (C.OUT / f'money_{tag}.json').write_text(json.dumps({n: {rm: slim(x) for rm, x in r.items()} for n, r in res.items()},
                                                        indent=1, default=float))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'old')
