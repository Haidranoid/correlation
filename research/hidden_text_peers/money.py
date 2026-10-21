"""The money test: score = the hidden-peer signal's percentile (rank / (n - 1), highest residualized hidden-peer
return = 1) among the day's panel firms (universe firms with at least 3 hidden text peers), every day from 2007-01-01
to 2022-05-31; NaN elsewhere. Graded by the judge's walk-forward with rebalance=21 against uniform and
volatility-matched random accounts.

    python3 -I money.py         # writes money_out.txt and money_out.json here
"""
import json
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import HERE, WORK, c2eval, log, np  # noqa: E402


def main():
    E = c2eval.load_export(c2eval.EXPORT_OLD, arrays=())
    days = E['days']
    N, T = len(E['tickers']), len(days)
    z = np.load(WORK / 'panel.npz')
    t, row, h = z['t'], z['row'], z['Hres']
    s = np.full((N, T), np.nan, np.float32)
    bounds = np.r_[0, np.nonzero(np.diff(t))[0] + 1, len(t)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        v = h[a:b]
        ok = np.isfinite(v)
        n = int(ok.sum())
        if n < 2:
            continue
        r = np.empty(n)
        r[np.argsort(v[ok], kind='stable')] = np.arange(n)
        s[row[a:b][ok], t[a]] = r / (n - 1)
    p = c2eval.save_scores(s, 'hidden_text_peers/scores/hidden_text_peers_old')
    fin = np.isfinite(s).sum(0)
    have = np.nonzero(fin)[0]
    lines = [f'scores {p}: finite from {c2eval.iso(days[have[0]])} to {c2eval.iso(days[have[-1]])}; stocks with a score '
             f'per day: median {int(np.median(fin[have]))}, min {int(fin[have].min())}, max {int(fin[have].max())}']
    for iso in ('2008-07-01', '2012-07-02', '2016-07-01', '2020-07-01', '2022-05-02'):
        d = c2eval.day_index(days, iso)
        top = np.argsort(-np.nan_to_num(s[:, d], nan=-1))[:20]
        lines.append(f'top 20 on {c2eval.iso(days[d])}: ' + ' '.join(E['tickers'][top]))
    del s
    log('\n'.join(lines))
    out = {}
    for rm in ('uniform', 'volatility'):
        txt, js = c2eval.judge(p, rebalance=21, random_match=rm)
        ch = [c2eval.change_over_random(js, i) for i in range(3)]
        out[rm] = dict(summary=c2eval.summary_line(js), change=ch, skipped=js.get('skipped'),
                       years=[dict(start=y['start'], end=y['end'], candidates=y.get('candidates'), pairs=y.get('pairs'),
                                   spy=y.get('spy_pct'), costs=y['costs']) for y in js['years']],
                       ls=dict(bps=js['summary']['ls_bps_per_day_mean'], z=js['summary']['ls_z'],
                               ic_ret_z=js['summary']['ic_ret_z'], turnover=js['summary']['ls_turnover_mean']))
        lines.append(f'\n== {rm} random accounts ==')
        lines.append(out[rm]['summary'])
        for i, cb in enumerate((0, 5, 20)):
            lines.append(f'{cb:>2} bps: change over random {ch[i]["change"]:+.2f} pts/yr (se {ch[i]["se"]:.2f}, ahead '
                         f'{ch[i]["ahead"]}/{ch[i]["years"]}); per year {ch[i]["per_year"]}')
        lines.append(f'skipped: {js.get("skipped")}')
        lines.append(txt)
        log('\n'.join(lines[-8:]))
    (HERE / 'money_out.txt').write_text('\n'.join(lines) + '\n')
    (HERE / 'money_out.json').write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
