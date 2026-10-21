"""The money test: score = the stock's similarity percentile (non-changers highest) among the universe stocks with a
known similarity, every day; NaN without one or outside the universe. Graded by the judge's walk-forward with
rebalance=21 against uniform and volatility-matched random accounts.

    python3 -I money.py        # writes money_out.txt and money_out.json here
"""
import json
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import HERE, WORK, c2eval, np  # noqa: E402
import panel as PN  # noqa: E402


def scores(E, tw, uni, sim):
    days, rebal = E['days'], tw['rebal']
    T = len(days)
    blk = np.searchsorted(rebal, np.arange(T), side='right') - 1
    s = np.full(sim.shape, np.nan, np.float32)
    for d in range(T):
        k = blk[d]
        if k < 0:
            continue
        rows = np.nonzero(uni[k] & np.isfinite(sim[:, d]))[0]
        n = len(rows)
        if n < 2:
            continue
        r = np.empty(n)
        r[np.argsort(sim[rows, d], kind='stable')] = np.arange(n)
        s[rows, d] = r / (n - 1)
    return s


def main():
    E, tw, uni, mdv = PN.load()
    P = PN.pairs()
    sim = PN.sim_panel(E, P, 'sim')
    s = scores(E, tw, uni, sim)
    del sim
    (WORK / 'scores').mkdir(parents=True, exist_ok=True)
    p = c2eval.save_scores(s, 'lazy_prices/scores/lazy_prices_old')
    fin = np.isfinite(s).sum(0)
    have = np.nonzero(fin)[0]
    lines = [f'scores {p}: finite from {c2eval.iso(E["days"][have[0]])} to {c2eval.iso(E["days"][have[-1]])}; '
             f'stocks with a score per day: median {int(np.median(fin[have]))}, min {int(fin[have].min())}, max {int(fin[have].max())}']
    # the top 20 at a few dates, to look at by hand
    for iso in ('2008-07-01', '2012-07-02', '2016-07-01', '2020-07-01', '2022-05-02'):
        d = c2eval.day_index(E['days'], iso)
        top = np.argsort(-np.nan_to_num(s[:, d], nan=-1))[:20]
        lines.append(f'top 20 on {c2eval.iso(E["days"][d])}: ' + ' '.join(E['tickers'][top]))
    del s
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
        print('\n'.join(lines[-8:]), flush=True)
    (HERE / 'money_out.txt').write_text('\n'.join(lines) + '\n')
    (HERE / 'money_out.json').write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
