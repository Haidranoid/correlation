"""Tables for experiments/+0.7-free-timing.md from the trade tables written by free_timing.py.

    python3 -I research/free_timing/report.py            # old export: tests 1-3 and everything "also reported"
    python3 -I research/free_timing/report.py unseen     # test 4 (new export), once, at the end
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = 5
PER_YEAR = 12          # 252 / 21 rebalances a year
SIGNALS = ('S1', 'S2', 'S3')
EXECS = ('open', 'close')
EXEC_NAME = {'open': 'next-open', 'close': 'same-close'}


def year_of(day):
    return (np.datetime64('1970-01-01') + day.astype('timedelta64[D]')).astype('datetime64[Y]').astype(int) + 1970


def iso(day):
    return str(np.datetime64('1970-01-01') + np.timedelta64(int(day), 'D'))


class Book:
    def __init__(self, path):
        z = np.load(path)
        self.z = {k: z[k] for k in z.files}
        z = self.z
        self.n = len(z['row'])
        self.k = z['k']
        self.side = z['side'].astype(np.float64)
        self.size = z['size']
        self.port = z['port']
        self.day = z['day']
        self.year = year_of(z['day'])
        self.dates = np.unique(self.k)
        self.date_day = {kk: z['day'][self.k == kk][0] for kk in self.dates}
        self.sigs = [s for s in SIGNALS if f'{s}_buy' in z]
        self.cap, self.cap_mr, self.valid = {}, {}, {}
        for e in EXECS:
            px, sp = z[f'px_{e}'], z[f'spy_{e}']
            with np.errstate(all='ignore'):
                raw = self.side[:, None] * np.log(z['END'][:, None] / px)
                mkt = self.side[:, None] * np.log(z['spy_END'][:, None] / sp)
            ok = np.isfinite(raw).all(1) & np.isfinite(mkt).all(1)
            self.valid[e] = ok
            self.cap[e] = np.where(ok[:, None], raw, np.nan)
            self.cap_mr[e] = np.where(ok[:, None], raw - mkt, np.nan)

    def kstar(self, sig):
        """Window day each trade executes on under the timing rule, and whether the signal can time it."""
        z = self.z
        fav = np.where(self.side[:, None] > 0, z[f'{sig}_buy'], z[f'{sig}_sell'])
        fav[:, W - 1] = True
        return fav.argmax(1), z[f'{sig}_fin'][:, 0]

    def delta(self, sig, e, mr=True, rule='timed'):
        """Per-trade Δ (fraction) and the mask of trades it is defined for."""
        c = (self.cap_mr if mr else self.cap)[e]
        base = c.mean(1)
        if rule == 'timed':
            ks, timable = self.kstar(sig)
            d = c[np.arange(self.n), ks] - base
            return d, self.valid[e] & timable
        if rule == 'day0':
            return c[:, 0] - base, self.valid[e]
        if rule == 'always':   # sanity: an always-favourable signal is day-0 execution
            return c[:, 0] - base, self.valid[e]
        raise ValueError(rule)


def wmean_cluster(d, w, g):
    """Weighted mean of d with weights w, standard error clustered by g (G/(G-1) correction)."""
    if len(d) == 0:
        return dict(mean=np.nan, se=np.nan, t=np.nan, n=0, G=0)
    W_ = w.sum()
    m = (w * d).sum() / W_
    u = w * (d - m)
    gs, inv = np.unique(g, return_inverse=True)
    ug = np.bincount(inv, weights=u)
    G = len(gs)
    se = np.sqrt(G / (G - 1) * (ug ** 2).sum()) / W_ if G > 1 else np.nan
    return dict(mean=m, se=se, t=m / se if se > 0 else np.nan, n=len(d), G=G)


def per_trade(B, sig, e, mr=True, rule='timed', sel=None):
    d, ok = B.delta(sig, e, mr, rule)
    if sel is not None:
        ok = ok & sel
    return wmean_cluster(d[ok], B.size[ok], B.k[ok])


def portfolio(B, sig, e, port, mr=False, rule='timed', date_sel=None):
    """Per-date effect x_g = sum of size x Δ over the portfolio's trades (Δ = 0 if untimable or unpriced)."""
    d, ok = B.delta(sig, e, mr, rule)
    contrib = np.where(ok & (B.port == port), B.size * d, 0.0)
    dates = B.dates if date_sel is None else B.dates[date_sel(B.dates)]
    pos = {kk: i for i, kk in enumerate(dates)}
    x = np.zeros(len(dates))
    for i in range(B.n):
        j = pos.get(B.k[i])
        if j is not None:
            x[j] += contrib[i]
    yrs = np.array([year_of(np.array(B.date_day[kk])) for kk in dates])
    G = len(x)
    num = PER_YEAR * x.mean() * 100
    se = PER_YEAR * x.std(ddof=1) / np.sqrt(G) * 100 if G > 1 else np.nan
    by_year = {int(y): float(x[yrs == y].sum() * 100) for y in np.unique(yrs)}
    nyr = {int(y): int((yrs == y).sum()) for y in np.unique(yrs)}
    ahead = sum(v > 0 for v in by_year.values())
    return dict(points=num, se=se, t=num / se if se > 0 else np.nan, G=G, by_year=by_year, n_reb=nyr, ahead=ahead,
                years=len(by_year))


def fmt_pt(r):
    return f"{r['mean'] * 1e4:+.2f} bps (se {r['se'] * 1e4:.2f}, t {r['t']:+.2f}; {r['n']:,} trades, {r['G']} dates)"


def main(which='old'):
    B = Book(os.path.join(HERE, f'trades_{which}.npz'))
    z = B.z
    out = {}
    lines = []
    P = lines.append
    P(f'# free timing report: {which} export')
    P(f"rebalances {len(B.dates)}: {iso(B.date_day[B.dates[0]])} .. {iso(B.date_day[B.dates[-1]])}; trades {B.n:,} "
      f"(P1 {(B.port == 1).sum():,}, P2 {(B.port == 2).sum():,}); buys {(B.side > 0).sum():,}, sells {(B.side < 0).sum():,}")
    for e in EXECS:
        P(f"priced ({EXEC_NAME[e]}): {B.valid[e].mean():.4f}")
    P(f"trades touching a bad open: {z['bad_open_hit'].sum()}; executions at an open equal to the close: {z['open_eq_close'].mean():.4f}")
    P(f"liquid set per rebalance: median {int(np.median(z['liq_size']))}, min {z['liq_size'].min()}, max {z['liq_size'].max()}")
    P(f"P1 turnover (sum of trade sizes) per rebalance: {B.size[B.port == 1].sum() / len(B.dates):.3f}; "
      f"P2: {B.size[B.port == 2].sum() / len(B.dates):.3f}")
    for s in B.sigs:
        ks, tim = B.kstar(s)
        P(f"{s}: timable {tim.mean():.3f} (P1 {tim[B.port == 1].mean():.3f}, P2 {tim[B.port == 2].mean():.3f}); "
          f"mean window day {ks[tim].mean():.2f}; executed before day 4: {(ks[tim] < 4).mean():.3f}; "
          f"day histogram {np.bincount(ks[tim], minlength=5).tolist()}")

    # sanity checks
    for e in EXECS:
        c = B.cap_mr[e][B.valid[e]]
        assert np.abs((c - c.mean(1, keepdims=True)).mean(1)).max() < 1e-12
        a, _ = B.delta('S1', e, True, 'always')
        b, _ = B.delta('S1', e, True, 'day0')
        assert np.allclose(a[B.valid[e]], b[B.valid[e]])
    P('sanity: mean over window days of (captured - random) = 0; always-favourable = day-0: ok')

    if which == 'old':
        y = B.year
        early = y <= 2019
        late = y >= 2020
        P('\n## Test 1: S1, next-open, P1+P2 pooled, 2008-2019, market-relative (pass: >= +5 bps and t >= 3)')
        r1 = per_trade(B, 'S1', 'open', True, sel=early)
        P(fmt_pt(r1))
        out['test1'] = r1
        P('\n## Test 2: same, 2020-2025-06 (pass: same sign, t >= 2)')
        r2 = per_trade(B, 'S1', 'open', True, sel=late)
        P(fmt_pt(r2))
        out['test2'] = r2
        P('\n## Test 3: P2 extra yearly return against random-day execution, S1, next-open, raw, 2008-2025 (pass: ahead in >= 2/3 of years)')
        r3 = portfolio(B, 'S1', 'open', 2)
        P(f"{r3['points']:+.2f} points a year (se {r3['se']:.2f}, t {r3['t']:+.2f}, {r3['G']} rebalances); ahead {r3['ahead']} of {r3['years']} years")
        out['test3'] = r3
        P('per year (points; rebalances): ' + ', '.join(f"{yy} {v:+.2f} ({r3['n_reb'][yy]})" for yy, v in r3['by_year'].items()))

        P('\n## Per trade, all signals and executions, P1+P2 pooled (bps; t)')
        P('| Signal | Execution | Measure | 2008-2019 | 2020-2025 | 2008-2025 |')
        P('|---|---|---|---|---|---|')
        tab = {}
        for s in B.sigs:
            for e in EXECS:
                for mr in (True, False):
                    cells = []
                    for nm, sel in (('early', early), ('late', late), ('all', None)):
                        r = per_trade(B, s, e, mr, sel=sel)
                        tab[(s, e, mr, nm)] = r
                        cells.append(f"{r['mean'] * 1e4:+.2f} ({r['t']:+.1f})" if r['n'] else 'n/a')
                    P(f"| {s} | {EXEC_NAME[e]} | {'market-relative' if mr else 'raw'} | " + ' | '.join(cells) + ' |')
        for e in EXECS:
            for mr in (True, False):
                cells = []
                for nm, sel in (('early', early), ('late', late), ('all', None)):
                    r = per_trade(B, 'S1', e, mr, rule='day0', sel=sel)
                    cells.append(f"{r['mean'] * 1e4:+.2f} ({r['t']:+.1f})")
                P(f"| day 0 | {EXEC_NAME[e]} | {'market-relative' if mr else 'raw'} | " + ' | '.join(cells) + ' |')
        out['per_trade'] = {f'{s}|{e}|{"mr" if mr else "raw"}|{nm}': v for (s, e, mr, nm), v in tab.items()}

        P('\n## P1 and P2 separately, per trade, market-relative, 2008-2025 (bps; t)')
        P('| Signal | Execution | P1 | P2 |')
        P('|---|---|---|---|')
        for s in B.sigs:
            for e in EXECS:
                c = [per_trade(B, s, e, True, sel=B.port == p) for p in (1, 2)]
                P(f"| {s} | {EXEC_NAME[e]} | " + ' | '.join(f"{r['mean'] * 1e4:+.2f} ({r['t']:+.1f})" for r in c) + ' |')
        for e in EXECS:
            c = [per_trade(B, 'S1', e, True, rule='day0', sel=B.port == p) for p in (1, 2)]
            P(f"| day 0 | {EXEC_NAME[e]} | " + ' | '.join(f"{r['mean'] * 1e4:+.2f} ({r['t']:+.1f})" for r in c) + ' |')

        P('\n## Buys and sells separately, P1+P2, 2008-2025 (bps; t): market-relative / raw')
        P('| Signal | Execution | Buys mr | Sells mr | Buys raw | Sells raw |')
        P('|---|---|---|---|---|---|')
        for s in list(B.sigs) + ['day 0']:
            for e in EXECS:
                rule = 'day0' if s == 'day 0' else 'timed'
                sg = 'S1' if s == 'day 0' else s
                c = [per_trade(B, sg, e, mr, rule=rule, sel=B.side * sd > 0) for mr in (True, False) for sd in (1, -1)]
                P(f"| {s} | {EXEC_NAME[e]} | " + ' | '.join(f"{r['mean'] * 1e4:+.2f} ({r['t']:+.1f})" for r in c) + ' |')

        P('\n## Portfolio effect: extra yearly return against random-day execution (points a year; se; years ahead), raw, 2008-2025')
        P('| Signal | Execution | P1 | P2 |')
        P('|---|---|---|---|')
        port_tab = {}
        for s in list(B.sigs) + ['day 0']:
            for e in EXECS:
                rule = 'day0' if s == 'day 0' else 'timed'
                sg = 'S1' if s == 'day 0' else s
                c = []
                for p in (1, 2):
                    r = portfolio(B, sg, e, p, rule=rule)
                    port_tab[f'{s}|{e}|P{p}'] = r
                    c.append(f"{r['points']:+.2f} (se {r['se']:.2f}; {r['ahead']}/{r['years']})")
                P(f"| {s} | {EXEC_NAME[e]} | " + ' | '.join(c) + ' |')
        out['portfolio'] = port_tab
        P('(S3 rows: its rebalances outside 2010-07..2024-06 add 0, as the trades are untimable there.)')

        # S3's span: rebalances where S3 exists on day 0 for most trades
        s3 = z['S3_fin'][:, 0]
        cov = {kk: s3[B.k == kk].mean() for kk in B.dates}
        span_dates = np.array([kk for kk in B.dates if cov[kk] > 0.5])
        span = np.isin(B.k, span_dates)
        P(f"\n## S3's span: {len(span_dates)} rebalances, {iso(B.date_day[span_dates[0]])} .. {iso(B.date_day[span_dates[-1]])}")
        P('| Signal | Execution | Per trade mr, pooled | Per trade raw, pooled | P2 portfolio (points a year) | P1 portfolio |')
        P('|---|---|---|---|---|---|')
        for s in list(B.sigs) + ['day 0']:
            for e in EXECS:
                rule = 'day0' if s == 'day 0' else 'timed'
                sg = 'S1' if s == 'day 0' else s
                a = per_trade(B, sg, e, True, rule=rule, sel=span)
                b = per_trade(B, sg, e, False, rule=rule, sel=span)
                c = [portfolio(B, sg, e, p, rule=rule, date_sel=lambda ds: np.isin(ds, span_dates)) for p in (2, 1)]
                P(f"| {s} | {EXEC_NAME[e]} | {a['mean'] * 1e4:+.2f} ({a['t']:+.1f}) | {b['mean'] * 1e4:+.2f} ({b['t']:+.1f}) | "
                  + ' | '.join(f"{r['points']:+.2f} (se {r['se']:.2f})" for r in c) + ' |')

        P('\n## Per year: S1, next-open')
        P('| Year | Rebalances | P2 portfolio, raw (points) | P1 portfolio, raw | Per trade mr, pooled (bps; t) | Per trade raw, pooled | Day 0, P2 portfolio |')
        P('|---|---|---|---|---|---|---|')
        p1y = portfolio(B, 'S1', 'open', 1)
        d0y = portfolio(B, 'S1', 'open', 2, rule='day0')
        py = {}
        for yy in sorted(r3['by_year']):
            a = per_trade(B, 'S1', 'open', True, sel=B.year == yy)
            b = per_trade(B, 'S1', 'open', False, sel=B.year == yy)
            py[yy] = dict(p2=r3['by_year'][yy], p1=p1y['by_year'][yy], mr=a, raw=b)
            P(f"| {yy} | {r3['n_reb'][yy]} | {r3['by_year'][yy]:+.2f} | {p1y['by_year'][yy]:+.2f} | {a['mean'] * 1e4:+.2f} ({a['t']:+.1f}) | "
              f"{b['mean'] * 1e4:+.2f} ({b['t']:+.1f}) | {d0y['by_year'][yy]:+.2f} |")
        out['per_year'] = py

        P('\n## Market-relative portfolio effect (points a year), for comparison with the raw one')
        for s in B.sigs:
            c = [portfolio(B, s, 'open', p, mr=True) for p in (1, 2)]
            P(f"{s} next-open: P1 {c[0]['points']:+.2f} (se {c[0]['se']:.2f}), P2 {c[1]['points']:+.2f} (se {c[1]['se']:.2f})")
    else:
        P('\n## Test 4: the unseen months, S1, next-open, P1+P2 pooled, market-relative (pass: same sign as test 1)')
        r4 = per_trade(B, 'S1', 'open', True)
        P(fmt_pt(r4))
        out['test4'] = r4
        P('\n## Also: per trade (bps; t), pooled, P1, P2, buys, sells')
        P('| Signal | Execution | Measure | Pooled | P1 | P2 | Buys | Sells |')
        P('|---|---|---|---|---|---|---|---|')
        for s in list(B.sigs) + ['day 0']:
            for e in EXECS:
                for mr in (True, False):
                    rule = 'day0' if s == 'day 0' else 'timed'
                    sg = 'S1' if s == 'day 0' else s
                    c = [per_trade(B, sg, e, mr, rule=rule, sel=sel) for sel in
                         (None, B.port == 1, B.port == 2, B.side > 0, B.side < 0)]
                    P(f"| {s} | {EXEC_NAME[e]} | {'market-relative' if mr else 'raw'} | " +
                      ' | '.join(f"{r['mean'] * 1e4:+.2f} ({r['t']:+.1f})" for r in c) + ' |')
        P('\n## Portfolio effect (points a year, annualized from the rebalances in the period; se)')
        P('| Signal | Execution | P1 | P2 |')
        P('|---|---|---|---|')
        pt = {}
        for s in list(B.sigs) + ['day 0']:
            for e in EXECS:
                rule = 'day0' if s == 'day 0' else 'timed'
                sg = 'S1' if s == 'day 0' else s
                c = [portfolio(B, sg, e, p, rule=rule) for p in (1, 2)]
                for p, r in zip((1, 2), c):
                    pt[f'{s}|{e}|P{p}'] = r
                P(f"| {s} | {EXEC_NAME[e]} | " + ' | '.join(f"{r['points']:+.2f} (se {r['se']:.2f})" for r in c) + ' |')
        out['portfolio'] = pt
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(HERE, f'report_{which}.txt'), 'w') as f:
        f.write(txt + '\n')
    with open(os.path.join(HERE, f'report_{which}.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == '__main__':
    main('new' if len(sys.argv) > 1 and sys.argv[1] == 'unseen' else 'old')
