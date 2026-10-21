"""Tests 1-3 of experiments/-0.1-twin-momentum.md on the old export (through 2025-06-27), the baselines, the
diagnostics, the worst months (momentum crashes, 2009) and the periods.

    python3 -I tests.py      # writes out/tests_old.txt, out/tests_old.json, out/monthly_old.tsv
"""
import json
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402
from checks import start_index  # noqa: E402

LINES = []
MAIN = ('twin', 'plain', 'mkt')


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def pct(x):
    return f'{100 * x:+.3f}'


def describe(ms, spy):
    g = np.array([m['gross'] for m in ms])
    n_ = np.array([m['net'] for m in ms])
    s = np.array([spy[m['t0']] for m in ms])
    out = dict(gross=C.tstat(g), net=C.tstat(n_), long=C.tstat([m['long'] for m in ms]),
               short=C.tstat([m['short'] for m in ms]), sd=float(g.std(ddof=1)),
               sharpe_gross=float(C.sharpe(g)), sharpe_net=float(C.sharpe(n_)),
               beta_spy=float(np.cov(g, s)[0, 1] / np.var(s, ddof=1)),
               skew=float(((g - g.mean()) ** 3).mean() / g.std() ** 3),
               stock_turnover=float(np.mean([m['stock_turnover'] for m in ms[1:]])),
               fund_turnover=float(np.mean([m['fund_turnover'] for m in ms[1:]])),
               cost_bps=float(1e4 * np.mean([m['cost'] for m in ms])))
    return out


def main():
    E, tw = C.load(c2eval.EXPORT_OLD)
    P = C.Panel(E, tw)
    masks = C.ranked_sets(P, [P.s_twin, P.s_plain, P.s_mkt, P.s_twin_raw, P.s_mkt_raw])
    cov = C.coverage(P, masks)
    k0 = start_index(P, cov)
    ks = [k for k in range(k0, P.K) if P.rebal[k] + C.HOLD <= P.T - 1]
    iso = lambda t: c2eval.iso(P.days[t])  # noqa: E731
    say(f'formations {iso(P.rebal[ks[0]])} .. {iso(P.rebal[ks[-1]])} ({len(ks)}); last return day '
        f'{iso(P.rebal[ks[-1]] + C.HOLD)}')
    strategies = {
        'twin': (P.s_twin, 'twin'),
        'plain': (P.s_plain, None),
        'mkt': (P.s_mkt, 'spy'),
        'mkt unscaled (diag)': (P.s_mkt_raw, 'spy'),
        'twin unscaled (diag)': (P.s_twin_raw, 'twin'),
        'twin signal, unhedged (diag)': (P.s_twin, None),
        'plain signal, twin-hedged (diag)': (P.s_plain, 'twin'),
    }
    runs = {name: C.run_strategy(P, ks, sig, hedge, masks) for name, (sig, hedge) in strategies.items()}
    for v in runs.values():
        assert [m['t0'] for m in v] == [m['t0'] for m in runs['twin']]
    spy = {int(P.rebal[k]): float(P.fund_growth(np.array([P.spy]), P.rebal[k], P.rebal[k] + C.HOLD)[0] - 1) for k in ks}
    say('strategies built')

    def sel(ms, lo, hi):
        return [m for m in ms if lo <= m['t0'] < hi]

    end_p1 = c2eval.day_index(P.days, '2020-01-01')
    periods = {'P1 2006-03..2019': (0, end_p1), 'P3 2020..2025-05': (end_p1, P.T), 'all 2006-03..2025-05': (0, P.T)}
    res = {'start': iso(P.rebal[k0]), 'formations': [iso(P.rebal[ks[0]]), iso(P.rebal[ks[-1]])], 'periods': {}}

    say('\n== Long-short by strategy: mean % per 21-day hold (t), $1 long + $1 short ==')
    for pname, (lo, hi) in periods.items():
        say(f'\n-- {pname}')
        say(f'{"":<34} {"gross":>8} {"t":>6} {"net":>8} {"t":>6} {"long":>7} {"short":>7} {"sd":>6} {"SR g":>6} '
            f'{"SR n":>6} {"b_SPY":>6} {"skew":>6} {"turn":>5} {"fund":>5} {"cost":>5} {"n":>4}')
        res['periods'][pname] = {}
        for name in strategies:
            d = describe(sel(runs[name], lo, hi), spy)
            res['periods'][pname][name] = d
            say(f'{name:<34} {pct(d["gross"][0]):>8} {d["gross"][1]:+6.2f} {pct(d["net"][0]):>8} {d["net"][1]:+6.2f} '
                f'{pct(d["long"][0]):>7} {pct(d["short"][0]):>7} {100 * d["sd"]:6.2f} {d["sharpe_gross"]:+6.2f} '
                f'{d["sharpe_net"]:+6.2f} {d["beta_spy"]:+6.2f} {d["skew"]:+6.2f} {d["stock_turnover"]:5.2f} '
                f'{d["fund_turnover"]:5.2f} {d["cost_bps"]:5.1f} {d["gross"][2]:4d}')

    say('\n== Sharpe of the twin book vs the baselines: paired circular block bootstrap (10,000 draws) ==')
    res['boot'] = {}
    for pname, (lo, hi) in periods.items():
        say(f'-- {pname}')
        res['boot'][pname] = {}
        a = sel(runs['twin'], lo, hi)
        for base in ('plain', 'mkt', 'plain signal, twin-hedged (diag)', 'twin signal, unhedged (diag)'):
            b = sel(runs[base], lo, hi)
            for kind in ('gross', 'net'):
                blocks = (C.BLOCK, 1, 12) if (base == 'plain' and kind == 'gross') else (C.BLOCK,)
                for bl in blocks:
                    r = C.boot_sharpe_diff([m[kind] for m in a], [m[kind] for m in b], block=bl)
                    res['boot'][pname][f'{base}|{kind}|{bl}'] = r
                    say(f'  twin - {base} {kind}, block {bl}: Sharpe diff {r["diff"]:+.3f}, p {r["p"]:.4f}, '
                        f'90% interval [{r["ci90"][0]:+.3f}, {r["ci90"][1]:+.3f}], n {r["n"]}')
            d = [x['gross'] - y['gross'] for x, y in zip(a, b)]
            mu, t, n = C.tstat(d)
            ov = np.mean([len(set(x['L']) & set(y['L'])) / len(x['L']) for x, y in zip(a, b)])
            ovs = np.mean([len(set(x['S']) & set(y['S'])) / len(x['S']) for x, y in zip(a, b)])
            cr = float(np.corrcoef([x['gross'] for x in a], [y['gross'] for y in b])[0, 1])
            res['boot'][pname][f'{base}|meandiff'] = dict(mean=mu, t=t, n=n, overlap=[float(ov), float(ovs)], corr=cr)
            say(f'  twin - {base}: mean diff gross {pct(mu)} (t {t:+.2f}); picks shared long {ov:.2f} / short {ovs:.2f}; '
                f'monthly correlation {cr:+.2f}')

    # worst months / crashes
    say('\n== Worst holds, 2006-03..2025-05 (formation date; gross %; long leg, short leg) ==')
    res['worst'] = {}
    crash_names = list(MAIN) + ['plain signal, twin-hedged (diag)', 'twin signal, unhedged (diag)']
    for name in crash_names:
        ms = runs[name]
        g = np.array([m['gross'] for m in ms])
        o = np.argsort(g)[:5]
        mdd, i, j = C.max_drawdown(g)
        y09 = [m for m in ms if iso(m['t0'])[:4] == '2009']
        mam = [m for m in ms if '2009-03' <= iso(m['t0'])[:7] <= '2009-05']
        res['worst'][name] = dict(
            worst=[(iso(ms[q]['t0']), float(g[q]), ms[q]['long'], ms[q]['short']) for q in o],
            y2009=dict(sum=float(sum(m['gross'] for m in y09)), n=len(y09),
                       months=[(iso(m['t0']), m['gross'], m['long'], m['short']) for m in y09]),
            mar_may_2009=float(sum(m['gross'] for m in mam)),
            max_dd=dict(size=mdd, peak=iso(ms[i]['t0']) if i < len(ms) else None, trough=iso(ms[j - 1]['t0'])),
            worst3=float(min(g[q:q + 3].sum() for q in range(len(g) - 2))))
        w = res['worst'][name]
        say(f'-- {name}: ' + '; '.join(f'{d} {100 * x:+.1f} (L {100 * l_:+.1f}, S {100 * s_:+.1f})' for d, x, l_, s_ in w['worst']))
        say(f'   2009 ({w["y2009"]["n"]} holds): sum {100 * w["y2009"]["sum"]:+.1f}%; formed Mar-May 2009: '
            f'{100 * w["mar_may_2009"]:+.1f}%; max drawdown (cumulative sum) {100 * mdd:.1f}% from {w["max_dd"]["peak"]} '
            f'to the hold formed {w["max_dd"]["trough"]}; worst 3 consecutive holds {100 * w["worst3"]:+.1f}%')
    say('\n2009 holds, gross % (twin | plain | mkt):')
    for a, b, c in zip(*[[m for m in runs[n] if iso(m['t0'])[:4] in ('2008', '2009', '2010') and
                          '2008-11' <= iso(m['t0'])[:7] <= '2010-01'] for n in MAIN]):
        say(f'  {iso(a["t0"])}  {100 * a["gross"]:+6.1f} | {100 * b["gross"]:+6.1f} (L {100 * b["long"]:+.1f} S '
            f'{100 * b["short"]:+.1f}) | {100 * c["gross"]:+6.1f}   SPY {100 * spy[a["t0"]]:+.1f}')

    # by period blocks
    say('\n== By period: gross % per hold (t); Sharpe gross ==')
    blocks = [('2006-03..2009', '2006-01-01', '2010-01-01'), ('2010..2014', '2010-01-01', '2015-01-01'),
              ('2015..2019', '2015-01-01', '2020-01-01'), ('2020..2022', '2020-01-01', '2023-01-01'),
              ('2023..2025-05', '2023-01-01', None)]
    res['blocks'] = {}
    for lab, a_, b_ in blocks:
        lo = c2eval.day_index(P.days, a_)
        hi = c2eval.day_index(P.days, b_) if b_ else P.T
        tw_, pl, mk = (sel(runs[n], lo, hi) for n in MAIN)
        row = dict(twin=C.tstat([m['gross'] for m in tw_]), twin_net=C.tstat([m['net'] for m in tw_]),
                   plain=C.tstat([m['gross'] for m in pl]), mkt=C.tstat([m['gross'] for m in mk]),
                   d_plain=C.tstat([x['gross'] - y['gross'] for x, y in zip(tw_, pl)]),
                   sr_twin=float(C.sharpe([m['gross'] for m in tw_])), sr_plain=float(C.sharpe([m['gross'] for m in pl])),
                   sr_mkt=float(C.sharpe([m['gross'] for m in mk])), sd_twin=float(np.std([m['gross'] for m in tw_], ddof=1)),
                   sd_plain=float(np.std([m['gross'] for m in pl], ddof=1)))
        res['blocks'][lab] = row
        say(f'{lab:<14} ' + '  '.join(f'{k} {100 * v[0]:+.2f} ({v[1]:+.1f})' for k, v in row.items() if isinstance(v, tuple))
            + f'  SR twin {row["sr_twin"]:+.2f} plain {row["sr_plain"]:+.2f} mkt {row["sr_mkt"]:+.2f}  n {row["twin"][2]}')

    say('\n== By calendar year of formation: mean % per hold (twin gross, twin net, plain gross, mkt gross) ==')
    res['years'] = {}
    for y in range(2006, 2026):
        lo, hi = c2eval.day_index(P.days, f'{y}-01-01'), c2eval.day_index(P.days, f'{y + 1}-01-01')
        r = {n: [m['gross'] for m in sel(runs[n], lo, hi)] for n in MAIN}
        tn = [m['net'] for m in sel(runs['twin'], lo, hi)]
        if not tn:
            continue
        res['years'][y] = dict(twin=float(np.mean(r['twin'])), twin_net=float(np.mean(tn)),
                               plain=float(np.mean(r['plain'])), mkt=float(np.mean(r['mkt'])), n=len(tn))
        say(f'{y}  twin {100 * np.mean(r["twin"]):+6.2f}  net {100 * np.mean(tn):+6.2f}  plain {100 * np.mean(r["plain"]):+6.2f}'
            f'  mkt {100 * np.mean(r["mkt"]):+6.2f}  holds {len(tn)}')

    # the tests (pass marks fixed before results; Test 3's "same sign" = the plan's direction, positive)
    p1, p3 = res['periods']['P1 2006-03..2019'], res['periods']['P3 2020..2025-05']
    t1 = dict(gross=p1['twin']['gross'], net=p1['twin']['net'])
    t1['pass'] = bool(t1['gross'][1] >= 2.5 and t1['net'][0] > 0)
    bt = res['boot']['P1 2006-03..2019'][f'plain|gross|{C.BLOCK}']
    t2 = dict(sharpe_twin=p1['twin']['sharpe_gross'], sharpe_plain=p1['plain']['sharpe_gross'], boot=bt)
    t2['pass'] = bool(t2['sharpe_twin'] > t2['sharpe_plain'] and bt['p'] < 0.10)
    t3 = dict(gross=p3['twin']['gross'], net=p3['twin']['net'])
    t3['pass_old_part'] = bool(t3['gross'][0] > 0)
    res['tests'] = {'1': t1, '2': t2, '3_old': t3}
    say('\n== Tests ==')
    say(f'Test 1 (2006-03..2019 twin book): gross {pct(t1["gross"][0])}%/hold t {t1["gross"][1]:+.2f}; net '
        f'{pct(t1["net"][0])} t {t1["net"][1]:+.2f} -> {"PASS" if t1["pass"] else "FAIL"}')
    say(f'Test 2 (Sharpe, gross): twin {t2["sharpe_twin"]:+.3f} vs plain {t2["sharpe_plain"]:+.3f}; bootstrap p '
        f'{bt["p"]:.4f} -> {"PASS" if t2["pass"] else "FAIL"}')
    say(f'Test 3, old part (2020..2025-05, positive): gross {pct(t3["gross"][0])} t {t3["gross"][1]:+.2f} -> '
        f'{"PASS" if t3["pass_old_part"] else "FAIL"}')

    names = list(strategies)
    with open(C.OUT / 'monthly_old.tsv', 'w') as f:
        f.write('formation\tSPY\t' + '\t'.join(f'{n}|{kind}' for n in names for kind in ('gross', 'net', 'long', 'short'))
                + '\n')
        for i, m0 in enumerate(runs['twin']):
            vals = []
            for n in names:
                m = runs[n][i]
                vals += [f'{m[kind]:.6f}' for kind in ('gross', 'net', 'long', 'short')]
            f.write(iso(m0['t0']) + f'\t{spy[m0["t0"]]:.6f}\t' + '\t'.join(vals) + '\n')
    (C.OUT / 'tests_old.json').write_text(json.dumps(res, indent=1, default=float))
    (C.OUT / 'tests_old.txt').write_text('\n'.join(LINES) + '\n')


if __name__ == '__main__':
    main()
