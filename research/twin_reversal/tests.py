"""Tests 1-3 of experiments/+1.1-twin-reversal.md on the old export (through 2025-06-27), plus the diagnostics:
twin coverage by year, sub-periods, the one-day-late check, news vs no-news, earnings-season months.

    OMP_NUM_THREADS=8 python3 -I tests.py      # writes out/tests_old.txt, out/tests_old.json, out/monthly_old.tsv
"""
import json
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402

LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def year_of(P, t):
    return int(c2eval.iso(P.days[t])[:4])


def earnings_season(P, t0):
    """The signal window d-20..d is an earnings-season month when at least half its days fall in Jan 15-Feb 15,
    Apr 15-May 15, Jul 15-Aug 15 or Oct 15-Nov 15 (fixed before results)."""
    n = 0
    for t in range(t0 - C.H + 1, t0 + 1):
        iso = c2eval.iso(P.days[t])
        m, d = int(iso[5:7]), int(iso[8:10])
        if (m in (1, 4, 7, 10) and d >= 15) or (m in (2, 5, 8, 11) and d <= 15):
            n += 1
    return n >= (C.H + 1) // 2


def stats_row(name, xs, pct=True):
    m, t, n = C.tstat(xs)
    return f'{name:<44} {100 * m:+7.3f}% {t:+6.2f} {n:4d}'


def main():
    E, tw = C.load(c2eval.EXPORT_OLD)
    P = C.Panel(E, tw)
    K = len(P.rebal)
    cov = P.cov
    # Test-1 start (decided before results): the first rebalance with a full 500-stock universe, more than one
    # distinct twin and SPY betas. Before 2005-02-25 the export holds ~320 stocks and one fund (QQQ).
    start_k = next(c['k'] for c in cov if c['universe'] >= C.N_UNIV and c['distinct_twins'] > 1 and c['spy_beta_ok'])
    end_p1 = c2eval.day_index(P.days, '2020-01-01')
    ks = [k for k in range(start_k, K) if P.rebal[k] + C.H + 1 <= P.T - 1]   # complete holds, on time and 1 day late
    say(f'start: rebalance {start_k} = {c2eval.iso(P.days[P.rebal[start_k]])}; last formation '
        f'{c2eval.iso(P.days[P.rebal[ks[-1]]])}; months {len(ks)}')

    masks = C.ranked_sets(P, [P.s_twin, P.s_plain, P.s_mkt, P.s_mkt_scaled, P.s_twin_raw])
    strategies = {
        'twin': (P.s_twin, 'twin'),
        'plain': (P.s_plain, None),
        'mkt': (P.s_mkt, 'spy'),
        'mkt_scaled (diag)': (P.s_mkt_scaled, 'spy'),
        'twin_unscaled (diag)': (P.s_twin_raw, 'twin'),
        'twin signal, SPY hedge (diag)': (P.s_twin, 'spy'),
    }
    runs = {}
    for name, (sig, hedge) in strategies.items():
        for delay in (0, 1):
            runs[(name, delay)] = C.run_strategy(P, ks, sig, hedge, delay, masks)
    say('strategies built')

    def sel(ms, lo, hi):
        return [m for m in ms if lo <= m['t0'] < hi]

    periods = {'P1 2006-02..2019': (0, end_p1), 'P3 2020..2025-06': (end_p1, P.T)}
    res = {'start': c2eval.iso(P.days[P.rebal[start_k]]), 'periods': {}, 'tests': {}}

    say('\n== Long-short by strategy (mean monthly return, t, months), 21-day holds, $1 long + $1 short ==')
    for pname, (lo, hi) in periods.items():
        say(f'\n-- {pname}')
        say(f'{"":<44} {"mean":>8} {"t":>6} {"n":>4}')
        res['periods'][pname] = {}
        for name in strategies:
            for delay in (0, 1):
                ms = sel(runs[(name, delay)], lo, hi)
                g = [m['gross'] for m in ms]
                n_ = [m['net'] for m in ms]
                lab = name + (' (1 day late)' if delay else '')
                say(stats_row(lab + ' gross', g))
                say(stats_row(lab + ' net', n_))
                if delay == 0:
                    say(stats_row(lab + ' long leg', [m['long'] for m in ms]))
                    say(stats_row(lab + ' short leg', [m['short'] for m in ms]))
                    st = np.mean([m['stock_turnover'] for m in ms[1:]]) if len(ms) > 1 else float('nan')
                    ft = np.mean([m['fund_turnover'] for m in ms[1:]]) if len(ms) > 1 else float('nan')
                    say(f'{"   turnover per rebalance (of $2 stock book)":<44} stock {st:.2f}  fund {ft:.2f}  '
                        f'cost {1e4 * np.mean([m["cost"] for m in ms]):.1f} bps/month')
                res['periods'][pname][f'{name}|{delay}'] = dict(
                    gross=C.tstat(g), net=C.tstat(n_), long=C.tstat([m['long'] for m in ms]),
                    short=C.tstat([m['short'] for m in ms]),
                    stock_turnover=float(np.mean([m['stock_turnover'] for m in ms[1:]])) if len(ms) > 1 else None,
                    fund_turnover=float(np.mean([m['fund_turnover'] for m in ms[1:]])) if len(ms) > 1 else None)

    say('\n== Paired monthly differences: twin minus baseline ==')
    for pname, (lo, hi) in periods.items():
        say(f'-- {pname}')
        for base in ('plain', 'mkt', 'mkt_scaled (diag)', 'twin_unscaled (diag)', 'twin signal, SPY hedge (diag)'):
            a = sel(runs[('twin', 0)], lo, hi)
            b = sel(runs[(base, 0)], lo, hi)
            assert [m['t0'] for m in a] == [m['t0'] for m in b]
            for kind in ('gross', 'net'):
                d = [x[kind] - y[kind] for x, y in zip(a, b)]
                say(stats_row(f'twin - {base} {kind}', d))
                res['periods'][pname][f'diff|{base}|{kind}'] = C.tstat(d)
            ov = np.mean([len(set(x['L']) & set(y['L'])) / len(x['L']) for x, y in zip(a, b)])
            ovs = np.mean([len(set(x['S']) & set(y['S'])) / len(x['S']) for x, y in zip(a, b)])
            say(f'{"   overlap of picks with twin (long / short)":<44} {ov:.2f} / {ovs:.2f}')
            res['periods'][pname][f'overlap|{base}'] = [float(ov), float(ovs)]

    # sub-periods
    say('\n== By period (gross mean %/month, t): twin, plain, mkt, twin-plain, twin-mkt; twin net ==')
    blocks = [('2006-02..2009', '2006-01-01', '2010-01-01'), ('2010..2014', '2010-01-01', '2015-01-01'),
              ('2015..2019', '2015-01-01', '2020-01-01'), ('2020..2022', '2020-01-01', '2023-01-01'),
              ('2023..2025-06', '2023-01-01', '2026-01-01')]
    res['blocks'] = {}
    for lab, a_, b_ in blocks:
        lo, hi = c2eval.day_index(P.days, a_), c2eval.day_index(P.days, b_) if b_ < '2025-07' else P.T
        tw_ = sel(runs[('twin', 0)], lo, hi)
        pl = sel(runs[('plain', 0)], lo, hi)
        mk = sel(runs[('mkt', 0)], lo, hi)
        row = dict(twin=C.tstat([m['gross'] for m in tw_]), plain=C.tstat([m['gross'] for m in pl]),
                   mkt=C.tstat([m['gross'] for m in mk]),
                   d_plain=C.tstat([x['gross'] - y['gross'] for x, y in zip(tw_, pl)]),
                   d_mkt=C.tstat([x['gross'] - y['gross'] for x, y in zip(tw_, mk)]),
                   twin_net=C.tstat([m['net'] for m in tw_]),
                   twin_late=C.tstat([m['gross'] for m in sel(runs[('twin', 1)], lo, hi)]))
        res['blocks'][lab] = row
        say(f'{lab:<14} ' + '  '.join(f'{k} {100 * v[0]:+.2f} ({v[1]:+.1f})' for k, v in row.items()) + f'  n {row["twin"][2]}')

    # by year (twin gross / net, plain gross, mkt gross)
    say('\n== By calendar year of formation: twin gross, twin net, plain gross, mkt gross (% per month) ==')
    res['years'] = {}
    for y in range(2006, 2026):
        lo, hi = c2eval.day_index(P.days, f'{y}-01-01'), c2eval.day_index(P.days, f'{y + 1}-01-01')
        r = {nm: [m['gross'] for m in sel(runs[(nm, 0)], lo, hi)] for nm in ('twin', 'plain', 'mkt')}
        tn = [m['net'] for m in sel(runs[('twin', 0)], lo, hi)]
        if not r['twin']:
            continue
        res['years'][y] = dict(twin=float(np.mean(r['twin'])), twin_net=float(np.mean(tn)), plain=float(np.mean(r['plain'])),
                               mkt=float(np.mean(r['mkt'])), n=len(tn))
        say(f'{y}  twin {100 * np.mean(r["twin"]):+6.2f}  net {100 * np.mean(tn):+6.2f}  plain {100 * np.mean(r["plain"]):+6.2f}'
            f'  mkt {100 * np.mean(r["mkt"]):+6.2f}  months {len(tn)}')

    # coverage by year
    say('\n== Twin coverage by year (mean over the year\'s rebalances) ==')
    say('year  eligible  universe  plain-top500-without-twin  plain-top500-fund-like  median-corr  distinct-twins  broad-index-share')
    res['coverage'] = {}
    for y in range(2003, 2026):
        cs = [c for c in cov if c['day'][:4] == str(y)]
        if not cs:
            continue
        row = {f: float(np.mean([c[f] for c in cs])) for f in
               ('eligible', 'universe', 'plain_no_twin', 'plain_fund_like', 'median_corr', 'distinct_twins', 'broad_share')}
        res['coverage'][y] = row
        say(f'{y}  {row["eligible"]:7.0f}  {row["universe"]:7.0f}  {row["plain_no_twin"]:7.1f}  {row["plain_fund_like"]:7.1f}'
            f'  {row["median_corr"]:.3f}  {row["distinct_twins"]:5.0f}  {row["broad_share"]:.2f}')
    # most common twins in the universe, two dates
    for k in (start_k, ks[len(ks) // 2], ks[-1]):
        fr = tw['fund_row'][k, P.universe[k]]
        u, n = np.unique(tw['F']['tickers'][fr], return_counts=True)
        o = np.argsort(-n)[:8]
        say(f'top twins {c2eval.iso(P.days[P.rebal[k]])}: ' + ', '.join(f'{u[i]} {n[i]}' for i in o))

    # news vs no-news (twin book; within-month split of the legs by the volume-spike flag)
    say('\n== News proxy: twin long-short within stocks with / without a volume spike (>= 3x) in the signal window ==')
    res['news'] = {}
    for pname, (lo, hi) in periods.items():
        ms = sel(runs[('twin', 0)], lo, hi)
        news_ls, quiet_ls, share = [], [], []
        for m in ms:
            fl = P.news_flag(m['t0'])
            nL = len(m['L'])
            fL, fS = fl[m['L']], fl[m['S']]
            legsL, legsS = m['legs'][:nL], m['legs'][nL:]
            share.append((fL.mean() + fS.mean()) / 2)
            if fL.any() and fS.any():
                news_ls.append(legsL[fL].mean() - legsS[fS].mean())
            if (~fL).any() and (~fS).any():
                quiet_ls.append(legsL[~fL].mean() - legsS[~fS].mean())
        d = [None]
        res['news'][pname] = dict(news=C.tstat(news_ls), quiet=C.tstat(quiet_ls), share_flagged=float(np.mean(share)))
        say(f'-- {pname}: share of picks flagged {np.mean(share):.2f}')
        say(stats_row('  spike (news) stocks', news_ls))
        say(stats_row('  no spike stocks', quiet_ls))
    say('\n== Earnings-season months (signal window mostly in Jan15-Feb15, Apr15-May15, Jul15-Aug15, Oct15-Nov15) ==')
    res['season'] = {}
    for pname, (lo, hi) in periods.items():
        res['season'][pname] = {}
        for nm in ('twin', 'plain', 'mkt'):
            ms = sel(runs[(nm, 0)], lo, hi)
            fl = np.array([earnings_season(P, m['t0']) for m in ms])
            g = np.array([m['gross'] for m in ms])
            res['season'][pname][nm] = dict(season=C.tstat(g[fl]), other=C.tstat(g[~fl]))
            say(f'-- {pname} {nm}: ' + stats_row('season', g[fl]) + ' | ' + stats_row('other', g[~fl]))
        ms_t = sel(runs[('twin', 0)], lo, hi)
        ms_p = sel(runs[('plain', 0)], lo, hi)
        fl = np.array([earnings_season(P, m['t0']) for m in ms_t])
        dd = np.array([x['gross'] - y['gross'] for x, y in zip(ms_t, ms_p)])
        res['season'][pname]['twin-plain'] = dict(season=C.tstat(dd[fl]), other=C.tstat(dd[~fl]))
        say(f'-- {pname} twin-plain: ' + stats_row('season', dd[fl]) + ' | ' + stats_row('other', dd[~fl]))

    # the tests
    p1, p3 = res['periods']['P1 2006-02..2019'], res['periods']['P3 2020..2025-06']
    t1 = dict(gross=p1['twin|0']['gross'], net=p1['twin|0']['net'])
    t1['pass'] = bool(t1['gross'][1] >= 3 and t1['net'][0] > 0)
    t2 = dict(vs_plain=p1['diff|plain|gross'], vs_mkt=p1['diff|mkt|gross'])
    t2['pass'] = bool(t2['vs_plain'][1] >= 2 and t2['vs_mkt'][1] >= 2)
    t3 = dict(gross=p3['twin|0']['gross'], late=p3['twin|1']['gross'], late_p1=p1['twin|1']['gross'])
    sign1 = 1.0     # "same sign": the hypothesized direction (reversal pays, positive), as recorded in the plan's decisions
    t3['pass'] = bool(np.sign(t3['gross'][0]) == sign1 and abs(t3['gross'][1]) >= 1.5 and t3['gross'][1] * sign1 > 0
                      and np.sign(t3['late'][0]) == sign1)
    res['tests'] = {'1': t1, '2': t2, '3': t3}
    say('\n== Tests ==')
    say(f'Test 1 (2006-02..2019 twin LS): gross {100 * t1["gross"][0]:+.3f}%/mo t {t1["gross"][1]:+.2f}; net '
        f'{100 * t1["net"][0]:+.3f}%/mo t {t1["net"][1]:+.2f} -> {"PASS" if t1["pass"] else "FAIL"}')
    say(f'Test 2 (paired gross differences): vs plain {100 * t2["vs_plain"][0]:+.3f} t {t2["vs_plain"][1]:+.2f}; vs mkt '
        f'{100 * t2["vs_mkt"][0]:+.3f} t {t2["vs_mkt"][1]:+.2f} -> {"PASS" if t2["pass"] else "FAIL"}')
    say(f'Test 3 (2020..2025-06): gross {100 * t3["gross"][0]:+.3f} t {t3["gross"][1]:+.2f}; one day late '
        f'{100 * t3["late"][0]:+.3f} t {t3["late"][1]:+.2f} -> {"PASS" if t3["pass"] else "FAIL"}')

    # monthly series
    with open(C.OUT / 'monthly_old.tsv', 'w') as f:
        names = list(strategies)
        f.write('formation\t' + '\t'.join(f'{n}|{d}|{kind}' for n in names for d in (0, 1) for kind in ('gross', 'net'))
                + '\tseason\n')
        by = {key: {m['t0']: m for m in v} for key, v in runs.items()}
        for k in ks:
            t0 = int(P.rebal[k])
            vals = []
            for n in names:
                for d in (0, 1):
                    m = by[(n, d)].get(t0)
                    vals += [f'{m["gross"]:.6f}' if m else 'nan', f'{m["net"]:.6f}' if m else 'nan']
            f.write(c2eval.iso(P.days[t0]) + '\t' + '\t'.join(vals) + f'\t{int(earnings_season(P, t0))}\n')
    (C.OUT / 'tests_old.json').write_text(json.dumps(res, indent=1, default=float))
    (C.OUT / 'tests_old.txt').write_text('\n'.join(LINES) + '\n')


if __name__ == '__main__':
    main()
