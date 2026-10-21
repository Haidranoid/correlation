"""Tests 1-3 of experiments/-1.3-lazy-prices.md, plus the reported breakdowns.

Book at each calendar day t0 (twins' calendar, every 21 trading days): rank the universe stocks with a known
similarity at the close of t0; long the top fifth (non-changers), short the bottom fifth (changers), equal weight,
bought at the close of t0 and held to the close of t0+21. Costs 5 bps per side on actual turnover (positions drifted
to the end of the previous hold).

    python3 -I tests.py        # writes tests_out.txt and tests_out.json here
"""
import json
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import (COST_BPS, FIFTH, HERE, N_LARGE, STEP, TEST1, TEST3, c2eval, ols, rf_daily, spy_close,  # noqa: E402
                    tstat, np)
import panel as PN  # noqa: E402


def book(k, t0, sim_t0, uni, mdv, cff, large=False):
    rows = np.nonzero(uni[k] & np.isfinite(sim_t0))[0]
    if large and len(rows) > N_LARGE:
        rows = rows[np.argsort(-np.nan_to_num(mdv[k, rows], nan=-1), kind='stable')[:N_LARGE]]
    n = len(rows)
    if n < 2 * FIFTH:
        return None
    order = rows[np.argsort(sim_t0[rows], kind='stable')]          # lowest similarity first
    q = (FIFTH * np.arange(n)) // n
    S, L = order[q == 0], order[q == FIFTH - 1]
    e, x = t0, t0 + STEP
    g_all = cff[rows, x] / cff[rows, e]
    gL, gS = cff[L, x] / cff[L, e], cff[S, x] / cff[S, e]
    out = dict(t0=int(t0), n=n, nL=len(L), nS=len(S), long=float(np.mean(gL - 1)), short=float(np.mean(gS - 1)),
               all=float(np.nanmean(g_all - 1)),
               start={**{int(j): 1.0 / len(L) for j in L}, **{int(j): -1.0 / len(S) for j in S}},
               end={**{int(j): float(g) / len(L) for j, g in zip(L, gL)}, **{int(j): -float(g) / len(S) for j, g in zip(S, gS)}},
               sim_L=float(np.mean(sim_t0[L])), sim_S=float(np.mean(sim_t0[S])), L=L.tolist(), S=S.tolist())
    out['gross'] = out['long'] - out['short']
    return out


def turnover(prev_end, new_start):
    keys = set(prev_end) | set(new_start)
    return float(sum(abs(new_start.get(j, 0.0) - prev_end.get(j, 0.0)) for j in keys))


def run(ks, rebal, panel_, uni, mdv, cff, large=False):
    months, prev = [], None
    for k in ks:
        t0 = int(rebal[k])
        b = book(k, t0, panel_[:, t0], uni, mdv, cff, large)
        if b is None:
            prev = None
            continue
        if prev is not None and prev['t0'] + STEP != t0:
            prev = None
        b['turnover'] = turnover(prev['end'] if prev else {}, b['start'])
        b['cost'] = COST_BPS * 1e-4 * b['turnover']
        b['net'] = b['gross'] - b['cost']
        months.append(b)
        prev = b
    return months


def stats(months, mkt, spy, rf, days):
    """Mean monthly return (gross, net) with t, and CAPM alphas of the net long-short on the EW market and SPY."""
    if not months:
        return None
    t0 = np.array([m['t0'] for m in months])
    net = np.array([m['net'] for m in months])
    gross = np.array([m['gross'] for m in months])
    mk = np.array([mkt[t] for t in t0]); sp = np.array([spy[t] for t in t0]); r = np.array([rf[t] for t in t0])
    mg, tg, n = tstat(gross)
    mn, tn, _ = tstat(net)
    out = dict(months=n, gross=mg, t_gross=tg, net=mn, t_net=tn, sd=float(net.std(ddof=1)) if n > 1 else float('nan'),
               long_vs_all=tstat([m['long'] - m['all'] for m in months])[:2],
               short_vs_all=tstat([m['short'] - m['all'] for m in months])[:2],
               turnover=float(np.mean([m['turnover'] for m in months])),
               n_book=float(np.mean([m['n'] for m in months])), n_leg=float(np.mean([m['nL'] for m in months])),
               sim_L=float(np.mean([m['sim_L'] for m in months])), sim_S=float(np.mean([m['sim_S'] for m in months])),
               first=c2eval.iso(days[t0[0]]), last=c2eval.iso(days[t0[-1]]))
    for name, f in (('ew', mk), ('spy', sp)):
        if n >= 6:
            b, t, nn = ols(net, f - r)
            out[f'alpha_{name}'] = (float(b[0]), float(t[0]), float(b[1]), nn)
        else:
            out[f'alpha_{name}'] = (float('nan'),) * 3 + (n,)
    return out


def in_window(rebal, days, a, b):
    lo, hi = c2eval.day_index(days, a), c2eval.day_index(days, b) - 1
    if c2eval.iso(days[min(hi + 1, len(days) - 1)]) <= b:
        hi += 1
    return [k for k, t in enumerate(rebal) if lo <= t <= hi and t + STEP < len(days)]


def fmt(name, s):
    if s is None:
        return f'{name}: no months'
    ae, asp = s['alpha_ew'], s['alpha_spy']
    return (f"{name}: {s['months']} months {s['first']}..{s['last']}; book {s['n_book']:.0f} stocks, legs {s['n_leg']:.0f}; "
            f"mean monthly gross {100 * s['gross']:+.3f}% (t {s['t_gross']:+.2f}), net {100 * s['net']:+.3f}% "
            f"(t {s['t_net']:+.2f}), sd {100 * s['sd']:.2f}%; alpha vs EW market {100 * ae[0]:+.3f}%/month "
            f"(t {ae[1]:+.2f}, beta {ae[2]:+.2f}); vs SPY {100 * asp[0]:+.3f}% (t {asp[1]:+.2f}, beta {asp[2]:+.2f}); "
            f"long - all {100 * s['long_vs_all'][0]:+.3f}% (t {s['long_vs_all'][1]:+.2f}), short - all "
            f"{100 * s['short_vs_all'][0]:+.3f}% (t {s['short_vs_all'][1]:+.2f}); turnover {s['turnover']:.2f}; "
            f"mean similarity long {s['sim_L']:.3f}, short {s['sim_S']:.3f}")


def main():
    E, tw, uni, mdv = PN.load()
    days, rebal = E['days'], tw['rebal']
    P = PN.pairs()
    cff = PN.price_paths(E)
    T = len(days)
    # monthly factor returns over each hold t0 -> t0+21
    m = np.nan_to_num(np.asarray(E['market'], np.float64))
    cm = np.concatenate([[0], np.cumsum(m)])
    rfd = np.nan_to_num(rf_daily(days))
    crf = np.concatenate([[0], np.cumsum(rfd)])
    spc = PN.ffill(spy_close(days)[None, :])[0]
    mkt, spy, rf = {}, {}, {}
    for t0 in rebal:
        t0 = int(t0)
        if t0 + STEP >= T:
            continue
        mkt[t0] = np.exp(cm[t0 + STEP + 1] - cm[t0 + 1]) - 1
        spy[t0] = spc[t0 + STEP] / spc[t0] - 1
        rf[t0] = crf[t0 + STEP + 1] - crf[t0 + 1]
    res, lines = {}, []
    k1 = in_window(rebal, days, *TEST1)
    k3 = in_window(rebal, days, *TEST3)
    periods = {'2007-2009': ('2007-01-01', '2009-12-31'), '2010-2014': ('2010-01-01', '2014-12-31'),
               '2015-2019': ('2015-01-01', '2019-12-31'), '2020-2021': ('2020-01-01', '2021-12-31')}
    panels = {}
    runs = [(m, 'june') for m in PN.MEASURES] + [('sim', 'april')]
    for meas, rule in runs:
        sp = PN.sim_panel(E, P, meas, rule)
        if rule == 'april':
            meas = 'sim_april'
        if meas == 'sim':
            panels[meas] = sp
        for tag, ks, large in (('test1', k1, False), ('test2_large', k1, True), ('test3_2022', k3, False)):
            mo = run(ks, rebal, sp, uni, mdv, cff, large)
            res[f'{meas}/{tag}'] = stats(mo, mkt, spy, rf, days)
            lines.append(fmt(f'{meas} {tag}', res[f'{meas}/{tag}']))
            if meas == 'sim' and tag in ('test1', 'test2_large'):
                for pn, (a, b) in periods.items():
                    sub = [x for x in mo if a <= c2eval.iso(days[x['t0']]) <= b]
                    res[f'{meas}/{tag}/{pn}'] = stats(sub, mkt, spy, rf, days)
                    lines.append('   ' + fmt(f'{meas} {tag} {pn}', res[f'{meas}/{tag}/{pn}']))
                res[f'{meas}/{tag}/monthly'] = [(c2eval.iso(days[x['t0']]), x['gross'], x['net'], x['long'], x['short'],
                                                 x['all'], x['n']) for x in mo]
            if meas == 'sim' and tag == 'test3_2022':
                res[f'{meas}/{tag}/monthly'] = [(c2eval.iso(days[x['t0']]), x['gross'], x['net'], x['long'], x['short'],
                                                 x['all'], x['n']) for x in mo]
            print(lines[-1], flush=True)
    # coverage: universe size and share with a similarity, by year (calendar days in test windows)
    cov = []
    sp = panels['sim']
    for k in in_window(rebal, days, '2005-01-01', '2022-12-31'):
        t0 = int(rebal[k])
        u = uni[k]
        cov.append((c2eval.iso(days[t0]), int(u.sum()), int((u & np.isfinite(sp[:, t0])).sum())))
    res['coverage'] = cov
    by_year = {}
    for d, nu, ns in cov:
        by_year.setdefault(d[:4], []).append((nu, ns))
    lines.append('coverage by year (mean universe, mean with similarity):')
    for y, v in sorted(by_year.items()):
        a = np.array(v)
        lines.append(f'  {y}: {a[:, 0].mean():.0f} universe, {a[:, 1].mean():.0f} with similarity ({a[:, 1].sum() / max(a[:, 0].sum(), 1):.0%})')
    # pairs actually used in test 1 books (sim): fiscal-year-end months
    used = set()
    rows_cik = {}
    for c, rows in PN.cik_rows().items():
        for r in rows:
            rows_cik[r] = c
    for k in k1:
        t0 = int(rebal[k])
        rows = np.nonzero(uni[k] & np.isfinite(sp[:, t0]))[0]
        for r in rows:
            used.add((rows_cik[int(r)], int(_epoch(days[t0]))))
    fy_month = {(int(c), int(f)): int(mo) for c, f, mo in zip(P['cik'], P['fy'], P['fye_month'])}
    months_used = np.array([fy_month.get(u, -1) for u in used])
    res['used_pairs'] = dict(n=len(used), known=int((months_used > 0).sum()),
                             non_december=int(((months_used > 0) & (months_used != 12)).sum()))
    lines.append(f"pairs used in test 1's universe: {len(used)}; fiscal-year end known {res['used_pairs']['known']}, "
                 f"non-December {res['used_pairs']['non_december']}")
    print('\n'.join(lines[-(len(by_year) + 2):]))
    (HERE / 'tests_out.txt').write_text('\n'.join(lines) + '\n')
    (HERE / 'tests_out.json').write_text(json.dumps(res, indent=1, default=float))


def _epoch(day):
    from common import epoch_of
    return epoch_of(day)


if __name__ == '__main__':
    main()
