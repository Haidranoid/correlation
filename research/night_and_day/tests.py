"""Night and day: Tests 1-4 of experiments/-6.0-night-and-day.md (IC by leg, the intraday and overnight decile
long-shorts, the 500 largest, the later years), on one export.

    python3 -I tests.py old      # 2008-2019 and 2020-2025-06
    python3 -I tests.py new      # the unseen months 2025-06-30 .. 2026-10-19 (once, at the end)

Per decision day t (trade day d = t+1): rank s(t) over the universe (or its 500 largest); long the top 10%, short
the bottom 10%, equal weight. Each book's daily return is the long leg's mean simple return minus the short leg's,
over the members with valid legs on d. The intraday book holds open(d) -> close(d), the overnight book close(t) ->
open(d), and "full day" close(t) -> close(d) (what the judge trades). Both books open and close every day, so costs
are 4 trades a day: net = gross - 4 x c bps at c bps per side; break-even c = gross / 4.

Diagnostic (not a test): legs hedged with the stock's contemporaneous Dimson beta b0(t) times SPY's same leg on d.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np = C.np
c2eval = C.c2eval
iso = c2eval.iso

tag = sys.argv[1] if len(sys.argv) > 1 else 'old'
PERIODS = {'old': [('2008-2019', '2008-01-01', '2019-12-31'), ('2020-2025-06', '2020-01-01', '2025-06-27')],
           'new': [('unseen 2025-06-30..2026-10-19', '2025-06-30', '2026-10-19')]}[tag]
COSTS = (0.0, 2.0, 5.0)
LEGS = ('on', 'in', 'full')
LEG_NAME = {'on': 'overnight (close t -> open t+1)', 'in': 'intraday (open t+1 -> close t+1)',
            'full': 'full day (close t -> close t+1)'}


def series(P, which):
    """Per trade day d: book returns (simple, long - short) for each leg, hedged versions, long/short excess over the
    universe's equal-weight mean, and the daily rank ICs."""
    days = P['days']
    T = len(days)
    first = c2eval.day_index(days, PERIODS[0][1])
    end = (np.datetime64(PERIODS[-1][2]) - np.datetime64('1970-01-01')).astype(int)
    last = int(np.searchsorted(days, end, side='right')) - 1
    sT = P['sT']; b0T = P['b0T']; onT = P['onT']; inT = P['inT']
    rec = {k: [] for k in ('d', 'n', 'nl', 'ns', 'm')}
    for lg in LEGS:
        for k in (f'ls_{lg}', f'ls_{lg}_h', f'lx_{lg}', f'sx_{lg}', f'ic_{lg}'):
            rec[k] = []
    for d in range(max(first, 1), last + 1):
        t = d - 1
        k = C.kidx(P['rebal'], t)
        if k < 0:
            continue
        U = P['uni'][k] if which == 'all' else P['big'][k]
        s = sT[t]
        U = U & np.isfinite(s)
        n = int(U.sum())
        if n < C.MIN_NAMES:
            continue
        rows = np.nonzero(U)[0]
        o = rows[np.argsort(-s[rows], kind='stable')]
        mm = int(round(C.DECILE * n))
        L, S = o[:mm], o[-mm:]
        lon, lin = onT[d].astype(np.float64), inT[d].astype(np.float64)
        V = np.isfinite(lon)
        lf = lon + lin
        spy = {'on': P['spy_on'][d], 'in': P['spy_in'][d]}
        spy['full'] = spy['on'] + spy['in']
        b0 = b0T[t].astype(np.float64)
        LV, SV, UV = L[V[L]], S[V[S]], rows[V[rows]]
        if len(LV) == 0 or len(SV) == 0:
            continue
        rec['d'].append(d); rec['n'].append(n); rec['nl'].append(len(LV)); rec['ns'].append(len(SV))
        rec['m'].append(P['m'][t])
        for lg, x in (('on', lon), ('in', lin), ('full', lf)):
            g = np.expm1(x)
            gl, gs, gu = g[LV].mean(), g[SV].mean(), g[UV].mean()
            rec[f'ls_{lg}'].append(gl - gs)
            rec[f'lx_{lg}'].append(gl - gu)
            rec[f'sx_{lg}'].append(gs - gu)
            sp = np.expm1(spy[lg]) if np.isfinite(spy[lg]) else np.nan
            hl = g[LV] - np.nan_to_num(b0[LV]) * sp
            hs = g[SV] - np.nan_to_num(b0[SV]) * sp
            rec[f'ls_{lg}_h'].append(hl.mean() - hs.mean())
            rec[f'ic_{lg}'].append(C.spearman(s[UV].astype(np.float64), x[UV]))
    return {k: np.array(v) for k, v in rec.items()}


def period_mask(R, days, a, b):
    da = days[R['d']]
    lo = (np.datetime64(a) - np.datetime64('1970-01-01')).astype(int)
    hi = (np.datetime64(b) - np.datetime64('1970-01-01')).astype(int)
    return (da >= lo) & (da <= hi)


def summarize(R, days, sel):
    out = {'days': int(sel.sum()), 'names_median': float(np.median(R['n'][sel])) if sel.any() else float('nan'),
           'leg_names_median': float(np.median(R['nl'][sel])) if sel.any() else float('nan')}
    for lg in LEGS:
        ic, ict, _ = C.tstat(R[f'ic_{lg}'][sel])
        g, gt, nd = C.tstat(R[f'ls_{lg}'][sel])
        x = R[f'ls_{lg}'][sel]
        sd = float(np.nanstd(x, ddof=1)) if nd > 2 else float('nan')
        h, ht, _ = C.tstat(R[f'ls_{lg}_h'][sel])
        lx, lxt, _ = C.tstat(R[f'lx_{lg}'][sel])
        sx, sxt, _ = C.tstat(R[f'sx_{lg}'][sel])
        net = {}
        for c in COSTS:
            y = x - 4 * c * 1e-4
            mu, tt, _ = C.tstat(y)
            net[f'{c:g}'] = {'bps_day': mu * 1e4, 't': tt, 'nw_t': C.nw_t(y), 'pct_year': 100 * 252 * mu,
                             'sharpe': mu / sd * np.sqrt(252) if sd > 0 else float('nan'),
                             'pos_frac': float((y > 0).mean()) if len(y) else float('nan')}
        out[lg] = {'ic_mean': ic, 'ic_t': ict, 'ic_nw_t': C.nw_t(R[f'ic_{lg}'][sel]),
                   'ic_pos_frac': float((R[f'ic_{lg}'][sel] > 0).mean()) if sel.any() else float('nan'),
                   'gross_bps': g * 1e4, 'gross_t': gt, 'gross_nw_t': C.nw_t(x), 'net': net,
                   'breakeven_bps_per_side': g * 1e4 / 4,
                   'hedged_bps': h * 1e4, 'hedged_t': ht,
                   'long_excess_bps': lx * 1e4, 'long_excess_t': lxt, 'short_excess_bps': sx * 1e4, 'short_excess_t': sxt}
    # by SPY's sign on day t (where the catch-up should be, both ways)
    up = sel & (R['m'] > 0)
    dn = sel & (R['m'] < 0)
    for nm, ss in (('spy_up', up), ('spy_down', dn)):
        out[nm] = {lg: {'gross_bps': C.tstat(R[f'ls_{lg}'][ss])[0] * 1e4, 't': C.tstat(R[f'ls_{lg}'][ss])[1],
                        'days': int(ss.sum())} for lg in LEGS}
    return out


def by_year(R, days, sel):
    yrs = np.array([int(iso(x)[:4]) for x in days[R['d']]])
    rows = []
    for y in np.unique(yrs[sel]):
        s = sel & (yrs == y)
        r = {'year': int(y), 'days': int(s.sum())}
        for lg in LEGS:
            mu, tt, _ = C.tstat(R[f'ls_{lg}'][s])
            r[lg] = (mu * 1e4, tt)
            r[f'ic_{lg}'] = C.tstat(R[f'ic_{lg}'][s])[0]
        rows.append(r)
    return rows


def fmt(x, p=2, sign=True):
    if x is None or not np.isfinite(x):
        return 'n/a'
    return f'{x:+.{p}f}' if sign else f'{x:.{p}f}'


def main():
    P = C.load(tag, mmap=False)
    C.log('loaded; transposing')
    P['sT'] = np.ascontiguousarray(np.asarray(P['s']).T)
    P['b0T'] = np.ascontiguousarray(np.asarray(P['b0']).T)
    P['onT'] = np.ascontiguousarray(np.asarray(P['on']).T)
    P['inT'] = np.ascontiguousarray(np.asarray(P['intra']).T)
    for a in ('s', 'b0', 'blag', 'on', 'intra'):
        P.pop(a)
    days = P['days']
    res = {'tag': tag, 'periods': {}}
    L = []
    W = L.append
    for which in ('all', 'big'):
        C.log('series', which)
        R = series(P, which)
        np.savez(C.WORK / f'series_{tag}_{which}.npz', **R)
        for nm, a, b in PERIODS:
            sel = period_mask(R, days, a, b)
            res['periods'][f'{which}|{nm}'] = summarize(R, days, sel)
            res['periods'][f'{which}|{nm}']['by_year'] = by_year(R, days, sel)
    # ---- tables
    title = {'all': 'All liquid stocks', 'big': 'The 500 largest'}
    W(f'# Night and day: tests on the {tag} export\n')
    W('## Test 1: IC of s with each leg (mean daily rank IC; t from the daily series = clustered by date; NW t, 5 lags)\n')
    W('| Set | Period | Days | Median names | Overnight IC (t) | Intraday IC (t) | Full day IC (t) |')
    W('|---|---|---|---|---|---|---|')
    for which in ('all', 'big'):
        for nm, _, _ in PERIODS:
            r = res['periods'][f'{which}|{nm}']
            cells = [f"{r[lg]['ic_mean']:+.4f} ({r[lg]['ic_t']:+.2f}; NW {r[lg]['ic_nw_t']:+.2f})" for lg in LEGS]
            W(f"| {title[which]} | {nm} | {r['days']} | {r['names_median']:.0f} | " + ' | '.join(cells) + ' |')
    W('\n## Tests 2-4: the decile long-shorts (bps a day, long minus short; t from daily returns; NW t, 5 lags)\n')
    W('| Set | Period | Book | Gross bps/day (t; NW t) | Net 2 bps (t) | Net 5 bps (t) | Break-even bps/side | Sharpe gross | Hedged gross (t) |')
    W('|---|---|---|---|---|---|---|---|---|')
    for which in ('all', 'big'):
        for nm, _, _ in PERIODS:
            r = res['periods'][f'{which}|{nm}']
            for lg in LEGS:
                q = r[lg]
                W(f"| {title[which]} | {nm} | {LEG_NAME[lg]} | {q['gross_bps']:+.2f} ({q['gross_t']:+.2f}; {q['gross_nw_t']:+.2f}) | "
                  f"{q['net']['2']['bps_day']:+.2f} ({q['net']['2']['t']:+.2f}) | {q['net']['5']['bps_day']:+.2f} ({q['net']['5']['t']:+.2f}) | "
                  f"{q['breakeven_bps_per_side']:+.2f} | {q['net']['0']['sharpe']:+.2f} | {q['hedged_bps']:+.2f} ({q['hedged_t']:+.2f}) |")
    W('\n## Long and short sides against the universe mean (bps a day, t)\n')
    W('| Set | Period | Book | Long minus universe | Short minus universe |')
    W('|---|---|---|---|---|')
    for which in ('all', 'big'):
        for nm, _, _ in PERIODS:
            r = res['periods'][f'{which}|{nm}']
            for lg in LEGS:
                q = r[lg]
                W(f"| {title[which]} | {nm} | {lg} | {q['long_excess_bps']:+.2f} ({q['long_excess_t']:+.2f}) | "
                  f"{q['short_excess_bps']:+.2f} ({q['short_excess_t']:+.2f}) |")
    W('\n## By SPY\'s direction on day t (gross bps a day, t)\n')
    W('| Set | Period | SPY on t | Days | Overnight | Intraday | Full day |')
    W('|---|---|---|---|---|---|---|')
    for which in ('all', 'big'):
        for nm, _, _ in PERIODS:
            r = res['periods'][f'{which}|{nm}']
            for sd in ('spy_up', 'spy_down'):
                q = r[sd]
                W(f"| {title[which]} | {nm} | {sd.split('_')[1]} | {q['on']['days']} | " +
                  ' | '.join(f"{q[lg]['gross_bps']:+.2f} ({q[lg]['t']:+.2f})" for lg in LEGS) + ' |')
    W('\n## By year (gross bps a day, t; and mean IC)\n')
    for which in ('all', 'big'):
        W(f'**{title[which]}**\n')
        W('| Year | Days | Overnight | Intraday | Full day | IC overnight | IC intraday |')
        W('|---|---|---|---|---|---|---|')
        for nm, _, _ in PERIODS:
            for y in res['periods'][f'{which}|{nm}']['by_year']:
                W(f"| {y['year']} | {y['days']} | {y['on'][0]:+.2f} ({y['on'][1]:+.2f}) | {y['in'][0]:+.2f} ({y['in'][1]:+.2f}) | "
                  f"{y['full'][0]:+.2f} ({y['full'][1]:+.2f}) | {y['ic_on']:+.4f} | {y['ic_in']:+.4f} |")
        W('')
    # ---- verdict lines
    W('## Pass/fail (rules fixed in the .md)\n')
    if tag == 'old':
        a = res['periods']['all|2008-2019']['in']['net']['2']
        b = res['periods']['big|2008-2019']['in']['net']['2']
        c = res['periods']['all|2020-2025-06']['in']['net']['2']
        W(f"- Test 2 (intraday, all, 2008-2019, net of 2 bps/side, t >= 3): {a['bps_day']:+.2f} bps/day, t {a['t']:+.2f} -> "
          f"{'PASS' if a['bps_day'] > 0 and a['t'] >= 3 else 'FAIL'}")
        W(f"- Test 3 (intraday, 500 largest, 2008-2019, net of 2 bps/side, positive with t >= 2): {b['bps_day']:+.2f} bps/day, "
          f"t {b['t']:+.2f} -> {'PASS' if b['bps_day'] > 0 and b['t'] >= 2 else 'FAIL'}")
        W(f"- Test 4a (intraday, all, 2020-2025-06, net of 2 bps/side, positive): {c['bps_day']:+.2f} bps/day, t {c['t']:+.2f} -> "
          f"{'PASS' if c['bps_day'] > 0 else 'FAIL'}")
    else:
        a = res['periods']['all|unseen 2025-06-30..2026-10-19']['in']['net']['2']
        W(f"- Test 4b (intraday, all, unseen months, net of 2 bps/side, positive): {a['bps_day']:+.2f} bps/day, t {a['t']:+.2f} -> "
          f"{'PASS' if a['bps_day'] > 0 else 'FAIL'}")
    txt = '\n'.join(L) + '\n'
    (C.OUT / f'tests_{tag}.md').write_text(txt)
    (C.OUT / f'tests_{tag}.json').write_text(json.dumps(res, indent=1, default=float))
    print(txt)


if __name__ == '__main__':
    main()
