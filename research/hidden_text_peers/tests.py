"""Tests 1-3 of experiments/+4.6-hidden-text-peers.md, plus the reported books and breakdowns.

Book at each twins-calendar day t0 (every 21 trading days) from 2007-01 to 2022-05: the firms in the day's panel
(universe firms with at least 3 hidden text peers, panel.npz from build.py), ranked by a residualized signal; long the
top tenth, short the bottom tenth, equal weight, bought at the close of t0 and held to the close of t0+21
(forward-filled closes, so a missing day earns 0). Both legs are equal weight, so the long-short is the same in
market-relative and in raw returns. Costs 5 bps per side on actual turnover (positions drifted to the end of the
previous hold); the first book pays full entry.

Signals (all residualized across the same firms on their own t-20..t return):
- H: hidden text peers (the test), V: visible text peers, A: all text peers;
- R: a stand-in for peer momentum's whole-day correlation peers: each firm's 20 most correlated universe stocks
  (daily market-relative log returns over t0-251..t0), reported only.

    python3 -I tests.py         # writes tests_out.txt and tests_out.json here
"""
import json
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import (CORR_W, COST_BPS, FORM, HERE, MIN_OVERLAP, STEP, TENTH, TEST1, TEST3, WORK, c2eval, ffill, log,  # noqa: E402
                    np, tstat, universe)
from build import returns  # noqa: E402

N_CORR_PEERS = 20
SIGNALS = ('H', 'V', 'A', 'R')
NAMES = {'H': 'Hidden text peers', 'V': 'Visible text peers', 'A': 'All text peers', 'R': 'Correlation peers (top 20)',
         'H-V': 'Hidden minus visible'}


def corr_peer_signal(mr, r21, U_rows, firms, t0):
    """Mean 21-day market-relative return of each firm's 20 most correlated universe stocks (pairwise-complete
    correlation over t0-251..t0, at least 227 common days)."""
    X = mr[U_rows, t0 - CORR_W + 1:t0 + 1].astype(np.float64)
    M = np.isfinite(X).astype(np.float64)
    X0 = np.where(M > 0, X, 0.0)
    n = M @ M.T
    sx = X0 @ M.T                      # sum of x over days where both are finite
    sxx = (X0 * X0) @ M.T
    sxy = X0 @ X0.T
    with np.errstate(all='ignore'):
        cxy = n * sxy - sx * sx.T
        cxx = n * sxx - sx * sx
        C = cxy / np.sqrt(cxx * cxx.T)
    C[(n < MIN_OVERLAP) | ~np.isfinite(C)] = -np.inf
    np.fill_diagonal(C, -np.inf)
    pos = {int(r): i for i, r in enumerate(U_rows)}
    out = np.full(len(firms), np.nan)
    rr = r21[U_rows, t0].astype(np.float64)
    for q, f in enumerate(firms):
        i = pos.get(int(f))
        if i is None:
            continue
        top = np.argpartition(-C[i], N_CORR_PEERS)[:N_CORR_PEERS]
        top = top[np.isfinite(C[i, top])]
        v = rr[top]
        v = v[np.isfinite(v)]
        if len(top) == N_CORR_PEERS and len(v):
            out[q] = v.mean()
    return out


def resid(y, x):
    ok = np.isfinite(y) & np.isfinite(x)
    out = np.full(len(y), np.nan)
    if ok.sum() < 10:
        return out
    X = np.column_stack([np.ones(ok.sum()), x[ok]])
    b, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    out[ok] = y[ok] - X @ b
    return out


def book(rows, sig, t0, cff):
    ok = np.isfinite(sig)
    rows, sig = rows[ok], sig[ok]
    n = len(rows)
    if n < 2 * TENTH:
        return None
    order = rows[np.argsort(sig, kind='stable')]
    q = (TENTH * np.arange(n)) // n
    S, L = order[q == 0], order[q == TENTH - 1]
    e, x = t0, t0 + STEP
    gL, gS = cff[L, x] / cff[L, e], cff[S, x] / cff[S, e]
    g_all = cff[rows, x] / cff[rows, e]
    out = dict(t0=int(t0), n=n, nL=len(L), nS=len(S), long=float(np.mean(gL - 1)), short=float(np.mean(gS - 1)),
               all=float(np.nanmean(g_all - 1)),
               start={**{int(j): 1.0 / len(L) for j in L}, **{int(j): -1.0 / len(S) for j in S}},
               end={**{int(j): float(g) / len(L) for j, g in zip(L, gL)}, **{int(j): -float(g) / len(S) for j, g in zip(S, gS)}})
    out['gross'] = out['long'] - out['short']
    return out


def turnover(prev_end, new_start):
    keys = set(prev_end) | set(new_start)
    return float(sum(abs(new_start.get(j, 0.0) - prev_end.get(j, 0.0)) for j in keys))


def run_books(forms, cff, key):
    months, prev = [], None
    for f in forms:
        b = book(f['rows'], f[key], f['t0'], cff)
        if b is None:
            prev = None
            continue
        if prev is not None and prev['t0'] + STEP != f['t0']:
            prev = None
        b['turnover'] = turnover(prev['end'] if prev else {}, b['start'])
        b['cost'] = COST_BPS * 1e-4 * b['turnover']
        b['net'] = b['gross'] - b['cost']
        del b['start']
        prev = b
        months.append(b)
    for m in months:
        m.pop('end', None)
    return months


def stats(months):
    if not months:
        return None
    net = np.array([m['net'] for m in months])
    gross = np.array([m['gross'] for m in months])
    mg, tg, n = tstat(gross)
    mn, tn, _ = tstat(net)
    return dict(months=n, gross=mg, t_gross=tg, net=mn, t_net=tn, sd=float(net.std(ddof=1)) if n > 1 else float('nan'),
                pos=int((net > 0).sum()), long_vs_all=tstat([m['long'] - m['all'] for m in months])[:2],
                short_vs_all=tstat([m['short'] - m['all'] for m in months])[:2],
                turnover=float(np.mean([m['turnover'] for m in months])), n_book=float(np.mean([m['n'] for m in months])),
                n_leg=float(np.mean([m['nL'] for m in months])))


def diff_stats(ma, mb):
    da = {m['t0']: m['net'] for m in ma}
    d = [da[m['t0']] - m['net'] for m in mb if m['t0'] in da]
    mean, t, n = tstat(d)
    return dict(months=n, net=mean, t_net=t, pos=int((np.array(d) > 0).sum()))


def main():
    E = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close', 'dollar_volume', 'market'))
    days = E['days']
    T = len(days)
    tw = c2eval.twins(E)
    rebal = tw['rebal']
    uni, big = universe(E, tw)
    mr, r21, cff = returns(E)
    z = np.load(WORK / 'panel.npz')
    P = {k: z[k] for k in z.files}
    a, b = c2eval.day_index(days, FORM[0]), c2eval.day_index(days, '2022-06-01')
    forms = []
    for k, t0 in enumerate(rebal):
        t0 = int(t0)
        if not (a <= t0 < b) or t0 + STEP >= T:
            continue
        s, e = np.searchsorted(P['t'], [t0, t0 + 1])
        if e - s == 0:
            continue
        rows = P['row'][s:e]
        U_rows = np.nonzero(uni[k])[0]
        rsig = corr_peer_signal(mr, r21, U_rows, rows, t0)
        f = dict(t0=t0, k=k, rows=rows, H=P['Hres'][s:e], V=P['Vres'][s:e], A=P['Ares'][s:e],
                 R=resid(rsig, P['own'][s:e].astype(np.float64)), big=big[k, rows])
        forms.append(f)
    log(f'{len(forms)} formations, {c2eval.iso(days[forms[0]["t0"]])}..{c2eval.iso(days[forms[-1]["t0"]])}; firms per '
        f'formation median {int(np.median([len(f["rows"]) for f in forms]))}')
    in_p = lambda f, p: p[0] <= c2eval.iso(days[f['t0']]) <= p[1]  # noqa: E731
    res, lines = {}, []
    periods = {'test1 2007-2019': TEST1, 'test3 2020-2022-05': TEST3, '2007-2009': ('2007-01-01', '2009-12-31'),
               '2010-2014': ('2010-01-01', '2014-12-31'), '2015-2019': ('2015-01-01', '2019-12-31'), 'all 2007-2022-05': FORM}
    books = {}
    for sub in ('all', 'big'):
        fs = forms if sub == 'all' else [dict(f, rows=f['rows'][f['big']], **{s: f[s][f['big']] for s in SIGNALS}) for f in forms]
        for s in SIGNALS:
            books[(sub, s)] = run_books(fs, cff, s)
    for sub in ('all', 'big'):
        for pn, p in periods.items():
            for s in SIGNALS:
                mo = [m for m in books[(sub, s)] if p[0] <= c2eval.iso(days[m['t0']]) <= p[1]]
                st = stats(mo)
                res[f'{sub}/{pn}/{s}'] = st
                if st:
                    lines.append(f"{sub:3} {pn:20} {NAMES[s]:28}: {st['months']} months, book {st['n_book']:.0f} (legs "
                                 f"{st['n_leg']:.0f}); gross {100 * st['gross']:+.3f}% (t {st['t_gross']:+.2f}), net "
                                 f"{100 * st['net']:+.3f}% (t {st['t_net']:+.2f}), sd {100 * st['sd']:.2f}%, {st['pos']} up; "
                                 f"long-all {100 * st['long_vs_all'][0]:+.3f}% (t {st['long_vs_all'][1]:+.2f}), short-all "
                                 f"{100 * st['short_vs_all'][0]:+.3f}% (t {st['short_vs_all'][1]:+.2f}); turnover {st['turnover']:.2f}")
            hm = [m for m in books[(sub, 'H')] if p[0] <= c2eval.iso(days[m['t0']]) <= p[1]]
            vm = [m for m in books[(sub, 'V')] if p[0] <= c2eval.iso(days[m['t0']]) <= p[1]]
            am = [m for m in books[(sub, 'A')] if p[0] <= c2eval.iso(days[m['t0']]) <= p[1]]
            rm = [m for m in books[(sub, 'R')] if p[0] <= c2eval.iso(days[m['t0']]) <= p[1]]
            for nm, (x, y) in (('H-V', (hm, vm)), ('H-A', (hm, am)), ('H-R', (hm, rm))):
                d = diff_stats(x, y)
                res[f'{sub}/{pn}/{nm}'] = d
                lines.append(f"{sub:3} {pn:20} {nm:28}: {d['months']} months, net {100 * d['net']:+.3f}% (t {d['t_net']:+.2f}), {d['pos']} up")
            lines.append('')
        print('\n'.join(lines[-40:]), flush=True)
    # by formation year (all firms)
    lines.append('by formation year (all firms), net % a month: H, V, A, R, H-V, firms per formation')
    for y in range(2007, 2023):
        p = (f'{y}-01-01', f'{y}-12-31')
        row = []
        for s in SIGNALS:
            mo = [m for m in books[('all', s)] if p[0] <= c2eval.iso(days[m['t0']]) <= p[1]]
            row.append(np.mean([m['net'] for m in mo]) if mo else np.nan)
        dv = diff_stats([m for m in books[('all', 'H')] if p[0] <= c2eval.iso(days[m['t0']]) <= p[1]],
                        [m for m in books[('all', 'V')] if p[0] <= c2eval.iso(days[m['t0']]) <= p[1]])
        nf = np.mean([len(f['rows']) for f in forms if in_p(f, p)])
        npos = sum(m['net'] > 0 for m in books[('all', 'H')] if p[0] <= c2eval.iso(days[m['t0']]) <= p[1])
        nmo = sum(1 for m in books[('all', 'H')] if p[0] <= c2eval.iso(days[m['t0']]) <= p[1])
        res[f'year/{y}'] = dict(H=row[0], V=row[1], A=row[2], R=row[3], HV=dv['net'], firms=nf, H_up=int(npos), months=nmo)
        lines.append(f'  {y}: ' + ', '.join(f'{100 * v:+.2f}' for v in row) + f', {100 * dv["net"]:+.2f}, {nf:.0f} '
                     f'(H up {npos} of {nmo})')
    print('\n'.join(lines[-17:]))
    res['monthly_H'] = [(c2eval.iso(days[m['t0']]), m['gross'], m['net'], m['long'], m['short'], m['all'], m['n'])
                        for m in books[('all', 'H')]]
    (HERE / 'tests_out.txt').write_text('\n'.join(lines) + '\n')
    (HERE / 'tests_out.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
