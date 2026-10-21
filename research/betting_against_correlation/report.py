"""Tables for tests 1-4 (and the unseen months on the new export).

    python3 -I report.py old     -> results_old.txt
    python3 -I report.py new     -> results_new.txt (unseen months only)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import c2eval, np, WORK, HERE, ols_nw

tag = _sys.argv[1]
z = np.load(WORK / f'series_{tag}.npz')
days = z['days']
rf, mkt, spy = z['rf'], z['mkt'], z['spy']
mkt_ex, spy_ex = mkt - rf, spy - rf
lines = []


def P(s=''):
    print(s)
    lines.append(s)


def mask(a, b):
    return (days >= c2eval.day_index(days, a) * 0 + (np.datetime64(a) - np.datetime64('1970-01-01')).astype(int)) & \
           (days <= (np.datetime64(b) - np.datetime64('1970-01-01')).astype(int))


def stats(y, M):
    ok = M & np.isfinite(y)
    r = y[ok]
    n = len(r)
    mean, sd = r.mean() * 252, r.std(ddof=1) * np.sqrt(252)
    b1, t1, _, _ = ols_nw(r, mkt_ex[ok])      # ols_nw drops days with a missing factor (SPY: 2 days in 2011)
    b2, t2, _, _ = ols_nw(r, spy_ex[ok])
    return dict(n=n, years=n / 252, mean=mean * 100, sd=sd * 100, sharpe=mean / sd, cum=(np.prod(1 + r) - 1) * 100,
                a_ew=b1[0] * 252 * 100, t_ew=t1[0], b_ew=b1[1], a_spy=b2[0] * 252 * 100, t_spy=t2[0], b_spy=b2[1])


HDR = f"{'':34s} {'yrs':>5s} {'ret%/yr':>8s} {'vol%':>6s} {'Sharpe':>7s} {'alpha EW':>9s} {'t':>6s} {'beta EW':>8s} {'alpha SPY':>10s} {'t':>6s} {'beta SPY':>9s}"


def row(label, s):
    P(f"{label:34s} {s['years']:5.1f} {s['mean']:8.2f} {s['sd']:6.2f} {s['sharpe']:7.2f} {s['a_ew']:9.2f} {s['t_ew']:6.2f} "
      f"{s['b_ew']:8.2f} {s['a_spy']:10.2f} {s['t_spy']:6.2f} {s['b_spy']:9.2f}")


if tag == 'old':
    T1 = mask('2008-03-06', '2019-12-31')
    T3 = mask('2020-01-02', '2025-06-27')
    FULL = mask('2008-03-06', '2025-06-27')
    THIN = mask('2006-01-03', '2008-03-05')
    first = c2eval.iso(days[z['bac__form'][0] + 1])
    P(f"first BAC return day (any universe >= 100): {first}; graded periods start 2008-03-06 (universe >= 500)")
    P()
    P('== Test 1: BAC, 2008-03-06..2019-12-31 (net = after 5 bps per side) ==')
    P(HDR)
    s1n = stats(z['bac__net'], T1)
    row('BAC net', s1n)
    row('BAC gross', stats(z['bac__gross'], T1))
    pass1 = s1n['t_ew'] >= 2 and s1n['t_spy'] >= 2 and s1n['sharpe'] >= 0.3
    P(f"PASS test 1 (both alpha t >= 2 and Sharpe >= 0.3, net): {pass1}")
    sign = np.sign(s1n['a_ew'])
    P()
    P('== By period (net, then gross) ==')
    P(HDR)
    periods = [('2006-01..2008-03 (299 stocks; not graded)', THIN), ('2008-03..2009', mask('2008-03-06', '2009-12-31')),
               ('2010-2014', mask('2010-01-01', '2014-12-31')), ('2015-2019', mask('2015-01-01', '2019-12-31')),
               ('2020-2025-06', T3), ('all 2008-03..2025-06', FULL)]
    for lab, M in periods:
        row('BAC net ' + lab, stats(z['bac__net'], M))
    for lab, M in periods:
        row('BAC gross ' + lab, stats(z['bac__gross'], M))
    P()
    P('== Test 2: the 500 largest (by 252-day median dollar volume), net ==')
    P(HDR)
    s2 = stats(z['bac500__net'], T1)
    row('BAC500 net 2008-03..2019', s2)
    row('BAC500 gross 2008-03..2019', stats(z['bac500__gross'], T1))
    for lab, M in periods[1:]:
        row('BAC500 net ' + lab, stats(z['bac500__net'], M))
    pass2 = np.sign(s2['a_ew']) == sign and np.sign(s2['a_spy']) == sign and s2['t_ew'] * sign >= 1.5 and s2['t_spy'] * sign >= 1.5
    P(f"PASS test 2 (same sign as test 1, both alpha t >= 1.5): {pass2}")
    P()
    P('== Test 3: 2020-01-02..2025-06-27 (old export), net ==')
    P(HDR)
    s3 = stats(z['bac__net'], T3)
    row('BAC net 2020-2025-06', s3)
    row('BAC gross 2020-2025-06', stats(z['bac__gross'], T3))
    row('BAC500 net 2020-2025-06', stats(z['bac500__net'], T3))
    pass3 = np.sign(s3['a_ew']) == sign and np.sign(s3['a_spy']) == sign
    P(f"PASS test 3 (both alphas with test 1's sign): {pass3}")
    P()
    P('== Controls and sensitivities (net) ==')
    P(HDR)
    for nm_, lab in (('bab', 'BAB'), ('lowvol', 'Low volatility'), ('bab500', 'BAB, 500 largest'), ('lowvol500', 'Low vol, 500 largest'),
                     ('bac_dollar', 'BAC, $1/$1 (no beta scaling)'), ('bac_flag', 'BAC, flag-only universe')):
        for lab2, M in (('2008-03..2019', T1), ('2020-2025-06', T3)):
            row(f'{lab} {lab2}', stats(z[nm_ + '__net'], M))
    P()
    P('== Test 4: BAC net on the EW market, BAB and low volatility (net), Newey-West 5 lags ==')
    for lab, M in (('2008-03..2019', T1), ('2020-2025-06', T3), ('all 2008-03..2025-06', FULL)):
        for xs, xl in (((mkt_ex, z['bab__net']), 'mkt+BAB'), ((mkt_ex, z['lowvol__net']), 'mkt+LowVol'),
                       ((mkt_ex, z['bab__net'], z['lowvol__net']), 'mkt+BAB+LowVol')):
            X = np.column_stack(xs)
            ok = M & np.isfinite(z['bac__net']) & np.isfinite(X).all(1)
            b, t, n, r2 = ols_nw(z['bac__net'][ok], X[ok])
            P(f"  {lab:22s} {xl:16s} alpha {b[0] * 252 * 100:6.2f}%/yr t {t[0]:5.2f} | loadings " +
              ', '.join(f'{v:+.2f} (t {tt:+.1f})' for v, tt in zip(b[1:], t[1:])) + f" | R2 {r2:.2f}")
        X = np.column_stack((mkt_ex, z['bab__net'], z['lowvol__net']))
        ok = M & np.isfinite(z['bac500__net']) & np.isfinite(X).all(1)
        X5 = np.column_stack((mkt_ex, z['bab500__net'], z['lowvol500__net']))
        ok5 = M & np.isfinite(z['bac500__net']) & np.isfinite(X5).all(1)
        b, t, n, r2 = ols_nw(z['bac500__net'][ok5], X5[ok5])
        P(f"  {lab:22s} BAC500 on mkt+BAB500+LowVol500 alpha {b[0] * 252 * 100:6.2f}%/yr t {t[0]:5.2f} | loadings " +
          ', '.join(f'{v:+.2f} (t {tt:+.1f})' for v, tt in zip(b[1:], t[1:])) + f" | R2 {r2:.2f}")
    P()
    P('== Book facts (formations 2008-03..2025-06) ==')
    for nm_ in ('bac', 'bac500', 'bab', 'lowvol', 'bac_dollar'):
        f = z[nm_ + '__form']
        sel = days[f] >= (np.datetime64('2008-03-05') - np.datetime64('1970-01-01')).astype(int)
        cst = z[nm_ + '__cost'][FULL]
        P(f"  {nm_:11s} stocks per leg {z[nm_ + '__nleg'][sel].mean():6.1f}; ex-ante leg beta long {z[nm_ + '__bL'][sel].mean():.2f} short "
          f"{z[nm_ + '__bH'][sel].mean():.2f}; turnover per rebalance {z[nm_ + '__turn'][sel][1:].mean():.2f} (gross book "
          f"{np.mean([1]) * 0 + (1 / z[nm_ + '__bL'][sel]).mean() + (1 / z[nm_ + '__bH'][sel]).mean() if nm_ != 'bac_dollar' else 2:.2f}); "
          f"cost {np.nansum(cst) / FULL.sum() * 252 * 100:.2f}%/yr")
    g = z['bac__gross'][FULL]
    P(f"  BAC daily gross: worst {np.nanmin(g) * 100:.2f}% on {c2eval.iso(days[FULL][np.nanargmin(g)])}, best {np.nanmax(g) * 100:.2f}% on "
      f"{c2eval.iso(days[FULL][np.nanargmax(g)])}")
    out = HERE / 'results_old.txt'
else:
    U = mask('2025-06-30', '2026-10-19')
    P('== Unseen months 2025-06-30..2026-10-19 (new export, rebuilt), Newey-West 5 lags ==')
    P(HDR)
    for nm_, lab in (('bac', 'BAC'), ('bac500', 'BAC, 500 largest'), ('bab', 'BAB'), ('lowvol', 'Low volatility'),
                     ('bac_dollar', 'BAC, $1/$1'), ('bac_flag', 'BAC, flag-only universe')):
        row(f'{lab} net', stats(z[nm_ + '__net'], U))
        if nm_ == 'bac':
            row(f'{lab} gross', stats(z[nm_ + '__gross'], U))
    s = stats(z['bac__net'], U)
    P(f"BAC net cumulative return over the unseen months: {s['cum']:.2f}% ({s['n']} days)")
    s = stats(z['bac__gross'], U)
    P(f"BAC gross cumulative return over the unseen months: {s['cum']:.2f}%")
    P(f"EW market cumulative {(np.prod(1 + mkt[U]) - 1) * 100:.2f}%, SPY {(np.prod(1 + spy[U][np.isfinite(spy[U])]) - 1) * 100:.2f}%")
    X = np.column_stack((mkt_ex, z['bab__net'], z['lowvol__net']))
    ok = U & np.isfinite(z['bac__net']) & np.isfinite(X).all(1)
    b, t, n, r2 = ols_nw(z['bac__net'][ok], X[ok])
    P(f"BAC on mkt+BAB+LowVol: alpha {b[0] * 252 * 100:.2f}%/yr t {t[0]:.2f}")
    # consistency of the rebuilt series with the old one on shared days
    zo = np.load(WORK / 'series_old.npz')
    T = len(zo['days'])
    a, b_ = zo['bac__net'][:T], z['bac__net'][:T]
    ok = np.isfinite(a) & np.isfinite(b_)
    P(f"check: new-export BAC vs old-export BAC on shared days: correlation {np.corrcoef(a[ok], b_[ok])[0, 1]:.4f}, "
      f"mean diff {np.mean(a[ok] - b_[ok]) * 252 * 100:+.3f}%/yr")
    out = HERE / 'results_new.txt'
out.write_text('\n'.join(lines) + '\n')
