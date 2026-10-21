"""Tables for tests 1-3, the loadings and the unseen months.

    python3 -I report.py old     -> results_old.txt
    python3 -I report.py new     -> results_new.txt (the unseen months, 2025-06-30..2026-10-19)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import c2eval, np, WORK, HERE, ols, ols_nw, tstat, day_num

tag = _sys.argv[1]
z = np.load(WORK / f'book_{tag}.npz')
days = z['days']
lines = []


def P(s=''):
    print(s)
    lines.append(s)


def months(name, a, b):
    """Indices of complete months whose first holding day falls in [a, b]."""
    t0 = z[f'{name}__m_t0']
    first = days[t0 + 1]
    return np.nonzero((first >= day_num(a)) & (first <= day_num(b)) & z[f'{name}__m_complete'])[0]


def mstats(name, idx, kind='net'):
    y = z[f'{name}__m_{kind}'][idx]
    mx = (z[f'{name}__m_mkt'] - z[f'{name}__m_rf'])[idx]
    mean, t, n = tstat(y)
    b, tb, _, r2 = ols(y, mx)
    sd = y.std(ddof=1)
    return dict(n=n, mean=100 * mean, t=t, yr=100 * mean * 12, sharpe=mean / sd * np.sqrt(12), alpha=100 * b[0] * 12,
                ta=tb[0], beta=b[1])


HDR = f"{'':40s} {'months':>6s} {'%/month':>8s} {'t':>6s} {'%/yr':>7s} {'Sharpe':>7s} {'alpha %/yr':>10s} {'t':>6s} {'beta':>6s}"


def row(label, s):
    P(f"{label:40s} {s['n']:6d} {s['mean']:8.3f} {s['t']:6.2f} {s['yr']:7.2f} {s['sharpe']:7.2f} {s['alpha']:10.2f} "
      f"{s['ta']:6.2f} {s['beta']:6.2f}")


PERIODS = [('2006-01..2007-12', '2006-01-01', '2007-12-31'), ('2008-01..2009-12', '2008-01-01', '2009-12-31'),
           ('2010-01..2014-12', '2010-01-01', '2014-12-31'), ('2015-01..2019-12', '2015-01-01', '2019-12-31'),
           ('2020-01..2025-06', '2020-01-01', '2025-06-27'), ('All 2006-01..2025-06', '2006-01-01', '2025-06-27')]

if tag == 'old':
    T1 = ('2006-01-01', '2019-12-31')
    T1b = ('2008-03-06', '2019-12-31')
    T3 = ('2020-01-01', '2025-06-27')
    i1 = months('loners', *T1)
    P(f"first formation {c2eval.iso(days[z['loners__m_t0'][0]])}; test-1 months {len(i1)} "
      f"(first holding days {c2eval.iso(days[z['loners__m_t0'][i1[0]] + 1])}..{c2eval.iso(days[z['loners__m_t0'][i1[-1]] + 1])})")
    P()
    P('== Test 1: the loners book, 2006-01..2019-12 (net = after 5 bps per side) ==')
    P(HDR)
    s1n = mstats('loners', i1)
    row('loners net', s1n)
    row('loners gross', mstats('loners', i1, 'gross'))
    ib = months('loners', *T1b)
    row('sensitivity: 2008-03..2019 (>=500 stocks), net', mstats('loners', ib))
    pass1 = s1n['t'] >= 2.5 and s1n['alpha'] > 0
    P(f"PASS test 1 (t >= 2.5 net and alpha > 0): {pass1}")
    P()
    P('== Test 2: the 500 largest, 2006-01..2019-12 ==')
    P(HDR)
    i2 = months('loners500', *T1)
    s2n = mstats('loners500', i2)
    row('loners500 net', s2n)
    row('loners500 gross', mstats('loners500', i2, 'gross'))
    row('sensitivity: 2008-03..2019, net', mstats('loners500', months('loners500', *T1b)))
    P(f"PASS test 2 (mean net > 0): {s2n['mean'] > 0}")
    P()
    P('== Test 3: 2020-01..2025-06 ==')
    P(HDR)
    i3 = months('loners', *T3)
    s3n = mstats('loners', i3)
    row('loners net', s3n)
    row('loners gross', mstats('loners', i3, 'gross'))
    row('loners500 net', mstats('loners500', months('loners500', *T3)))
    P(f"PASS test 3, old export part (mean net > 0): {s3n['mean'] > 0}")
    P()
    P('== By period, net ==')
    for nm in ('loners', 'loners500'):
        P(f'-- {nm}')
        P(HDR)
        for lab, a, b in PERIODS:
            idx = months(nm, a, b)
            if len(idx) > 2:
                row(lab, mstats(nm, idx))
        P(HDR.replace('%/month', 'gross'))
        for lab, a, b in PERIODS:
            idx = months(nm, a, b)
            if len(idx) > 2:
                row(lab + ' gross', mstats(nm, idx, 'gross'))
    P()
    P('== The book: legs (means over test-1 months) ==')
    for nm in ('loners', 'loners500'):
        idx = months(nm, *T1)
        g = lambda k: z[f'{nm}__m_{k}'][idx]
        P(f"{nm}: universe {np.mean(g('nuniv')):.0f}, cells {np.mean(g('ncell')):.2f}, stocks per leg {np.mean(g('nleg')):.1f}, "
          f"turnover {np.mean(g('turn')):.2f} per month (cost {100 * np.mean(g('cost')) * 12:.2f}%/yr)")
        P(f"   c long {np.mean(g('cL')):.3f} short {np.mean(g('cH')):.3f}; median dollar volume long ${np.mean(g('dvL')) / 1e6:.1f}M "
          f"short ${np.mean(g('dvH')) / 1e6:.1f}M; volatility (daily) long {np.mean(g('volL')):.4f} short {np.mean(g('volH')):.4f}; "
          f"BAC ex-ante beta long {np.nanmean(g('betaL')):.2f} short {np.nanmean(g('betaH')):.2f}")
    P()
    P('== Cells (dollar-volume tercile x volatility tercile, 0 = low), gross long-short %/month (t), test-1 months ==')
    cells = z['loners__m_cells'][i1]
    for a in range(3):
        P('  dv ' + str(a) + ': ' + '  '.join(
            f"vol {b}: {100 * np.nanmean(cells[:, 3 * a + b]):+.3f} ({tstat(cells[:, 3 * a + b])[1]:+.2f})" for b in range(3)))
    P()
    P('== Loadings: monthly net loners book on the EW market - rf, low volatility and BAC (gross factors) ==')
    for lab, a, b in (('2006-01..2019-12', *T1), ('2020-01..2025-06', *T3), ('All 2006-01..2025-06', '2006-01-01', '2025-06-27')):
        idx = months('loners', a, b)
        t0 = z['loners__m_t0'][idx]
        # factor months matched by formation day
        fac = {}
        for f in ('bac', 'lowvol'):
            ft0 = z[f'{f}__m_t0']
            pos = np.searchsorted(ft0, t0)
            okf = (pos < len(ft0)) & (ft0[np.minimum(pos, len(ft0) - 1)] == t0)
            v = np.full(len(t0), np.nan)
            v[okf] = z[f'{f}__m_gross'][pos[okf]]
            fac[f] = v
        y = z['loners__m_net'][idx]
        mx = (z['loners__m_mkt'] - z['loners__m_rf'])[idx]
        P(f'-- {lab}: factor months {int(np.isfinite(fac["bac"]).sum())} of {len(idx)}; BAC mean {100 * np.nanmean(fac["bac"]):+.3f}%/month '
          f'(t {tstat(fac["bac"])[1]:+.2f}), low vol {100 * np.nanmean(fac["lowvol"]):+.3f}% (t {tstat(fac["lowvol"])[1]:+.2f}); '
          f'corr(loners, BAC) {np.corrcoef(y[np.isfinite(fac["bac"])], fac["bac"][np.isfinite(fac["bac"])])[0, 1]:+.2f}, '
          f'corr(loners, low vol) {np.corrcoef(y[np.isfinite(fac["lowvol"])], fac["lowvol"][np.isfinite(fac["lowvol"])])[0, 1]:+.2f}')
        for flab, X in (('mkt + low vol', np.column_stack([mx, fac['lowvol']])), ('mkt + BAC', np.column_stack([mx, fac['bac']])),
                        ('mkt + low vol + BAC', np.column_stack([mx, fac['lowvol'], fac['bac']]))):
            b_, t_, n_, r2 = ols(y, X)
            names = flab.split(' + ')
            P(f"   {flab:22s} n {n_:3d}: alpha {100 * 12 * b_[0]:+6.2f}%/yr (t {t_[0]:+.2f}); " +
              '; '.join(f'{nm_} {b_[i + 1]:+.2f} (t {t_[i + 1]:+.2f})' for i, nm_ in enumerate(names)) + f'; R2 {r2:.2f}')
    (HERE / 'results_old.txt').write_text('\n'.join(lines) + '\n')

else:
    a, b = day_num('2025-06-30'), day_num('2026-10-19')
    D = (days >= a) & (days <= b)
    mkt_ex = z['mkt'] - z['rf']
    P(f'== The unseen months: daily P&L of the continuing book, {c2eval.iso(days[D][0])}..{c2eval.iso(days[D][-1])} '
      f'({int(D.sum())} days) ==')
    for nm in ('loners', 'loners500', 'bac', 'lowvol'):
        for kind in ('net', 'gross'):
            y = z[f'{nm}__d_{kind}']
            ok = D & np.isfinite(y)
            r = y[ok]
            tot = r.sum() if nm.startswith('loners') else np.prod(1 + r) - 1
            bb, tt, n, _ = ols_nw(r, mkt_ex[ok])
            P(f"{nm:10s} {kind:5s}: days {n}, total {100 * tot:+.2f}% ({100 * r.mean() * 21:+.3f}% per 21 days, "
              f"t {r.mean() / (r.std(ddof=1) / np.sqrt(len(r))):+.2f}); alpha vs EW market {100 * 252 * bb[0]:+.2f}%/yr "
              f"(NW t {tt[0]:+.2f}), beta {bb[1]:+.2f}")
    P()
    P('-- each holding period of the loners book inside the window (net, % of the $1/$1 book):')
    t0s, t1s = z['loners__m_t0'], z['loners__m_t1']
    y = z['loners__d_net']
    y5 = z['loners500__d_net']
    for t0, t1 in zip(t0s, t1s):
        dd = np.arange(t0 + 1, t1 + 1)
        dd = dd[D[dd]]
        if len(dd):
            P(f'  formed {c2eval.iso(days[t0])}: {c2eval.iso(days[dd[0]])}..{c2eval.iso(days[dd[-1]])} ({len(dd)} days) '
              f'loners {100 * np.nansum(y[dd]):+.2f}%  500 largest {100 * np.nansum(y5[dd]):+.2f}%')
    # factor loadings over the unseen days (daily, NW)
    yl = z['loners__d_net']
    X = np.column_stack([mkt_ex, z['lowvol__d_gross'], z['bac__d_gross']])
    ok = D & np.isfinite(yl) & np.isfinite(X).all(1)
    bb, tt, n, r2 = ols_nw(yl[ok], X[ok])
    P(f'-- loadings over the unseen days (daily, NW 5 lags): alpha {100 * 252 * bb[0]:+.2f}%/yr (t {tt[0]:+.2f}); mkt {bb[1]:+.2f} '
      f'(t {tt[1]:+.2f}); low vol {bb[2]:+.2f} (t {tt[2]:+.2f}); BAC {bb[3]:+.2f} (t {tt[3]:+.2f}); R2 {r2:.2f}')
    (HERE / 'results_new.txt').write_text('\n'.join(lines) + '\n')
