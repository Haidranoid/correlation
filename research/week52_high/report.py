"""Tests 1-3, the by-period table, the 500 largest, the skip variant, alphas and the published-claim rows.

    python3 -I -B report.py old   -> out/results_old.txt
    python3 -I -B report.py new   -> out/results_new.txt (the unseen months; once, at the end)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import c2eval, np, WORK, OUT, ols, TEST1, TEST3, UNSEEN_FROM  # noqa: E402

FAC = ('Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD')
NW = 6
VARS = ('pth', 'pth_skip', 'mom', 'jt6', 'pth_big', 'pth_skip_big', 'mom_big')
LABEL = {'pth': 'PTH book (the plan)', 'pth_skip': 'PTH, one-week skip', 'mom': '12-1 momentum (baseline)',
         'jt6': "GH's JT: 6-month return, no skip", 'pth_big': 'PTH, 500 largest', 'pth_skip_big': 'PTH skip, 500 largest',
         'mom_big': '12-1 momentum, 500 largest'}


def load(tag, divfix=False):
    s = np.load(WORK / f"series_{tag}{'_divfix' if divfix else ''}.npz")
    S = {k: s[k] for k in s.files}
    S['end'] = np.array([c2eval.iso(S['days'][b]) for b in S['pb']])
    S['start'] = np.array([c2eval.iso(S['days'][a]) for a in S['pa']])
    S['F'] = np.column_stack([S[f'ff_{f}'] for f in FAC])
    # January periods: at least 11 of the period's days in January
    d = S['days']
    S['jan'] = np.array([sum(c2eval.iso(d[t])[5:7] == '01' for t in range(a + 1, b + 1)) >= 11
                         for a, b in zip(S['pa'], S['pb'])])
    return S


def win(S, a, b, var='pth'):
    return (S['end'] >= a) & (S['end'] <= b) & np.isfinite(S[f'{var}_net'])


def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return dict(n=n, mean=np.nan, t=np.nan, nwt=np.nan, sd=np.nan, sharpe=np.nan, tot=np.nan)
    b, t, _, _ = ols(x)
    _, tn, _, _ = ols(x, nw_lags=NW)
    sd = x.std(ddof=1)
    return dict(n=n, mean=100 * x.mean(), t=float(t[0]), nwt=float(tn[0]), sd=100 * sd * np.sqrt(12),
                sharpe=x.mean() / sd * np.sqrt(12), tot=100 * (np.prod(1 + x) - 1))


def alpha(x, F):
    ok = np.isfinite(x) & np.isfinite(F).all(1)
    if ok.sum() < 10:
        return dict(n=int(ok.sum()), a=np.nan, t=np.nan, nwt=np.nan, mkt=np.nan, umd=np.nan)
    b, t, n, r2 = ols(x[ok], F[ok])
    _, tn, _, _ = ols(x[ok], F[ok], nw_lags=NW)
    return dict(n=n, a=100 * b[0], t=float(t[0]), nwt=float(tn[0]), mkt=float(b[1]), umd=float(b[6]), r2=r2)


def fmt_s(s):
    return f"{s['mean']:+.3f} ({s['t']:+.2f}; NW {s['nwt']:+.2f})"


def fmt_a(a):
    return f"{a['a']:+.3f} ({a['t']:+.2f}; NW {a['nwt']:+.2f})" + ('' if a['n'] is None else f", n={a['n']}")


def row(S, mask, var, kind='net'):
    x = S[f'{var}_{kind}'][mask]
    return stats(x), alpha(x, S['F'][mask])


def old():
    R, D = load('old'), load('old', True)
    L = []
    p = L.append
    p('All returns are per 21-day period ("month"), in percent; t plain (NW 6 lags). Alphas: FF5 + UMD intercept.')
    m1 = win(R, *TEST1)
    p(f"\nTEST 1 window: periods ending {R['end'][m1][0]}..{R['end'][m1][-1]}, n = {int(m1.sum())}, "
      f"cohorts active min {int(R['pth_ncoh'][m1].min())}")
    for var in VARS:
        for kind in ('net', 'gross'):
            s, a = row(R, m1, var, kind)
            p(f"  {LABEL[var]:38s} {kind:5s} mean {fmt_s(s)}  %/yr {12 * s['mean']:+.2f}  vol {s['sd']:.1f}  "
              f"Sharpe {s['sharpe']:+.2f}  alpha {fmt_a(a)}  mkt {a['mkt']:+.2f} umd {a['umd']:+.2f}")
        lg = stats(R[f'{var}_long'][m1] - R[f'{var}_mkt'][m1])
        sh = stats(R[f'{var}_mkt'][m1] - R[f'{var}_short'][m1])
        p(f"  {'':38s} long - market {fmt_s(lg)}; market - short {fmt_s(sh)}")
    s1 = stats(R['pth_net'][m1])
    p(f"\nTEST 1 RESULT: mean {s1['mean']:+.3f}% per period, t {s1['t']:+.2f} (NW {s1['nwt']:+.2f}) -> "
      f"{'PASS' if s1['t'] >= 2 else 'FAIL'} (needs t >= 2)")

    p('\nTEST 2: book net on baseline net, OLS, test 1 window')
    for y, xv, lab in (('pth', 'mom', 'PTH on 12-1 (the test)'), ('pth', 'jt6', "PTH on GH's JT 6-month (reported)"),
                       ('pth_skip', 'mom', 'PTH skip on 12-1'), ('pth_big', 'mom_big', '500 largest: PTH on 12-1'),
                       ('mom', 'pth', 'reverse: 12-1 on PTH')):
        for nm, M in (('2006-2019', m1), ('2020-2025-06', win(R, *TEST3)), ('2006-2025-06', win(R, '2006-01-01', '2025-06-27'))):
            yy, xx = R[f'{y}_net'][M], R[f'{xv}_net'][M]
            b, t, n, r2 = ols(yy, xx)
            _, tn, _, _ = ols(yy, xx, nw_lags=NW)
            p(f"  {lab:36s} {nm:13s} intercept {100 * b[0]:+.3f}% (t {t[0]:+.2f}; NW {tn[0]:+.2f}) slope {b[1]:+.3f} "
              f"(t {t[1]:+.1f}) R2 {r2:.3f} n {n}  corr {np.corrcoef(yy, xx)[0, 1]:+.3f}")
    b, t, n, r2 = ols(R['pth_net'][m1], R['mom_net'][m1])
    p(f"TEST 2 RESULT: intercept {100 * b[0]:+.3f}%, t {t[0]:+.2f} -> {'PASS' if (b[0] > 0 and t[0] >= 2) else 'FAIL'}")

    m3 = win(R, *TEST3)
    s3 = stats(R['pth_net'][m3])
    p(f"\nTEST 3 window: periods ending {R['end'][m3][0]}..{R['end'][m3][-1]}, n = {int(m3.sum())}: "
      f"mean {fmt_s(s3)} -> same sign as test 1: {np.sign(s3['mean']) == np.sign(s1['mean'])}")
    s3d = stats(D['pth_net'][m3])
    p(f"  with the dividend correction: {fmt_s(s3d)}")

    p('\nBY PERIOD (net). raw / dividend-corrected where the window reaches 2022-07')
    wins = [('2006-2007', '2006-01-01', '2007-12-31'), ('2008-2009', '2008-01-01', '2009-12-31'),
            ('2010-2014', '2010-01-01', '2014-12-31'), ('2015-2019', '2015-01-01', '2019-12-31'),
            ('2006-2019', *TEST1), ('2020-2025-06', *TEST3), ('2022-07..2025-06', '2022-07-01', '2025-06-27'),
            ('2006-2025-06', '2006-01-01', '2025-06-27')]
    for var in VARS:
        p(f"  {LABEL[var]}")
        for nm, a, b in wins:
            for tag, S in (('raw', R), ('divfix', D)):
                if tag == 'divfix' and b < '2022-07-01':
                    continue
                M = win(S, a, b, var)
                s, al = row(S, M, var)
                p(f"    {nm:17s} {tag:6s} n {s['n']:3d} mean {fmt_s(s)} Sharpe {s['sharpe']:+.2f} total {s['tot']:+.1f}% "
                  f"alpha {fmt_a(al)} mkt {al['mkt']:+.2f} umd {al['umd']:+.2f}")
    p('\nJANUARY SPLIT (net, test 1 window and 2006-2025-06)')
    for var in ('pth', 'mom', 'jt6'):
        for nm, a, b in (('2006-2019', *TEST1), ('2006-2025-06', '2006-01-01', '2025-06-27')):
            M = win(R, a, b, var)
            sj, sn = stats(R[f'{var}_net'][M & R['jan']]), stats(R[f'{var}_net'][M & ~R['jan']])
            p(f"  {LABEL[var]:38s} {nm:13s} January periods n {sj['n']} {fmt_s(sj)}; others n {sn['n']} {fmt_s(sn)}")
    p('\nWORST AND BEST PERIODS, PTH net, 2006-2025-06')
    M = win(R, '2006-01-01', '2025-06-27')
    x = R['pth_net'][M]
    o = np.argsort(x)
    for i in list(o[:6]) + list(o[-4:]):
        p(f"  {R['start'][M][i]}..{R['end'][M][i]} PTH {100 * x[i]:+.2f}% mom {100 * R['mom_net'][M][i]:+.2f}% "
          f"long-mkt {100 * (R['pth_long'][M][i] - R['pth_mkt'][M][i]):+.2f} mkt-short {100 * (R['pth_mkt'][M][i] - R['pth_short'][M][i]):+.2f}")
    p('\nCHECK: correlation of the 12-1 baseline (gross) with French UMD over the same periods')
    for nm, a, b in (('2006-2019', *TEST1), ('2020-2025-06', *TEST3)):
        M = win(R, a, b) & np.isfinite(R['ff_UMD'])
        p(f"  {nm}: mom {np.corrcoef(R['mom_gross'][M], R['ff_UMD'][M])[0, 1]:+.3f}, "
          f"PTH {np.corrcoef(R['pth_gross'][M], R['ff_UMD'][M])[0, 1]:+.3f}; UMD mean {100 * np.mean(R['ff_UMD'][M]):+.3f}% "
          f"(t {stats(R['ff_UMD'][M])['t']:+.2f})")
    p('\nCOSTS: mean net - gross, % per period, test 1: ' +
      ', '.join(f"{v} {100 * np.mean(R[f'{v}_net'][m1] - R[f'{v}_gross'][m1]):+.3f}" for v in ('pth', 'mom')))
    p('LEGS: median stocks per leg: ' + ', '.join(f"{v} {int(np.median(R[f'{v}_nleg'][R[f'{v}_nleg'] > 0]))}" for v in VARS))
    (OUT / 'results_old.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def new():
    R, D = load('new'), load('new', True)
    O = load('old')
    L = []
    p = L.append
    M = (R['start'] >= c2eval.iso(R['days'][c2eval.day_index(R['days'], UNSEEN_FROM) - 1])) & np.isfinite(R['pth_net'])
    p(f"UNSEEN periods: {int(M.sum())}, {R['start'][M][0]}..{R['end'][M][-1]}; lengths " +
      ', '.join(str(int(b - a)) for a, b in zip(R['pa'][M], R['pb'][M])))
    for var in VARS:
        for tag, S in (('raw', R), ('divfix', D)):
            s, al = row(S, M, var)
            p(f"  {LABEL[var]:38s} {tag:6s} n {s['n']} mean {fmt_s(s)} total {s['tot']:+.2f}%  alpha {fmt_a(al)} "
              f"mkt {al['mkt']:+.2f} umd {al['umd']:+.2f}")
        lg = stats(R[f'{var}_long'][M] - R[f'{var}_mkt'][M])
        sh = stats(R[f'{var}_mkt'][M] - R[f'{var}_short'][M])
        p(f"  {'':38s} long - market {fmt_s(lg)}; market - short {fmt_s(sh)}")
    for tag, S in (('raw', R), ('divfix', D)):
        b, t, n, r2 = ols(S['pth_net'][M], S['mom_net'][M])
        p(f"  test-2 regression in the unseen periods ({tag}): intercept {100 * b[0]:+.3f}% (t {t[0]:+.2f}) slope {b[1]:+.3f} n {n}")
    p('  per period (raw): ' + '; '.join(f"{e} PTH {100 * x:+.2f} mom {100 * y:+.2f}" for e, x, y in
                                          zip(R['end'][M], R['pth_net'][M], R['mom_net'][M])))
    # old vs new on shared full periods
    key = {(a, b): i for i, (a, b) in enumerate(zip(O['pa'], O['pb']))}
    io, inew = [], []
    for i, (a, b) in enumerate(zip(R['pa'], R['pb'])):
        if (a, b) in key and np.isfinite(R['pth_net'][i]) and np.isfinite(O['pth_net'][key[(a, b)]]):
            io.append(key[(a, b)])
            inew.append(i)
    x, y = O['pth_net'][io], R['pth_net'][inew]
    p(f"CHECK old vs new export, PTH net on {len(io)} shared periods: corr {np.corrcoef(x, y)[0, 1]:.4f}, "
      f"mean old {100 * x.mean():+.3f} new {100 * y.mean():+.3f}")
    (OUT / 'results_new.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    {'old': old, 'new': new}[sys.argv[1]]()
