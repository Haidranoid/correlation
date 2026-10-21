"""Tables for the tests, Novy-Marx & Velikov's checks, the leverage cost and the published-claim comparison.
Run only after fidelity.py has passed.

    python3 -I -B report.py old     -> results_old.txt
    python3 -I -B report.py new     -> results_new.txt (the unseen months; once, at the end)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import np, WORK, HERE, ols, month_mask, TEST1, TEST3, UNSEEN  # noqa: E402
import pubdata  # noqa: E402

tag = sys.argv[1]
lines = []


def P(s=''):
    print(s)
    lines.append(s)


# French monthly factors, keyed by 'YYYY-MM'
fd5, f5 = pubdata.ff5_monthly()
fdu, fu = pubdata.umd_monthly()
FAC = {}
for i, d in enumerate(fd5):
    FAC[str(d)[:7]] = [f5['Mkt-RF'][i], f5['SMB'][i], f5['HML'][i], f5['RMW'][i], f5['CMA'][i], np.nan]
for i, d in enumerate(fdu):
    k = str(d)[:7]
    if k in FAC:
        FAC[k][5] = fu['UMD'][i]
ad, ac = pubdata.aqr_bab_us_monthly()
AQR = {str(d)[:7]: v for d, v in zip(ad, ac['BAB_US'])}


def factors(months):
    return np.array([FAC.get(m, [np.nan] * 6) for m in months], float)


def stats(y, months, a, b):
    """Monthly mean (%), plain t, Newey-West(3) t, annualized sd, Sharpe, FF5+UMD alpha (% a month), its t, market
    loading, months used, and the months the alpha could use."""
    M = month_mask(months, a, b) & np.isfinite(y)
    r, mo = y[M], months[M]
    n = len(r)
    if n < 3:
        return None
    mean, sd = r.mean(), r.std(ddof=1)
    _, tnw, _, _ = ols(r, None, nw_lags=3)
    X = factors(mo)
    bb, tt, na, _ = ols(r, X)
    return dict(n=n, first=mo[0], last=mo[-1], mean=mean * 100, t=mean / (sd / np.sqrt(n)), tnw=tnw[0],
                ann=mean * 1200, sd=sd * np.sqrt(12) * 100, sr=mean / sd * np.sqrt(12), alpha=bb[0] * 100,
                ta=tt[0], bmkt=bb[1], na=na, cum=(np.prod(1 + r) - 1) * 100)


HDR = (f"{'':44s} {'months':>13s} {'n':>4s} {'mean%/mo':>8s} {'t':>6s} {'t NW':>6s} {'%/yr':>6s} {'sd%':>5s} "
       f"{'Sharpe':>6s} {'FF5+UMD a%/mo':>13s} {'t':>6s} {'b_mkt':>6s} {'n_a':>4s}")


def row(label, s):
    if s is None:
        P(f'{label:44s} (no months)')
        return
    P(f"{label:44s} {s['first']}..{s['last']} {s['n']:4d} {s['mean']:8.3f} {s['t']:6.2f} {s['tnw']:6.2f} {s['ann']:6.2f} "
      f"{s['sd']:5.1f} {s['sr']:6.2f} {s['alpha']:13.3f} {s['ta']:6.2f} {s['bmkt']:6.2f} {s['na']:4d}")


def breakeven(z, s, a, b):
    """Break-even borrowing spread over the T-bill (% a year) on the borrowed (1/beta_L - 1) dollars of the levered
    low-beta leg: 12 x mean net / mean(1/beta_L - 1); also the spread at which the mean's t falls to 2, and the
    break-even if the whole long leg (1/beta_L) were borrowed."""
    y, L = z[f'{s}__net'], z[f'{s}__levL']
    M = month_mask(z['months'], a, b) & np.isfinite(y)
    y, L = y[M], L[M]
    be = 12 * y.mean() / (L - 1).mean()
    be_all = 12 * y.mean() / L.mean()

    def t_at(sp):
        x = y - sp / 12 * (L - 1)
        return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    s2 = float('nan')
    if t_at(0.0) >= 2:
        lo, hi = 0.0, be
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if t_at(mid) >= 2 else (lo, mid)
        s2 = lo
    return dict(levL=L.mean(), borrowed=(L - 1).mean(), be=be * 100, be_all=be_all * 100, s_t2=s2 * 100)


months_all = None
if tag == 'old':
    z = np.load(WORK / 'series_old.npz')
    zf = np.load(WORK / 'series_old_divfix.npz')
    months = z['months']
    aqr = np.array([AQR.get(m, np.nan) for m in months])
    periods = [('2008-03..2009', '2008-03', '2009-12'), ('2010-2014', '2010-01', '2014-12'), ('2015-2019', '2015-01', '2019-12'),
               ('2020-2025-06', '2020-01', '2025-06'), ('2022-07..2025-06', '2022-07', '2025-06'),
               ('all 2008-03..2025-06', '2008-03', '2025-06')]

    P('Monthly returns (decimal series -> % a month). t = plain mean / se; t NW = Newey-West, 3 lags; alpha = intercept')
    P('of monthly net returns on French Mkt-RF, SMB, HML, RMW, CMA, UMD (plain OLS t). French factors end 2026-08.')
    P('June 2025 is a partial month on the old export (through 2025-06-27).')
    P()
    P('== Test 1: BAB, 2008-03..2019-12, net of costs ==')
    P(HDR)
    s1 = stats(z['bab__net'], months, *TEST1)
    row('BAB net', s1)
    row('BAB gross', stats(z['bab__gross'], months, *TEST1))
    pass1 = s1['t'] >= 2 and s1['alpha'] > 0
    P(f"PASS test 1 (t >= 2 and FF5+UMD alpha > 0): {pass1}")
    sign = np.sign(s1['mean'])
    P()
    P('== Test 2: BAB on the 500 largest, 2008-03..2019-12, net ==')
    P(HDR)
    s2 = stats(z['bab500__net'], months, *TEST1)
    row('BAB500 net', s2)
    row('BAB500 gross', stats(z['bab500__gross'], months, *TEST1))
    pass2 = np.sign(s2['mean']) == sign
    P(f"PASS test 2 (mean net excess return with test 1's sign): {pass2}  (alpha sign {'same' if np.sign(s2['alpha']) == np.sign(s1['alpha']) else 'opposite'})")
    P()
    P('== Test 3: BAB, 2020-01..2025-06, net (store closes as they are; dividend-corrected below) ==')
    P(HDR)
    s3 = stats(z['bab__net'], months, *TEST3)
    row('BAB net', s3)
    row('BAB gross', stats(z['bab__gross'], months, *TEST3))
    row('BAB net, dividend-corrected', stats(zf['bab__net'], months, *TEST3))
    row('BAB500 net', stats(z['bab500__net'], months, *TEST3))
    row('BAB500 net, dividend-corrected', stats(zf['bab500__net'], months, *TEST3))
    pass3 = np.sign(s3['mean']) == sign
    P(f"PASS test 3 (mean net excess return with test 1's sign): {pass3}; dividend-corrected: "
      f"{np.sign(stats(zf['bab__net'], months, *TEST3)['mean']) == sign}")
    P()
    P('== By period, net (rows from 2022-07 also dividend-corrected) ==')
    P(HDR)
    for s, lab in (('bab', 'BAB'), ('bab500', 'BAB500')):
        for pl, a, b in periods:
            row(f'{lab} {pl}', stats(z[f'{s}__net'], months, a, b))
            if b >= '2022-07':
                row(f'{lab} {pl}, div-corrected', stats(zf[f'{s}__net'], months, a, b))
    P()
    P("== Novy-Marx & Velikov's checks: unlevered ($1 long / $1 short) low-beta third minus high-beta third, net ==")
    P(HDR)
    for s, lab in (('nmv_ew', 'EW thirds'), ('nmv_dv', 'DV-weighted thirds'), ('bab_unlev', 'rank-weighted, unlevered')):
        for pl, a, b in [('2008-03..2019', *TEST1), ('2020-2025-06', *TEST3), ('all 2008-03..2025-06', '2008-03', '2025-06')]:
            row(f'{lab} {pl}', stats(z[f'{s}__net'], months, a, b))
        row(f'{lab} 2020-2025-06, div-corrected', stats(zf[f'{s}__net'], months, *TEST3))
        row(f'{lab} 2008-03..2019 gross', stats(z[f'{s}__gross'], months, *TEST1))
    P()
    P('== Leverage cost: break-even borrowing spread over the T-bill on the levered low-beta leg (net BAB) ==')
    for pl, a, b in [('2008-03..2019', *TEST1), ('2020-2025-06', *TEST3), ('all 2008-03..2025-06', '2008-03', '2025-06')]:
        for zz, lab in ((z, ''), (zf, ', div-corrected')):
            if lab and b < '2022-07':
                continue
            e = breakeven(zz, 'bab', a, b)
            P(f"  {pl}{lab}: mean leverage of the long leg {e['levL']:.2f} (borrowed {e['borrowed']:.2f} per $1 of capital); "
              f"break-even spread {e['be']:+.2f}% a year; spread at which t falls to 2: {e['s_t2']:.2f}%; "
              f"break-even if the whole long leg is borrowed {e['be_all']:+.2f}%")
    P()
    P("== AQR's US BAB (CRSP universe), same months, and its post-publication record ==")
    P(HDR)
    row('AQR BAB_US 2008-03..2019-12', stats(aqr, months, *TEST1))
    row('AQR BAB_US 2020-01..2025-06', stats(aqr, months, *TEST3))
    am = np.array(list(AQR.keys()))
    av = np.array(list(AQR.values()))
    row('AQR BAB_US 2013-01..2025-12 (live)', stats(av, am, '2013-01', '2025-12'))
    row('AQR BAB_US 1931-01..2012-03 (FP sample)', stats(av, am, '1931-01', '2012-03'))
    row('AQR BAB_US 2013-01..2026-08 (all live)', stats(av, am, '2013-01', '2026-08'))
    P()
    P('== Book facts (formations 2008-02..2025-05) ==')
    M = month_mask(months, '2008-03', '2025-06')
    for s in ('bab', 'bab500', 'nmv_ew', 'nmv_dv', 'bab_unlev'):
        cst = z[f'{s}__cost'][M]
        P(f"  {s:10s} names per leg {np.nanmean(z[f'{s}__nL'][M]):5.0f}/{np.nanmean(z[f'{s}__nH'][M]):5.0f}; ex-ante leg betas "
          f"{np.nanmean(z[f'{s}__bL'][M]):.2f}/{np.nanmean(z[f'{s}__bH'][M]):.2f}; leverage {np.nanmean(z[f'{s}__levL'][M]):.2f}/"
          f"{np.nanmean(z[f'{s}__levH'][M]):.2f}; turnover per month {np.nanmean(z[f'{s}__turn'][M][1:]):.2f}; "
          f"cost {np.nanmean(cst) * 1200:.2f}% a year")
    g = z['bab__net'][M]
    P(f"  BAB net: worst month {np.nanmin(g) * 100:.2f}% ({months[M][np.nanargmin(g)]}), best {np.nanmax(g) * 100:.2f}% "
      f"({months[M][np.nanargmax(g)]})")
    for s in ('bab', 'bab500', 'nmv_ew', 'nmv_dv'):
        ok = M & np.isfinite(z[f'{s}__gross']) & np.isfinite(aqr)
        P(f"  {s:10s} gross correlation with AQR BAB_US over 2008-03..2025-06: {np.corrcoef(z[f'{s}__gross'][ok], aqr[ok])[0, 1]:.3f}")
    P()
    P('VERDICT CHAIN: test 1 ' + str(pass1) + ', test 2 ' + str(pass2) + ', test 3 ' + str(pass3))
    out = HERE / 'results_old.txt'
else:
    z = np.load(WORK / 'series_new.npz')
    zf = np.load(WORK / 'series_new_divfix.npz')
    zo = np.load(WORK / 'series_old.npz')
    months = z['months']
    aqr = np.array([AQR.get(m, np.nan) for m in months])
    P('== The unseen months: return months 2025-07..2026-10 (October partial, through 2026-10-19), new export ==')
    P('French factors and AQR end 2026-08, so alphas and AQR use 2025-07..2026-08.')
    P(HDR)
    for s, lab in (('bab', 'BAB'), ('bab500', 'BAB500'), ('nmv_ew', 'EW thirds'), ('nmv_dv', 'DV-weighted thirds'),
                   ('bab_unlev', 'rank-weighted, unlevered')):
        row(f'{lab} net', stats(z[f'{s}__net'], months, *UNSEEN))
        row(f'{lab} net, div-corrected', stats(zf[f'{s}__net'], months, *UNSEEN))
    row('BAB gross', stats(z['bab__gross'], months, *UNSEEN))
    row('AQR BAB_US, same months', stats(aqr, months, *UNSEEN))
    s = stats(z['bab__net'], months, *UNSEEN)
    sf = stats(zf['bab__net'], months, *UNSEEN)
    P(f"BAB net cumulative over the unseen months: {s['cum']:.2f}% (div-corrected {sf['cum']:.2f}%)")
    P(f"same sign as test 1 (positive): {s['mean'] > 0} (div-corrected {sf['mean'] > 0})")
    for zz, lab in ((z, ''), (zf, ', div-corrected')):
        e = breakeven(zz, 'bab', *UNSEEN)
        P(f"  break-even spread{lab}: {e['be']:+.2f}% a year (leverage {e['levL']:.2f})")
    # consistency with the old export on shared months
    mo = zo['months']
    shared = [m for m in mo if m in set(months) and m < '2025-06']
    a = np.array([zo['bab__gross'][list(mo).index(m)] for m in shared])
    b = np.array([z['bab__gross'][list(months).index(m)] for m in shared])
    ok = np.isfinite(a) & np.isfinite(b)
    P(f"check: new-export BAB vs old-export BAB, gross, shared months 2008-03..2025-05: correlation "
      f"{np.corrcoef(a[ok], b[ok])[0, 1]:.4f}, mean difference {np.mean(a[ok] - b[ok]) * 1200:+.3f}% a year")
    out = HERE / 'results_new.txt'
out.write_text('\n'.join(lines) + '\n')
