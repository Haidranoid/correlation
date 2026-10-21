"""Tables and the plan's tests from build.py's series (the .md, "Tests" and "Decisions"). Run only after fidelity.py
has passed.

    python3 -I -B report.py old     -> report_old.txt / .json  (2006-2019, 2020-2025-06, 2006-2025-06, ...)
    python3 -I -B report.py new     -> report_new.txt / .json  (the unseen months 2025-07..2026-10; once, at the end)

Sharpe ratios for the tests and the file-name number are from daily returns x sqrt(252) (the plan's bootstrap is in
21-day blocks); the paper's convention, monthly x sqrt(12), is reported beside them. WML and RMOM are zero-cost
spreads: no risk-free rate is subtracted.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (np, WORK, HERE, pubdata, ols, month_mask, TEST1, TEST3, FULL, POST22, UNSEEN, VOL_W,  # noqa: E402
                    TARGET)

B, BLOCK, SEED = 10000, 21, 0
PRIMARY = 'divfix'          # fixed before results: the dividend-corrected panel is primary (identical before 2022-07)
tag = sys.argv[1]
lines = []
RES = {}


def P(s=''):
    print(s, flush=True)
    lines.append(s)


# ---- French factors, monthly keyed by 'YYYY-MM'
fd5, f5 = pubdata.ff5_monthly()
fdu, fu = pubdata.umd_monthly()
FAC = {}
for i, d in enumerate(fd5):
    FAC[str(d)[:7]] = [f5['Mkt-RF'][i], f5['SMB'][i], f5['HML'][i], f5['RMW'][i], f5['CMA'][i], np.nan]
UMD_M = {str(d)[:7]: x for d, x in zip(fdu, fu['UMD'])}
for k in FAC:
    FAC[k][5] = UMD_M.get(k, np.nan)


def factors(months):
    return np.array([FAC.get(m, [np.nan] * 6) for m in months], float)


def maxdd_sum(r):
    """Largest peak-to-trough fall of cumulative (summed, uncompounded) returns: a spread's P&L per $1 a side."""
    c = np.concatenate([[0.0], np.cumsum(r)])
    return float((c - np.maximum.accumulate(c)).min())


def mstats(y, months, a, b):
    M = month_mask(months, a, b) & np.isfinite(y)
    r, mo = y[M], months[M]
    n = len(r)
    if n < 3:
        return None
    mean, sd = r.mean(), r.std(ddof=1)
    _, tnw, _, _ = ols(r, None, nw_lags=3)
    X = factors(mo)
    if np.isfinite(X).all(1).sum() >= 12:
        bb, tt, na, _ = ols(r, X)
        alpha, ta = bb[0] * 100, tt[0]
    else:
        alpha, ta, na = np.nan, np.nan, 0
    w = int(np.argmin(r))
    return dict(n=n, first=str(mo[0]), last=str(mo[-1]), mean=mean * 100, t=mean / (sd / np.sqrt(n)), tnw=float(tnw[0]),
                sd=sd * np.sqrt(12) * 100, sr_m=mean / sd * np.sqrt(12), alpha=alpha, ta=ta, na=int(na),
                worst=r[w] * 100, worst_month=str(mo[w]), maxdd=maxdd_sum(r) * 100)


def day_mask(z, a, b):
    """Days whose (calendar) month is within [a, b]."""
    dm = np.array([str(np.datetime64(int(d), 'D'))[:7] for d in z['days']])
    return (dm >= a) & (dm <= b)


def sr_d(x):
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252))


def boot_idx(rng, n, b):
    nb = -(-n // BLOCK)
    st = rng.integers(0, n, (b, nb))
    return ((st[:, :, None] + np.arange(BLOCK)[None, None, :]) % n).reshape(b, -1)[:, :n]


def boot_dsharpe(rm, wm, seed=SEED):
    """Paired circular 21-day block bootstrap of Sharpe(RMOM) - Sharpe(WML), daily x sqrt(252)."""
    rng = np.random.default_rng(seed)
    n = len(rm)
    out = []
    for a in range(0, B, 500):
        idx = boot_idx(rng, n, min(500, B - a))
        X, Y = rm[idx], wm[idx]
        out.append(X.mean(1) / X.std(1, ddof=1) * np.sqrt(252) - Y.mean(1) / Y.std(1, ddof=1) * np.sqrt(252))
    return np.concatenate(out)


def p_greater(stat_b, stat):
    """One-sided p for H1: stat > 0 (centred bootstrap). Copied from research/correlation_portfolios/report.py."""
    return float(np.mean((stat_b - stat) >= stat))


def p_two(stat_b, stat):
    return float(np.mean(np.abs(stat_b - stat) >= abs(stat)))


HDR = ('| Series | Months | Mean %/month | t | t NW(3) | Volatility %/yr | Sharpe (monthly) | Sharpe (daily) | '
       'FF5+UMD alpha %/month (t) | Worst month | Max drawdown (pts) |')
SEP = '|---|---|---|---|---|---|---|---|---|---|---|'


def period(z, v, a, b, label, key):
    """All the numbers for one variant and period; prints a table and returns a dict."""
    months = z['months']
    D = day_mask(z, a, b)
    rec = {}
    P(f'#### {label} ({v}: {"liquid set" if v == "all" else "500 largest"})')
    P()
    P(HDR)
    P(SEP)
    for s in ('wml', 'rmom'):
        for kind in ('gross', 'net'):
            y = z[f'{v}__{s}__{kind}_m']
            st = mstats(y, months, a, b)
            # daily Sharpe over the same months' days, only where both series exist (RMOM's months)
            okd = D & np.isfinite(z[f'{v}__rmom__{kind}_d']) & np.isfinite(z[f'{v}__wml__{kind}_d'])
            st['sr_d'] = sr_d(z[f'{v}__{s}__{kind}_d'][okd])
            st['days'] = int(okd.sum())
            rec[f'{s}_{kind}'] = st
            P(f"| {s.upper()} {kind} | {st['first']}..{st['last']} ({st['n']}) | {st['mean']:+.3f} | {st['t']:+.2f} | "
              f"{st['tnw']:+.2f} | {st['sd']:.1f} | {st['sr_m']:+.2f} | {st['sr_d']:+.2f} | "
              f"{st['alpha']:+.3f} ({st['ta']:+.2f}; {st['na']} mo) | {st['worst']:+.1f}% ({st['worst_month']}) | "
              f"{st['maxdd']:+.1f} |")
    P()
    # bootstrap of the Sharpe difference, daily
    for kind in ('net', 'gross'):
        rm, wm = z[f'{v}__rmom__{kind}_d'], z[f'{v}__wml__{kind}_d']
        ok = D & np.isfinite(rm) & np.isfinite(wm)
        d = sr_d(rm[ok]) - sr_d(wm[ok])
        bs = boot_dsharpe(rm[ok], wm[ok])
        rec[f'dsharpe_{kind}'] = dict(d=d, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                                      p_greater=p_greater(bs, d), p_two=p_two(bs, d), days=int(ok.sum()),
                                      d_monthly=rec[f'rmom_{kind}']['sr_m'] - rec[f'wml_{kind}']['sr_m'])
        r = rec[f'dsharpe_{kind}']
        P(f"Sharpe RMOM − WML, {kind}, daily: {d:+.3f} (95% {r['lo']:+.3f} to {r['hi']:+.3f}), one-sided p "
          f"{r['p_greater']:.3f}, two-sided p {r['p_two']:.3f}, {r['days']} days; monthly convention "
          f"{r['d_monthly']:+.3f}.")
    # scale and costs
    sc = z[f'{v}__scale']
    M = month_mask(months, a, b) & np.isfinite(sc)
    s = sc[M]
    rec['scale'] = dict(mean=float(s.mean()), min=float(s.min()), max=float(s.max()),
                        min_month=str(months[M][np.argmin(s)]), max_month=str(months[M][np.argmax(s)]),
                        share_above_1=float((s > 1).mean()))
    sig = z[f'{v}__sigma'][M]
    for s_ in ('wml', 'rmom'):
        rec[f'{s_}_cost_per_year'] = float(np.nanmean(z[f'{v}__{s_}__cost'][M]) * 12 * 100)
        rec[f'{s_}_turn'] = float(np.nanmean(z[f'{v}__{s_}__turn'][M]))
    rec['sigma_mean'] = float(sig.mean())
    P(f"Scale (12% ÷ σ̂, set at each month-end): mean {rec['scale']['mean']:.2f}, min {rec['scale']['min']:.2f} "
      f"({rec['scale']['min_month']}), max {rec['scale']['max']:.2f} ({rec['scale']['max_month']}), above 1 in "
      f"{rec['scale']['share_above_1']:.0%} of months; mean σ̂ {rec['sigma_mean']:.1%}. Turnover a month (sum of "
      f"|Δposition|): WML {rec['wml_turn']:.2f}, RMOM {rec['rmom_turn']:.2f}; costs a year: WML "
      f"{rec['wml_cost_per_year']:.2f}%, RMOM {rec['rmom_cost_per_year']:.2f}%.")
    # RMOM against WML (Barroso & Santa-Clara's spanning regression), net, plus FF5 + UMD
    y = z[f'{v}__rmom__net_m']
    x = z[f'{v}__wml__net_m']
    Mm = month_mask(months, a, b) & np.isfinite(y) & np.isfinite(x)
    bb, tt, n_, r2 = ols(y[Mm], x[Mm])
    X6 = factors(months[Mm])
    okf = np.isfinite(X6).all(1)
    if okf.sum() >= 12:
        b2, t2, n2, _ = ols(y[Mm][okf], np.column_stack([x[Mm][okf], X6[okf]]))
    else:
        b2, t2, n2 = [np.nan] * 2, [np.nan] * 2, 0
    rec['span'] = dict(alpha=bb[0] * 100, t=tt[0], beta=bb[1], r2=r2, n=n_, alpha_ff=b2[0] * 100, t_ff=t2[0], n_ff=n2)
    P(f"RMOM net on WML net: alpha {bb[0] * 100:+.3f}% a month (t {tt[0]:+.2f}), beta {bb[1]:.2f}, R² {r2:.2f}; "
      f"adding FF5 + UMD: alpha {b2[0] * 100:+.3f}% (t {t2[0]:+.2f}, {n2} months).")
    P()
    RES[key] = rec
    return rec


def crash(z, v):
    """March-May 2009: compounded monthly returns, legs and scales."""
    months = z['months']
    M = month_mask(months, '2009-03', '2009-05')
    out = {}
    for s in ('wml', 'rmom'):
        for kind in ('gross', 'net'):
            y = z[f'{v}__{s}__{kind}_m'][M]
            out[f'{s}_{kind}'] = float(np.prod(1 + y) - 1)
    out['RW'] = float(np.prod(1 + z[f'{v}__RW_m'][M]) - 1)
    out['RL'] = float(np.prod(1 + z[f'{v}__RL_m'][M]) - 1)
    out['scale'] = [float(x) for x in z[f'{v}__scale'][M]]
    out['monthly_wml'] = [float(x) for x in z[f'{v}__wml__gross_m'][M]]
    return out


def french_scaled(a_date, b_date):
    """Barroso & Santa-Clara's rule on French's own UMD (daily for sigma, monthly for returns): Sharpe of UMD and of
    scaled UMD, monthly convention, over return months a..b (YYYY-MM)."""
    ud, uc = pubdata.umd_daily()
    u = uc['UMD']
    ym_d = np.array([str(d)[:7] for d in ud])
    mo, mu = list(UMD_M.keys()), np.array(list(UMD_M.values()))
    last = {}
    for i, m in enumerate(ym_d):
        last[m] = i                        # index of the month's last trading day in the daily file
    um, sm, mm = [], [], []
    for m, r in zip(mo, mu):
        # previous month
        y, mth = int(m[:4]), int(m[5:])
        pm = f'{y - (mth == 1)}-{(mth - 2) % 12 + 1:02d}'
        if pm not in last:
            continue
        e = last[pm]
        if e - VOL_W + 1 < 0:
            continue
        x = u[e - VOL_W + 1:e + 1]
        w = TARGET / np.sqrt(252 * np.mean(x * x))
        um.append(r)
        sm.append(w * r)
        mm.append(m)
    um, sm, mm = np.array(um), np.array(sm), np.array(mm)
    M = (mm >= a_date) & (mm <= b_date)
    f = lambda x: float(x.mean() / x.std(ddof=1) * np.sqrt(12))  # noqa: E731
    return dict(n=int(M.sum()), sr_umd=f(um[M]), sr_scaled=f(sm[M]), worst_umd=float(um[M].min()),
                worst_umd_month=str(mm[M][np.argmin(um[M])]), worst_scaled=float(sm[M].min()),
                worst_scaled_month=str(mm[M][np.argmin(sm[M])]), mean_umd=float(um[M].mean() * 12),
                mean_scaled=float(sm[M].mean() * 12))


def yearly(z, v):
    months = z['months']
    P(f'Yearly ({v}), summed monthly returns, %: year | WML gross | WML net | RMOM gross | RMOM net | mean scale')
    for yr in range(2005, 2027):
        M = np.array([m[:4] == str(yr) for m in months]) & np.isfinite(z[f'{v}__rmom__net_m'])
        if not M.any():
            continue
        P(f"  {yr} ({int(M.sum())} mo) | {np.nansum(z[f'{v}__wml__gross_m'][M]) * 100:+6.1f} | "
          f"{np.nansum(z[f'{v}__wml__net_m'][M]) * 100:+6.1f} | {np.nansum(z[f'{v}__rmom__gross_m'][M]) * 100:+6.1f} | "
          f"{np.nansum(z[f'{v}__rmom__net_m'][M]) * 100:+6.1f} | {np.nanmean(z[f'{v}__scale'][M]):.2f}")
    P()


def main():
    Z = {px: dict(np.load(WORK / f'series_{tag}_{px}.npz')) for px in ('plain', 'divfix')}
    if tag == 'old':
        P('## Old export (through 2025-06-27)')
        P()
        P('### Fidelity (from fidelity.py, run first)')
        P((HERE / 'fidelity.txt').read_text())
        for px in (PRIMARY, 'plain'):
            z = Z[px]
            P(f'### Prices: {px}')
            P()
            for v in ('all', 'big'):
                periods = [(TEST1, '2006-2019'), (TEST3, '2020-2025-06'), (FULL, '2006-2025-06'), (POST22, '2022-07..2025-06')]
                if px == 'plain':
                    periods = [(TEST3, '2020-2025-06'), (FULL, '2006-2025-06'), (POST22, '2022-07..2025-06')]
                for (a, b), label in periods:
                    period(z, v, a, b, f'{label}, {px}', f'{px}|{v}|{label}')
        z = Z[PRIMARY]
        for v in ('all', 'big'):
            c = crash(z, v)
            RES[f'crash|{v}'] = c
            P(f"Crash, March-May 2009 ({v}), compounded: WML gross {c['wml_gross']:+.1%} net {c['wml_net']:+.1%}; RMOM "
              f"gross {c['rmom_gross']:+.1%} net {c['rmom_net']:+.1%}; winners {c['RW']:+.1%}, losers {c['RL']:+.1%}; "
              f"scales {', '.join(f'{x:.2f}' for x in c['scale'])}; WML by month "
              f"{', '.join(f'{x:+.1%}' for x in c['monthly_wml'])}")
        P()
        for v in ('all', 'big'):
            yearly(z, v)
        # French
        P('### French UMD (post-publication record) and the same rule on it')
        P()
        P('| Period | Months | UMD mean %/yr | UMD Sharpe | Scaled UMD mean %/yr | Scaled UMD Sharpe | UMD worst month | Scaled worst month |')
        P('|---|---|---|---|---|---|---|---|')
        for a, b, lab in (('1927-01', '2011-12', '1927-2011 (B&SC sample, check)'), ('2006-01', '2019-12', '2006-2019'),
                          ('2020-01', '2025-06', '2020-2025-06'), ('2006-01', '2025-06', '2006-2025-06'),
                          ('2006-01', '2025-12', '2006-2025')):
            f = french_scaled(a, b)
            RES[f'french|{lab}'] = f
            P(f"| {lab} | {f['n']} | {f['mean_umd'] * 100:+.1f} | {f['sr_umd']:+.2f} | {f['mean_scaled'] * 100:+.1f} | "
              f"{f['sr_scaled']:+.2f} | {f['worst_umd'] * 100:+.1f}% ({f['worst_umd_month']}) | "
              f"{f['worst_scaled'] * 100:+.1f}% ({f['worst_scaled_month']}) |")
        P()
        # tests
        t1 = RES[f'{PRIMARY}|all|2006-2019']
        t1_mean_t = t1['rmom_net']['t']
        d1 = t1['dsharpe_net']
        pass1 = (t1_mean_t >= 2) and (d1['d'] > 0) and (d1['p_greater'] < 0.10)
        c = RES['crash|all']
        loss_r, loss_w = -c['rmom_net'], -c['wml_net']
        b1 = RES[f'{PRIMARY}|big|2006-2019']['dsharpe_net']
        pass2 = (loss_w > 0) and (loss_r < 0.5 * loss_w) and (np.sign(b1['d']) == np.sign(d1['d']))
        t3 = RES[f'{PRIMARY}|all|2020-2025-06']['dsharpe_net']
        t3p = RES['plain|all|2020-2025-06']['dsharpe_net']
        pass3a = np.sign(t3['d']) == np.sign(d1['d'])
        full = RES[f'{PRIMARY}|all|2006-2025-06']
        number = (full['rmom_gross']['sr_d'] - full['wml_gross']['sr_d']) * 15
        fullp = RES['plain|all|2006-2025-06']
        number_plain = (fullp['rmom_gross']['sr_d'] - fullp['wml_gross']['sr_d']) * 15
        P('### The tests')
        P()
        P(f"Test 1 (2006-2019, net): RMOM mean {t1['rmom_net']['mean']:+.3f}% a month, t {t1_mean_t:+.2f} (needs >= 2); "
          f"Sharpe RMOM − WML {d1['d']:+.3f}, one-sided p {d1['p_greater']:.3f} (needs > 0 and p < 0.10) -> "
          f"{'PASS' if pass1 else 'FAIL'}")
        P(f"Test 2: Mar-May 2009 loss RMOM {loss_r:.1%} vs WML {loss_w:.1%} (needs < half: {0.5 * loss_w:.1%}); 500 largest "
          f"Sharpe difference {b1['d']:+.3f} vs {d1['d']:+.3f} (same sign needed) -> {'PASS' if pass2 else 'FAIL'}")
        P(f"Test 3a (2020-2025-06, net, {PRIMARY}): Sharpe difference {t3['d']:+.3f} (plain {t3p['d']:+.3f}); same sign as "
          f"test 1 needed -> {'PASS' if pass3a else 'FAIL'} (the unseen months: report_new)")
        P(f"The number: (Sharpe RMOM − Sharpe WML, gross, daily, 2006-2025-06, {PRIMARY}) x 15 = "
          f"({full['rmom_gross']['sr_d']:.3f} − {full['wml_gross']['sr_d']:.3f}) x 15 = {number:+.1f} (plain prices "
          f"{number_plain:+.1f}; monthly convention {(full['rmom_gross']['sr_m'] - full['wml_gross']['sr_m']) * 15:+.1f}); "
          f"gross difference one-sided p {full['dsharpe_gross']['p_greater']:.3f}")
        RES['tests'] = dict(pass1=bool(pass1), pass2=bool(pass2), pass3a=bool(pass3a), number=number,
                            number_plain=number_plain, sign1=float(np.sign(d1['d'])))
    else:
        P('## New export: the unseen months (return months 2025-07..2026-10, through 2026-10-19)')
        P()
        for px in (PRIMARY, 'plain'):
            for v in ('all', 'big'):
                period(Z[px], v, UNSEEN[0], UNSEEN[1], f'unseen, {px}', f'{px}|{v}|unseen')
            yearly(Z[px], 'all')
        old = json.loads((HERE / 'report_old.json').read_text())
        s1 = old['tests']['sign1']
        u = RES[f'{PRIMARY}|all|unseen']['dsharpe_net']
        up = RES['plain|all|unseen']['dsharpe_net']
        pass3b = np.sign(u['d']) == s1
        P(f"Test 3b (unseen months, net, {PRIMARY}): Sharpe difference {u['d']:+.3f} (plain {up['d']:+.3f}); same sign as "
          f"test 1 ({s1:+.0f}) needed -> {'PASS' if pass3b else 'FAIL'}")
        RES['tests'] = dict(pass3b=bool(pass3b))
    (HERE / f'report_{tag}.txt').write_text('\n'.join(lines) + '\n')
    (HERE / f'report_{tag}.json').write_text(json.dumps(RES, indent=1, default=float))


if __name__ == '__main__':
    main()
