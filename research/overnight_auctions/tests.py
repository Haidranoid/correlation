"""Overnight auctions: Tests 1-3 (the hedged decile book, net of auction costs), the raw version, the tug of war and
the 500 largest, on one export.

    python3 -I tests.py old      # 2008-2019, 2020-2025-06 and the dividend sensitivities (2008-2022, 2020-2022, 2023-2025-06)
    python3 -I tests.py new      # the unseen months 2025-06-30 .. 2026-10-19 (once, at the end)

Per signal day t (exit day d = t+1): rank S(t) over the ranked set; long the top round(10% n), short the bottom,
equal weight, $1 a side. Twin book: each position paired with -beta x its twin, bought in t's closing auction and
sold in d's opening auction. Raw book: S_raw, hedged with SPY at the stock's SPY beta. Tug of war: the same positions'
intraday leg on d (open -> close), hedged the same way, gross. Costs: 2 bps per side on stocks, both sides daily
(8 bps a day), plus 2 x 1 bp x sum over funds of |net fund weight|. Break-even = (gross - hedge cost) / 4 per side.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import build as B  # noqa: E402
np = C.np
c2eval = C.c2eval
iso = c2eval.iso

tag = sys.argv[1] if len(sys.argv) > 1 else 'old'
if tag == 'old':
    PERIODS = C.PERIODS_OLD + [('2008-2022 (dividends adjusted)', '2008-01-01', '2022-12-31'),
                               ('2020-2022', '2020-01-01', '2022-12-31'), ('2023-2025-06 (raw store)', '2023-01-01', '2025-06-27')]
else:
    PERIODS = C.PERIODS_NEW
BOOKS = [('twin', 'all'), ('twin', 'big'), ('raw', 'all'), ('raw', 'big')]
FIELDS = ('gross', 'hcost', 'net', 'tug', 'lx', 'sx', 'n', 'nl', 'ns')


def run(P, d_lo, d_hi):
    """Daily records for each book over exit days d_lo..d_hi (export indices)."""
    c0 = P['c0']
    rebal = P['rebal']
    ST, SrT = P['S'].T, P['Sraw'].T
    onT, inT, okT = P['on_s'].T, P['in_s'].T, P['ok_s'].T
    fonT, finT, fokT = P['on_f'].T, P['in_f'].T, P['ok_f'].T
    spy = P['spy_row']
    rec = {b: {k: [] for k in ('d',) + FIELDS} for b in BOOKS}
    side = C.STOCK_SIDE_BPS * 1e-4
    hside = C.TWIN_SIDE_BPS * 1e-4
    for d in range(d_lo, d_hi + 1):
        t = d - 1
        j = d - c0
        if j < 1:
            continue
        k = C.kidx(rebal, t - 1)
        if k < 0:
            continue
        has_twin = P['fund_row'][k] >= 0
        ok_d, on_d, in_d = np.asarray(okT[j]), np.asarray(onT[j], np.float64), np.asarray(inT[j], np.float64)
        fok, fon, fin = np.asarray(fokT[j]), np.asarray(fonT[j], np.float64), np.asarray(finT[j], np.float64)
        for sig, which in BOOKS:
            U = P['uni'][k] & has_twin
            if which == 'big':
                U = U & P['big'][k]
            if sig == 'twin':
                s = np.asarray(ST[t], np.float64)
                R = U & np.isfinite(s)
                frow = P['fund_row'][k].astype(np.int64)
                beta = P['beta'][k].astype(np.float64)
            else:
                s = np.asarray(SrT[t], np.float64)
                beta = P['bspy'][k].astype(np.float64)
                R = U & np.isfinite(s) & np.isfinite(beta)
                frow = np.full(len(s), spy, np.int64)
            rows = np.nonzero(R)[0]
            n = len(rows)
            if n < C.MIN_NAMES:
                continue
            o = rows[np.argsort(-s[rows], kind='stable')]
            m = int(round(C.DECILE * n))
            Lg, Sh = o[:m], o[-m:]
            fr = frow[rows]
            valid = np.zeros(len(s), bool)
            valid[rows] = ok_d[rows] & fok[fr]
            LV, SV, UV = Lg[valid[Lg]], Sh[valid[Sh]], rows[valid[rows]]
            if len(LV) == 0 or len(SV) == 0:
                continue
            r_on = np.full(len(s), np.nan)
            r_in = np.full(len(s), np.nan)
            r_on[UV] = np.expm1(on_d[UV]) - beta[UV] * np.expm1(fon[frow[UV]])
            r_in[UV] = np.expm1(in_d[UV]) - beta[UV] * np.expm1(fin[frow[UV]])
            gross = r_on[LV].mean() - r_on[SV].mean()
            tug = r_in[LV].mean() - r_in[SV].mean()
            # netted hedge legs: longs hold -beta/nL of their fund, shorts +beta/nS
            f_all = np.concatenate([frow[LV], frow[SV]])
            w_all = np.concatenate([-beta[LV] / len(LV), beta[SV] / len(SV)])
            uf, inv = np.unique(f_all, return_inverse=True)
            hcost = 2 * hside * np.abs(np.bincount(inv, weights=w_all, minlength=len(uf))).sum()
            um = r_on[UV].mean()
            R_ = rec[(sig, which)]
            R_['d'].append(d); R_['gross'].append(gross); R_['hcost'].append(hcost)
            R_['net'].append(gross - 4 * side - hcost); R_['tug'].append(tug)
            R_['lx'].append(r_on[LV].mean() - um); R_['sx'].append(r_on[SV].mean() - um)
            R_['n'].append(n); R_['nl'].append(len(LV)); R_['ns'].append(len(SV))
    return {b: {k: np.array(v) for k, v in r.items()} for b, r in rec.items()}


def summarize(R, days, a, b):
    da = days[R['d']]
    sel = (da >= C.iso_to_day(a)) & (da <= C.iso_to_day(b))
    out = {'days': int(sel.sum())}
    if not sel.any():
        return out
    g, n_, h = R['gross'][sel], R['net'][sel], R['hcost'][sel]
    gm, gt, _ = C.tstat(g)
    nm, nt, _ = C.tstat(n_)
    tm, tt, _ = C.tstat(R['tug'][sel])
    lm, lt, _ = C.tstat(R['lx'][sel])
    sm, st, _ = C.tstat(R['sx'][sel])
    sd = float(np.std(n_, ddof=1))
    out.update({'names_median': float(np.median(R['n'][sel])), 'leg_median': float(np.median(R['nl'][sel])),
                'gross_bps': gm * 1e4, 'gross_t': gt, 'gross_nw_t': C.nw_t(g), 'hcost_bps': float(h.mean()) * 1e4,
                'net_bps': nm * 1e4, 'net_t': nt, 'net_nw_t': C.nw_t(n_), 'net_pct_year': 100 * 252 * nm,
                'net_sharpe': nm / sd * np.sqrt(252) if sd > 0 else float('nan'),
                'breakeven_bps_side': (gm - float(h.mean())) * 1e4 / 4,
                'tug_bps': tm * 1e4, 'tug_t': tt, 'long_x_bps': lm * 1e4, 'long_x_t': lt, 'short_x_bps': sm * 1e4, 'short_x_t': st})
    return out


def by_year(R, days):
    yrs = np.array([int(iso(x)[:4]) for x in days[R['d']]])
    rows = []
    for y in np.unique(yrs):
        s = yrs == y
        gm, gt, _ = C.tstat(R['gross'][s])
        nm, nt, _ = C.tstat(R['net'][s])
        tm, tt, _ = C.tstat(R['tug'][s])
        rows.append({'year': int(y), 'days': int(s.sum()), 'gross_bps': gm * 1e4, 'gross_t': gt, 'net_bps': nm * 1e4,
                     'net_t': nt, 'tug_bps': tm * 1e4, 'tug_t': tt})
    return rows


def f(x, p=2):
    return 'n/a' if x is None or not np.isfinite(x) else f'{x:+.{p}f}'


def main():
    P = B.load(tag)
    days = P['days']
    C.log('loaded', tag)
    d_lo = c2eval.day_index(days, PERIODS[0][1] if tag == 'new' else C.FIRST)
    d_hi = int(np.searchsorted(days, C.iso_to_day(PERIODS[0][2] if tag == 'new' else '2025-06-27'), side='right')) - 1
    REC = run(P, d_lo, d_hi)
    C.log('books done')
    np.savez(C.WORK / f'books_{tag}.npz', **{f'{s}_{w}_{k}': v for (s, w), r in REC.items() for k, v in r.items()})
    res = {'tag': tag, 'periods': {}, 'by_year': {}}
    for b in BOOKS:
        key = f'{b[0]}|{b[1]}'
        for nm, a, z in PERIODS:
            res['periods'][f'{key}|{nm}'] = summarize(REC[b], days, a, z)
        res['by_year'][key] = by_year(REC[b], days)
    (C.OUT / f'tests_{tag}.json').write_text(json.dumps(res, indent=1))
    L = []
    W = L.append
    name = {('twin', 'all'): 'Twin, all stocks', ('twin', 'big'): 'Twin, 500 largest', ('raw', 'all'): 'Raw (SPY-hedged), all stocks',
            ('raw', 'big'): 'Raw (SPY-hedged), 500 largest'}
    W(f'# Overnight auctions: tests on the {tag} export\n')
    W('Bps a day, $1 long and $1 short; t plain (NW = Newey-West, 5 lags). Net = gross - 8 bps (stocks, 2 bps a side, both sides daily) - netted hedge cost (1 bp a side).\n')
    W('| Book | Period | Days | Names | Leg | Gross (t; NW) | Hedge cost | Net (t; NW) | Break-even bps/side | Net % a year | Sharpe net | Tug of war gross (t) | Long - universe (t) | Short - universe (t) |')
    W('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for b in BOOKS:
        for nm, _, _ in PERIODS:
            r = res['periods'][f'{b[0]}|{b[1]}|{nm}']
            if r['days'] == 0:
                continue
            W(f"| {name[b]} | {nm} | {r['days']} | {r['names_median']:.0f} | {r['leg_median']:.0f} | {f(r['gross_bps'])} ({f(r['gross_t'])}; {f(r['gross_nw_t'])}) | "
              f"{r['hcost_bps']:.2f} | {f(r['net_bps'])} ({f(r['net_t'])}; {f(r['net_nw_t'])}) | {f(r['breakeven_bps_side'])} | {f(r['net_pct_year'], 1)} | "
              f"{f(r['net_sharpe'])} | {f(r['tug_bps'])} ({f(r['tug_t'])}) | {f(r['long_x_bps'])} ({f(r['long_x_t'])}) | {f(r['short_x_bps'])} ({f(r['short_x_t'])}) |")
    W('\n## By calendar year (exit day), gross and net bps a day (t), tug of war gross\n')
    for b in BOOKS:
        W(f'\n### {name[b]}\n')
        W('| Year | Days | Gross (t) | Net (t) | Tug of war (t) |')
        W('|---|---|---|---|---|')
        for r in res['by_year'][f'{b[0]}|{b[1]}']:
            W(f"| {r['year']} | {r['days']} | {f(r['gross_bps'])} ({f(r['gross_t'])}) | {f(r['net_bps'])} ({f(r['net_t'])}) | {f(r['tug_bps'])} ({f(r['tug_t'])}) |")
    txt = '\n'.join(L) + '\n'
    (C.OUT / f'tests_{tag}.md').write_text(txt)
    print(txt)


if __name__ == '__main__':
    main()
