"""Overnight auctions, dividend sensitivity (b) of decision D1: in the unseen months the store is raw, so an ex-date
overnight drops by the dividend. The twin gap fade check's Yahoo daily file (y1d.csv, 2026-06-01..2026-10-19, its
symbols only; read-only) gives dividend and split ex-dates. Drop every ex-date overnight of a stock or a twin fund
(in the signal and as an exit), recompute S, the books and the money account, and compare with the headline on exit
days 2026-07-01..2026-10-19 (so every signal window lies inside the covered dates). Same seed for the random accounts.

    python3 -I divunseen.py      # -> out/divunseen.md, .json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import build as B  # noqa: E402
import tests as TT  # noqa: E402
import money as M  # noqa: E402
np = C.np
c2eval = C.c2eval
iso = c2eval.iso
BIG = c2eval.WORK / 'twin_gap_fade_check'


def main():
    import csv
    P = B.load('new')
    days = P['days']
    T = len(days)
    c0 = P['c0']
    E = c2eval.load_export(C.EXPORTS['new'], arrays=('close',))
    tw = c2eval.twins(E)
    tw.pop('F')
    pos = np.load(BIG / 'positions.npz')
    smap = dict(zip(pos['stk_i'].tolist(), pos['stk_sym'].tolist()))
    fmap = dict(zip(pos['fnd_f'].tolist(), pos['fnd_sym'].tolist()))
    covered, evset, n_events = set(), {}, 0
    with open(BIG / 'y1d.csv') as fh:
        for r in csv.DictReader(fh):
            covered.add(r['sym'])
            dv = float(r['dividends']) if r['dividends'] not in ('', 'nan') else 0.0
            sp = float(r['stock_splits']) if r['stock_splits'] not in ('', 'nan') else 0.0
            if dv > 0 or sp > 0:
                evset.setdefault(r['sym'], []).append(r['date'])
                n_events += 1
    day_of = {iso(x): j for j, x in enumerate(days)}
    P2 = dict(P)
    for key_on, key_ok, key_st, n_rows, names in (('on_s', 'ok_s', 'st_s', len(E['tickers']),
                                                   [smap.get(i, str(E['tickers'][i]).replace('.', '-')) for i in range(len(E['tickers']))]),
                                                  ('on_f', 'ok_f', 'st_f', len(tw['fund']),
                                                   [fmap.get(f, str(tw['fund_ticker'][f])) for f in range(len(tw['fund']))])):
        on, ok, st = P[key_on].copy(), P[key_ok].copy(), P[key_st].copy()
        n_ev = 0
        for r, sym in enumerate(names):
            for dt in evset.get(sym, []):
                d = day_of.get(dt)
                if d is None or d - c0 < 1:
                    continue
                on[r, d - c0] = np.nan; ok[r, d - c0] = False; st[r, d - c0] = False
                n_ev += 1
        P2[key_on], P2[key_ok], P2[key_st] = on, ok, st
        print(key_on, 'ex-date overnights dropped', n_ev)
    ks = B.blocks_needed(P['rebal'], c0, T)
    P2['S'], P2['Sraw'] = B.signal(P2['on_s'], P2['on_f'], tw, P['uni'], c0, T, ks)
    d_lo = c2eval.day_index(days, '2026-07-01')
    d_hi = int(np.searchsorted(days, C.iso_to_day('2026-10-19'), side='right')) - 1
    # coverage: ranked stock-days (exit days in the window) whose symbol is in the Yahoo file
    tot = cov = 0
    for d in range(d_lo, d_hi + 1):
        k = C.kidx(P['rebal'], d - 2)
        U = P['uni'][k] & (P['fund_row'][k] >= 0) & np.isfinite(P['S'][:, d - 1])
        rows = np.nonzero(U)[0]
        tot += len(rows)
        cov += sum(1 for i in rows if smap.get(int(i), str(E['tickers'][i]).replace('.', '-')) in covered)
    res = {'window': f'{iso(days[d_lo])}..{iso(days[d_hi])}', 'ranked_stock_days': tot, 'covered_share': cov / max(tot, 1),
           'ex_dates_in_file': n_events}
    for lab, PP in (('headline', P), ('ex_dates_dropped', P2)):
        R = TT.run(PP, d_lo, d_hi)
        r = R[('twin', 'all')]
        g, gt, _ = C.tstat(r['gross']); n_, nt, _ = C.tstat(r['net'])
        D = M.daily(PP, d_lo, d_hi)
        sel = np.ones(len(D['d']), bool)
        row = M.yearly(D, days, 'acc', 'rnd', 0.0, [('window', sel)])[0]
        ex = D['acc'] - D['rnd'].astype(np.float64).mean(1)
        m, t_, _ = C.tstat(ex)
        res[lab] = {'book_gross_bps': g * 1e4, 'book_gross_t': gt, 'book_net_bps': n_ * 1e4, 'book_net_t': nt,
                    'account_pct': row['account_pct'], 'random_median_pct': row['random_median_pct'], 'diff': row['diff'],
                    'p': row['p'], 'daily_excess_bps': m * 1e4, 'daily_excess_t': t_, 'days': int(len(D['d']))}
    (C.OUT / 'divunseen.json').write_text(json.dumps(res, indent=1))
    L = ['# Overnight auctions: Yahoo ex-dates dropped in the unseen months (2026-07-01..2026-10-19 exits)\n',
         f"Ranked stock-days {tot}, symbol in the Yahoo file {res['covered_share']:.1%}; dividend or split ex-dates in the file {n_events}.\n",
         '| Version | Twin book gross (t) | Twin book net (t) | Account % | Median random % | Difference | p | Daily excess over mean random (t) |',
         '|---|---|---|---|---|---|---|---|']
    for lab in ('headline', 'ex_dates_dropped'):
        r = res[lab]
        L.append(f"| {lab} | {r['book_gross_bps']:+.2f} ({r['book_gross_t']:+.2f}) | {r['book_net_bps']:+.2f} ({r['book_net_t']:+.2f}) | "
                 f"{r['account_pct']:+.1f} | {r['random_median_pct']:+.1f} | {r['diff']:+.1f} | {r['p']:.3f} | {r['daily_excess_bps']:+.2f} ({r['daily_excess_t']:+.2f}) |")
    txt = '\n'.join(L) + '\n'
    (C.OUT / 'divunseen.md').write_text(txt)
    print(txt)


if __name__ == '__main__':
    main()
