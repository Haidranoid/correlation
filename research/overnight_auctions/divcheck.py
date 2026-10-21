"""Overnight auctions: how the store adjusts opens and closes for dividends (a data check; no signal, no outcomes).

Stooq back-adjusts a symbol's history for dividends by multiplying every earlier open, high, low and close by the same
factor, and stores adjusted prices to 6 significant digits. Raw prices sit on the cent grid; adjusted ones mostly
don't (raw prices can still carry an odd sub-penny open, e.g. KO's 72.025 on 2025-06-02). A day is off-grid when at
least 2 of its open, high, low and close are off the cent grid. Back-adjustment is cumulative, so each symbol has
one raw boundary B: days < B adjusted (off-grid), days >= B raw (on-grid). B is fitted as the one-step change point
minimizing (on-grid days before B) + (off-grid days from B), the earliest such B on ties.

For an overnight d (close_{d-1} -> open_d): if day d-1 is in the raw region, a dividend going ex on d is NOT adjusted
(the open drops by the dividend against an unadjusted previous close). If d-1 < B, it is adjusted (the previous close
is scaled down, so the overnight is a total return).

    python3 -I divcheck.py old|new     # -> out/divcheck_<tag>.txt, raw boundaries -> WORK/rawb_<tag>.npz
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np = C.np
c2eval = C.c2eval


def on_grid(p):
    """Price on the cent grid (float32 tolerance). Prices >= $1000 have no sub-cent digits at 6 significant digits,
    so they always count as on the grid (raw): conservative for the raw-region sensitivity."""
    p = p.astype(np.float64)
    return np.abs(p * 100 - np.round(p * 100)) < 0.01


def raw_boundary(S, ids, days):
    """[len(ids)] index into days of each symbol's raw boundary B (see the module docstring); len(days) when the
    best fit has no raw region."""
    out = np.zeros(len(ids), np.int64)
    for k, i in enumerate(ids):
        a, b = S['offs'][i], S['offs'][i + 1]
        if b <= a:
            out[k] = 0
            continue
        offc = ((~on_grid(S['open'][a:b])).astype(int) + (~on_grid(S['high'][a:b])) + (~on_grid(S['low'][a:b]))
                + (~on_grid(S['close'][a:b])))
        off = offc >= 2
        n = len(off)
        # errors(j) for a boundary at row j (rows < j adjusted, rows >= j raw), j = 0..n
        on_before = np.concatenate([[0], np.cumsum(~off)])
        off_after = np.concatenate([np.cumsum(off[::-1])[::-1], [0]])
        j = int(np.argmin(on_before + off_after))
        if j >= n:
            out[k] = len(days)
        else:
            out[k] = int(np.searchsorted(days, S['date'][a + j], side='left'))
    return out


def main(tag):
    E = c2eval.load_export(C.EXPORTS[tag], arrays=('close', 'dollar_volume'))
    days = E['days']
    S = c2eval.load_store(C.STORES[tag])
    tw = c2eval.twins(E)
    rb_s = raw_boundary(S, E['ids'], days)
    rb_f = raw_boundary(S, tw['fund'], days)
    # a "payer": the store has an adjusted (off-grid) region for the symbol, i.e. Stooq adjusted at least one event;
    # first[k] = export index of the symbol's first store day
    first = np.array([int(np.searchsorted(days, S['date'][S['offs'][i]], side='left')) if S['offs'][i + 1] > S['offs'][i]
                      else len(days) for i in E['ids']])
    payer = rb_s > first
    np.savez(C.WORK / f'rawb_{tag}.npz', stock=rb_s, fund=rb_f, payer=payer, first=first)
    L = []
    W = L.append
    W(f'# Dividend adjustment in the {tag} store (cent-grid check)\n')
    # examples
    for tk in ('SPY', 'KO', 'XOM', 'T', 'AAPL', 'MSFT', 'JPM', 'O'):
        j = np.nonzero(E['tickers'] == tk)[0]
        if len(j):
            b = rb_s[j[0]]
            W(f'{tk} (stock row): raw from {c2eval.iso(days[min(b, len(days) - 1)]) if b < len(days) else "never"}')
        j = np.nonzero(tw['fund_ticker'] == tk)[0]
        if len(j):
            b = rb_f[j[0]]
            W(f'{tk} (fund): raw from {c2eval.iso(days[min(b, len(days) - 1)]) if b < len(days) else "never"}')
    # universe (no twin filter needed for this view): share of universe stock-days whose overnight is in the raw region
    uni, big = C.universe(E, tw)
    rebal = tw['rebal']
    yrs = np.array([int(c2eval.iso(x)[:4]) for x in days])
    W('\nShare of universe stock-days (on days d >= 2008) whose overnight d is in the raw region (day d-1 >= B): an')
    W('ex-date there is not adjusted. Also the share of universe stocks whose raw region starts by the year\'s end.\n')
    W('| Year | Universe stock-days | Raw-region share | 500 largest raw share | Twin overnights raw share | Payers (adjusted region exists) | Payers in raw region |')
    W('|---|---|---|---|---|---|---|')
    for y in range(2008, int(yrs[-1]) + 1):
        ds = np.nonzero(yrs == y)[0]
        n = rw = nb = rwb = nf = rwf = npy = rwp = 0
        for d in ds:
            k = C.kidx(rebal, d - 1)
            if k < 0:
                continue
            u = uni[k] & (tw['fund_row'][k] >= 0)
            raw = rb_s <= d - 1
            n += u.sum(); rw += (u & raw).sum()
            npy += (u & payer).sum(); rwp += (u & payer & raw).sum()
            nb += (u & big[k]).sum(); rwb += (u & big[k] & raw).sum()
            fr = tw['fund_row'][k][u]
            nf += len(fr); rwf += (rb_f[fr] <= d - 1).sum()
        if n:
            W(f'| {y} | {n} | {rw / n:.3f} | {rwb / max(nb, 1):.3f} | {rwf / max(nf, 1):.3f} | {npy / n:.3f} | {rwp / n:.3f} |')
    txt = '\n'.join(L) + '\n'
    (C.OUT / f'divcheck_{tag}.txt').write_text(txt)
    print(txt)


if __name__ == '__main__':
    main(sys.argv[1])
