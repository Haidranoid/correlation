"""Positions of the twin gap fade's books on the recent days of the new export, for the intraday-bar check
(experiments/-16.2-twin-gap-fade.md, "Independent check with intraday bars").

Uses the original code read-only: books.py's load_rows/day_slices and its constants, and books.books() as the
reference. For every day from FROM on, it re-runs the deciles exactly as books.books() does (stable argsort of z,
round(10% x n) per leg, intraday bad-print rows dropped) and records which stocks were long and short, so the same
positions can be re-priced at other entry times. Checks: the gross and headline returns per day equal books.books()
and research/twin_gap_fade/out/daily_new.tsv.

Also writes every ranked row on those days (for the variant that recomputes the signal from Yahoo's first-bar open),
with the trailing sd of u recovered from the store (the same computation as build.py), and the Yahoo symbols to fetch.

    python3 -I positions.py      # -> WORK/twin_gap_fade_check/positions.npz, out/positions.txt, out/tickers_*.txt
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, '/home/bhoult/programming/correlation/research/twin_gap_fade')
import books as B  # noqa: E402  (read-only import of the original code)
from books import c2eval, np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
BIG = c2eval.WORK / 'twin_gap_fade_check'
FROM = '2026-07-01'
LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def yahoo_symbol(t):
    """Store ticker -> Yahoo symbol: class shares use '-', any '.US'-style suffix stripped."""
    t = str(t).upper()
    if t.endswith('.US'):
        t = t[:-3]
    return t.replace('.', '-').replace('/', '-')


def deciles(R, sub):
    """Positions of the twin book, day by day, as books.books() forms them. Returns arrays over positions:
    t, i, f, beta, side (+1 long, -1 short), h (hedged open-to-close return), z, row (index into R)."""
    keep = np.ones(len(R['t']), bool)
    if sub == '500':
        keep &= R['in500']
    keep &= ~R['idbad']
    idx = np.nonzero(keep)[0]
    t = R['t'][idx]
    u, st, en = B.day_slices(t)
    out = {k: [] for k in ('t', 'row', 'side')}
    for d in range(len(u)):
        a, b = st[d], en[d]
        n = b - a
        if n < B.MIN_N:
            continue
        nl = int(round(B.FRAC * n))
        rows = idx[a:b]
        o = np.argsort(R['z'][rows], kind='stable')
        lo, hi = o[:nl], o[n - nl:]
        for side, sel in ((1, lo), (-1, hi)):
            out['t'].append(np.full(nl, u[d], np.int32))
            out['row'].append(rows[sel].astype(np.int64))
            out['side'].append(np.full(nl, side, np.int8))
    P = {k: np.concatenate(v) for k, v in out.items()}
    r = P['row']
    for k in ('i', 'f', 'beta', 'z', 'g', 'rs', 'rf', 'in500', 'in100'):
        P[k] = R[k][r]
    P['h'] = P['rs'] - P['beta'].astype(np.float64) * P['rf']
    return P


def book_from_positions(P):
    """Daily gross, netted twin cost, from positions (mean long hedged return - mean short)."""
    u = np.unique(P['t'])
    g = np.zeros(len(u))
    tc = np.zeros(len(u))
    for j, t in enumerate(u):
        m = P['t'] == t
        L = m & (P['side'] == 1)
        S = m & (P['side'] == -1)
        g[j] = P['h'][L].mean() - P['h'][S].mean()
        nl = L.sum()
        fi = np.concatenate([P['f'][L], P['f'][S]])
        w = np.concatenate([-P['beta'][L].astype(np.float64) / nl, P['beta'][S].astype(np.float64) / S.sum()])
        tc[j] = B.fund_cost(fi, w)
    return u, g, tc


def trailing_sd(on_s, on_f, si, fj, beta, t_days):
    """sd of u = overnight_stock - beta x overnight_twin over days t-252..t-1, needing >= 189 finite values,
    exactly as build.py (bad prints already masked in the inputs)."""
    sd = np.full(len(si), np.nan)
    for k in range(len(si)):
        t = t_days[k]
        uh = on_s[si[k], t - B_SD_W:t].astype(np.float64) - beta[k] * on_f[fj[k], t - B_SD_W:t].astype(np.float64)
        uh = uh[np.isfinite(uh)]
        if len(uh) >= B_SD_MIN:
            s = uh.std(ddof=1)
            if s > 0:
                sd[k] = s
    return sd


B_SD_W, B_SD_MIN, BAD = 252, 189, 0.5


def main():
    OUT.mkdir(exist_ok=True)
    BIG.mkdir(parents=True, exist_ok=True)
    R = B.load_rows('new')
    days = R['days']
    t0 = c2eval.day_index(days, FROM)
    m = R['t'] >= t0
    R = {k: (v[m] if k != 'days' else v) for k, v in R.items()}
    say(f'rows from {FROM}: {m.sum()}, days {len(np.unique(R["t"]))}, {c2eval.iso(days[R["t"].min()])}..{c2eval.iso(days[R["t"].max()])}')

    # the reference: daily_new.tsv
    ref = {}
    for line in open('/home/bhoult/programming/correlation/research/twin_gap_fade/out/daily_new.tsv').read().splitlines()[1:]:
        f = line.split('\t')
        ref[f[0]] = dict(gross=float(f[2]), head=float(f[4]), top500_gross=float(f[7]), top500_head=float(f[9]))

    res = {}
    for sub in ('all', '500'):
        P = deciles(R, sub)
        Bk = B.books(R, sub)  # the original code on the same rows
        u, g, tc = book_from_positions(P)
        assert np.array_equal(u, Bk['t'])
        dg = np.abs(g - Bk['twin_gross']).max()
        dtc = np.abs(tc - Bk['twin_tc']).max()
        key = 'gross' if sub == 'all' else 'top500_gross'
        keyh = 'head' if sub == 'all' else 'top500_head'
        rg = np.array([ref[c2eval.iso(days[t])][key] for t in u])
        rh = np.array([ref[c2eval.iso(days[t])][keyh] for t in u])
        head = g - 2 * B.P_HEAD - tc
        say(f'{sub}: {len(u)} days, {len(P["t"])} positions; max |gross - books.books()| {dg:.2e}, max |twin cost - books.books()| {dtc:.2e}; '
            f'max |gross - daily_new.tsv| {np.abs(g - rg).max():.2e}, max |headline - daily_new.tsv| {np.abs(head - rh).max():.2e} '
            f'(tsv has 6 decimals)')
        assert np.abs(g - rg).max() < 1e-6 and np.abs(head - rh).max() < 1e-6
        res[sub] = P

    # every ranked row on these days (for the variant), with the trailing sd of u recomputed from the store
    keep = ~R['idbad']
    rk = {k: R[k][keep] for k in ('t', 'i', 'f', 'beta', 'z', 'g', 'rs', 'rf', 'in500', 'in100')}
    E = c2eval.load_export(c2eval.EXPORT_NEW, arrays=('close', 'dollar_volume'))
    assert np.array_equal(E['days'], days)
    tw = c2eval.twins(E)
    tw.pop('F')
    S = c2eval.load_store(c2eval.STORE_NEW)
    lo = t0 - B_SD_W - 2
    dsub = days[lo:]
    sids = np.unique(rk['i'])
    fids = np.unique(np.concatenate([rk['f'], res['all']['f'], res['500']['f']]))
    so = c2eval.store_panel(S, E['ids'][sids], dsub, 'open')
    sc = c2eval.store_panel(S, E['ids'][sids], dsub, 'close')
    fo = c2eval.store_panel(S, tw['fund'][fids], dsub, 'open')
    fc = c2eval.store_panel(S, tw['fund'][fids], dsub, 'close')

    def masked_on(o, c):
        with np.errstate(all='ignore'):
            on = np.full(o.shape, np.nan)
            on[:, 1:] = np.log(o[:, 1:].astype(np.float64) / c[:, :-1])
            idr = np.log(c.astype(np.float64) / o)
        bad = (np.abs(np.nan_to_num(on, nan=0)) > BAD) | (np.abs(np.nan_to_num(idr, nan=0)) > BAD)
        on[bad | ~np.isfinite(on)] = np.nan
        return on

    on_s, on_f = masked_on(so, sc), masked_on(fo, fc)
    srow = np.searchsorted(sids, rk['i'])
    frow = np.searchsorted(fids, rk['f'])
    tl = rk['t'] - lo
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sd = trailing_sd(on_s, on_f, srow, frow, rk['beta'].astype(np.float64), tl)
    # check: z = (g - beta x twin gap) / sd reproduces the rows' z
    gf = on_f[frow, tl]
    z2 = (rk['g'].astype(np.float64) - rk['beta'] * gf) / sd
    dz = np.abs(z2 - rk['z'])
    say(f'ranked rows {len(sd)}; recomputed z vs rows z: max abs diff {np.nanmax(dz):.2e}, '
        f'99.9th pct {np.nanpercentile(dz, 99.9):.2e}, non-finite {int((~np.isfinite(z2)).sum())}')
    rk['sd'] = sd
    rk['gf'] = gf
    # store prices for every ranked stock-day and its twin (scaled as the store has them)
    rk['s_open'] = so[srow, tl]
    rk['s_close'] = sc[srow, tl]
    rk['s_prevclose'] = sc[srow, tl - 1]
    rk['f_open'] = fo[frow, tl]
    rk['f_close'] = fc[frow, tl]
    rk['f_prevclose'] = fc[frow, tl - 1]
    # checks: the rows' returns equal the store's
    with np.errstate(all='ignore'):
        drs = np.abs(rk['s_close'].astype(np.float64) / rk['s_open'] - 1 - rk['rs'])
        drf = np.abs(rk['f_close'].astype(np.float64) / rk['f_open'] - 1 - rk['rf'])
    say(f'store open-to-close vs rows: stock max {np.nanmax(drs):.2e}, twin max {np.nanmax(drf):.2e}')

    # symbols
    stk = {int(i): yahoo_symbol(E['tickers'][i]) for i in np.unique(rk['i'])}
    fnd = {int(f): yahoo_symbol(tw['fund_ticker'][f]) for f in fids}
    exch = {s['id']: s['exchange'] for s in S['syms']}
    rk['exchange'] = np.array([exch[int(E['ids'][i])] for i in rk['i']])
    pos_all = set(int(i) for i in res['all']['i']) | set(int(i) for i in res['500']['i'])
    say(f'distinct stocks: ranked {len(stk)}, in positions (all or 500 book) {len(pos_all)}; distinct twins {len(fnd)}')
    np.savez(BIG / 'positions.npz', days=days, t0=t0,
             **{f'all_{k}': v for k, v in res['all'].items()}, **{f'b500_{k}': v for k, v in res['500'].items()},
             **{f'rk_{k}': v for k, v in rk.items()},
             stk_i=np.array(sorted(stk)), stk_sym=np.array([stk[i] for i in sorted(stk)]),
             fnd_f=np.array(sorted(fnd)), fnd_sym=np.array([fnd[f] for f in sorted(fnd)]),
             fund_store_id=tw['fund'][np.array(sorted(fnd))], stock_store_id=E['ids'][np.array(sorted(stk))])
    (OUT / 'positions.txt').write_text('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
