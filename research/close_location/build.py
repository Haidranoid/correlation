"""Build the close-location signal P and the weekly ranked sets (experiments/+2.5-close-location.md).

    python3 -I build.py old|new

Writes, under ~/.local/share/correlation2/experiments/close_location/:
  rows_<tag>.npz  one row per (formation week, stock in the ranked set): t (export day index), i (stock row), f (twin
                  fund row), P, Pst (stock-only 5-day mean CLV, a diagnostic), r1 (same-day log return), r5 (5-day log
                  return), y (next 5-day simple return, close d -> d+5), m5 (the export market's 5-day simple return
                  over the same days), in500.
  P_<tag>.npy     the unsigned money-test scores: P on every day from the first full 2008 window, for the universe in
                  force that day and with an export close that day; NaN elsewhere.
and out/build_<tag>.txt: data quality and coverage (no forward returns are looked at here).
"""
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np = C.np
c2eval = C.c2eval

LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def ffill_cols(X):
    """Forward-fill NaNs along axis 1 (uses only earlier columns)."""
    fin = np.isfinite(X)
    idx = np.where(fin, np.arange(X.shape[1], dtype=np.int32)[None, :], 0).astype(np.int32)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return np.take_along_axis(X, idx, 1)


def main(tag):
    t_start = time.time()
    exp, sto = C.SRC[tag]
    E = c2eval.load_export(exp, arrays=('close', 'dollar_volume', 'v', 'market'))
    days = E['days']
    N, T = len(E['ids']), len(days)
    say(f'{tag}: export {N} x {T}, {c2eval.iso(days[0])}..{c2eval.iso(days[-1])}')
    tw = c2eval.twins(E)
    F = tw.pop('F')
    del F
    excl = c2eval.excluded(E, tw)
    rebal = tw['rebal']
    K = len(rebal)
    nF = len(tw['fund'])
    c0 = c2eval.day_index(days, C.FIRST)
    anchor = c0 + C.WIN - 1  # first day whose 5-day window lies entirely in 2008
    TT = T - c0
    say(f'CLV from {c2eval.iso(days[c0])} (index {c0}); first P / week anchor {c2eval.iso(days[anchor])} (index {anchor}); funds {nF}')

    # ---- store high / low / close, stocks and twin funds, from 2008
    S = c2eval.load_store(sto)
    dd = days[c0:]
    Hs = c2eval.store_panel(S, E['ids'], dd, 'high')
    Ls = c2eval.store_panel(S, E['ids'], dd, 'low')
    Cs = c2eval.store_panel(S, E['ids'], dd, 'close')
    clv_s, bad_s = C.clv(Hs, Ls, Cs)
    del Hs, Ls
    Hf = c2eval.store_panel(S, tw['fund'], dd, 'high')
    Lf = c2eval.store_panel(S, tw['fund'], dd, 'low')
    Cf = c2eval.store_panel(S, tw['fund'], dd, 'close')
    clv_f, bad_f = C.clv(Hf, Lf, Cf)
    del Hf, Lf, Cf
    say(f'store panels done ({time.time() - t_start:.0f}s)')

    # ---- universe per twin rebalance
    blk = np.searchsorted(rebal, np.arange(T), side='right') - 1  # rebal_index for every day
    uni = np.zeros((K, N), bool)
    in500 = np.zeros((K, N), bool)
    blocks = []
    dv = E['dollar_volume']
    for k in range(K):
        t0 = int(rebal[k])
        t1 = int(rebal[k + 1]) if k + 1 < K else T
        if t1 <= anchor:
            continue
        L = C.liquid(E, t0)
        U = L & (tw['fund_row'][k] >= 0) & ~excl[k]
        si = np.nonzero(U)[0]
        uni[k] = U
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mdv = np.nan_to_num(np.nanmedian(np.asarray(dv[si, t0 + 1 - C.TOP_W:t0 + 1]), 1))
        in500[k, si[np.argsort(-mdv, kind='stable')[:C.TOP]]] = True
        fr = tw['fund_row'][k, si]
        blocks.append(dict(k=k, t0=t0, day=c2eval.iso(days[t0]), liquid=int(L.sum()), universe=int(len(si)),
                           funds=int(len(np.unique(fr))), corr_median=float(np.median(tw['corr'][k, si]))))
    say(f'universe done ({time.time() - t_start:.0f}s)')

    # ---- daily difference D = CLV stock - CLV twin (twin of that day's rebalance), and P
    D = np.full((N, TT), np.nan, np.float32)
    Udays = np.zeros((N, TT), bool)        # stock-days in the universe (for data-quality counts)
    Fdays = np.zeros((nF, TT), bool)       # fund-days that are a universe member's twin
    for k in range(K):
        t0 = int(rebal[k])
        t1 = int(rebal[k + 1]) if k + 1 < K else T
        a, b = max(t0, c0), t1
        if b <= a:
            continue
        cols = np.arange(a - c0, b - c0)
        rows = np.nonzero(tw['fund_row'][k] >= 0)[0]
        fr = tw['fund_row'][k, rows]
        D[np.ix_(rows, cols)] = clv_s[np.ix_(rows, cols)] - clv_f[np.ix_(fr, cols)]
        ur = np.nonzero(uni[k])[0]
        Udays[np.ix_(ur, cols)] = True
        Fdays[np.ix_(np.unique(tw['fund_row'][k, ur]), cols)] = True

    def roll(X):
        fin = np.isfinite(X)
        z = np.zeros((X.shape[0], 1))
        cs = np.concatenate([z, np.cumsum(np.where(fin, X, 0).astype(np.float64), 1)], 1)
        cn = np.concatenate([z, np.cumsum(fin, 1, dtype=np.float64)], 1)
        out = np.full(X.shape, np.nan, np.float32)
        j = np.arange(C.WIN - 1, X.shape[1])
        s = cs[:, j + 1] - cs[:, j + 1 - C.WIN]
        n = cn[:, j + 1] - cn[:, j + 1 - C.WIN]
        with np.errstate(all='ignore'):
            out[:, j] = np.where(n >= C.MIN_OK, s / n, np.nan)
        return out

    P = roll(D)
    Pst = roll(clv_s)
    say(f'P done ({time.time() - t_start:.0f}s)')

    # ---- data quality on universe stock-days and twin fund-days (2008 on; no forward returns)
    yr = np.array([c2eval.iso(x)[:4] for x in dd])
    years = sorted(set(yr))
    Ce = np.asarray(E['close'][:, c0:], np.float64)
    say('\nBad prints (share of days, %): universe stock-days | twin fund-days')
    say('year  stock-days  missing  range>0.5  H<C  L>C  H=L  all-bad | fund-days  missing  range>0.5  H<C  L>C  H=L  all-bad')
    for y in years:
        cm = yr == y
        u = Udays[:, cm]
        f = Fdays[:, cm]
        nu, nf = u.sum(), f.sum()
        su = [100 * (bad_s[r][:, cm] & u).sum() / nu for r in ('missing', 'range>0.5', 'H<C', 'L>C', 'H=L')]
        sf = [100 * (bad_f[r][:, cm] & f).sum() / nf for r in ('missing', 'range>0.5', 'H<C', 'L>C', 'H=L')]
        au = 100 * (~np.isfinite(clv_s[:, cm]) & u).sum() / nu
        af = 100 * (~np.isfinite(clv_f[:, cm]) & f).sum() / nf
        say(f'{y}  {nu:>9}  ' + '  '.join(f'{x:.3f}' for x in su) + f'  {au:.3f} | {nf:>7}  '
            + '  '.join(f'{x:.3f}' for x in sf) + f'  {af:.3f}')
    with np.errstate(all='ignore'):
        rel = np.abs(Cs / Ce - 1)
    both = Udays & np.isfinite(rel)
    say(f'\nStore close vs export close on universe stock-days with both: n {int(both.sum())}, '
        f'differ > 0.1%: {100 * (both & (rel > 1e-3)).sum() / both.sum():.3f}%, > 1%: {100 * (both & (rel > 1e-2)).sum() / both.sum():.3f}%; '
        f'export close missing on universe stock-days: {100 * (Udays & ~np.isfinite(Ce)).sum() / Udays.sum():.3f}%')
    del rel, both, Cs
    say('\nCLV on universe stock-days and twin fund-days, and P coverage')
    say('year  stock CLV mean  sd  at +1 %  at -1 % | twin CLV mean  sd | P finite % of universe days  P mean  P sd')
    for y in years:
        cm = yr == y
        u = Udays[:, cm]
        f = Fdays[:, cm]
        xs = clv_s[:, cm][u]
        xs = xs[np.isfinite(xs)]
        xf = clv_f[:, cm][f]
        xf = xf[np.isfinite(xf)]
        pp = P[:, cm][u]
        pf = pp[np.isfinite(pp)]
        say(f'{y}  {xs.mean():+.3f}  {xs.std():.3f}  {100 * (xs == 1).mean():.1f}  {100 * (xs == -1).mean():.1f} | '
            f'{xf.mean():+.3f}  {xf.std():.3f} | {100 * len(pf) / len(pp):.1f}  {pf.mean():+.3f}  {pf.std():.3f}')
    del Udays, Fdays, clv_s, clv_f, D

    # ---- returns from export closes (forward-filled for the holding and look-back legs)
    Cx = np.asarray(E['close'][:, c0 - C.WIN - 1:], np.float64)  # column 0 = day c0-6
    off = c0 - C.WIN - 1
    Cff = ffill_cols(Cx)
    mk = np.asarray(E['market'], np.float64)
    cmk = np.concatenate([[0.0], np.cumsum(np.nan_to_num(mk))])
    weeks = np.arange(anchor, T - C.WIN, C.WIN)  # formation days with a full hold inside the export
    cols = {k: [] for k in ('t', 'i', 'f', 'P', 'Pst', 'r1', 'r5', 'y', 'm5', 'in500')}
    wk_info = []
    for d in weeks:
        k = int(blk[d])
        j = d - c0
        x = d - off
        with np.errstate(all='ignore'):
            r1 = np.log(Cff[:, x] / Cff[:, x - 1])
            r5 = np.log(Cff[:, x] / Cff[:, x - 5])
            y = Cff[:, x + 5] / Cff[:, x] - 1
        ok = (uni[k] & np.isfinite(P[:, j]) & np.isfinite(Cx[:, x]) & np.isfinite(r1) & np.isfinite(r5) & np.isfinite(y))
        si = np.nonzero(ok)[0]
        m5 = float(np.expm1(cmk[d + 6] - cmk[d + 1]))  # market log returns of days d+1..d+5
        cols['t'].append(np.full(len(si), d, np.int32))
        cols['i'].append(si.astype(np.int32))
        cols['f'].append(tw['fund_row'][k, si].astype(np.int32))
        cols['P'].append(P[si, j])
        cols['Pst'].append(Pst[si, j])
        cols['r1'].append(r1[si].astype(np.float32))
        cols['r5'].append(r5[si].astype(np.float32))
        cols['y'].append(y[si])
        cols['m5'].append(np.full(len(si), m5))
        cols['in500'].append(in500[k, si])
        wk_info.append((d, int(uni[k].sum()), len(si), int(in500[k, si].sum())))
    rows = {k: np.concatenate(v) for k, v in cols.items()}
    C.BIG.mkdir(parents=True, exist_ok=True)
    np.savez(C.BIG / f'rows_{tag}.npz', days=days, **rows)
    say(f'\nrows {len(rows["t"])} over {len(weeks)} weeks ({c2eval.iso(days[weeks[0]])}..{c2eval.iso(days[weeks[-1]])}); '
        f'last formation with a full hold {c2eval.iso(days[weeks[-1]])}')

    # ---- money-test scores (unsigned): P for the universe in force that day, with an export close that day
    sc = np.full((N, T), np.nan, np.float32)
    for k in range(K):
        t0 = int(rebal[k])
        t1 = int(rebal[k + 1]) if k + 1 < K else T
        a = max(t0, anchor)
        if t1 <= a:
            continue
        ur = np.nonzero(uni[k])[0]
        pc = P[np.ix_(ur, np.arange(a - c0, t1 - c0))]
        ce = np.asarray(E['close'][:, a:t1])[ur]
        sc[np.ix_(ur, np.arange(a, t1))] = np.where(np.isfinite(ce), pc, np.nan)
    p = c2eval.save_scores(sc, f'close_location/P_{tag}', export=exp)
    per_day = np.isfinite(sc).sum(0)
    say(f'scores -> {p}; first scored day {c2eval.iso(days[int(np.argmax(per_day > 0))])}; '
        f'median names per scored day {int(np.median(per_day[per_day > 0]))}')
    del sc

    # ---- coverage by year
    by = {}
    for bl in blocks:
        by.setdefault(bl['day'][:4], []).append(bl)
    wy = {}
    for d, nu, nr, n5 in wk_info:
        wy.setdefault(c2eval.iso(days[d])[:4], []).append((nu, nr, n5))
    say('\nyear  twin rebalances  liquid  universe  distinct twins  median twin corr | weeks  ranked/week (median)  ranked 500/week')
    for y in sorted(set(by) | set(wy)):
        bl = by.get(y, [])
        w = wy.get(y, [])
        a = (f'{len(bl):>3}  {np.mean([x["liquid"] for x in bl]):>6.0f}  {np.mean([x["universe"] for x in bl]):>6.0f}  '
             f'{np.mean([x["funds"] for x in bl]):>5.0f}  {np.mean([x["corr_median"] for x in bl]):.2f}') if bl else '-'
        b = (f'{len(w):>3}  {np.median([x[1] for x in w]):>6.0f}  {np.median([x[2] for x in w]):>5.0f}') if w else '-'
        say(f'{y}  {a} | {b}')
    small = [(c2eval.iso(days[d]), nr, n5) for d, nu, nr, n5 in wk_info if nr < C.MIN_NAMES or n5 < C.MIN_NAMES]
    say(f'weeks with fewer than {C.MIN_NAMES} ranked names (all / 500 largest): {small}')
    C.OUT.mkdir(exist_ok=True)
    (C.OUT / f'build_{tag}.txt').write_text('\n'.join(LOG) + '\n')
    (C.OUT / f'build_{tag}_blocks.json').write_text(json.dumps(blocks, indent=0))
    say(f'done in {time.time() - t_start:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1])
