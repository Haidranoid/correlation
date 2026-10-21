"""The daily signal panel, point in time (everything at day t uses closes through t).

At each trading day t from 2007-01-01 to 2022-05-31:
- fiscal year y = epoch_of(t) (last year's from June 1, else the year before); only FY y's text peers count;
- a text pair (i, j) counts from the later of the two filings' dates (common.avail_day);
- text peers of firm i: the filings whose similarity with i's is at or above the year's top-2% cutoff, the 50 most
  similar of all that year's filings, kept when the peer's CIK is in the ticker map;
- both i and its peers must be in the liquid universe at the latest twins-calendar day <= t;
- the correlation of daily market-relative log returns (log return - the export's equal-weight market; |log| > 0.5
  is a bad print and missing) over t-251..t, pairwise complete, with at least 227 common days;
- hidden peers: correlation strictly below the median of the firm's peer correlations that day; visible: the rest;
- a firm needs at least 3 hidden peers;
- H, V, A: the hidden, visible and all peers' mean market-relative log return over t-20..t (forward-filled closes,
  log(C_t / C_t-21) minus the market's sum over the same 21 days);
- each is residualized across the firms that have all three and their own 21-day return (OLS with intercept at each t).

    python3 -I build.py         # writes panel.npz and formation_counts.npz in the work folder
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import (CAP, CORR_W, FORM, FY_FIRST, FY_LAST, MAX_LOG, MIN_HIDDEN, MIN_OVERLAP, PEERS, SIG_W, WORK,  # noqa: E402
                    c2eval, cik_rows, epoch_end, ffill, log, np, universe)

CHUNK = 4000


def returns(E):
    close = np.asarray(E['close'], np.float64)
    m = np.nan_to_num(np.asarray(E['market'], np.float64))
    lr = np.full(close.shape, np.nan)
    with np.errstate(all='ignore'):
        lr[:, 1:] = np.log(close[:, 1:] / close[:, :-1])
    lr[~np.isfinite(lr) | (np.abs(lr) > MAX_LOG)] = np.nan
    mr = (lr - m[None, :]).astype(np.float32)
    del lr
    cff = ffill(close)
    del close
    cm = np.concatenate([[0.0], np.cumsum(m)])           # cm[t+1] = market sum through t
    r21 = np.full(cff.shape, np.nan, np.float32)
    with np.errstate(all='ignore'):
        r21[:, SIG_W:] = np.log(cff[:, SIG_W:] / cff[:, :-SIG_W]) - (cm[SIG_W + 1:] - cm[1:-SIG_W])[None, :]
    r21[~np.isfinite(r21)] = np.nan
    return mr, r21, cff


def rolling_corr(mr, a, b, d0, d1):
    """Correlation of rows a and b (arrays of pairs) over the CORR_W days ending at each t in [d0, d1). [P, d1-d0]."""
    out = np.full((len(a), d1 - d0), np.nan, np.float32)
    s = d0 - CORR_W + 1
    for c0 in range(0, len(a), CHUNK):
        X = mr[a[c0:c0 + CHUNK], s:d1].astype(np.float64)
        Y = mr[b[c0:c0 + CHUNK], s:d1].astype(np.float64)
        M = np.isfinite(X) & np.isfinite(Y)
        X = np.where(M, X, 0.0)
        Y = np.where(M, Y, 0.0)

        def rs(Z):
            cs = np.zeros((Z.shape[0], Z.shape[1] + 1))
            np.cumsum(Z, 1, out=cs[:, 1:])
            return cs[:, CORR_W:] - cs[:, :-CORR_W]      # column i <-> window ending at d0 + i
        n = rs(M.astype(np.float64))
        sx, sy = rs(X), rs(Y)
        with np.errstate(all='ignore'):
            cxy = n * rs(X * Y) - sx * sy
            cxx = n * rs(X * X) - sx * sx
            cyy = n * rs(Y * Y) - sy * sy
            r = cxy / np.sqrt(cxx * cyy)
        r[(n < MIN_OVERLAP) | ~np.isfinite(r)] = np.nan
        out[c0:c0 + CHUNK] = r
    return out


def main():
    E = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close', 'dollar_volume', 'market'))
    days = E['days']
    N, T = len(E['tickers']), len(days)
    tw = c2eval.twins(E)
    rebal = tw['rebal']
    uni, big = universe(E, tw)
    log('universe ready; median size', int(np.median(uni.sum(1)[uni.sum(1) > 0])))
    mr, r21, cff = returns(E)
    del cff
    log('returns ready')
    rows_of = cik_rows()
    f0, f1 = c2eval.day_index(days, FORM[0]), c2eval.day_index(days, '2022-06-01')
    blk = np.searchsorted(rebal, np.arange(T), side='right') - 1
    rebal_set = set(int(x) for x in rebal)
    cols = {k: [] for k in ('t', 'row', 'n_peers', 'n_hidden', 'H', 'V', 'A', 'corr_h', 'corr_v', 'own')}
    fc = {k: [] for k in ('t', 'row', 'n_text', 'n_mapped', 'n_univ', 'n_corr', 'n_hidden', 'fy')}
    for fy in range(FY_FIRST, FY_LAST + 1):
        z = np.load(PEERS / f'{fy}.npz')
        cik, avail, mapped = z['cik'], z['avail'], z['mapped']
        keep = z['rank_all'] < CAP
        pi, pj = z['pi'][keep], z['pj'][keep]
        pm = mapped[pj]
        pi, pj = pi[pm], pj[pm]
        # expand to export rows (a CIK may have several rows; pairs of the same CIK cannot occur)
        A, B, AV = [], [], []
        for i, j in zip(pi, pj):
            for ra in rows_of[int(cik[i])]:
                for rb in rows_of[int(cik[j])]:
                    A.append(ra); B.append(rb); AV.append(max(avail[i], avail[j]))
        A, B, AV = np.array(A, np.int64), np.array(B, np.int64), np.array(AV, np.int64)
        lo = max(int(np.searchsorted(days, epoch_end(fy - 1))), f0)
        hi = min(int(np.searchsorted(days, epoch_end(fy))), f1)
        if hi <= lo:
            continue
        # unordered pairs for the correlations
        ua, ub = np.minimum(A, B), np.maximum(A, B)
        key = ua * N + ub
        uk, inv = np.unique(key, return_inverse=True)
        C = rolling_corr(mr, uk // N, uk % N, lo, hi)
        AVd = np.searchsorted(days, AV)                         # first trading day on or after the date
        # mapped filings of this year and their row lists (for the formation-day counts)
        mi = np.nonzero(mapped)[0]
        f_rows = np.array([r for i in mi for r in rows_of[int(cik[i])]], np.int64)
        f_avd = np.array([np.searchsorted(days, avail[i]) for i in mi for _ in rows_of[int(cik[i])]], np.int64)
        f_ntext = np.array([min(int(z['n_above'][i]), CAP) for i in mi for _ in rows_of[int(cik[i])]], np.int64)
        n_days = 0
        for t in range(lo, hi):
            k = blk[t]
            U = uni[k]
            di = t - lo
            act = (AVd <= t) & U[A] & U[B]
            c = C[inv, di]
            act_c = act & np.isfinite(c)
            a, b, cc = A[act_c], B[act_c], c[act_c]
            if t in rebal_set:
                fm = (f_avd <= t) & U[f_rows]
                rows_t = f_rows[fm]
                n_map = np.bincount(A[AVd <= t], minlength=N)
                n_un = np.bincount(A[act], minlength=N)
                n_co = np.bincount(a, minlength=N)
            if len(a):
                o = np.lexsort((cc, a))
                a, b, cc = a[o], b[o], cc[o]
                starts = np.r_[0, np.nonzero(np.diff(a))[0] + 1]
                ns = np.diff(np.r_[starts, len(a)])
                g = np.repeat(np.arange(len(starts)), ns)
                med = (cc[starts + (ns - 1) // 2] + cc[starts + ns // 2]) / 2.0
                hid = cc < med[g]
                ret = r21[b, t].astype(np.float64)
                fin = np.isfinite(ret)
                G = len(starts)

                def gmean(mask):
                    m2 = mask & fin
                    cnt = np.bincount(g[m2], minlength=G)
                    s = np.bincount(g[m2], weights=ret[m2], minlength=G)
                    with np.errstate(all='ignore'):
                        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
                nh = np.bincount(g[hid], minlength=G)
                H, V, Aall = gmean(hid), gmean(~hid), gmean(np.ones(len(a), bool))
                with np.errstate(all='ignore'):
                    ch = np.bincount(g[hid], weights=cc[hid], minlength=G) / np.maximum(nh, 1)
                    cv = np.bincount(g[~hid], weights=cc[~hid], minlength=G) / np.maximum(ns - nh, 1)
                firm = a[starts]
                ok = nh >= MIN_HIDDEN
                for name, v in (('t', np.full(ok.sum(), t)), ('row', firm[ok]), ('n_peers', ns[ok]), ('n_hidden', nh[ok]),
                                ('H', H[ok]), ('V', V[ok]), ('A', Aall[ok]), ('corr_h', ch[ok]), ('corr_v', cv[ok]),
                                ('own', r21[firm[ok], t])):
                    cols[name].append(np.asarray(v))
            if t in rebal_set:
                nh_full = np.zeros(N, np.int64)
                if len(a):
                    nh_full[firm] = nh
                fc['t'].append(np.full(len(rows_t), t)); fc['row'].append(rows_t)
                fc['n_text'].append(f_ntext[fm]); fc['n_mapped'].append(n_map[rows_t]); fc['n_univ'].append(n_un[rows_t])
                fc['n_corr'].append(n_co[rows_t]); fc['n_hidden'].append(nh_full[rows_t]); fc['fy'].append(np.full(len(rows_t), fy))
            n_days += 1
        log(f'FY{fy}: {len(pi)} mapped text pairs -> {len(A)} row pairs ({len(uk)} unordered); days '
            f'{c2eval.iso(days[lo])}..{c2eval.iso(days[hi - 1])} ({n_days}); firm-days so far {sum(len(x) for x in cols["t"])}')
        del C
    P = {k: np.concatenate(v) for k, v in cols.items()}
    # residualize across firms at each t: H, V, A on the firm's own 21-day return (OLS with intercept)
    ok = np.isfinite(P['H']) & np.isfinite(P['V']) & np.isfinite(P['A']) & np.isfinite(P['own'])
    P = {k: v[ok] for k, v in P.items()}
    for name in ('H', 'V', 'A'):
        P[name + 'res'] = np.full(len(P['t']), np.nan)
    ts = P['t']
    order = np.argsort(ts, kind='stable')
    P = {k: v[order] for k, v in P.items()}
    ts = P['t']
    bounds = np.r_[0, np.nonzero(np.diff(ts))[0] + 1, len(ts)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s < 10:
            continue
        x = P['own'][s:e].astype(np.float64)
        X = np.column_stack([np.ones(e - s), x])
        for name in ('H', 'V', 'A'):
            y = P[name][s:e].astype(np.float64)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            P[name + 'res'][s:e] = y - X @ beta
    np.savez(WORK / 'panel.npz', **P)
    F = {k: np.concatenate(v) for k, v in fc.items()}
    np.savez(WORK / 'formation_counts.npz', **F)
    log(f'panel: {len(P["t"])} firm-days, {len(np.unique(P["t"]))} days; median firms a day '
        f'{int(np.median(np.diff(bounds)))}')


if __name__ == '__main__':
    main()
