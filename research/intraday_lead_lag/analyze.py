"""The intraday lead-lag test (experiments/-100.0-intraday-lead-lag.md) on one bar size.

    python -I analyze.py 5m|1m     # reads WORK/panel_<mode>.npz and WORK/daily.npz; writes out/results_<mode>.txt/.json

Rules: the plan and out/decisions.txt. In short: within-day log returns between consecutive bar closes (the first
bar's return and anything overnight excluded; a bar missing from Yahoo's output makes the returns into and out of
it missing; no forward filling); full days only (SPY has every bar); the 2% close rule; the first half of the days
selects, the second tests; market-adjusted returns r - beta x SPY's same-bar return (beta from the selection
half); lead-lag = exact pairwise-complete Pearson corr(r_A(k), r_B(k+1)) within the day, from masked matrix
products; null = B's days rotated by s within the half; threshold = 99.9th percentile of the pooled null |corr|.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path.home() / '.local/share/correlation2/experiments/intraday_lead_lag'
FUNDS = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY',
         'SMH', 'KRE', 'XBI', 'GDX', 'XOP']
MIN_N = 1000
Q = 99.9
MAX_SHIFTS = 20
MIN_FOLLOWERS = 10
COST_BP = 1.0
LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def pw_corr(X, Y, min_n=MIN_N):
    """Exact pairwise-complete Pearson correlation of every row of X [S, M] with every row of Y [S, M] (NaN =
    missing), over the columns where both exist. Returns corr [S, S] (NaN where n < min_n or no variance), n."""
    Mx, My = np.isfinite(X), np.isfinite(Y)
    X0, Y0 = np.where(Mx, X, 0.0), np.where(My, Y, 0.0)
    mx, my = Mx.astype(np.float64), My.astype(np.float64)
    n = mx @ my.T
    sx = X0 @ my.T
    sy = mx @ Y0.T
    sxx = (X0 * X0) @ my.T
    syy = mx @ (Y0 * Y0).T
    sxy = X0 @ Y0.T
    with np.errstate(all='ignore'):
        cov = sxy - sx * sy / n
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        c = cov / np.sqrt(vx * vy)
    c[(n < min_n) | ~np.isfinite(c) | (vx <= 0) | (vy <= 0)] = np.nan
    return c, n


def direct_corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def avg_rank(x):
    o = np.argsort(x, kind='stable')
    xs = x[o]
    starts = np.r_[0, np.nonzero(np.diff(xs) != 0)[0] + 1]
    ends = np.r_[starts[1:], len(xs)]
    r = np.empty(len(x))
    r[o] = np.repeat((starts + ends - 1) / 2.0, ends - starts)
    return r


def spearman(a, b):
    ra, rb = avg_rank(a), avg_rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def clustered_t(x, day):
    """Mean of x over bars and its t with the standard error clustered by day."""
    ok = np.isfinite(x)
    x, day = x[ok], day[ok]
    if len(x) < 3:
        return float('nan'), float('nan'), len(x), 0
    e = x - x.mean()
    g = np.bincount(day, weights=e)
    se = np.sqrt((g * g).sum()) / len(x)
    return float(x.mean()), float(x.mean() / se), int(len(x)), int((np.bincount(day) > 0).sum())


def lagged(A):
    """[S, D, R] returns -> X = r(k), Y = r(k+1) as [S, D, R-1] (same day only)."""
    return A[:, :, :-1], A[:, :, 1:]


def flat(Z):
    return Z.reshape(Z.shape[0], -1)


def selection_stats(A, valid_sym, shifts, label):
    """Lead-lag corr on the half A [S, D, R]; the null over day rotations; threshold; counts."""
    X3, Y3 = lagged(A)
    X = flat(X3)
    C, n = pw_corr(X, flat(Y3))
    S = A.shape[0]
    pair_ok = valid_sym[:, None] & valid_sym[None, :] & ~np.eye(S, dtype=bool)
    C[~pair_ok] = np.nan
    nulls = []
    for s in shifts:
        Cs, _ = pw_corr(X, flat(np.roll(Y3, -s, axis=1)))  # A's day i with B's day i+s (wrapping)
        Cs[~pair_ok] = np.nan
        nulls.append(Cs)
    N = np.stack(nulls)
    pooled = np.abs(N[np.isfinite(N)])
    thr = float(np.percentile(pooled, Q))
    count = int((np.abs(C) > thr).sum())
    null_counts = np.array([(np.abs(Ns) > thr).sum() for Ns in N])
    null_count = float(null_counts.mean())
    n_pairs = int(np.isfinite(C).sum())
    say(f'[{label}] pairs with >= {MIN_N} bar pairs: {n_pairs:,}; null shifts {list(shifts)}; threshold |corr| > {thr:.4f} '
        f'(99.9th pct of {len(pooled):,} null values)')
    say(f'[{label}] above threshold: {count:,} pairs; null mean {null_count:.1f} per shift (min {null_counts.min()}, '
        f'max {null_counts.max()}); ratio {count / null_count:.3f}; (count/null - 1) x 100 = {(count / null_count - 1) * 100:+.1f}; '
        f'shifts with a count >= the actual: {int((null_counts >= count).sum())} of {len(null_counts)}')
    return {'C': C, 'n': n, 'N': N, 'thr': thr, 'count': count, 'null_count': null_count,
            'null_counts': null_counts.tolist(), 'n_pairs': n_pairs, 'X': X, 'Y3': Y3}


def main():
    mode = sys.argv[1]
    P = np.load(WORK / f'panel_{mode}.npz')
    syms = [str(s) for s in P['syms']]
    dates = P['dates']
    close = P['close'].copy()
    volume = P['volume']
    S, D, B = close.shape
    spy = syms.index('SPY')
    is_fund = np.array([s in FUNDS for s in syms])
    say(f'== {mode}: {S} symbols, {D} dates {dates[0]}..{dates[-1]}, {B} bars a day; panel rows {int(P["n_rows"])}, '
        f'outside session {int(P["n_outside"])}, other UTC offset {int(P["n_other_offset"])}, NaN close {int(P["n_nan_close"])}, '
        f'duplicates {int(P["n_dup"])}')
    # a volume-0 bar at the previous bar's close shows no trade: missing (the panel keeps closes only, so this
    # stands in for "flat volume-0 bar at the prior close")
    prevc = np.full_like(close, np.nan)
    prevc[:, :, 1:] = close[:, :, :-1]
    stale = np.isfinite(close) & (volume == 0) & (close == prevc)
    say(f'volume-0 bars {int((np.isfinite(close) & (volume == 0)).sum()):,} of {int(np.isfinite(close).sum()):,}; '
        f'of them at the previous close (set missing) {int(stale.sum())}')
    close[stale] = np.nan
    # full days: SPY has every bar
    full = np.isfinite(close[spy]).all(1)
    say(f'full days (SPY has all {B} bars): {int(full.sum())} of {D}; dropped {[str(x) for x in dates[~full]]}')
    close, volume, dates = close[:, full], volume[:, full], dates[full]
    D = len(dates)
    # 2% rule against Yahoo's daily close
    Dl = np.load(WORK / 'daily.npz')
    dsyms = [str(s) for s in Dl['syms']]
    assert dsyms == syms
    dmap = {str(x): i for i, x in enumerate(Dl['dates'])}
    dclose = np.full((S, D), np.nan)
    for j, x in enumerate(dates):
        if str(x) in dmap:
            dclose[:, j] = Dl['close'][:, dmap[str(x)]]
    has_bars = np.isfinite(close).any(2)
    last_idx = np.where(has_bars, B - 1 - np.argmax(np.isfinite(close[:, :, ::-1]), 2), 0)
    last_close = np.take_along_axis(close, last_idx[:, :, None], 2)[:, :, 0]
    with np.errstate(all='ignore'):
        dev = np.abs(last_close / dclose - 1)
    no_daily = has_bars & ~np.isfinite(dclose)
    off2 = has_bars & np.isfinite(dclose) & (dev > 0.02)
    drop = no_daily | off2
    say(f'2% rule: symbol-days with bars {int(has_bars.sum()):,}; without a daily close {int(no_daily.sum())} '
        f'({sorted(set(syms[i] for i in np.nonzero(no_daily)[0]))}); last-bar close off by > 2%: {int(off2.sum())} '
        f'({sorted(set((syms[i], str(dates[j])) for i, j in zip(*np.nonzero(off2))))[:20]})')
    close[drop] = np.nan
    missing_frac = 1 - np.isfinite(close[:, :, :]).sum() / (S * D * B)
    say(f'bars missing after the rules (absent, or symbol-day dropped): {missing_frac:.4%}; absent bars only on kept '
        f'symbol-days: {1 - np.isfinite(close[~drop & has_bars]).mean():.4%}')
    # within-day log returns at bars 1..B-1 (the 09:30 bar's return from the prior close is excluded)
    with np.errstate(all='ignore'):
        R = np.log(close[:, :, 1:] / close[:, :, :-1])
    R[~np.isfinite(R)] = np.nan
    Rn = R.shape[2]
    Dsel = D // 2
    sel, tst = slice(0, Dsel), slice(Dsel, D)
    say(f'days: selection {Dsel} ({dates[0]}..{dates[Dsel - 1]}), test {D - Dsel} ({dates[Dsel]}..{dates[-1]}); '
        f'{Rn} returns a day, {Rn - 1} lag-1 slots')
    # market adjustment, beta from the selection half
    rs = flat(R[:, sel])
    m = rs[spy]
    beta = np.full(S, np.nan)
    for i in range(S):
        ok = np.isfinite(rs[i]) & np.isfinite(m)
        if ok.sum() >= MIN_N:
            x, y = m[ok], rs[i][ok]
            beta[i] = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    A = R - beta[:, None, None] * R[spy][None, :, :]
    A[spy] = np.nan  # identically zero; out of the adjusted pair set
    valid = np.isfinite(beta) & (np.arange(S) != spy)
    say(f'beta (selection half): median {np.nanmedian(beta):.2f}, 10-90% {np.nanpercentile(beta, 10):.2f}..'
        f'{np.nanpercentile(beta, 90):.2f}; symbols in the adjusted pair set {int(valid.sum())}')
    Dh = Dsel
    shifts = list(range(1, min(MAX_SHIFTS, Dh - 1) + 1))
    res = {'mode': mode, 'days': D, 'days_sel': Dsel, 'days_test': D - Dsel, 'dates': [str(x) for x in dates],
           'symbols_in_pairs': int(valid.sum()), 'shifts': shifts}

    # ---------------- Test 1: selection half
    say('\n-- Test 1: selection half, market-adjusted returns')
    T1 = selection_stats(A[:, sel], valid, shifts, 'adjusted')
    C, thr = T1['C'], T1['thr']
    number = (T1['count'] / T1['null_count'] - 1) * 100
    res['test1'] = {k: T1[k] for k in ('thr', 'count', 'null_count', 'null_counts', 'n_pairs')}
    res['test1']['number'] = number
    res['test1']['pass'] = bool(T1['count'] >= 2 * T1['null_count'])
    say(f'TEST 1: {"PASS" if res["test1"]["pass"] else "FAIL"} (needs count >= 2 x null = {2 * T1["null_count"]:.1f}); '
        f'the number {number:+.1f}')

    # fast path vs direct for a few pairs (real and one shifted)
    rng = np.random.default_rng(0)
    X, Y3 = T1['X'], T1['Y3']
    Yf = flat(Y3)
    cand = np.argwhere(np.isfinite(C))
    worst = 0.0
    picks = cand[rng.choice(len(cand), 6, replace=False)].tolist() + [list(np.unravel_index(np.nanargmax(np.abs(C)), C.shape))]
    for a, b in picks:
        c_d, n_d = direct_corr(X[a], Yf[b])
        worst = max(worst, abs(c_d - C[a, b]))
        assert n_d == int(T1['n'][a, b])
    s0 = shifts[0]
    Ys = flat(np.roll(Y3, -s0, axis=1))
    for a, b in picks[:3]:
        c_d, _ = direct_corr(X[a], Ys[b])
        worst = max(worst, abs(c_d - T1['N'][0][a, b]))
    say(f'fast path vs direct computation on {len(picks)} real pairs and 3 shifted ones: max |diff| {worst:.2e}')
    assert worst < 1e-9
    res['check_max_diff'] = worst

    # top 20 pairs
    abs_c = np.where(np.isfinite(C), np.abs(C), -1)
    order = np.argsort(-abs_c, axis=None)[:20]
    Ct, _ = pw_corr(flat(lagged(A[:, tst])[0]), flat(lagged(A[:, tst])[1]))
    # Yahoo dollar volume per symbol over the selection half (for the "who leads" context)
    dv = np.nansum(close[:, sel] * volume[:, sel], axis=(1, 2))
    dv_rank = np.argsort(np.argsort(-dv))  # 0 = most traded
    say('\nTop 20 ordered pairs (leader -> follower) by |corr| in the selection half:')
    say(f'{"leader":>6} -> {"follower":<8} {"corr":>8} {"n":>6} {"test corr":>9}  dv rank L/F')
    top = []
    for f in order:
        a, b = np.unravel_index(f, C.shape)
        top.append({'leader': syms[a], 'follower': syms[b], 'corr': float(C[a, b]), 'n': int(T1['n'][a, b]),
                    'test_corr': float(Ct[a, b]), 'dv_rank_leader': int(dv_rank[a]), 'dv_rank_follower': int(dv_rank[b])})
        say(f'{syms[a]:>6} -> {syms[b]:<8} {C[a, b]:+8.4f} {int(T1["n"][a, b]):6d} {Ct[a, b]:+9.4f}  {dv_rank[a]}/{dv_rank[b]}')
    res['top20'] = top

    # selected pairs
    selm = np.isfinite(C) & (np.abs(C) > thr)
    W = np.where(selm, np.sign(C), 0.0)
    la, fb = np.nonzero(selm)
    same_sign = np.mean(np.sign(Ct[la, fb]) == np.sign(C[la, fb])) if len(la) else np.nan
    mean_signed_test = np.nanmean(np.sign(C[la, fb]) * Ct[la, fb]) if len(la) else np.nan
    lead_more_traded = np.mean(dv[la] > dv[fb]) if len(la) else np.nan
    say(f'\nselected pairs {len(la)}: leaders {len(set(la))}, followers {len(set(fb))}; positive corr '
        f'{np.mean(C[la, fb] > 0):.1%}; in the test half same sign {same_sign:.1%}, mean sign-aligned test corr '
        f'{mean_signed_test:+.4f}; leader more traded than follower (Yahoo $ volume, selection half) {lead_more_traded:.1%}')
    res['selected'] = {'pairs': int(len(la)), 'leaders': len(set(la.tolist())), 'followers': len(set(fb.tolist())),
                       'pos_share': float(np.mean(C[la, fb] > 0)) if len(la) else None,
                       'test_same_sign': float(same_sign), 'test_mean_aligned_corr': float(mean_signed_test),
                       'leader_more_traded': float(lead_more_traded)}

    # funds vs stocks
    say('\nFunds vs stocks (selection half): share of pairs above the threshold, actual vs null mean; mean |corr|')
    cells = {}
    N = T1['N']
    for lname, lm in (('fund', is_fund), ('stock', ~is_fund)):
        for fname, fm in (('stock', ~is_fund), ('fund', is_fund)):
            cm = lm[:, None] & fm[None, :] & np.isfinite(C)
            npairs = int(cm.sum())
            act = float((np.abs(C[cm]) > thr).mean())
            nul = float(np.mean([(np.abs(Ns[cm & np.isfinite(Ns)]) > thr).mean() for Ns in N]))
            mabs = float(np.abs(C[cm]).mean())
            mabs_null = float(np.mean([np.abs(Ns[cm & np.isfinite(Ns)]).mean() for Ns in N]))
            cells[f'{lname}->{fname}'] = {'pairs': npairs, 'above': int((np.abs(C[cm]) > thr).sum()), 'share': act,
                                          'null_share': nul, 'mean_abs': mabs, 'null_mean_abs': mabs_null,
                                          'mean_signed': float(C[cm].mean())}
            say(f'  {lname:>5} -> {fname:<5}: pairs {npairs:7,}; above {int((np.abs(C[cm]) > thr).sum()):5d}; share '
                f'{act:.4%} vs null {nul:.4%} (x{act / nul if nul > 0 else float("nan"):.2f}); mean |corr| {mabs:.4f} vs '
                f'null {mabs_null:.4f}; mean corr {C[cm].mean():+.4f}')
    res['cells'] = cells

    # same-bar correlation (context)
    say('\nSame-bar correlation corr(r_A(k), r_B(k)), selection half, pairs A < B:')
    same = {}
    for name, Z in (('raw', R[:, sel]), ('adjusted', A[:, sel])):
        Cs0, _ = pw_corr(flat(Z), flat(Z))
        iu = np.triu_indices(S, 1)
        v = Cs0[iu]
        v = v[np.isfinite(v)]
        same[name] = {'pairs': int(len(v)), 'mean': float(v.mean()), 'median': float(np.median(v)),
                      'p10': float(np.percentile(v, 10)), 'p90': float(np.percentile(v, 90)),
                      'max': float(v.max())}
        if name == 'raw':
            sv = Cs0[spy][np.arange(S) != spy]
            same['spy_raw_median'] = float(np.nanmedian(sv))
        say(f'  {name}: pairs {len(v):,}; mean {v.mean():+.4f}, median {np.median(v):+.4f}, 10-90% '
            f'{np.percentile(v, 10):+.4f}..{np.percentile(v, 90):+.4f}, max {v.max():+.4f}')
    say(f'  median raw same-bar corr of a symbol with SPY: {same["spy_raw_median"]:+.4f}')
    res['same_bar'] = same

    # robustness variant: per-symbol same-day sd normalization (pre-registered, not graded)
    say('\n-- Robustness (not graded): adjusted returns divided by each symbol\'s own same-day sd')
    with np.errstate(all='ignore'):
        sd = np.nanstd(A[:, sel], axis=2, ddof=1, keepdims=True)
        An = A[:, sel] / sd
    An[~np.isfinite(An)] = np.nan
    T1n = selection_stats(An, valid, shifts, 'day-normalized')
    res['robust_daynorm'] = {k: T1n[k] for k in ('thr', 'count', 'null_count', 'n_pairs')}
    res['robust_daynorm']['number'] = (T1n['count'] / T1n['null_count'] - 1) * 100
    del T1n

    # replication in the test half (context, not graded)
    say('\n-- Context (not graded): the same count in the test half with its own null')
    T1t = selection_stats(A[:, tst], valid, list(range(1, min(MAX_SHIFTS, (D - Dsel) - 1) + 1)), 'test half')
    sel_t = np.isfinite(T1t['C']) & (np.abs(T1t['C']) > T1t['thr'])
    overlap = int((sel_t & selm).sum())
    agree = int((sel_t & selm & (np.sign(T1t['C']) == np.sign(C))).sum())
    exp_overlap = len(la) * sel_t.sum() / max(1, np.isfinite(C).sum())
    say(f'[test half] pairs above threshold in both halves: {overlap} (same sign {agree}); expected by chance {exp_overlap:.2f}')
    res['test_half_count'] = {'thr': T1t['thr'], 'count': T1t['count'], 'null_count': T1t['null_count'],
                              'overlap': overlap, 'overlap_same_sign': agree, 'overlap_expected': float(exp_overlap)}
    del T1t

    # ---------------- Test 2: out of sample
    say('\n-- Test 2: test half, out of sample')
    At, Rt = A[:, tst], R[:, tst]
    Xa = flat(At[:, :, :-1])            # leaders' adjusted r(k)
    tgt_adj = flat(At[:, :, 1:])        # followers' adjusted r(k+1)
    tgt_raw = flat(Rt[:, :, 1:])
    spy_k = flat(Rt[spy][None, :, :-1])[0]
    day_of = np.repeat(np.arange(D - Dsel), Rn - 1)
    Mx = np.isfinite(Xa)
    pred = W.T @ np.where(Mx, Xa, 0.0)
    npres = (np.abs(W).T @ Mx.astype(np.float64))
    pred[npres == 0] = np.nan
    followers = np.abs(W).sum(0) > 0
    # baseline: lag-1 beta of raw r_B(k+1) on raw SPY r(k), selection half
    Rs3 = R[:, sel]
    xk = flat(Rs3[spy][None, :, :-1])[0]
    yk = flat(Rs3[:, :, 1:])
    blag = np.full(S, np.nan)
    for i in range(S):
        ok = np.isfinite(xk) & np.isfinite(yk[i])
        if ok.sum() >= MIN_N:
            x, y = xk[ok], yk[i][ok]
            blag[i] = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    base = blag[:, None] * spy_k[None, :]
    M = Xa.shape[1]
    ic = np.full(M, np.nan)
    ic_raw = np.full(M, np.nan)
    ic_base = np.full(M, np.nan)
    nf = np.zeros(M, int)
    gross = np.full(M, np.nan)
    wprev = np.zeros(S)
    turn = np.zeros(M)
    for j in range(M):
        f = followers & np.isfinite(pred[:, j]) & np.isfinite(tgt_adj[:, j]) & np.isfinite(base[:, j])
        idx = np.nonzero(f)[0]
        w = np.zeros(S)
        if len(idx) >= MIN_FOLLOWERS:
            nf[j] = len(idx)
            p = pred[idx, j]
            ic[j] = spearman(p, tgt_adj[idx, j]) if np.ptp(p) > 0 else 0.0
            ic_raw[j] = spearman(p, tgt_raw[idx, j]) if np.ptp(p) > 0 else 0.0
            bb = base[idx, j]
            ic_base[j] = spearman(bb, tgt_adj[idx, j]) if np.ptp(bb) > 0 else 0.0
            # money: followers with a prediction and a raw return in bar k+1
            g = np.nonzero(followers & np.isfinite(pred[:, j]) & np.isfinite(tgt_raw[:, j]))[0]
            if len(g) >= MIN_FOLLOWERS:
                o = g[np.argsort(pred[g, j], kind='stable')]
                kq = max(1, int(round(0.1 * len(g))))
                lo, hi = o[:kq], o[-kq:]
                gross[j] = tgt_raw[hi, j].mean() - tgt_raw[lo, j].mean()
                w[hi] += 1.0 / kq
                w[lo] -= 1.0 / kq
        # turnover: the day starts flat; the book closes at the day's last slot
        first = (j % (Rn - 1)) == 0
        if first:
            wprev = np.zeros(S)
        turn[j] = np.abs(w - wprev).sum()
        if (j % (Rn - 1)) == Rn - 2:
            turn[j] += np.abs(w).sum()
        wprev = w
    m_ic, t_ic, nb, nd = clustered_t(ic, day_of)
    m_icr, t_icr, _, _ = clustered_t(ic_raw, day_of)
    m_b, t_b, _, _ = clustered_t(ic_base, day_of)
    m_d, t_d, nbd, _ = clustered_t(ic - ic_base, day_of)
    say(f'followers with a selected leader: {int(followers.sum())}; bars with >= {MIN_FOLLOWERS} followers: {nb} on {nd} days; '
        f'median followers per bar {int(np.median(nf[nf > 0])) if (nf > 0).any() else 0}')
    say(f'rank IC (adjusted target): mean {m_ic:+.5f}, day-clustered t {t_ic:+.2f}; raw target: {m_icr:+.5f}, t {t_icr:+.2f}')
    say(f'baseline SPY lag-1 beta IC (same followers): mean {m_b:+.5f}, t {t_b:+.2f}; paired difference {m_d:+.5f}, t {t_d:+.2f}')
    pass2 = bool(np.isfinite(t_ic) and t_ic >= 3 and np.isfinite(t_d) and t_d >= 2)
    say(f'TEST 2: {"PASS" if pass2 else "FAIL"} (needs IC t >= 3 and paired t >= 2)')
    res['test2'] = {'followers': int(followers.sum()), 'bars': nb, 'days': nd, 'ic': m_ic, 't_ic': t_ic, 'ic_raw': m_icr,
                    't_ic_raw': t_icr, 'ic_base': m_b, 't_base': t_b, 'diff': m_d, 't_diff': t_d, 'pass': pass2,
                    'median_followers': int(np.median(nf[nf > 0])) if (nf > 0).any() else 0}

    # ---------------- Test 3: money
    say('\n-- Test 3: money (raw returns, long top tenth / short bottom tenth, one bar)')
    gm, gt, ngb, _ = clustered_t(gross * 1e4, day_of)
    traded = np.isfinite(gross)
    mean_turn = float(turn[traded].sum() / max(1, traded.sum()))
    cost = COST_BP * mean_turn
    net = gm - cost
    be = gm / mean_turn if mean_turn > 0 else float('nan')
    say(f'bars traded {ngb}; gross {gm:+.3f} bp/bar (day-clustered t {gt:+.2f}); mean turnover {mean_turn:.2f} per bar '
        f'(4 = whole book); cost at {COST_BP} bp/side {cost:.3f} bp/bar; net {net:+.3f} bp/bar; break-even cost '
        f'{be:+.3f} bp per side')
    pass3 = bool(np.isfinite(net) and net > 0)
    say(f'TEST 3: {"PASS" if pass3 else "FAIL"} (needs net > 0)')
    res['test3'] = {'bars': ngb, 'gross_bp': gm, 't_gross': gt, 'turnover': mean_turn, 'cost_bp': cost, 'net_bp': net,
                    'breakeven_bp_per_side': be, 'pass': pass3}
    verdict = 'test 1' if not res['test1']['pass'] else ('test 2' if not pass2 else ('test 3' if not pass3 else None))
    res['first_failure'] = verdict
    say(f'\nfirst failure: {verdict}')
    (HERE / 'out' / f'results_{mode}.json').write_text(json.dumps(res, indent=1, default=float))
    (HERE / 'out' / f'results_{mode}.txt').write_text('\n'.join(LINES) + '\n')


if __name__ == '__main__':
    main()
