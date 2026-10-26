"""Hypothesis A of experiments/untested-luca-market-geometry.md: the aggregation ladder (tests A1 and A2).

    ladder.py old    build the ladder on the old export, fit, score 2008-01..2025-06, write the A results
    ladder.py new    build the ladder on the new export only (unseen_a.py scores and reports the unseen months)

The rules are the plan's ("A. The aggregation ladder"); every choice the plan leaves open is listed, with its timing,
under "Decisions" in research/market_geometry/results_A.md. Point in time throughout: at decision k (close t0 on the
twins' fixed calendar, export day 252 + 21k) the levels, signals and volatilities use closes through t0 only; the
outcome runs from the close of t0+1 to the close of t0+1+h (one skipped day); a forecaster fitted at k uses only
pairs whose outcomes ended at or before t0.

Stages: build() -> rows_<tag>.npz (one row per level member per decision date, scaled signals and outcomes);
score() -> per-month loss contributions per level and forecaster; report() -> the tests and tables.
numpy only (system python3). Run under the memory-capped scope (see results_A.md).
"""
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '3')

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data')
import c2eval  # noqa: E402
import pubdata  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = c2eval.WORK / 'market_geometry' / 'a'
WORK.mkdir(parents=True, exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}

# ---- the plan's settings ----------------------------------------------------------------------------------------
STEP = 21                 # decision calendar: every 21 trading days (tw['rebal'])
N_UNIV = 500              # the 500 most-traded stocks, minus c2eval.excluded
UNIV_WIN = 63             # universe: median dollar volume and >= 90% returns over the last 63 days (as states())
MIN_FRAC = 0.9
BASKETS = (5, 25, 100)    # random equal-weight baskets, 1,000 of each per period
N_BASKET = 1000
MIN_GROUP = 5             # groups: stocks sharing a twin fund, at least 5
N_MATCH = 20              # matched-basket replicates per group per period (decision: see results_A.md)
GRID_Q = 5                # 5 x 5 grid of trailing-252-day market beta and volatility
GRID_WIN = 252
VOL_WIN = 63              # members' signals and outcomes are divided by their own trailing 63-day volatility
HORIZONS = (5, 21)
SKIP = 1                  # one day skipped between the decision close and the start of the move
TREND_A, TREND_B = 252, 21  # trend: log return from close t0-252 to close t0-21 (12 months skipping the last)
REV_LEN = 5               # reversal: minus the last 5-day return
FFILL_LIMIT = 5           # closes forward-filled over gaps of at most 5 days
SCORE_FROM, SCORE_TO = '2008-01-01', '2019-12-31'   # the test window (decision dates)
EVAL_FROM, EVAL_TO = '2020-01-01', '2025-06-30'     # the chronological evaluation period, reported
UNSEEN_FROM = '2025-06-30'
N_BOOT = 10_000
BLOCK = 6                 # bootstrap blocks of 6 months (decision dates)
NW_LAGS = 2
P_BAR = 0.0013            # one-sided p the plan's t >= 3 implies
T_BAR = 3.0
SEED = 20261023

# Window columns: column j holds the daily log return of export day t0 - 251 + j (j = 251 is day t0).
J0 = 251
NCOL = J0 + 1 + SKIP + max(HORIZONS) + 1          # through day t0 + 1 + 21 + ... -> 274 columns (last j = 273)

LEVELS = ['stock', 'b5', 'b25', 'b100', 'group', 'matched', 'fund', 'market', 'spy']
LV = {n: i for i, n in enumerate(LEVELS)}
FEATS = ['x_trend', 'x_rev', 'x_ar5', 'x_ar21', 'y5', 'y21', 'y5n', 'y21n', 'l21', 'l63', 'lf5', 'lf21']
FI = {n: i for i, n in enumerate(FEATS)}


# ---- data ---------------------------------------------------------------------------------------------------------
def ffill(C, limit=FFILL_LIMIT):
    """Forward-fill each row's closes over gaps of at most `limit` days (NaN beyond, and before the first close)."""
    N, T = C.shape
    out = np.full(C.shape, np.nan)
    last = np.full(N, np.nan)
    age = np.full(N, 10 ** 6)
    for d in range(T):
        v = C[:, d]
        ok = np.isfinite(v)
        last = np.where(ok, v, last)
        age = np.where(ok, 0, age + 1)
        out[:, d] = np.where(age <= limit, last, np.nan)
    return out


def log_returns(Cf):
    r = np.full(Cf.shape, np.nan, np.float32)
    with np.errstate(all='ignore'):
        r[:, 1:] = np.log(Cf[:, 1:] / Cf[:, :-1])
    r[~np.isfinite(r)] = np.nan
    return r


def liquid_count(C, DV, T):
    """Reconstruction of the export's `market` universe (crates/c2-census/src/variable.rs): a stock is liquid on rows
    d .. d+20 (d = 252, 273, ...) if, over rows d-252 .. d-1, it has >= 126 finite closes, median close >= $5 and
    median dollar volume >= $5M. Returns (liquid [N, T] bool, count of liquid stocks with a raw daily return [T],
    the reconstructed equal-weight mean log return [T])."""
    N = C.shape[0]
    liq = np.zeros((N, T), bool)
    d = 252
    while d < T:
        c = C[:, d - 252:d]
        v = DV[:, d - 252:d]
        fin = np.isfinite(c)
        with np.errstate(all='ignore'):
            mc = np.nanmedian(c, 1)
            mv = np.nanmedian(np.where(fin, v, np.nan), 1)
        ok = (fin.sum(1) >= 126) & (np.nan_to_num(mc) >= 5.0) & (np.nan_to_num(mv) >= 5e6)
        liq[:, d:d + 21] = ok[:, None]
        d += 21
    with np.errstate(all='ignore'):
        rr = np.full(C.shape, np.nan)
        rr[:, 1:] = np.log(C[:, 1:] / C[:, :-1])
    rr[~np.isfinite(rr)] = np.nan
    use = liq & np.isfinite(rr)
    cnt = use.sum(0)
    with np.errstate(all='ignore'):
        mkt = np.where(use, rr, 0).sum(0) / cnt
    return liq, cnt, mkt


def spy_yahoo(days):
    """SPY's Yahoo adjusted close on the export's days, forward-filled over gaps of at most 5 days."""
    d, adj = pubdata.etf_adjclose(('SPY',))
    s = pubdata.on_days(d, adj['SPY'], days)
    return ffill(s[None, :])[0]


def load(tag):
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume', 'market'))
    tw = c2eval.twins(E)
    excl = c2eval.excluded(E, tw)
    C = np.array(E['close'], np.float64)
    DV = np.array(E['dollar_volume'], np.float64)
    T = C.shape[1]
    R = log_returns(ffill(C))
    liq, cnt, mkt_rec = liquid_count(C, DV, T)
    del liq
    F = tw['F']
    FR = log_returns(ffill(np.asarray(F['close'], np.float64)))
    FR[np.abs(FR) > 0.5] = np.nan                     # unrepaired fund prices: c2eval's +-50% rule
    market = np.asarray(E['market'], np.float64)
    spy = spy_yahoo(E['days'])
    spy_r = log_returns(spy[None, :])[0]
    return dict(E=E, tw=tw, excl=excl, C=C, DV=DV, R=R, FR=FR, market=market, spy_r=spy_r, cnt=cnt,
                mkt_rec=mkt_rec, T=T)


# ---- per-member features ------------------------------------------------------------------------------------------
def window(X, t0, T):
    """Columns t0-251 .. t0+22 of a [m, T] daily-return array (NaN beyond the data)."""
    a, b = t0 - J0, t0 - J0 + NCOL
    out = np.full((X.shape[0], NCOL), np.nan, np.float32)
    lo, hi = max(a, 0), min(b, T)
    out[:, lo - a:hi - a] = X[:, lo:hi]
    return out


def _sum(W, a, b):
    X = W[:, a:b + 1]
    ok = np.isfinite(X)
    s = np.where(ok, X, 0.0).sum(1, dtype=np.float64)
    return np.where(ok.sum(1) >= MIN_FRAC * (b - a + 1), s, np.nan)


def _rms(W, a, b):
    X = W[:, a:b + 1]
    ok = np.isfinite(X)
    n = ok.sum(1)
    with np.errstate(all='ignore'):
        s = np.sqrt(np.where(ok, X.astype(np.float64) ** 2, 0.0).sum(1) / n)
    return np.where(n >= MIN_FRAC * (b - a + 1), s, np.nan)


def feats(W):
    """[m, NCOL] daily log returns -> [m, len(FEATS)] float32 scaled signals and outcomes (NaN where unavailable).
    Signals use columns <= J0 (closes through t0); outcomes start after the skipped day t0+1 (y5n/y21n: no skip,
    the reported upper bound)."""
    j = J0
    trend = _sum(W, j - TREND_A + 1, j - TREND_B)              # days t0-251 .. t0-21
    last5 = _sum(W, j - REV_LEN + 1, j)
    last21 = _sum(W, j - 21 + 1, j)
    s63 = _rms(W, j - VOL_WIN + 1, j)
    s21 = _rms(W, j - 21 + 1, j)
    out = {}
    with np.errstate(all='ignore'):
        out['x_trend'] = trend / (s63 * math.sqrt(TREND_A - TREND_B))
        out['x_rev'] = -last5 / (s63 * math.sqrt(REV_LEN))
        out['x_ar5'] = last5 / (s63 * math.sqrt(5))
        out['x_ar21'] = last21 / (s63 * math.sqrt(21))
        for h in HORIZONS:
            out[f'y{h}'] = _sum(W, j + 1 + SKIP, j + SKIP + h) / (s63 * math.sqrt(h))   # days t0+2 .. t0+1+h
            out[f'y{h}n'] = _sum(W, j + 1, j + h) / (s63 * math.sqrt(h))               # days t0+1 .. t0+h
            out[f'lf{h}'] = np.log(_rms(W, j + 1 + SKIP, j + SKIP + h))
        out['l21'] = np.log(s21)
        out['l63'] = np.log(s63)
    M = np.stack([out[f] for f in FEATS], 1).astype(np.float32)
    M[~np.isfinite(M)] = np.nan
    return M


def basket_returns(Wm, members):
    """Equal-weight daily mean of members' log returns (as the export's market), [nb, NCOL]; NaN on days when fewer
    than half the members have a return. members: [nb, s] row indices into Wm."""
    nb, s = members.shape
    n = Wm.shape[0]
    Mm = np.zeros((nb, n), np.float32)
    np.put_along_axis(Mm, members, 1.0, axis=1)
    ok = np.isfinite(Wm)
    num = Mm @ np.where(ok, Wm, 0.0).astype(np.float32)
    den = Mm @ ok.astype(np.float32)
    with np.errstate(all='ignore'):
        B = num / den
    B[den < 0.5 * s] = np.nan
    return B


def basket_returns_lists(Wm, lists):
    """Same as basket_returns for baskets of different sizes (list of index arrays)."""
    n = Wm.shape[0]
    Mm = np.zeros((len(lists), n), np.float32)
    sizes = np.zeros(len(lists))
    for i, L in enumerate(lists):
        Mm[i, L] = 1.0
        sizes[i] = len(L)
    ok = np.isfinite(Wm)
    num = Mm @ np.where(ok, Wm, 0.0).astype(np.float32)
    den = Mm @ ok.astype(np.float32)
    with np.errstate(all='ignore'):
        B = num / den
    B[den < 0.5 * sizes[:, None]] = np.nan
    return B


# ---- the levels at one decision date ----------------------------------------------------------------------------
def universe(D, k, t0):
    """The 500 most-traded eligible stocks at t0, exactly as research/market_timing/common.py states() picks them."""
    C, DV = D['C'], D['DV']
    c = C[:, t0 - UNIV_WIN:t0 + 1]
    with np.errstate(all='ignore'):
        r = np.log(c[:, 1:] / c[:, :-1])
        r[~np.isfinite(r)] = np.nan
        mdv = np.nanmedian(DV[:, t0 - UNIV_WIN + 1:t0 + 1], 1)
        sd = np.nanstd(r, 1)
    ok = (np.isfinite(r).mean(1) >= MIN_FRAC) & (np.nan_to_num(mdv) > 0) & ~D['excl'][k] & (np.nan_to_num(sd) > 0)
    order = np.argsort(-np.where(ok, mdv, -np.inf), kind='stable')
    return order[:min(N_UNIV, int(ok.sum()))]


def grid_cells(Wu, Wmkt):
    """5 x 5 cell (0..24) of each universe stock's trailing-252-day beta to the export's market and volatility
    (days t0-251 .. t0; at least 126 common finite days), -1 where unavailable. Quintile cuts over the universe."""
    X = Wu[:, :J0 + 1].astype(np.float64)
    m = Wmkt[0, :J0 + 1].astype(np.float64)
    ok = np.isfinite(X) & np.isfinite(m)[None, :]
    n = ok.sum(1)
    Xz = np.where(ok, X, 0.0)
    mz = np.where(ok, m[None, :], 0.0)
    with np.errstate(all='ignore'):
        mx, mm = Xz.sum(1) / n, mz.sum(1) / n
        cov = (Xz * mz).sum(1) / n - mx * mm
        vm = (mz * mz).sum(1) / n - mm * mm
        beta = cov / vm
        vol = np.sqrt((Xz * Xz).sum(1) / n - mx * mx)
    valid = (n >= 126) & np.isfinite(beta) & np.isfinite(vol)
    cell = np.full(len(X), -1)
    if valid.sum() >= 2 * GRID_Q:
        qb = np.quantile(beta[valid], np.linspace(0, 1, GRID_Q + 1)[1:-1])
        qv = np.quantile(vol[valid], np.linspace(0, 1, GRID_Q + 1)[1:-1])
        cb = np.searchsorted(qb, beta, side='right')
        cv = np.searchsorted(qv, vol, side='right')
        cell = np.where(valid, cb * GRID_Q + cv, -1)
    return cell


def matched_members(rng, group_idx, cell, n, stats):
    """One matched basket for a group: for each member, a non-member of the universe from the same grid cell, without
    repeats while the cell has unused non-members (then with repeats); a member or cell without a usable match falls
    back to the nearest cell with non-members (Chebyshev, then Manhattan distance on the grid), and a member with
    no cell to any non-member."""
    inside = np.zeros(n, bool)
    inside[group_idx] = True
    cand_all = np.nonzero(~inside & (cell >= 0))[0]
    if len(cand_all) == 0:                      # no grid (the market series is too short) or no valid non-member
        cand_all = np.nonzero(~inside)[0]
        if len(cand_all) == 0:
            stats['no_nonmember_groups'] += 1
            return None
        stats['no_grid_members'] += len(group_idx)
        return rng.choice(cand_all, len(group_idx), replace=len(cand_all) < len(group_idx))
    by_cell = {c: list(cand_all[cell[cand_all] == c]) for c in range(GRID_Q * GRID_Q)}
    used = set()
    out = []
    for i in group_idx:
        c = cell[i]
        if c < 0:
            stats['no_cell'] += 1
            out.append(int(rng.choice(cand_all)))
            continue
        pool = by_cell[c]
        if not pool:
            stats['nearest_cell'] += 1
            bi, vi = divmod(c, GRID_Q)
            best = None
            for cc in range(GRID_Q * GRID_Q):
                if by_cell[cc]:
                    bj, vj = divmod(cc, GRID_Q)
                    key = (max(abs(bi - bj), abs(vi - vj)), abs(bi - bj) + abs(vi - vj), cc)
                    best = key if best is None or key < best else best
            pool = by_cell[best[2]]
        fresh = [p for p in pool if p not in used]
        if fresh:
            pick = int(fresh[rng.integers(len(fresh))])
        else:
            stats['repeat'] += 1
            pick = int(pool[rng.integers(len(pool))])
        used.add(pick)
        out.append(pick)
    return np.array(out)


def build(tag, verbose=True):
    """One row per level member per decision date: level, replicate, size, fund row, k and the scaled features."""
    t_start = time.time()
    D = load(tag)
    E, tw = D['E'], D['tw']
    T = D['T']
    rebal = np.asarray(tw['rebal'])
    assert np.all(np.diff(rebal) == STEP) and rebal[0] == 252
    Wmkt_all = D['market'][None, :]
    Wspy_all = D['spy_r'][None, :]
    rows = {'lvl': [], 'rep': [], 'size': [], 'frow': [], 'k': [], 'X': []}
    meta = {'n_univ': [], 'n_groups': [], 'group_sizes': [],
            'match': {'no_cell': 0, 'nearest_cell': 0, 'repeat': 0, 'members': 0, 'no_grid_members': 0,
                      'no_nonmember_groups': 0}}

    def add(lvl, rep, size, frow, k, X):
        m = X.shape[0]
        rows['lvl'].append(np.full(m, LV[lvl], np.int8))
        rows['rep'].append(np.full(m, rep, np.int8))
        rows['size'].append(np.broadcast_to(np.asarray(size, np.int16), (m,)).copy())
        rows['frow'].append(np.broadcast_to(np.asarray(frow, np.int32), (m,)).copy())
        rows['k'].append(np.full(m, k, np.int16))
        rows['X'].append(X)

    for k, t0 in enumerate(rebal):
        t0 = int(t0)
        sel = universe(D, k, t0)
        n = len(sel)
        meta['n_univ'].append(n)
        Wu = window(D['R'][sel], t0, T)
        Wm = window(Wmkt_all, t0, T)
        rng = np.random.default_rng([SEED, k])
        # 1. single stocks
        add('stock', 0, 1, -1, k, feats(Wu))
        # 2. random equal-weight baskets, drawn uniformly (without repeats inside a basket)
        for s in BASKETS:
            if n < s:
                continue
            mem = np.argsort(rng.random((N_BASKET, n)), 1)[:, :s]
            add(f'b{s}', 0, s, -1, k, feats(basket_returns(Wu, mem)))
        # 3. groups: universe stocks sharing a twin fund, at least 5
        fr = tw['fund_row'][k][sel]
        funds_k, counts = np.unique(fr[fr >= 0], return_counts=True)
        gfunds = funds_k[counts >= MIN_GROUP]
        glists = [np.nonzero(fr == f)[0] for f in gfunds]
        meta['n_groups'].append(len(glists))
        meta['group_sizes'].append([len(g) for g in glists])
        if glists:
            gs = np.array([len(g) for g in glists])
            add('group', 0, gs, gfunds, k, feats(basket_returns_lists(Wu, glists)))
            # 4. matched random baskets: N_MATCH replicates, each one matched basket per group
            cell = grid_cells(Wu, Wm)
            c0 = dict(meta['match'])
            for r in range(N_MATCH):
                mlists = [matched_members(rng, g, cell, n, meta['match']) for g in glists]
                keep = [i for i, ml in enumerate(mlists) if ml is not None]
                meta['match']['members'] += int(gs[keep].sum())
                if keep:
                    add('matched', r, gs[keep], gfunds[keep], k,
                        feats(basket_returns_lists(Wu, [mlists[i] for i in keep])))
            meta.setdefault('match_by_k', {})[k] = {key: meta['match'][key] - c0[key] for key in c0}
            # 5. the groups' twin funds, on their own prices
            add('fund', 0, gs, gfunds, k, feats(window(D['FR'][gfunds], t0, T)))
        # 6. the export's equal-weight market, and SPY
        add('market', 0, int(D['cnt'][t0]), -1, k, feats(Wm))
        add('spy', 0, 500, -1, k, feats(window(Wspy_all, t0, T)))
        if verbose and k % 20 == 0:
            print(f'  {c2eval.iso(E["days"][t0])} k {k} univ {n} groups {len(glists)} '
                  f'({time.time() - t_start:.0f}s)', flush=True)
    out = {key: np.concatenate(v) for key, v in rows.items()}
    out['rebal'] = rebal
    out['days'] = np.asarray(E['days'])
    out['cnt'] = D['cnt']
    out['fund_ticker'] = np.asarray(tw['F']['tickers'])
    np.savez(WORK / f'rows_{tag}.npz', **out)
    meta['seconds'] = time.time() - t_start
    # check: the reconstructed market universe against the export's market series
    m, rec = D['market'], D['mkt_rec']
    ok = np.isfinite(m) & np.isfinite(rec)
    meta['market_check'] = {'days': int(ok.sum()), 'corr': float(np.corrcoef(m[ok], rec[ok])[0, 1]),
                            'max_abs_diff': float(np.abs(m[ok] - rec[ok]).max()),
                            'median_abs_diff': float(np.median(np.abs(m[ok] - rec[ok])))}
    # check: Yahoo SPY against the store's SPY over their overlap (daily log returns)
    S = c2eval.load_store(c2eval.STORE_FOR[str(Path(E['path']))])
    sid = [s['id'] for s in S['syms'] if s['ticker'] == 'SPY' and s['kind'] == 'etf']
    sc = c2eval.store_panel(S, sid, E['days'], 'close')[0].astype(np.float64)
    sr = log_returns(ffill(sc[None, :]))[0]
    ok = np.isfinite(sr) & np.isfinite(D['spy_r'])
    meta['spy_check'] = {'days': int(ok.sum()), 'first_store_day': c2eval.iso(E['days'][int(np.argmax(np.isfinite(sc)))]),
                         'corr': float(np.corrcoef(sr[ok], D['spy_r'][ok])[0, 1]),
                         'mean_diff_bps_per_day': float((D['spy_r'][ok] - sr[ok]).mean() * 1e4)}
    (WORK / f'build_{tag}.json').write_text(json.dumps(meta))
    return out, meta


# ---- fitting and scoring ------------------------------------------------------------------------------------------
RET_FORECASTERS = [(sig, h) for sig in ('trend', 'reversal', 'ar1') for h in HORIZONS]


def sig_col(sig, h):
    return {'trend': 'x_trend', 'reversal': 'x_rev', 'ar1': f'x_ar{h}'}[sig]


def units():
    """Fitting units: each level is one unit; each matched replicate is its own unit (its own forecaster)."""
    u = [(l, 0) for l in LEVELS if l != 'matched']
    u += [('matched', r) for r in range(N_MATCH)]
    return u


def _lag(rebal, end_offset):
    """Months back to the latest decision whose outcome (ending end_offset days after it) ended at or before t0."""
    L = int(math.ceil(end_offset / STEP))
    assert np.all(np.diff(rebal) == STEP) and end_offset <= L * STEP and end_offset > (L - 1) * STEP
    return L


def score_return(rows, K, lvl, rep, xcol, ycol, end_offset, rebal, ranks=True):
    """Expanding least squares y = a + b x on the unit's pooled pairs whose outcomes ended at or before t0, and the
    per-month loss contributions of its forecast and of the historical mean. Returns a dict of [K] arrays."""
    m = (rows['lvl'] == LV[lvl]) & (rows['rep'] == rep)
    x = rows['X'][m, FI[xcol]].astype(np.float64)
    y = rows['X'][m, FI[ycol]].astype(np.float64)
    k = rows['k'][m].astype(int)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, k = x[ok], y[ok], k[ok]
    bc = lambda w: np.bincount(k, weights=w, minlength=K)  # noqa: E731
    n, sx, sy, sxx, sxy, syy = bc(np.ones_like(x)), bc(x), bc(y), bc(x * x), bc(x * y), bc(y * y)
    L = _lag(rebal, end_offset)
    cum = lambda v: np.concatenate([np.zeros(L), np.cumsum(v)[:-L]]) if L < K else np.zeros(K)  # noqa: E731
    Cn, Cx, Cy, Cxx, Cxy = cum(n), cum(sx), cum(sy), cum(sxx), cum(sxy)
    with np.errstate(all='ignore'):
        vx = Cxx - Cx * Cx / Cn
        b = np.where((Cn >= 3) & (vx > 0), (Cxy - Cx * Cy / Cn) / vx, np.nan)
        a = Cy / Cn - b * Cx / Cn
        ybar = Cy / Cn
    sse = syy - 2 * a * sy - 2 * b * sxy + n * a * a + 2 * a * b * sx + b * b * sxx
    sst = syy - 2 * ybar * sy + n * ybar * ybar
    d = ybar - a
    cw = sst - sse + (n * d * d - 2 * d * b * sx + b * b * sxx)
    # direction: the raw signal's sign, and the calibrated forecast's sign, against the outcome minus the hist. mean
    yb_row, a_row, b_row = ybar[k], a[k], b[k]
    dev = y - yb_row
    fin = np.isfinite(dev) & (dev != 0)
    hit_sig = bc(((np.sign(x) == np.sign(dev)) & fin).astype(float))
    fdev = a_row + b_row * x - yb_row
    fin2 = fin & np.isfinite(fdev) & (fdev != 0)
    hit_mod = bc(((np.sign(fdev) == np.sign(dev)) & fin2).astype(float))
    n_hit, n_hit_mod = bc(fin.astype(float)), bc(fin2.astype(float))
    # rank correlation across the unit's members each period (needs >= 5 members)
    rank = np.full(K, np.nan)
    if ranks and lvl not in ('market', 'spy'):
        order = np.argsort(k, kind='stable')
        ks, xs, ys = k[order], x[order], y[order]
        cuts = np.searchsorted(ks, np.arange(K + 1))
        for kk in range(K):
            a0, b0 = cuts[kk], cuts[kk + 1]
            if b0 - a0 >= 5:
                rx = np.argsort(np.argsort(xs[a0:b0])).astype(float)
                ry = np.argsort(np.argsort(ys[a0:b0])).astype(float)
                rank[kk] = np.corrcoef(rx, ry)[0, 1]
    return dict(n=n, sse=sse, sst=sst, ss0=syy, cw=cw, a=a, b=b, ybar=ybar, ntrain=Cn, hit_sig=hit_sig,
                hit_mod=hit_mod, n_hit=n_hit, n_hit_mod=n_hit_mod, rank=rank)


def score_vol(rows, K, lvl, rep, h, scaled, rebal):
    """Two-term HAR: y = c + b1 log rv21 + b2 log rv63, expanding least squares on the unit's past pairs. Outcome:
    log realized volatility over days t0+2 .. t0+1+h, minus log of the member's trailing 63-day volatility when
    scaled (the plan's division by own trailing volatility); undivided when not."""
    m = (rows['lvl'] == LV[lvl]) & (rows['rep'] == rep)
    X3 = np.stack([np.ones(m.sum()), rows['X'][m, FI['l21']], rows['X'][m, FI['l63']]], 1).astype(np.float64)
    y = rows['X'][m, FI[f'lf{h}']].astype(np.float64)
    if scaled:
        y = y - X3[:, 2]
    k = rows['k'][m].astype(int)
    ok = np.isfinite(X3).all(1) & np.isfinite(y)
    X3, y, k = X3[ok], y[ok], k[ok]
    XX = np.zeros((K, 3, 3))
    Xy = np.zeros((K, 3))
    np.add.at(XX, k, X3[:, :, None] * X3[:, None, :])
    np.add.at(Xy, k, X3 * y[:, None])
    syy = np.bincount(k, weights=y * y, minlength=K)
    sy = Xy[:, 0]
    n = XX[:, 0, 0]
    L = _lag(rebal, SKIP + h)
    cXX = np.concatenate([np.zeros((L, 3, 3)), np.cumsum(XX, 0)[:-L]])
    cXy = np.concatenate([np.zeros((L, 3)), np.cumsum(Xy, 0)[:-L]])
    beta = np.full((K, 3), np.nan)
    for kk in range(K):
        if cXX[kk, 0, 0] >= 10:
            beta[kk] = np.linalg.lstsq(cXX[kk], cXy[kk], rcond=None)[0]
    with np.errstate(all='ignore'):
        ybar = cXy[:, 0] / cXX[:, 0, 0]
    sse = syy - 2 * (beta * Xy).sum(1) + np.einsum('ki,kij,kj->k', beta, XX, beta)
    sst = syy - 2 * ybar * sy + n * ybar * ybar
    sumf = (beta * XX[:, 0, :]).sum(1)                     # sum of forecasts
    sumf2 = np.einsum('ki,kij,kj->k', beta, XX, beta)     # sum of squared forecasts
    cw = sst - sse + (n * ybar * ybar - 2 * ybar * sumf + sumf2)
    return dict(n=n, sse=sse, sst=sst, ss0=syy, cw=cw, ybar=ybar, ntrain=cXX[:, 0, 0], beta=beta)


def score(rows):
    """Every unit x forecaster. Keys: ('ret', sig, h, variant) with variant 'skip' (the test) or 'noskip' (the
    upper bound), and ('vol', h, 'scaled'|'raw')."""
    rebal = rows['rebal']
    K = len(rebal)
    out = {}
    for sig, h in RET_FORECASTERS:
        for variant, ycol, off in (('skip', f'y{h}', SKIP + h), ('noskip', f'y{h}n', h)):
            for lvl, rep in units():
                out[('ret', sig, h, variant, lvl, rep)] = score_return(rows, K, lvl, rep, sig_col(sig, h), ycol, off, rebal,
                                                                          ranks=(variant == 'skip'))
    for h in HORIZONS:
        for scaled in (True, False):
            for lvl, rep in units():
                out[('vol', h, 'scaled' if scaled else 'raw', lvl, rep)] = score_vol(rows, K, lvl, rep, h, scaled, rebal)
    return out


# ---- statistics ---------------------------------------------------------------------------------------------------
def nw_t(f, lags=NW_LAGS):
    f = np.asarray(f, float)
    f = f[np.isfinite(f)]
    n = len(f)
    if n < 5:
        return float('nan'), float('nan'), float('nan')
    mu = f.mean()
    e = f - mu
    s = e @ e / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    se = math.sqrt(max(s, 1e-300) / n)
    return float(mu / se), float(mu), float(se)


def p_norm(t):
    """One-sided p for a positive effect from a normal t."""
    return float(0.5 * math.erfc(t / math.sqrt(2))) if np.isfinite(t) else float('nan')


def boot_index(M, seed, n=N_BOOT, block=BLOCK):
    """Circular moving-block bootstrap of M decision dates: [n, M] indices."""
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(M / block))
    starts = rng.integers(0, M, (n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % M
    return idx.reshape(n, nb * block)[:, :M]


def r2(sse, sst):
    return 1.0 - sse.sum(-1) / sst.sum(-1)


def slope_on(xs, ys):
    """OLS slope of ys [..., L] on xs [L]."""
    xc = xs - xs.mean()
    return (ys * xc).sum(-1) / (xc @ xc) if ys.ndim == 1 else (ys - ys.mean(-1, keepdims=True)) @ xc / (xc @ xc)


def delta_slope_t(SSE, SST, xs):
    """Newey-West t of the slope of R2 on xs by the delta method: SSE, SST [L, M] per-month contributions. The
    per-month influence terms sum to zero by construction, so they give the standard error only; the t is the
    point slope over that standard error."""
    M = SSE.shape[1]
    A, B = SSE.sum(1), SST.sum(1)
    u = -(SSE - (A / B)[:, None] * SST) / (B / M)[:, None]   # per-month influence on each R2
    xc = xs - xs.mean()
    s = (xc / (xc @ xc)) @ u
    se = nw_t(s + 1.0)[2]                                     # shift: nw_t's se doesn't depend on the mean
    return float(slope_on(xs, 1 - A / B) / se)


def delta_diff_t(SSE, SST):
    """NW t of R2[0] - R2[1] by the delta method (SSE, SST [2, M]): the point difference over its NW se."""
    M = SSE.shape[1]
    A, B = SSE.sum(1), SST.sum(1)
    u = -(SSE - (A / B)[:, None] * SST) / (B / M)[:, None]
    se = nw_t(u[0] - u[1] + 1.0)[2]
    R = 1 - A / B
    return float((R[0] - R[1]) / se)


def pooled(res, key_prefix, lvl):
    """Per-month contributions for a level (matched replicates summed)."""
    if lvl == 'matched':
        parts = [res[key_prefix + ('matched', r)] for r in range(N_MATCH)]
        return {f: sum(p[f] for p in parts) for f in ('n', 'sse', 'sst', 'ss0', 'cw', 'hit_sig', 'hit_mod', 'n_hit',
                                                          'n_hit_mod') if f in parts[0]} | \
            ({'rank': np.nanmean(np.stack([p['rank'] for p in parts]), 0)} if 'rank' in parts[0] else {}) | \
            {'ntrain': parts[0]['ntrain']}
    return res[key_prefix + (lvl, 0)]


SLOPE_LEVELS = ['stock', 'b5', 'b25', 'b100', 'market']
REPORT_LEVELS = ['stock', 'b5', 'b25', 'b100', 'group', 'matched', 'fund', 'market', 'spy']


def level_sizes(rows, mk):
    """Mean member count per level over the months mk (bool [K])."""
    out = {}
    for l in REPORT_LEVELS:
        m = (rows['lvl'] == LV[l]) & (rows['rep'] == 0) & mk[rows['k'].astype(int)]
        out[l] = float(rows['size'][m].mean()) if m.any() else float('nan')
    return out


def period_stats(rows, res, mk, seed, boot=True):
    """All the A statistics over the decision months mk (bool [K])."""
    sizes = level_sizes(rows, mk)
    xs = np.log(np.array([sizes[l] for l in SLOPE_LEVELS]))
    M = int(mk.sum())
    idx = boot_index(M, seed) if boot and M >= 2 * BLOCK else None
    out = {'months': M, 'sizes': sizes, 'log_sizes_slope': xs.tolist(), 'ret': {}, 'vol': {}}

    def one(prefix, is_ret):
        rec = {'levels': {}}
        SSE, SST = {}, {}
        for l in REPORT_LEVELS:
            p = pooled(res, prefix, l)
            sse, sst, ss0 = p['sse'][mk], p['sst'][mk], p['ss0'][mk]
            good = np.isfinite(sse) & np.isfinite(sst)
            sse_g, sst_g = np.where(good, sse, 0.0), np.where(good, sst, 0.0)
            SSE[l], SST[l] = sse_g, sst_g
            L = {'n_pairs': int(p['n'][mk][good].sum()), 'months_scored': int(good.sum()),
                 'r2_hist': float(r2(sse_g, sst_g)) if good.any() else float('nan'),
                 'r2_zero': float(1 - sse_g.sum() / np.where(good, ss0, 0).sum()) if good.any() else float('nan')}
            cwt, cwm, _ = nw_t(np.where(good, p['cw'][mk], np.nan))
            L['cw_t'], L['cw_p'] = cwt, p_norm(cwt)
            if is_ret:
                nh, nhm = p['n_hit'][mk].sum(), p['n_hit_mod'][mk].sum()
                L['hit_signal'] = float(p['hit_sig'][mk].sum() / nh) if nh else float('nan')
                L['hit_model'] = float(p['hit_mod'][mk].sum() / nhm) if nhm else float('nan')
                with np.errstate(all='ignore'):
                    hm = p['hit_sig'][mk] / p['n_hit'][mk] - 0.5
                L['hit_signal_t'] = nw_t(hm)[0]
                rk = p['rank'][mk]
                L['rank_mean'] = float(np.nanmean(rk)) if np.isfinite(rk).any() else float('nan')
                L['rank_t'] = nw_t(rk)[0] if np.isfinite(rk).sum() >= 5 else float('nan')
                L['months_with_rank'] = int(np.isfinite(rk).sum())
            if idx is not None and good.all():
                bs = 1 - SSE[l][idx].sum(1) / SST[l][idx].sum(1)
                L['r2_hist_ci90'] = [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
                cwb = np.where(good, p['cw'][mk], 0.0)[idx].mean(1)
                L['cw_boot_p'] = float((1 + (cwb <= 0).sum()) / (1 + len(cwb)))
            rec['levels'][l] = L
        ys = np.array([rec['levels'][l]['r2_hist'] for l in SLOPE_LEVELS])
        rec['slope'] = float(slope_on(xs, ys))
        Sse = np.stack([SSE[l] for l in SLOPE_LEVELS])
        Sst = np.stack([SST[l] for l in SLOPE_LEVELS])
        rec['slope_nw_t'] = delta_slope_t(Sse, Sst, xs)
        rec['slope4'] = float(slope_on(xs[:4], ys[:4]))
        rec['slope4_nw_t'] = delta_slope_t(Sse[:4], Sst[:4], xs[:4])
        dg = rec['levels']['group']['r2_hist'] - rec['levels']['matched']['r2_hist']
        rec['a2_diff'] = float(dg)
        rec['a2_nw_t'] = delta_diff_t(np.stack([SSE['group'], SSE['matched']]), np.stack([SST['group'], SST['matched']]))
        if idx is not None:
            bR = 1 - Sse[:, idx].sum(2) / Sst[:, idx].sum(2)        # [L, n]
            bsl = slope_on(xs, bR.T)
            rec['slope_ci90'] = [float(np.percentile(bsl, 5)), float(np.percentile(bsl, 95))]
            rec['slope_boot_p'] = float((1 + (bsl <= 0).sum()) / (1 + len(bsl)))
            rec['slope_boot_se'] = float(bsl.std(ddof=1))
            rec['slope_boot_t'] = float(rec['slope'] / bsl.std(ddof=1))
            bg = 1 - SSE['group'][idx].sum(1) / SST['group'][idx].sum(1)
            bm = 1 - SSE['matched'][idx].sum(1) / SST['matched'][idx].sum(1)
            rec['a2_ci90'] = [float(np.percentile(bg - bm, 5)), float(np.percentile(bg - bm, 95))]
            rec['a2_boot_p'] = float((1 + ((bg - bm) <= 0).sum()) / (1 + N_BOOT))
            bsl4 = slope_on(xs[:4], bR[:4].T)
            rec['slope4_ci90'] = [float(np.percentile(bsl4, 5)), float(np.percentile(bsl4, 95))]
            rec['slope4_boot_p'] = float((1 + (bsl4 <= 0).sum()) / (1 + len(bsl4)))
            rec['boot_slopes_hist'] = np.histogram(bsl, bins=40)[0].tolist()
            rec['boot_slopes_edges'] = np.histogram(bsl, bins=40)[1].tolist()
        return rec

    for sig, h in RET_FORECASTERS:
        for variant in ('skip', 'noskip'):
            out['ret'][f'{sig}_h{h}_{variant}'] = one(('ret', sig, h, variant), True)
    for h in HORIZONS:
        for v in ('scaled', 'raw'):
            out['vol'][f'har_h{h}_{v}'] = one(('vol', h, v), False)
    # training months at the first scored month (the market and SPY: one pair per month)
    first = int(np.argmax(mk))
    for sig, h in RET_FORECASTERS:
        for l in ('market', 'spy', 'stock'):
            out['ret'][f'{sig}_h{h}_skip']['levels'][l]['train_pairs_at_start'] = float(
                res[('ret', sig, h, 'skip', l, 0)]['ntrain'][first])
    return out


DISTINCT = [('trend', 5), ('trend', 21), ('reversal', 5), ('reversal', 21), ('ar1', 21)]


def signflip_null(rows, mk, n_draws=200, seed=SEED + 7):
    """Reading aid, not a pass rule: the ladder under no predictability that keeps each month's cross-sectional
    structure. Each draw flips the sign of every member's signal in month k by one random s_k = +-1 (the same flip
    for every level, so the levels stay paired), refits every forecaster from scratch and rescores the five slope
    levels over the months mk. Returns, per distinct forecaster, the null's R2 per level and its slopes."""
    rebal = rows['rebal']
    K = len(rebal)
    rng = np.random.default_rng(seed)
    S = rng.choice([-1.0, 1.0], (n_draws, K))
    out = {}
    for sig, h in DISTINCT:
        xcol, ycol, off = sig_col(sig, h), f'y{h}', SKIP + h
        Lg = _lag(rebal, off)
        per = {}
        for lvl in SLOPE_LEVELS:
            m = (rows['lvl'] == LV[lvl]) & (rows['rep'] == 0)
            x = rows['X'][m, FI[xcol]].astype(np.float64)
            y = rows['X'][m, FI[ycol]].astype(np.float64)
            k = rows['k'][m].astype(int)
            ok = np.isfinite(x) & np.isfinite(y)
            x, y, k = x[ok], y[ok], k[ok]
            bc = lambda w: np.bincount(k, weights=w, minlength=K)  # noqa: E731
            n, sy, syy, sxx = bc(np.ones_like(x)), bc(y), bc(y * y), bc(x * x)
            sx0, sxy0 = bc(x), bc(x * y)
            r2s = np.empty(n_draws)
            for d in range(n_draws):
                sx, sxy = sx0 * S[d], sxy0 * S[d]
                cum = lambda v: np.concatenate([np.zeros(Lg), np.cumsum(v)[:-Lg]])  # noqa: E731
                Cn, Cx, Cy, Cxx, Cxy = cum(n), cum(sx), cum(sy), cum(sxx), cum(sxy)
                with np.errstate(all='ignore'):
                    vx = Cxx - Cx * Cx / Cn
                    b = np.where((Cn >= 3) & (vx > 0), (Cxy - Cx * Cy / Cn) / vx, np.nan)
                    a = Cy / Cn - b * Cx / Cn
                    ybar = Cy / Cn
                sse = syy - 2 * a * sy - 2 * b * sxy + n * a * a + 2 * a * b * sx + b * b * sxx
                sst = syy - 2 * ybar * sy + n * ybar * ybar
                r2s[d] = 1 - sse[mk].sum() / sst[mk].sum()
            per[lvl] = r2s
        R = np.stack([per[l] for l in SLOPE_LEVELS], 1)          # [draws, 5]
        out[f'{sig}_h{h}'] = {'r2': R, 'slope': None}
    return out


def a1_decision(st):
    """A1 as the plan writes it, per calibrated return forecaster: the slope's bootstrap one-sided p below 0.0013 and
    the market's R2 above zero by Clark-West (the primary bar, t >= 3). p of the conclusion for a forecaster: the
    larger of the two one-sided p's (both must hold); A1's raw p: the smallest over the distinct forecasters."""
    rows = []
    for sig, h in RET_FORECASTERS:
        r = st['ret'][f'{sig}_h{h}_skip']
        mk = r['levels']['market']
        ok_slope = r['slope'] > 0 and r['slope_boot_p'] < P_BAR
        ok_cw = mk['cw_t'] >= T_BAR
        rows.append({'forecaster': f'{sig} h{h}', 'slope': r['slope'], 'slope_boot_p': r['slope_boot_p'],
                     'slope_nw_t': r['slope_nw_t'], 'market_r2': mk['r2_hist'], 'market_cw_t': mk['cw_t'],
                     'market_cw_p': mk['cw_p'], 'p_conclusion': max(r['slope_boot_p'], mk['cw_p']),
                     'pass_slope': bool(ok_slope), 'pass_cw': bool(ok_cw), 'pass': bool(ok_slope and ok_cw),
                     'duplicate_of': 'reversal h5' if (sig, h) == ('ar1', 5) else None})
    distinct = [r for r in rows if r['duplicate_of'] is None]
    best = min(distinct, key=lambda r: r['p_conclusion'])
    return {'per_forecaster': rows, 'n_distinct': len(distinct), 'raw_p': best['p_conclusion'],
            'raw_p_forecaster': best['forecaster'], 'bonferroni_p': min(1.0, best['p_conclusion'] * len(distinct)),
            'pass': any(r['pass'] for r in rows)}


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[j]))
        adj[j] = run
    return adj


def a2_decision(st):
    rows = []
    for sig, h in RET_FORECASTERS:
        r = st['ret'][f'{sig}_h{h}_skip']
        rows.append({'forecaster': f'{sig} h{h}', 'group_r2': r['levels']['group']['r2_hist'],
                     'matched_r2': r['levels']['matched']['r2_hist'], 'diff': r['a2_diff'], 'ci90': r.get('a2_ci90'),
                     'boot_p': r.get('a2_boot_p'), 'nw_t': r['a2_nw_t'],
                     'duplicate_of': 'reversal h5' if (sig, h) == ('ar1', 5) else None})
    distinct = [r for r in rows if r['duplicate_of'] is None]
    adj = holm(np.array([r['boot_p'] for r in distinct]))
    for r, a in zip(distinct, adj):
        r['holm_within_a2'] = float(a)
    best = min(distinct, key=lambda r: r['boot_p'])
    return {'per_forecaster': rows, 'raw_p': best['boot_p'], 'raw_p_forecaster': best['forecaster'],
            'holm_min_within_a2': float(adj.min()), 'pass_unadjusted_0.05': bool(adj.min() < 0.05)}


def months_mask(rows, lo, hi):
    """Decision dates t0 with lo <= t0 <= hi (ISO dates) whose 21-day outcome (through t0+22) is in the data."""
    days, rebal = rows['days'], rows['rebal']
    t0d = days[rebal]
    ep = np.datetime64('1970-01-01')
    lo_d, hi_d = (np.datetime64(lo) - ep).astype(int), (np.datetime64(hi) - ep).astype(int)
    return (t0d >= lo_d) & (t0d <= hi_d) & (rebal + SKIP + max(HORIZONS) <= len(days) - 1)


def year_masks(rows, years):
    days, rebal = rows['days'], rows['rebal']
    yr = np.array([int(c2eval.iso(d)[:4]) for d in days[rebal]])
    ok = rebal + SKIP + max(HORIZONS) <= len(days) - 1
    return {y: (yr == y) & ok for y in years}


def report(tag='old'):
    rows = dict(np.load(WORK / f'rows_{tag}.npz'))
    res = score(rows)
    periods = {
        'test 2008-01..2019-12': (SCORE_FROM, SCORE_TO),
        '2008-01..2009-12': ('2008-01-01', '2009-12-31'),
        '2010-01..2014-12': ('2010-01-01', '2014-12-31'),
        '2015-01..2019-12': ('2015-01-01', '2019-12-31'),
        'eval 2020-01..2025-06': (EVAL_FROM, '2025-06-28'),
    }
    out = {'periods': {}, 'years': {}}
    for i, (name, (lo, hi)) in enumerate(periods.items()):
        mk = months_mask(rows, lo, hi)
        out['periods'][name] = period_stats(rows, res, mk, SEED + i)
    for y, mk in year_masks(rows, range(2008, 2026)).items():
        if mk.sum() >= 6:
            out['years'][str(y)] = period_stats(rows, res, mk, SEED + y, boot=False)
    test = out['periods']['test 2008-01..2019-12']
    mk_test = months_mask(rows, SCORE_FROM, SCORE_TO)
    xs = np.log(np.array([test['sizes'][l] for l in SLOPE_LEVELS]))
    out['signflip'] = {}
    for f, v in signflip_null(rows, mk_test).items():
        sl = slope_on(xs, v['r2'])
        sl4 = slope_on(xs[:4], v['r2'][:, :4])
        real = test['ret'][f + '_skip']
        out['signflip'][f] = {
            'draws': int(len(sl)),
            'slope_null_mean': float(sl.mean()), 'slope_null_sd': float(sl.std(ddof=1)),
            'slope_real': real['slope'], 'share_null_slopes_ge_real': float((sl >= real['slope']).mean()),
            'slope4_null_mean': float(sl4.mean()), 'slope4_null_sd': float(sl4.std(ddof=1)),
            'slope4_real': real['slope4'], 'share_null_slopes4_ge_real': float((sl4 >= real['slope4']).mean()),
            'r2_null_mean': {l: float(v['r2'][:, i].mean()) for i, l in enumerate(SLOPE_LEVELS)},
            'r2_null_sd': {l: float(v['r2'][:, i].std(ddof=1)) for i, l in enumerate(SLOPE_LEVELS)},
            'slope_null_hist': np.histogram(sl, bins=30)[0].tolist(),
            'slope_null_edges': np.histogram(sl, bins=30)[1].tolist()}
    out['A1'] = a1_decision(test)
    out['A2'] = a2_decision(test)
    out['control'] = {}
    for h in HORIZONS:
        for v in ('scaled', 'raw'):
            r = test['vol'][f'har_h{h}_{v}']
            mk_ = r['levels']['market']
            out['control'][f'har_h{h}_{v}'] = {'slope': r['slope'], 'slope_boot_p': r.get('slope_boot_p'),
                                               'slope_nw_t': r['slope_nw_t'], 'market_r2': mk_['r2_hist'],
                                               'market_cw_t': mk_['cw_t'],
                                               'pass': bool(r['slope'] > 0 and r.get('slope_boot_p', 1) < P_BAR
                                                            and mk_['cw_t'] >= T_BAR)}
    # fitted coefficients over time for the market and single stocks (reported)
    rebal = rows['rebal']
    out['coef'] = {}
    for sig, h in RET_FORECASTERS:
        for l in ('stock', 'b100', 'market'):
            r = res[('ret', sig, h, 'skip', l, 0)]
            out['coef'][f'{sig}_h{h}_{l}'] = {'b_first_scored': float(r['b'][months_mask(rows, SCORE_FROM, SCORE_TO)][0]),
                                              'b_last_scored': float(r['b'][months_mask(rows, SCORE_FROM, SCORE_TO)][-1])}
    out['months'] = {name: [c2eval.iso(d) for d in rows['days'][rebal[months_mask(rows, lo, hi)]]][:1] +
                     [c2eval.iso(d) for d in rows['days'][rebal[months_mask(rows, lo, hi)]]][-1:]
                     for name, (lo, hi) in periods.items()}
    return out, res, rows


def dump(obj, path):
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else round(float(o), 8)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return conv(o.tolist())
        return o
    Path(path).write_text(json.dumps(conv(obj), separators=(',', ':')))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'old'
    t = time.time()
    if not (WORK / f'rows_{tag}.npz').exists() or '--rebuild' in sys.argv:
        _, meta = build(tag)
        print(f'build {tag}: {meta["seconds"]:.0f}s; market check {meta["market_check"]}; spy check {meta["spy_check"]}; '
              f'matched {meta["match"]}', flush=True)
    if tag == 'old' and '--build-only' not in sys.argv:
        out, _, _ = report('old')
        dump(out, HERE / 'a_results_old.json')
        print(f'report old written ({time.time() - t:.0f}s)')


if __name__ == '__main__':
    main()
