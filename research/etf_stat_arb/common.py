"""ETF stat arb (experiments/-6.8-etf-stat-arb.md): Avellaneda & Lee's s-score rule against each stock's twin fund.

Shared pieces: universe, returns, the vectorized 60-day fits (rolling OLS on the twin, cumulative residual X_k, AR(1)
on X_k, all from rolling sums), a direct per-window fit for checking, the s-score, the band state machine and the
book's daily P&L with costs.

At the close of day t, for a stock in the trading set with twin f (the twin of the latest c2eval.twins rebalance):
  - OLS of the stock's daily simple returns on f's over the 60 return days t-59..t, with an intercept; a day counts
    only if both returns exist (and |log return| <= 0.5 for each, a point-in-time bad-print rule); >= 57 such days.
  - X_k = cumulative residual (0 on days that don't count), k = 1..60.
  - AR(1) X_{k+1} = a + b X_k + zeta, k = 1..59; Var(zeta) = SSR / 59.
  - passes if 0 < b < exp(-1/30)  (kappa = -ln(b) * 252 > 252/30).
  - m = a / (1 - b); sigma_eq = sqrt(Var(zeta) / (1 - b^2)); s = -(m - mbar) / sigma_eq, mbar = mean m over the
    trading set's passing stocks that day.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)
WORK = c2eval.WORK / 'etf_stat_arb'
WORK.mkdir(parents=True, exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
STORES = {'old': c2eval.STORE_OLD, 'new': c2eval.STORE_NEW}

W = 60                    # regression window (return days)
MIN_VALID = 57            # = 60 - max(1, round(0.05 * 60)), as c2eval.twin_residual's max_missing
NAR = W - 1               # AR(1) observations
B_MAX = float(np.exp(-1.0 / 30.0))   # kappa > 252/30
MAX_LOG = 0.5             # |daily log move| above this: the day doesn't count in the fit (stock or fund)
S_OPEN = 1.25
S_CLOSE_LONG = -0.50      # close a long if s > -0.50
S_CLOSE_SHORT = 0.75      # close a short if s < +0.75
STOCK_BPS = 5.0
FUND_BPS = 1.0

# universe (copied from research/night_and_day/common.py)
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6
DV_LARGE_W, N_LARGE = 63, 500

PERIODS = {'2006-2019': ('2006-01-01', '2019-12-31'), '2020-2025-06': ('2020-01-01', '2025-06-27'),
           'post-2022-07': ('2022-07-01', '2025-06-27'), 'unseen': ('2025-06-30', '2026-10-19'),
           'unseen-ff': ('2025-06-30', '2026-08-31')}

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


# ---------------------------------------------------------------------------------------------------------------
def universe(E, tw):
    """[K, N] bool universe and [K, N] bool 500 largest on tw['rebal'] days (night_and_day's rule) AND a twin."""
    close, dv = E['close'], E['dollar_volume']
    rebal = tw['rebal']
    N = close.shape[0]
    excl = c2eval.excluded(E, tw)
    uni = np.zeros((len(rebal), N), bool)
    big = np.zeros((len(rebal), N), bool)
    for k, t0 in enumerate(rebal):
        if t0 < DATA_W - 1:
            continue
        c = np.asarray(close[:, t0 - DATA_W + 1:t0 + 1], np.float64)
        fin = np.isfinite(c).mean(1) >= DATA_FRAC
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mc = np.nanmedian(c[:, -MED_W:], 1)
            v = np.asarray(dv[:, t0 - MED_W + 1:t0 + 1], np.float64)
            mv = np.nanmedian(v, 1)
            mv63 = np.nanmedian(v[:, -DV_LARGE_W:], 1)
        u = fin & (np.nan_to_num(mc) >= MIN_PRICE) & (np.nan_to_num(mv) >= MIN_DV) & ~excl[k] & (tw['fund_row'][k] >= 0)
        uni[k] = u
        cand = np.nonzero(u)[0]
        o = cand[np.argsort(-np.nan_to_num(mv63[cand], nan=-1.0), kind='stable')]
        big[k, o[:N_LARGE]] = True
    return uni, big


def load(tag):
    """Export (close, dollar_volume), twins and universe for 'old' or 'new'. Cached universe under WORK."""
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume'))
    tw = c2eval.twins(E)
    p = WORK / f'universe_{tag}.npz'
    if p.exists():
        z = np.load(p)
        uni, big = z['uni'], z['big']
    else:
        uni, big = universe(E, tw)
        np.savez(p, uni=uni, big=big)
    return E, tw, uni, big


def stock_close(E, tag, source):
    """[N, T] float64 closes: 'export' (repaired) or 'store' (the store's raw, unrepaired closes)."""
    if source == 'export':
        return np.asarray(E['close'], np.float64)
    S = c2eval.load_store(STORES[tag])
    return c2eval.store_panel(S, E['ids'], E['days'], 'close').astype(np.float64)


def log_returns(c):
    r = np.full(c.shape, np.nan)
    with np.errstate(all='ignore'):
        r[:, 1:] = np.log(c[:, 1:] / c[:, :-1])
    r[~np.isfinite(r)] = np.nan
    return r


def ffill(X):
    X = np.asarray(X, np.float64)
    ok = np.isfinite(X)
    idx = np.where(ok, np.arange(X.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return X[np.arange(X.shape[0])[:, None], idx]


def signal_returns(lr):
    """Simple returns for the fit from daily log returns: NaN where missing or |log| > MAX_LOG."""
    r = np.expm1(lr)
    r[~np.isfinite(lr) | (np.abs(lr) > MAX_LOG)] = np.nan
    return r


def pnl_returns(close, fix=None):
    """Simple daily returns for P&L from forward-filled closes (a missing day is 0, the move lands on the next close);
    fix: optional [N, T] log-return correction added on top (dividends)."""
    lr = log_returns(ffill(close))
    lr[~np.isfinite(lr)] = 0.0
    if fix is not None:
        lr = lr + np.nan_to_num(fix)
    return np.expm1(lr)


# ---------------------------------------------------------------------------------------------------------------
def fit_block(y, x):
    """Vectorized fits for every window in a slice. y, x: [n, L] daily simple returns (NaN = day doesn't count).
    Window j (j = 0..J-1, J = L - W + 1) is columns j..j+W-1. Returns dict of [n, J] float64 arrays:
    nv, beta, alpha, a, b, var_zeta, m, sigeq (NaN where nv < MIN_VALID or degenerate)."""
    n, L = y.shape
    J = L - W + 1
    V = np.isfinite(y) & np.isfinite(x)
    y0 = np.where(V, y, 0.0)
    x0 = np.where(V, x, 0.0)
    v0 = V.astype(np.float64)

    def cum(z):
        c = np.zeros((z.shape[0], z.shape[1] + 1))
        np.cumsum(z, 1, out=c[:, 1:])
        return c

    def wsum(z):  # sum over columns j..j+W-1 for each window j
        c = cum(z)
        return c[:, W:W + J] - c[:, 0:J]

    nv = wsum(v0)
    Sx, Sy = wsum(x0), wsum(y0)
    Sxx, Sxy = wsum(x0 * x0), wsum(x0 * y0)
    with np.errstate(all='ignore'):
        beta = (Sxy - Sx * Sy / nv) / (Sxx - Sx * Sx / nv)
        alpha = (Sy - beta * Sx) / nv
    # G[q] = (CY, CN, CX)[q] = sums over columns < q, q = 0..L. X_k (window j) = w . (G[j+k] - G[j]), w = (1, -alpha, -beta)
    G = [cum(y0), cum(v0), cum(x0)]
    w = [np.ones((n, J)), -alpha, -beta]

    def qsum(f, lo, hi):  # sum_{q=j+lo}^{j+hi} f[:, q]
        c = cum(f)
        return c[:, hi + 1:hi + 1 + J] - c[:, lo:lo + J]

    def lin(lo, hi):
        return sum(w[c] * qsum(G[c], lo, hi) for c in range(3))

    def quad(lo, hi):
        tot = 0.0
        for c in range(3):
            for d in range(c, 3):
                f = 1.0 if c == d else 2.0
                tot = tot + f * w[c] * w[d] * qsum(G[c] * G[d], lo, hi)
        return tot

    def lagq(lo, hi):  # sum_{q=j+lo}^{j+hi} (w.G[q]) (w.G[q+1])
        tot = 0.0
        for c in range(3):
            for d in range(3):
                tot = tot + w[c] * w[d] * qsum(G[c][:, :-1] * G[d][:, 1:], lo, hi)
        return tot

    c0 = sum(w[c] * G[c][:, 0:J] for c in range(3))      # w . G[j]
    LU, LV = lin(1, NAR), lin(2, W)
    SU = LU - NAR * c0
    SV = LV - NAR * c0
    SUU = quad(1, NAR) - 2 * c0 * LU + NAR * c0 * c0
    SVV = quad(2, W) - 2 * c0 * LV + NAR * c0 * c0
    SUV = lagq(1, NAR) - c0 * LV - c0 * LU + NAR * c0 * c0
    with np.errstate(all='ignore'):
        b = (SUV - SU * SV / NAR) / (SUU - SU * SU / NAR)
        a = (SV - b * SU) / NAR
        ssr = SVV - a * SV - b * SUV
        vz = np.maximum(ssr, 0.0) / NAR
        m = a / (1 - b)
        sigeq = np.sqrt(vz / (1 - b * b))
    bad = (nv < MIN_VALID) | ~np.isfinite(beta) | ~np.isfinite(b)
    out = dict(nv=nv, beta=beta, alpha=alpha, a=a, b=b, var_zeta=vz, m=m, sigeq=sigeq)
    for k in out:
        if k != 'nv':
            out[k] = np.where(bad, np.nan, out[k])
    return out


def fit_direct(yw, xw):
    """Direct fit of one window (60 returns each) with np.linalg.lstsq; same conventions as fit_block."""
    V = np.isfinite(yw) & np.isfinite(xw)
    A = np.column_stack([np.ones(V.sum()), xw[V]])
    (alpha, beta), *_ = np.linalg.lstsq(A, yw[V], rcond=None)
    e = np.zeros(len(yw))
    e[V] = yw[V] - alpha - beta * xw[V]
    X = np.cumsum(e)
    A2 = np.column_stack([np.ones(NAR), X[:-1]])
    (a, b), *_ = np.linalg.lstsq(A2, X[1:], rcond=None)
    z = X[1:] - a - b * X[:-1]
    vz = float(z @ z) / NAR
    return dict(nv=int(V.sum()), beta=beta, alpha=alpha, a=a, b=b, var_zeta=vz, m=a / (1 - b),
                sigeq=np.sqrt(vz / (1 - b * b)), X60=float(X[-1]))


def fit_panel(rs, rf, tw, uni):
    """Fits for every day t and every stock in the universe of t's block, against the block's twin.
    rs [N, T] stock simple returns for the fit, rf [F, T] fund ones (NaN = doesn't count).
    Returns float32 [N, T] arrays beta, m, sigeq, b (NaN where no fit) and nv (int16)."""
    N, T = rs.shape
    out = {k: np.full((N, T), np.nan, np.float32) for k in ('beta', 'm', 'sigeq', 'b', 'a', 'var_zeta')}
    out['nv'] = np.zeros((N, T), np.int16)
    rebal = tw['rebal']
    for k, t0 in enumerate(rebal):
        t1 = int(rebal[k + 1]) if k + 1 < len(rebal) else T
        si = np.nonzero(uni[k])[0]
        if len(si) == 0 or t0 - W + 1 < 1:
            continue
        fj = tw['fund_row'][k, si]
        y = rs[si, t0 - W + 1:t1]
        x = rf[fj, t0 - W + 1:t1]
        f = fit_block(y, x)
        for key in ('beta', 'm', 'sigeq', 'b', 'a', 'var_zeta'):
            out[key][si, t0:t1] = f[key]
        out['nv'][si, t0:t1] = f['nv']
    return out


def s_score(fit, set_by_day):
    """s [N, T] float32 for a trading set (set_by_day: [N, T] bool, or a callable k -> [N] mask via block).
    s is NaN where the stock isn't in the set or fails the fit/kappa filter. mbar = mean m over the set's passing
    stocks that day."""
    b = fit['b']
    ok = set_by_day & np.isfinite(b) & (b > 0) & (b < B_MAX) & np.isfinite(fit['m']) & (fit['sigeq'] > 0)
    m = np.where(ok, fit['m'].astype(np.float64), np.nan)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mbar = np.nanmean(m, 0)
        s = -(m - mbar[None, :]) / fit['sigeq']
    return s.astype(np.float32), ok


def by_day(mask_k, rebal, T):
    """Expand a [K, N] per-rebalance mask to [N, T] (False before the first rebalance)."""
    blk = np.searchsorted(rebal, np.arange(T), side='right') - 1
    out = np.zeros((mask_k.shape[1], T), bool)
    good = blk >= 0
    out[:, good] = mask_k[blk[good]].T
    return out


def fund_row_by_day(tw, T):
    blk = np.searchsorted(tw['rebal'], np.arange(T), side='right') - 1
    fr = np.full((tw['fund_row'].shape[1], T), -1, np.int32)
    good = blk >= 0
    fr[:, good] = tw['fund_row'][blk[good]].T
    return fr


# ---------------------------------------------------------------------------------------------------------------
def state_path(s, has_close, in_set):
    """The band state machine. s [N, T]; has_close [N, T] (a trade on day d needs the stock's close on d).
    Returns p [N, T] int8 (position after the close of d) and exit reasons counts/holding-time lists."""
    N, T = s.shape
    p = np.zeros((N, T), np.int8)
    cur = np.zeros(N, np.int8)
    entry = np.full(N, -1, np.int64)
    hold = {'band': [], 'fit': [], 'universe': []}
    nflip = 0
    for d in range(T):
        sd = s[:, d]
        hc = has_close[:, d]
        fin = np.isfinite(sd)
        new = cur.copy()
        # closes
        lng, sht = cur == 1, cur == -1
        close_band = hc & fin & ((lng & (sd > S_CLOSE_LONG)) | (sht & (sd < S_CLOSE_SHORT)))
        close_nan = hc & ~fin & (cur != 0)
        for reason, msk in (('band', close_band), ('fit', close_nan & in_set[:, d]), ('universe', close_nan & ~in_set[:, d])):
            idx = np.nonzero(msk)[0]
            if len(idx):
                hold[reason].extend((d - entry[idx]).tolist())
        new[close_band | close_nan] = 0
        # opens (also right after a band close: a flip)
        flat = new == 0
        op_l = hc & fin & flat & (sd < -S_OPEN)
        op_s = hc & fin & flat & (sd > S_OPEN)
        nflip += int(((op_l | op_s) & close_band).sum())
        new[op_l] = 1
        new[op_s] = -1
        entry[op_l | op_s] = d
        cur = new
        p[:, d] = cur
    return p, {'hold': {k: np.array(v) for k, v in hold.items()}, 'flips': nflip}


def delay_path(p, has_close):
    """Execute each decision at the next close: q[d] = p[d-1] if the stock has a close on d, else q[d-1]."""
    N, T = p.shape
    q = np.zeros_like(p)
    for d in range(1, T):
        q[:, d] = np.where(has_close[:, d], p[:, d - 1], q[:, d - 1])
    return q


def run_book(p, beta, fund_row, rs_pnl, rf_pnl, nF, d_start, beta_shift=0):
    """Daily book from positions p [N, T] (after each close). Stock legs are buy-and-hold from entry ($1 at entry,
    drifting); the twin hedge is reset each close to -beta x the stock leg's value, netted per fund.
    beta: [N, T] window betas (the held beta is carried forward where NaN); beta_shift=1 uses the previous day's beta
    and twin (the delayed book). Costs: STOCK_BPS on stock trades, FUND_BPS on netted fund trades, charged to the
    next day's return. Returns dict of daily arrays (index d = return earned from close d-1 to close d)."""
    N, T = p.shape
    V = np.zeros(N)                 # signed stock leg value
    H = np.zeros(nF)                # fund leg values
    bheld = np.full(N, np.nan)
    fheld = np.full(N, -1, np.int64)
    out = {k: np.zeros(T) for k in ('gross', 'cost_stock', 'cost_fund', 'n', 'n_long', 'n_short', 'stock_traded',
                                     'fund_traded', 'pnl_stock', 'pnl_fund')}
    pend_cs = pend_cf = 0.0
    nprev = 0
    prev = np.zeros(N, np.int8)
    for d in range(d_start, T):
        # P&L over close d-1 -> close d
        rsd = rs_pnl[:, d]
        pnl_s = float(V @ rsd)
        V = V * (1 + rsd)
        rfd = rf_pnl[:, d]
        pnl_f = float(H @ rfd)
        H = H * (1 + rfd)
        den = max(nprev, 1)
        out['gross'][d] = (pnl_s + pnl_f) / den
        out['pnl_stock'][d] = pnl_s / den
        out['pnl_fund'][d] = pnl_f / den
        out['cost_stock'][d] = pend_cs / den
        out['cost_fund'][d] = pend_cf / den
        out['n'][d] = nprev
        # trades at close d
        pd_ = p[:, d]
        chg = pd_ != prev
        traded = 0.0
        if chg.any():
            idx = np.nonzero(chg)[0]
            traded += float(np.abs(V[idx]).sum())          # close the old leg at its current value
            newv = pd_[idx].astype(np.float64)             # open $1 new legs
            traded += float(np.abs(newv).sum())
            V[idx] = newv
        bd = beta[:, d - beta_shift]
        fd = fund_row[:, d - beta_shift]
        upd = np.isfinite(bd) & (fd >= 0)
        bheld[upd] = bd[upd]
        fheld[upd] = fd[upd]
        held = np.nonzero(V != 0)[0]
        Hn = np.zeros(nF)
        hb = np.nan_to_num(bheld[held])
        hf = fheld[held]
        okf = hf >= 0
        np.add.at(Hn, hf[okf], -hb[okf] * V[held][okf])
        ftraded = float(np.abs(Hn - H).sum())
        H = Hn
        pend_cs = STOCK_BPS * 1e-4 * traded
        pend_cf = FUND_BPS * 1e-4 * ftraded
        out['stock_traded'][d] = traded
        out['fund_traded'][d] = ftraded
        nprev = len(held)
        out['n_long'][d] = int((pd_ == 1).sum())
        out['n_short'][d] = int((pd_ == -1).sum())
        prev = pd_.copy()
    out['net'] = out['gross'] - out['cost_stock'] - out['cost_fund']
    return out


# ---------------------------------------------------------------------------------------------------------------
def tstat(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float('nan'), float('nan'), len(x)
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)


def nw_t(x, lags=5):
    """Newey-West (Bartlett) t of the mean of a daily series (NaNs dropped)."""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < lags + 3:
        return float('nan')
    u = x - x.mean()
    v = (u @ u) / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (u[L:] @ u[:-L]) / n
    return float(x.mean() / np.sqrt(v / n))


def period_mask(days, name):
    a, b = PERIODS[name]
    iso = np.array([c2eval.iso(d) for d in days])
    return (iso >= a) & (iso <= b)
