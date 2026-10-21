"""Twin momentum: shared pieces (experiments/-0.1-twin-momentum.md).

Universe, the three signals (twin-residual momentum, plain 12-1 momentum, market-residual momentum), buy-and-hold
hedged decile long-shorts, the cost model and the Sharpe bootstrap, for one export at a time. The portfolio calendar is
c2eval.twins' own rebalance calendar (export day indices 252, 273, ...), so a stock's twin, beta and the universe stay
fixed through each 21-day hold. Adapted from research/twin_reversal/common.py (copied, not imported).

    import common as C; E, tw = C.load(C.c2eval.EXPORT_OLD); P = C.Panel(E, tw)

Every signal array s is [N, T] float32 with s[:, t] = the value used for a decision at the close of day t; it uses
closes through day t - 21 only.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.dont_write_bytecode = True   # importing stockai must not write .pyc files into its tree
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
sys.path.insert(0, '/home/bhoult/programming/correlation/stockai')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402
from stockai.baselines.momentum import momentum_scores  # noqa: E402  (read-only reuse)

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)

HOLD = 21            # hold, trading days
SKIP = 21            # the most recent month is skipped
LOOK = 252           # the window starts 252 days back: returns of days t-251 .. t-21
WIN = LOOK - SKIP    # 231 daily returns in the window
MAX_MISSING = 12     # ~5% of the window (the plan)
N_UNIV = 500
DV_WIN = 63
MIN_PRICE = 5.0
DECILE = 0.10
STOCK_BPS = 5.0      # per side
FUND_BPS = 1.0       # per side
MAX_FUND_LOG = 0.5   # daily fund moves beyond +-50% (log) are bad prints: zero return that day
MIN_RANKED = 450     # Test 1 starts at the first 2006 rebalance with >= 450 universe stocks carrying every signal
BLOCK = 6            # bootstrap block length, months
N_BOOT = 10000
BROAD = {'SPY', 'IVV', 'VOO', 'VTI', 'IWM', 'IWB', 'IWV', 'QQQ', 'DIA', 'MDY', 'IJH', 'IJR', 'RSP', 'IWN', 'IWO',
         'IWD', 'IWF', 'IWR', 'IWS', 'IWP', 'SCHB', 'SCHX', 'VB', 'VO', 'VV', 'VTV', 'VUG', 'SPLG', 'OEF', 'ITOT'}


def gap_days(close, frac=0.5, look=10):
    """Market-wide data gaps: days where fewer than half as many series have a close as the median of the previous
    10 days (point in time). On the old export: 2002-08-15, 2002-09-16, 2011-02-17."""
    n = np.isfinite(np.asarray(close)).sum(0)
    return [d for d in range(look, len(n)) if n[d] < frac * np.median(n[d - look:d])]


def fill_gaps(close):
    """Copy of close (float32) with each gap day's missing closes set to the previous day's close (past data only)."""
    c = np.array(close, np.float32, copy=True)
    gd = gap_days(c)
    for d in gd:
        miss = ~np.isfinite(c[:, d]) & np.isfinite(c[:, d - 1])
        c[miss, d] = c[miss, d - 1]
    return c, gd


def load(export=c2eval.EXPORT_OLD):
    """The export and its (cached) twins, with market-wide gap days forward-filled in the stocks' and the funds'
    closes. The twins themselves are c2eval's, chosen on the unfilled data."""
    E = c2eval.load_export(export)
    tw = c2eval.twins(E)
    E = dict(E)
    E['close'], gs = fill_gaps(E['close'])
    F = dict(tw['F'])
    F['close'], gf = fill_gaps(F['close'])
    tw = dict(tw)
    tw['F'] = F
    E['gap_days'] = [c2eval.iso(E['days'][d]) for d in gs]
    E['fund_gap_days'] = [c2eval.iso(E['days'][d]) for d in gf]
    return E, tw


def ffill(X):
    """Forward-fill NaNs along axis 1 (leading NaNs stay NaN). float64."""
    X = np.asarray(X, np.float64)
    ok = np.isfinite(X)
    idx = np.where(ok, np.arange(X.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return X[np.arange(X.shape[0])[:, None], idx]


def log_returns(close):
    c = np.asarray(close, np.float64)
    r = np.full(c.shape, np.nan)
    with np.errstate(all='ignore'):
        r[:, 1:] = np.log(c[:, 1:] / c[:, :-1])
    r[~np.isfinite(r)] = np.nan
    return r


def shift(x, k):
    """out[:, t] = x[:, t - k] (float32), NaN for t < k."""
    out = np.full(x.shape, np.nan, np.float32)
    out[:, k:] = x[:, :-k]
    return out


def market_betas(sr, spy_r, rebal, window=252, min_frac=0.9):
    """[K, N] beta to SPY and the sd of the daily residual at each twin rebalance, estimated as c2eval.twins estimates
    the twin's beta: correlation (cosine of returns standardized over their own finite days, missing days 0) x
    sd_stock / sd_SPY over the trailing 252 days through the rebalance close."""
    K, N = len(rebal), sr.shape[0]
    beta = np.full((K, N), np.nan, np.float32)
    rsd = np.full((K, N), np.nan, np.float32)
    for k, t0 in enumerate(rebal):
        a, b = t0 - window + 1, t0 + 1
        S, m = sr[:, a:b], spy_r[a:b]
        if np.isfinite(m).mean() < min_frac:
            continue
        ok = np.isfinite(S).mean(1) >= min_frac
        Sx = S[ok]
        with np.errstate(all='ignore'):
            sd_s = np.nanstd(Sx, 1)
            Z = (Sx - np.nanmean(Sx, 1, keepdims=True)) / sd_s[:, None]
            Z[~np.isfinite(Z)] = 0.0
            Z /= np.linalg.norm(Z, axis=1, keepdims=True)
            sd_m = np.nanstd(m)
            zm = (m - np.nanmean(m)) / sd_m
            zm[~np.isfinite(zm)] = 0.0
            zm /= np.linalg.norm(zm)
            r = Z @ zm
            bet = r * sd_s / sd_m
            res = Sx - bet[:, None] * m[None, :]
            beta[k, ok] = bet
            rsd[k, ok] = np.nanstd(res, 1)
    return beta, rsd


class Panel:
    """Everything the tests need for one export: universe per rebalance, the signals at every day, price paths."""

    def __init__(self, E, tw, verbose=True, diagnostics=True):
        self.E, self.tw = E, tw
        self.days = E['days']
        self.T = len(self.days)
        self.rebal = tw['rebal']
        self.K = len(self.rebal)
        F = tw['F']
        self.spy = int(np.nonzero(F['tickers'] == 'SPY')[0][0])
        self.close = E['close']                      # float32, gap days filled
        self.N = self.close.shape[0]
        self.cff = ffill(self.close)                  # forward-filled stock closes for buy-and-hold
        fc = ffill(F['close'])
        fr = log_returns(fc)
        fr[~np.isfinite(fr) | (np.abs(fr) > MAX_FUND_LOG)] = 0.0
        self.flog = np.cumsum(fr, 1)                  # fund log value; V_f(t)/V_f(e) = exp(flog[t] - flog[e])
        del fc, fr
        if verbose:
            print(f'panel: prices ready ({self.N} stocks, {F["close"].shape[0]} funds, {self.T} days)', flush=True)
        say = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
        # (1) the plan's signal: the scaled 231-day twin residual ending at t-21, read at t
        self.s_twin = shift(c2eval.twin_residual(E, tw, WIN, scaled=True, max_missing=MAX_MISSING), SKIP)
        say('panel: twin signal ready')
        # (a) plain 12-1 momentum: ln(close[t-21] / close[t-252]), stockai's function
        self.s_plain = momentum_scores(self.close, SKIP, LOOK).astype(np.float32)
        say('panel: plain momentum ready')
        # (b) market-residual momentum: the same twin_residual code with SPY as every stock's "twin" at its market beta
        sr = log_returns(self.close)
        spy_r = log_returns(np.asarray(F['close'], np.float64)[self.spy:self.spy + 1])[0]
        spy_r[np.abs(spy_r) > MAX_FUND_LOG] = np.nan
        self.mbeta, self.mresid_sd = market_betas(sr, spy_r, self.rebal)
        del sr
        fake = {'rebal': self.rebal, 'fund_row': np.where(np.isfinite(self.mbeta), 0, -1).astype(np.int32),
                'beta': self.mbeta, 'resid_sd': self.mresid_sd,
                'F': {'close': np.asarray(F['close'][self.spy:self.spy + 1], np.float32)}}
        self.s_mkt = shift(c2eval.twin_residual(E, fake, WIN, scaled=True, max_missing=MAX_MISSING), SKIP)
        say('panel: market-residual signal ready')
        if diagnostics:
            self.s_twin_raw = shift(c2eval.twin_residual(E, tw, WIN, scaled=False, max_missing=MAX_MISSING), SKIP)
            self.s_mkt_raw = shift(c2eval.twin_residual(E, fake, WIN, scaled=False, max_missing=MAX_MISSING), SKIP)
            say('panel: unscaled diagnostics ready')
        self.excl = c2eval.excluded(E, tw)
        self.universe = self._universe()
        say('panel: universe ready')

    def block_index(self):
        """[T] index into rebal of the latest rebalance at or before each day (-1 before the first)."""
        return np.searchsorted(self.rebal, np.arange(self.T), side='right') - 1

    def _universe(self):
        """Per rebalance k: the 500 most-traded stocks (median dollar volume over the 63 days through the close)
        among those with close >= $5, with a twin at that rebalance and not excluded (c2eval.excluded). Also coverage
        counts (filled in later by coverage())."""
        uni = np.zeros((self.K, self.N), bool)
        self.cov = []
        tick = self.tw['F']['tickers']
        dv = self.E['dollar_volume']
        for k, t0 in enumerate(self.rebal):
            if t0 - DV_WIN + 1 < 0:
                continue
            with np.errstate(all='ignore'):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    mdv = np.nanmedian(np.asarray(dv[:, t0 - DV_WIN + 1:t0 + 1], np.float64), 1)
            price_ok = np.nan_to_num(self.close[:, t0]) >= MIN_PRICE
            has = self.tw['fund_row'][k] >= 0
            ex = self.excl[k]
            base = price_ok & np.isfinite(mdv)
            elig = base & has & ~ex
            order = np.argsort(-np.where(elig, mdv, -np.inf))
            sel = order[:min(N_UNIV, int(elig.sum()))]
            uni[k, sel] = True
            plain = np.argsort(-np.where(base, mdv, -np.inf))[:min(N_UNIV, int(base.sum()))]
            fr = self.tw['fund_row'][k, sel]
            tk = tick[fr]
            self.cov.append(dict(k=k, day=c2eval.iso(self.days[t0]), eligible=int(elig.sum()), universe=int(len(sel)),
                                 plain_no_twin=int((~has[plain]).sum()), plain_excluded=int(ex[plain].sum()),
                                 median_corr=float(np.median(self.tw['corr'][k, sel])) if len(sel) else float('nan'),
                                 distinct_twins=int(len(set(fr.tolist()))),
                                 broad_share=float(np.mean([t in BROAD for t in tk])) if len(sel) else float('nan'),
                                 spy_beta_ok=bool(np.isfinite(self.mbeta[k]).any())))
        return uni

    def stock_growth(self, rows, e, x):
        return self.cff[rows, x] / self.cff[rows, e]

    def fund_growth(self, frows, e, x):
        return np.exp(self.flog[frows, x] - self.flog[frows, e])


def ranked_sets(P, signals):
    """[K, N] bool: universe stocks with every listed signal finite at the rebalance close."""
    M = P.universe.copy()
    for k, t0 in enumerate(P.rebal):
        for s in signals:
            M[k] &= np.isfinite(s[:, t0])
    return M


def coverage(P, masks):
    """Adds to P.cov: ranked-set size and the share of the ranked set whose twin at k differs from the twin at k-1
    (the signal's residual uses k-1's twin, the hedge uses k's)."""
    for c in P.cov:
        k = c['k']
        rows = np.nonzero(masks[k])[0]
        c['ranked'] = int(len(rows))
        if k >= 1 and len(rows):
            a, b = P.tw['fund_row'][k, rows], P.tw['fund_row'][k - 1, rows]
            c['twin_changed'] = float(np.mean(a != b))
        else:
            c['twin_changed'] = float('nan')
    return P.cov


def month_book(P, k, signal, hedge, set_mask, delay=0):
    """The decile long-short formed at rebalance k (decided at close t0) and held from close t0+delay for 21 days.

    signal: [N, T] (higher = buy). hedge: 'twin' (-beta x twin fund, both from rebalance k), 'spy' (-market beta x SPY)
    or None. set_mask: [N] bool, the ranked set. Returns the long/short rows, the legs' hedged returns, the gross
    long-short return, and start/end positions for the cost model."""
    t0 = int(P.rebal[k])
    e, x = t0 + delay, t0 + delay + HOLD
    if x >= P.T:
        return None
    rows = np.nonzero(set_mask)[0]
    s = signal[rows, t0]
    n = len(rows)
    m = int(round(DECILE * n))
    if m == 0:
        return None
    order = np.argsort(-s, kind='stable')
    L, S = rows[order[:m]], rows[order[-m:]]
    out = {'t0': t0, 'e': e, 'x': x, 'L': L, 'S': S, 'n': n}
    w = np.concatenate([np.full(m, 1.0 / m), np.full(m, -1.0 / m)])
    srows = np.concatenate([L, S])
    g = P.stock_growth(srows, e, x)
    if hedge == 'twin':
        fr = P.tw['fund_row'][k, srows].astype(int)
        b = P.tw['beta'][k, srows].astype(np.float64)
    elif hedge == 'spy':
        fr = np.full(len(srows), P.spy)
        b = P.mbeta[k, srows].astype(np.float64)
    else:
        fr, b = None, None
    if fr is not None:
        gf = P.fund_growth(fr, e, x)
        legs = (g - 1) - b * (gf - 1)
        fw = -w * b
        uf, inv = np.unique(fr, return_inverse=True)
        fstart = np.bincount(inv, weights=fw, minlength=len(uf))
        fend = fstart * P.fund_growth(uf, e, x)
        out['fund_start'] = dict(zip(uf.tolist(), fstart.tolist()))
        out['fund_end'] = dict(zip(uf.tolist(), fend.tolist()))
    else:
        legs = g - 1
        out['fund_start'], out['fund_end'] = {}, {}
    out['legs'] = legs
    out['long'] = float(legs[:m].mean())
    out['short'] = float(legs[m:].mean())
    out['gross'] = out['long'] - out['short']
    out['stock_start'] = dict(zip(srows.tolist(), w.tolist()))
    out['stock_end'] = dict(zip(srows.tolist(), (w * g).tolist()))
    return out


def turnover(prev_end, new_start):
    keys = set(prev_end) | set(new_start)
    return float(sum(abs(new_start.get(j, 0.0) - prev_end.get(j, 0.0)) for j in keys))


def run_strategy(P, ks, signal, hedge, set_masks, delay=0):
    """Monthly books for consecutive rebalances ks, with costs on actual turnover (positions drifted to the end of the
    previous hold; fund positions netted across stocks). Returns a list of month dicts with gross, cost, net."""
    months = []
    prev = None
    for k in ks:
        b = month_book(P, k, signal, hedge, set_masks[k], delay)
        if b is None:
            continue
        if prev is not None and prev['x'] != b['e']:
            prev = None
        st = turnover(prev['stock_end'] if prev else {}, b['stock_start'])
        ft = turnover(prev['fund_end'] if prev else {}, b['fund_start'])
        b['stock_turnover'], b['fund_turnover'] = st, ft
        b['cost'] = STOCK_BPS * 1e-4 * st + FUND_BPS * 1e-4 * ft
        b['net'] = b['gross'] - b['cost']
        b['k'] = k
        del b['stock_start'], b['fund_start']
        months.append(b)
        prev = dict(b)
    return months


def tstat(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float('nan'), float('nan'), len(x)
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)


def sharpe(x, axis=None):
    """Annualized Sharpe of 21-day holds (12 a year): mean / sd x sqrt(12)."""
    x = np.asarray(x, np.float64)
    return x.mean(axis) / x.std(axis, ddof=1) * np.sqrt(252 / HOLD)


def boot_sharpe_diff(a, b, block=BLOCK, n_boot=N_BOOT, seed=20261021):
    """Paired circular block bootstrap of the monthly pairs (a_i, b_i). Statistic: Sharpe(a) - Sharpe(b), annualized.
    One-sided p for H0 'Sharpe(a) <= Sharpe(b)', centered: p = (1 + #{d* - d_obs >= d_obs}) / (n_boot + 1).
    Also the 90% percentile interval of d*."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    n = len(a)
    d_obs = float(sharpe(a) - sharpe(b))
    rng = np.random.default_rng(seed)
    nb = -(-n // block)
    starts = rng.integers(0, n, (n_boot, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(n_boot, -1)[:, :n]
    d = sharpe(a[idx], 1) - sharpe(b[idx], 1)
    p = float((1 + np.sum(d - d_obs >= d_obs)) / (n_boot + 1))
    lo, hi = np.percentile(d, [5, 95])
    return dict(diff=d_obs, p=p, ci90=[float(lo), float(hi)], block=block, n=n)


def max_drawdown(x):
    """Largest peak-to-trough fall of the cumulative sum of monthly returns, and the formation indices."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    peak = np.maximum.accumulate(c)
    dd = peak - c
    j = int(np.argmax(dd))
    i = int(np.argmax(c[:j + 1])) if j > 0 else 0
    return float(dd[j]), i, j
