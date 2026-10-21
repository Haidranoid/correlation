"""Twin reversal: shared pieces (experiments/+1.1-twin-reversal.md).

Universe, signals, buy-and-hold hedged decile long-shorts and the cost model, for one export at a time. The portfolio
calendar is c2eval.twins' own rebalance calendar (export day indices 252, 273, ...), so a stock's twin, beta and the
universe stay fixed through each 21-day hold.

    import common as C; E, tw = C.load(C.c2eval.EXPORT_OLD); P = C.Panel(E, tw)
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)

H = 21               # signal window and hold, trading days
N_UNIV = 500         # stocks in the universe
DV_WIN = 63          # median dollar volume window for "most traded"
MIN_PRICE = 5.0
DECILE = 0.10
STOCK_BPS = 5.0      # per side
FUND_BPS = 1.0       # per side
MAX_FUND_LOG = 0.5   # daily fund moves beyond +-50% (log) are bad prints: zero return that day
SPIKE_X = 3.0        # news proxy: max daily share volume in the signal window >= 3x the median of the 63 days before it
BROAD = {'SPY', 'IVV', 'VOO', 'VTI', 'IWM', 'IWB', 'IWV', 'QQQ', 'DIA', 'MDY', 'IJH', 'IJR', 'RSP', 'IWN', 'IWO',
         'IWD', 'IWF', 'IWR', 'IWS', 'IWP', 'SCHB', 'SCHX', 'VB', 'VO', 'VV', 'VTV', 'VUG', 'SPLG', 'OEF', 'ITOT'}


def gap_days(close, frac=0.5, look=10):
    """Market-wide data gaps: days where fewer than half as many series have a close as the median of the previous
    10 days (point in time). On the old export: 2002-08-15, 2002-09-16, 2011-02-17."""
    n = np.isfinite(np.asarray(close)).sum(0)
    return [d for d in range(look, len(n)) if n[d] < frac * np.median(n[d - look:d])]


def fill_gaps(close):
    """Copy of close with each gap day's missing closes set to the previous day's close (only past data). Without
    this, every 21-day window that spans 2011-02-17 has two missing returns for nearly every stock and fund, and
    c2eval.twin_residual (max_missing=1) returns NaN for the whole month."""
    c = np.array(close, np.float32, copy=True)
    gd = gap_days(c)
    for d in gd:
        miss = ~np.isfinite(c[:, d]) & np.isfinite(c[:, d - 1])
        c[miss, d] = c[miss, d - 1]
    return c, gd


def load(export=c2eval.EXPORT_OLD, cache=True):
    """The export and its twins, with market-wide gap days forward-filled in the stocks' and the funds' closes (the
    twins themselves are c2eval's, chosen on the unfilled data). cache=False for the new export: c2eval's twin cache
    lives outside this experiment's folders."""
    E = c2eval.load_export(export)
    tw = c2eval.twins(E, cache=cache)
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
    """Forward-fill NaNs along axis 1 (leading NaNs stay NaN)."""
    X = np.asarray(X, np.float64)
    ok = np.isfinite(X)
    idx = np.where(ok, np.arange(X.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return X[np.arange(X.shape[0])[:, None], idx]     # leading NaNs map to column 0, which is NaN


def log_returns(close):
    c = np.asarray(close, np.float64)
    r = np.full(c.shape, np.nan)
    with np.errstate(all='ignore'):
        r[:, 1:] = np.log(c[:, 1:] / c[:, :-1])
    r[~np.isfinite(r)] = np.nan
    return r


def window_sum(r, h, max_missing=1):
    """[N, T] sum of daily log returns over the h days ending at d (missing days count 0); NaN where more than
    max_missing of the h returns are missing (as c2eval.twin_residual does)."""
    N, T = r.shape
    cs = np.zeros((N, T + 1))
    np.cumsum(np.nan_to_num(r), 1, out=cs[:, 1:])
    cm = np.zeros((N, T + 1))
    np.cumsum(~np.isfinite(r), 1, out=cm[:, 1:])
    out = np.full((N, T), np.nan)
    out[:, h:] = cs[:, h + 1:] - cs[:, 1:T - h + 1]
    miss = np.full((N, T), np.inf)
    miss[:, h:] = cm[:, h + 1:] - cm[:, 1:T - h + 1]
    out[miss > max_missing] = np.nan
    return out


class Panel:
    """Everything the tests need for one export: universe per rebalance, the signals at every day, price paths."""

    def __init__(self, E, tw, verbose=True):
        self.E, self.tw = E, tw
        self.days = E['days']
        self.T = len(self.days)
        self.rebal = tw['rebal']
        F = tw['F']
        self.spy = int(np.nonzero(F['tickers'] == 'SPY')[0][0])
        close = np.asarray(E['close'], np.float64)
        self.close = close
        self.N = close.shape[0]
        # price paths for buy-and-hold: forward-filled stock closes; fund log-price from forward-filled closes with
        # bad daily prints (|log move| > 0.5) zeroed
        self.cff = ffill(close)
        fc = ffill(F['close'])
        fr = log_returns(fc)
        fr[~np.isfinite(fr) | (np.abs(fr) > MAX_FUND_LOG)] = 0.0
        self.flog = np.cumsum(fr, 1)          # fund log value; V_f(t)/V_f(e) = exp(flog[t] - flog[e])
        self.fund_ok = np.isfinite(fc)
        if verbose:
            print('panel: prices ready', flush=True)
        sr = log_returns(close)
        # signals at every day (positive = fell against its benchmark = buy)
        self.s_twin = -np.asarray(c2eval.twin_residual(E, tw, H, scaled=True), np.float64)
        self.s_twin_raw = -np.asarray(c2eval.twin_residual(E, tw, H, scaled=False), np.float64)
        r21 = window_sum(sr, H)
        self.s_plain = -r21
        spy_r = log_returns(np.asarray(F['close'], np.float64)[self.spy:self.spy + 1])
        spy_r[np.abs(spy_r) > MAX_FUND_LOG] = np.nan
        spy21 = window_sum(spy_r, H)[0]
        self.mbeta, self.mresid_sd = self._market_betas(sr, spy_r[0])
        bm = self.mbeta[self._block_index()]          # [T, N] beta of the latest rebalance
        bsd = self.mresid_sd[self._block_index()]
        mres = r21 - bm.T * spy21[None, :]
        self.s_mkt = -mres
        self.s_mkt_scaled = -mres / (bsd.T * np.sqrt(H))
        del bm, bsd, mres, r21, sr
        if verbose:
            print('panel: signals ready', flush=True)
        # news proxy: max daily share volume over the signal window vs the median of the 63 days before it
        with np.errstate(all='ignore'):
            vol = np.asarray(E['dollar_volume'], np.float64) / close
        self.vol = vol
        self.dv = np.asarray(E['dollar_volume'], np.float64)
        self.universe = self._universe()
        if verbose:
            print('panel: universe ready', flush=True)

    def _block_index(self):
        """[T] index into rebal of the latest rebalance at or before each day (-1 before the first)."""
        return np.searchsorted(self.rebal, np.arange(self.T), side='right') - 1

    def _market_betas(self, sr, spy_r, window=252, min_frac=0.9):
        """beta to SPY at each twin rebalance, estimated as c2eval.twins estimates the twin's beta: correlation (cosine
        of returns standardized over their own finite days, missing days 0) x sd_stock / sd_SPY over the trailing 252
        days through the rebalance close. Also the sd of the daily residual. Row K (extra) is all NaN for days before
        the first rebalance."""
        K = len(self.rebal)
        beta = np.full((K + 1, self.N), np.nan)
        rsd = np.full((K + 1, self.N), np.nan)
        for k, t0 in enumerate(self.rebal):
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

    def _universe(self):
        """Per rebalance k: the 500 most-traded stocks (median dollar volume over the 63 days through the close)
        among those with close >= $5, not fund-like and with a twin at that rebalance. Also the coverage counts."""
        K = len(self.rebal)
        uni = np.zeros((K, self.N), bool)
        self.cov = []
        tick = self.tw['F']['tickers']
        for k, t0 in enumerate(self.rebal):
            if t0 - DV_WIN + 1 < 0:
                continue
            with np.errstate(all='ignore'):
                mdv = np.nanmedian(self.dv[:, t0 - DV_WIN + 1:t0 + 1], 1)
            price_ok = np.nan_to_num(self.close[:, t0]) >= MIN_PRICE
            has = self.tw['fund_row'][k] >= 0
            like = self.tw['fund_like'][k].astype(bool)
            base = price_ok & np.isfinite(mdv)
            elig = base & has & ~like
            order = np.argsort(-np.where(elig, mdv, -np.inf))
            sel = order[:min(N_UNIV, int(elig.sum()))]
            uni[k, sel] = True
            # coverage: the plain 500 most traded (price >= $5 only): how many are fund-like, how many lack a twin
            plain = np.argsort(-np.where(base, mdv, -np.inf))[:min(N_UNIV, int(base.sum()))]
            fr = self.tw['fund_row'][k, sel]
            tk = tick[fr]
            self.cov.append(dict(k=k, day=c2eval.iso(self.days[t0]), eligible=int(elig.sum()), universe=int(len(sel)),
                                 plain_no_twin=int((~has[plain]).sum()), plain_fund_like=int(like[plain].sum()),
                                 median_corr=float(np.median(self.tw['corr'][k, sel])) if len(sel) else float('nan'),
                                 distinct_twins=int(len(set(fr.tolist()))),
                                 broad_share=float(np.mean([t in BROAD for t in tk])) if len(sel) else float('nan'),
                                 spy_beta_ok=bool(np.isfinite(self.mbeta[k]).any())))
        return uni

    def news_flag(self, d):
        """[N] bool: the stock had a volume spike in the signal window d-20..d (max daily share volume >= 3x the
        median over d-83..d-21)."""
        with np.errstate(all='ignore'):
            pre = np.nanmedian(self.vol[:, d - H - DV_WIN + 1:d - H + 1], 1)
            mx = np.nanmax(self.vol[:, d - H + 1:d + 1], 1)
        return np.nan_to_num(mx) >= SPIKE_X * np.where(np.isfinite(pre) & (pre > 0), pre, np.inf)

    # -------------------------------------------------------------------------------------------------------------
    def stock_growth(self, rows, e, x):
        """Gross buy-and-hold growth V(x)/V(e) of stocks from close e to close x."""
        return self.cff[rows, x] / self.cff[rows, e]

    def fund_growth(self, frows, e, x):
        return np.exp(self.flog[frows, x] - self.flog[frows, e])


def month_book(P, k, signal, hedge, delay=0, set_mask=None):
    """The decile long-short formed at rebalance k (decided at close t0) and held from close t0+delay for 21 days.

    signal: [N, T] score array (positive = buy). hedge: 'twin' (-beta x twin fund), 'spy' (-market beta x SPY) or
    None. set_mask: [N] bool, the ranked set (universe stocks with every compared signal finite).
    Returns dict with the long/short rows, legs' hedged returns, gross long-short return, and the end positions
    (stock rows -> drifted weight, fund rows -> drifted weight) and start positions for the cost model."""
    t0 = int(P.rebal[k])
    e, x = t0 + delay, t0 + delay + H
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
        b = P.mbeta[k, srows]
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


def run_strategy(P, ks, signal, hedge, delay=0, set_masks=None):
    """Monthly books for consecutive rebalances ks, with costs on actual turnover (positions drifted to the end of
    the previous hold; fund positions netted across stocks). Returns list of month dicts with gross, cost, net."""
    months = []
    prev = None
    for k in ks:
        b = month_book(P, k, signal, hedge, delay, set_masks[k])
        if b is None:
            continue
        if prev is not None and prev['x'] != b['e']:
            prev = None      # not contiguous: start from cash
        st = turnover(prev['stock_end'] if prev else {}, b['stock_start'])
        ft = turnover(prev['fund_end'] if prev else {}, b['fund_start'])
        b['stock_turnover'], b['fund_turnover'] = st, ft
        b['cost'] = STOCK_BPS * 1e-4 * st + FUND_BPS * 1e-4 * ft
        b['net'] = b['gross'] - b['cost']
        b['k'] = k
        months.append(b)
        prev = b
    return months


def tstat(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float('nan'), float('nan'), len(x)
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)


def ranked_sets(P, signals):
    """[K, N] bool: universe stocks with every listed signal finite at the rebalance close."""
    K = len(P.rebal)
    M = P.universe.copy()
    for k, t0 in enumerate(P.rebal):
        for s in signals:
            M[k] &= np.isfinite(s[:, t0])
    return M
