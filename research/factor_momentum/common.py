"""Statistical-factor momentum (experiments/-3.8-factor-momentum.md): shared pieces.

The universe (copied from research/night_and_day/common.py), the gap fill (copied from research/twin_momentum),
the eigen-decomposition at each formation, the eigenportfolios and the book, the two baselines, buy-and-hold hold
returns and the cost model. One export at a time.

Calendar: c2eval.twins' rebalance days (export day indices 252 + 21k). A formation at day t uses closes through the
close of t only; the book is held from close t to close t+21.

    import common as C; D = C.load('old'); F = C.formation(D, k)
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)
WORK = c2eval.WORK / 'factor_momentum'
WORK.mkdir(parents=True, exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}

# universe (night_and_day's judge-like liquid set)
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6
DV_LARGE_W, N_LARGE = 63, 500

# the plan
WIN = 252            # daily returns t-251 .. t
MIN_FRAC = 0.9       # a stock needs returns on >= 90% of the window's days
SKIP = 21            # 12-1: returns t-251 .. t-21 (231 days); 1-month: t-20 .. t (21 days)
FIRST, LAST = 2, 21  # components kept (1-based, by eigenvalue, descending); component 1 (the market) is left out
NK = LAST - FIRST + 1
TARGET_VOL = 0.10    # each factor scaled to 10% annualized volatility
BOOK_GROSS = 2.0
HOLD = 21
BPS = 5.0            # per side, on turnover
DECILE = 0.10
N_STAB = 50          # stability: best match among the previous formation's top 50 components
MAX_FUND_LOG = 0.5   # fund daily log moves beyond +-50% are bad prints (c2eval's convention): zero that day
# baseline (b) counts at a formation only with enough twins (rule fixed from coverage, before any returns)
B_MIN_FRAC, B_MIN_DISTINCT = 0.8, 10

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# data

def gap_days(close, frac=0.5, look=10):
    """Market-wide data gaps: days where fewer than half as many series have a close as the median of the previous
    10 days (point in time). Copied from research/twin_momentum/common.py."""
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


def ffill(X):
    """Forward-fill NaNs along axis 1 (leading NaNs stay NaN). float64."""
    X = np.asarray(X, np.float64)
    ok = np.isfinite(X)
    idx = np.where(ok, np.arange(X.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return X[np.arange(X.shape[0])[:, None], idx]


def universe(E, tw):
    """[K, N] bool universe and [K, N] bool 500 largest on tw['rebal'] days. Copied from
    research/night_and_day/common.py: closes on >= 90% of the trailing 756 days, median close >= $5 and median dollar
    volume >= $5M over the trailing 252 days, minus c2eval.excluded(); the 500 largest by median dollar volume over
    the trailing 63 days."""
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
        u = fin & (np.nan_to_num(mc) >= MIN_PRICE) & (np.nan_to_num(mv) >= MIN_DV) & ~excl[k]
        uni[k] = u
        cand = np.nonzero(u)[0]
        o = cand[np.argsort(-np.nan_to_num(mv63[cand], nan=-1.0), kind='stable')]
        big[k, o[:N_LARGE]] = True
    return uni, big


def load(tag):
    """Export (gap days filled), its twins, the universe (cached), SPY's store close and forward-filled closes."""
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume'))
    tw = c2eval.twins(E)
    key = WORK / f"universe_{tag}_{E['meta']['fingerprint']}.npz"
    if key.exists():
        z = np.load(key)
        uni, big = z['uni'], z['big']
    else:
        uni, big = universe(E, tw)
        np.savez(key, uni=uni, big=big)
    raw_close = E['close']
    cf, gd = fill_gaps(raw_close)
    D = {'tag': tag, 'E': E, 'tw': tw, 'uni': uni, 'big': big, 'rebal': tw['rebal'], 'days': E['days'],
         'gap_days': [c2eval.iso(E['days'][d]) for d in gd]}
    D['close'] = cf                                   # float32, gap days filled (formation windows)
    D['cff'] = ffill(cf).astype(np.float64)           # forward-filled: a stock with no close keeps its last one
    D['N'], D['T'] = cf.shape
    D['K'] = len(D['rebal'])
    F = tw['F']
    spy = int(np.nonzero(F['tickers'] == 'SPY')[0][0])
    D['spy_close'] = ffill(np.asarray(F['close'][spy:spy + 1], np.float64))[0]
    D['fclose'] = F['close']                          # [F, T] fund store closes (unfilled)
    D['ftick'] = F['tickers']
    return D


def first_k(D):
    """First rebalance with a universe (the judge-like set needs 756 days of history: 2004-12-31, about 300 stocks)."""
    return int(np.nonzero(D['big'].sum(1) > 0)[0][0])


def first_k500(D):
    """First rebalance with a 500-stock universe (2007-12-03 on the old export)."""
    return int(np.nonzero(D['big'].sum(1) >= N_LARGE)[0][0])


def iso(D, t):
    return c2eval.iso(D['days'][int(t)])


# ---------------------------------------------------------------------------------------------------------------
# the eigen-decomposition and the book at one formation

def formation(D, k, keep_vectors=N_STAB):
    """Everything the plan builds at rebalance k (close t = rebal[k]); uses closes t-252 .. t only.

    Returns dict with rows [n] (stock rows, ascending), sd [n] (daily log-return sd), lam [n] (eigenvalues,
    descending), V [n, keep_vectors] (eigenvectors of the top components, columns descending), w [n, 20] (gross-1
    eigenportfolio weights of components 2-21), f [20, 252] (in-sample daily factor returns), past12, past1, vol
    (annualized), sign, scale, B [n, 20] (each component's signed, vol-scaled weights after the book's $2 scaling),
    W [n] (the book: B.sum(1)), W1 [n] (the 1-month-sign diagnostic book, also $2 gross), r [n, 252] (daily log
    returns of the window, NaN where missing)."""
    t = int(D['rebal'][k])
    rows = np.nonzero(D['big'][k])[0]
    c = np.asarray(D['close'][rows, t - WIN:t + 1], np.float64)        # closes t-252 .. t
    with np.errstate(all='ignore'):
        r = np.log(c[:, 1:] / c[:, :-1])                               # returns of days t-251 .. t
    r[~np.isfinite(r)] = np.nan
    ok = np.isfinite(r).mean(1) >= MIN_FRAC
    rows, r = rows[ok], r[ok]
    n = len(rows)
    mu = np.nanmean(r, 1)
    sd = np.nanstd(r, 1)
    z = (r - mu[:, None]) / sd[:, None]
    z[~np.isfinite(z)] = 0.0
    u = z / np.linalg.norm(z, axis=1, keepdims=True)                   # unit-norm rows: correlation = cosine
    Cm = u @ u.T
    lam, V = np.linalg.eigh(Cm)                                        # ascending
    lam, V = lam[::-1], V[:, ::-1]                                     # descending
    v = V[:, FIRST - 1:LAST]                                           # components 2..21
    w = v / sd[:, None]
    w = w / np.abs(w).sum(0, keepdims=True)                            # gross 1 per eigenportfolio
    rs = np.expm1(r)
    rs[~np.isfinite(rs)] = 0.0                                         # missing daily return: 0
    f = w.T @ rs                                                       # [20, 252] daily factor returns, in sample
    past12 = f[:, :WIN - SKIP].sum(1)                                  # days t-251 .. t-21
    past1 = f[:, WIN - SKIP:].sum(1)                                   # days t-20 .. t
    vol = f.std(1, ddof=1) * np.sqrt(252)
    scale = TARGET_VOL / vol
    sign = np.sign(past12)
    B = w * (sign * scale)[None, :]
    Wb = B.sum(1)
    norm = BOOK_GROSS / np.abs(Wb).sum()
    B *= norm
    Wb = Wb * norm
    W1 = (w * (np.sign(past1) * scale)[None, :]).sum(1)
    W1 *= BOOK_GROSS / np.abs(W1).sum()
    return dict(k=k, t=t, rows=rows, n=n, sd=sd, lam=lam, V=V[:, :keep_vectors].astype(np.float32), w=w, f=f,
                past12=past12, past1=past1, vol=vol, sign=sign, scale=scale, B=B, W=Wb, W1=W1, norm=norm, r=r)


def hold_growth(D, rows, t):
    """Buy-and-hold growth - 1 from close t to close t+21 (forward-filled closes: a stock with no later close keeps
    its last close, so zero return from there). NaN-free for stocks with a close at or before t."""
    return D['cff'][rows, t + HOLD] / D['cff'][rows, t] - 1.0


def spy_hold(D, t):
    return D['spy_close'][t + HOLD] / D['spy_close'][t] - 1.0


# ---------------------------------------------------------------------------------------------------------------
# baselines

def mom12_1(D, rows, t):
    """Plain 12-1 momentum at decision day t, as stockai/stockai/baselines/momentum.py computes it (read, copied):
    ln(close[t-21] / close[t-252]); NaN where either close is missing or <= 0. Gap-filled closes."""
    old = np.asarray(D['close'][rows, t - 252], np.float64)
    new = np.asarray(D['close'][rows, t - 21], np.float64)
    out = np.full(len(rows), np.nan)
    ok = np.isfinite(old) & np.isfinite(new) & (old > 0) & (new > 0)
    out[ok] = np.log(new[ok] / old[ok])
    return out


def fund_mom12_1(D, frows, t):
    """The twin fund's 12-1 log return from store closes: the sum of its daily log returns over days t-251 .. t-21,
    a daily move beyond +-50% (log) or a missing day counting as zero."""
    c = np.asarray(D['fclose'][frows, t - 252:t - 21 + 1], np.float64)
    with np.errstate(all='ignore'):
        r = np.log(c[:, 1:] / c[:, :-1])
    r[~np.isfinite(r) | (np.abs(r) > MAX_FUND_LOG)] = 0.0
    return r.sum(1)


def decile_weights(s, frac=DECILE):
    """Equal-weight decile long-short over the finite entries of s: $1 long the top round(10% n) 'slots', $1 short
    the bottom ones. A group of tied values that straddles the decile edge shares the remaining slots equally
    (fractional weights), so ties never depend on row order. NaN entries get weight 0."""
    w = np.zeros(len(s))
    idx = np.nonzero(np.isfinite(s))[0]
    n = len(idx)
    m = int(round(frac * n))
    if m == 0:
        return w
    x = s[idx]
    for side, sgn in ((-x, 1.0), (x, -1.0)):
        vals, inv = np.unique(side, return_inverse=True)                 # ascending in side = best first
        filled = 0.0
        for g in range(len(vals)):
            members = idx[inv == g]
            room = m - filled
            if room <= 0:
                break
            take = min(len(members), room)
            w[members] += sgn * (take / len(members)) / m
            filled += take
    return w


# ---------------------------------------------------------------------------------------------------------------
# returns and costs

def turnover(prev_end, new_start):
    keys = set(prev_end) | set(new_start)
    return float(sum(abs(new_start.get(j, 0.0) - prev_end.get(j, 0.0)) for j in keys))


class Book:
    """Consecutive buy-and-hold holds of a weight vector; costs on turnover against the drifted positions."""

    def __init__(self, D):
        self.D = D
        self.prev_end = None
        self.prev_x = None
        self.months = []

    def add(self, k, rows, wts, extra=None):
        D = self.D
        t = int(D['rebal'][k])
        if t + HOLD > D['T'] - 1:
            return None
        nz = wts != 0
        rows, wts = rows[nz], wts[nz]
        g = hold_growth(D, rows, t)
        assert np.isfinite(g).all(), 'missing hold return'
        start = dict(zip(rows.tolist(), wts.tolist()))
        end = dict(zip(rows.tolist(), (wts * (1 + g)).tolist()))
        if self.prev_x is not None and self.prev_x != t:
            self.prev_end = None
        to = turnover(self.prev_end or {}, start)
        gross = float(wts @ g)
        cost = BPS * 1e-4 * to
        m = dict(k=k, t=t, gross=gross, cost=cost, net=gross - cost, turnover=to,
                 long=float(wts[wts > 0] @ g[wts > 0]), short=float(wts[wts < 0] @ g[wts < 0]),
                 long_exp=float(wts[wts > 0].sum()), short_exp=float(-wts[wts < 0].sum()), n_pos=int(len(rows)),
                 spy=float(spy_hold(D, t)))
        if extra:
            m.update(extra)
        self.months.append(m)
        self.prev_end, self.prev_x = end, t + HOLD
        return m


# ---------------------------------------------------------------------------------------------------------------
# statistics

def tstat(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float('nan'), float('nan'), len(x)
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)


def ols(y, X, nw_lags=0):
    """OLS of y on [1, X]; returns coefficients, plain t, Newey-West t (Bartlett, nw_lags), R^2, n."""
    y = np.asarray(y, np.float64)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, np.float64) for x in X])
    n, p = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    s2 = e @ e / (n - p)
    t_plain = b / np.sqrt(np.diag(XtX_inv) * s2)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, nw_lags + 1):
        wl = 1 - L / (nw_lags + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += wl * (G + G.T)
    Vnw = XtX_inv @ S @ XtX_inv * n / (n - p)
    t_nw = b / np.sqrt(np.diag(Vnw))
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return dict(b=b.tolist(), t=t_plain.tolist(), t_nw=t_nw.tolist(), r2=float(r2), n=int(n))


def sharpe(x):
    x = np.asarray(x, np.float64)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252 / HOLD))


def max_drawdown(x):
    c = np.concatenate([[0.0], np.cumsum(x)])
    peak = np.maximum.accumulate(c)
    dd = peak - c
    j = int(np.argmax(dd))
    i = int(np.argmax(c[:j + 1])) if j > 0 else 0
    return float(dd[j]), i, j
