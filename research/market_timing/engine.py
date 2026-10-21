"""Co-movement market timing: the monthly engine shared by the backtest (old export) and the unseen months (new).

A strategy is a weight in SPY per calendar month (0.5 or 1 for the rules; anything in [0, 1] for the volatility-managed
baseline), bought at the close of t0 and held to the close of t0 + 21, the rest in cash at 0.
"""
import numpy as np

import common as C

DPY = 252  # trading days a year


def rel_paths(P, t0s):
    """[M, 22] SPY price relative to the month's start: P[t0 + j] / P[t0], j = 0..21."""
    idx = np.asarray(t0s)[:, None] + np.arange(C.H + 1)[None, :]
    R = P[idx] / P[np.asarray(t0s)][:, None]
    assert np.isfinite(R).all()
    return R


def daily_returns(w, rel):
    """Daily returns [..., M, 21] of buying w of SPY at t0 and holding (no daily rebalancing)."""
    v = 1.0 + np.asarray(w, float)[..., None] * (rel - 1.0)
    return v[..., 1:] / v[..., :-1] - 1.0


def measures(dr):
    """Annual return (compound), Sharpe (rf 0, daily, x sqrt 252), maximum drawdown from the daily path, for [M, 21]."""
    f = dr.ravel()
    lg = np.log1p(f).sum()
    cagr = np.exp(lg * DPY / len(f)) - 1.0
    sharpe = f.mean() / f.std(ddof=1) * np.sqrt(DPY)
    v = np.concatenate([[1.0], np.cumprod(1.0 + f)])
    mdd = float(np.max(1.0 - v / np.maximum.accumulate(v)))
    return {'annual': float(cagr), 'sharpe': float(sharpe), 'mdd': mdd}


def summarize(w, rel):
    m = measures(daily_returns(w, rel))
    m['exposure'] = float(np.mean(w))
    return m


def vol_weights(var, target):
    """min(1, c / var) with c chosen by bisection so the mean weight equals target. Returns (weights, c)."""
    var = np.asarray(var, float)
    lo, hi = np.log(var.min() * 1e-6), np.log(var.max() * 1e6)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.minimum(1.0, np.exp(mid) / var).mean() < target:
            lo = mid
        else:
            hi = mid
    c = np.exp(0.5 * (lo + hi))
    return np.minimum(1.0, c / var), float(c)


def random_timing(w, rel, n=C.N_RANDOM, seed=C.SEED, chunk=500):
    """Random orderings of the rule's monthly weights (same months, same count of each weight). Per draw: annual
    return, Sharpe and maximum drawdown."""
    rng = np.random.default_rng(seed)
    w = np.asarray(w, float)
    M = len(w)
    out = {'annual': np.empty(n), 'sharpe': np.empty(n), 'mdd': np.empty(n)}
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        W = np.stack([rng.permutation(w) for _ in range(b - a)])                 # [D, M]
        dr = daily_returns(W, rel[None, :, :]).reshape(b - a, M * C.H)           # [D, M*21]
        lg = np.log1p(dr).sum(1)
        out['annual'][a:b] = np.exp(lg * DPY / dr.shape[1]) - 1.0
        out['sharpe'][a:b] = dr.mean(1) / dr.std(1, ddof=1) * np.sqrt(DPY)
        v = np.concatenate([np.ones((b - a, 1)), np.cumprod(1.0 + dr, 1)], 1)
        out['mdd'][a:b] = np.max(1.0 - v / np.maximum.accumulate(v, 1), 1)
    return out


def random_p(rule_value, draws):
    """One-sided p: (1 + draws >= the rule's) / (1 + n)."""
    return float((1 + np.sum(draws >= rule_value)) / (1 + len(draws)))


def block_bootstrap_sharpe(dr_a, dr_b, n=C.N_BOOT, block=C.BLOCK, seed=C.SEED + 1):
    """Paired circular block bootstrap of months for Sharpe(a) - Sharpe(b). dr_*: [M, 21] daily returns. Returns the
    point difference, p = share of draws with difference <= 0, and the 5th and 95th percentiles."""
    rng = np.random.default_rng(seed)
    M = dr_a.shape[0]
    s1a, s2a = dr_a.sum(1), (dr_a ** 2).sum(1)
    s1b, s2b = dr_b.sum(1), (dr_b ** 2).sum(1)
    nb = int(np.ceil(M / block))
    starts = rng.integers(0, M, size=(n, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % M).reshape(n, nb * block)[:, :M]
    nd = M * dr_a.shape[1]

    def sharpe(s1, s2):
        S1, S2 = s1[idx].sum(1), s2[idx].sum(1)
        mu = S1 / nd
        var = (S2 - nd * mu ** 2) / (nd - 1)
        return mu / np.sqrt(var) * np.sqrt(DPY)

    d = sharpe(s1a, s2a) - sharpe(s1b, s2b)
    point = measures(dr_a)['sharpe'] - measures(dr_b)['sharpe']
    return {'diff': float(point), 'p': float(np.mean(d <= 0)), 'lo5': float(np.percentile(d, 5)),
            'hi95': float(np.percentile(d, 95))}


def signal_weights(x, k_list, valid, direction):
    """Monthly weights for the calendar indices k_list: 1 when x[k] is strictly above (direction +1) or strictly below
    (direction -1) the median of the previous values (only those where valid), else 0.5. Also the medians."""
    xv = np.where(valid, x, np.nan)
    med = np.array([C.trailing_median(xv, k) for k in k_list])
    cur = x[np.asarray(k_list)]
    hi = cur > med if direction > 0 else cur < med
    w = np.where(hi, 1.0, 0.5)
    w[~np.isfinite(med)] = np.nan
    return w, med


def switches(w):
    return int(np.sum(np.abs(np.diff(w)) > 0))


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float('nan'), float('nan')
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float(d), float(d / se)


def corr_t(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    return r, float(r * np.sqrt(n - 2) / np.sqrt(1 - r * r))
