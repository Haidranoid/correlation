"""Causal (no look-ahead) seasonal adjustment for NSA monthly or quarterly series.

For period length p (12 monthly, 4 quarterly):
  dev[t]  = v[t] - centred 2xp moving average at t   (needs v[t-p/2 .. t+p/2], all present)
  s[t]    = median of dev[t-p], dev[t-2p], ..., dev[t-K*p] over the available ones (>= min_years)
  sa[t]   = v[t] - s[t]
v = x (additive, for rates) or log(x) (multiplicative, for levels; result exp'ed back).

Causality: dev[t-k*p] uses v up to t-k*p+p/2 <= t-p/2, i.e. only data at least 6 months
(2 quarters) older than t, all published before x[t] itself. The factors are NOT renormalised to
sum to zero within a year, because that would mix in factors computed from later data.
The median over the last K=5 years damps one-off shocks (April 2020) in the factors.
"""
import numpy as np


def centred_ma(v, p):
    n = len(v)
    h = p // 2
    out = np.full(n, np.nan)
    w = np.ones(p + 1)
    w[0] = w[-1] = 0.5
    w /= p
    for t in range(h, n - h):
        seg = v[t - h:t + h + 1]
        if np.all(np.isfinite(seg)):
            out[t] = float(np.dot(w, seg))
    return out


def causal_sa(x, p=12, log=False, K=5, min_years=3):
    x = np.asarray(x, dtype=float)
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(x > 0, np.log(x), np.nan)
    else:
        v = x.copy()
    dev = v - centred_ma(v, p)
    n = len(v)
    s = np.full(n, np.nan)
    for t in range(n):
        past = [dev[t - k * p] for k in range(1, K + 1) if t - k * p >= 0 and np.isfinite(dev[t - k * p])]
        if len(past) >= min_years:
            s[t] = float(np.median(past))
    sa = v - s
    return np.exp(sa) if log else sa


def causal_sa_matrix(X, p=12, log=False, K=5, min_years=3):
    """Vectorised over rows (areas) of X (areas x periods)."""
    X = np.asarray(X, dtype=float)
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            V = np.where(X > 0, np.log(X), np.nan)
    else:
        V = X.copy()
    a, n = V.shape
    h = p // 2
    w = np.ones(p + 1)
    w[0] = w[-1] = 0.5
    w /= p
    MA = np.full((a, n), np.nan)
    for t in range(h, n - h):
        seg = V[:, t - h:t + h + 1]
        ok = np.all(np.isfinite(seg), axis=1)
        MA[ok, t] = seg[ok] @ w
    D = V - MA
    S = np.full((a, n), np.nan)
    for t in range(n):
        ks = [t - k * p for k in range(1, K + 1) if t - k * p >= 0]
        if len(ks) < min_years:
            continue
        past = D[:, ks]
        cnt = np.sum(np.isfinite(past), axis=1)
        with np.errstate(all="ignore"):
            med = np.nanmedian(np.where(np.isfinite(past), past, np.nan), axis=1) if past.size else np.full(a, np.nan)
        med[cnt < min_years] = np.nan
        S[:, t] = med
    SA = V - S
    return np.exp(SA) if log else SA


if __name__ == "__main__":
    # Self-test: a known seasonal pattern + trend + noise is recovered, and no future values are used.
    rng = np.random.default_rng(0)
    n = 240
    t = np.arange(n)
    seas = np.tile([0.8, 0.6, 0.3, -0.1, -0.4, -0.3, 0.1, 0.0, -0.3, -0.4, -0.2, 0.0], n // 12)
    trend = 5 + np.cumsum(rng.normal(0, 0.05, n))
    x = trend + seas + rng.normal(0, 0.05, n)
    sa = causal_sa(x)
    err = (sa - trend)[60:]
    print("additive: rmse vs trend", float(np.sqrt(np.nanmean(err ** 2))), "raw rmse", float(np.sqrt(np.mean((x - trend)[60:] ** 2))))
    # look-ahead check: changing the future must not change the past SA values
    x2 = x.copy()
    x2[150:] += 3.0
    sa2 = causal_sa(x2)
    first_change = int(np.argmax(np.abs(np.nan_to_num(sa2 - sa)) > 1e-12))
    print("first SA index changed by perturbing t>=150:", first_change, "(must be >= 150)")
    m = causal_sa_matrix(np.vstack([x, x2]))
    print("matrix == scalar:", bool(np.allclose(np.nan_to_num(m[0]), np.nan_to_num(sa))), bool(np.allclose(np.nan_to_num(m[1]), np.nan_to_num(sa2))))
    q = np.exp(trend[::3] / 5 + np.tile([0.02, 0.05, -0.01, -0.06], n // 12))
    qs = causal_sa(q, p=4, log=True)
    print("quarterly log: rmse", float(np.sqrt(np.nanmean((np.log(qs) - trend[::3] / 5)[20:] ** 2))))
