"""Point-in-time price features for the LightGBM baseline (and any other model that wants them).

Every value at decision day d uses data through the close of day d only: windows end at d inclusive, gaps in closes
are filled forward only, and nothing is normalized over the time axis of the panel. So a feature at day d never
changes when data after d changes (tests/test_baselines_features.py checks this bit for bit).

Features at day d, with h the window in trading days:

- vsum_h (h = 1, 5, 21, 63, 252): sum of the census variable v over days d-h+1 .. d, divided by sqrt(h). Missing v
  counts as 0, as in correlation2's census sums. vsum_1 is v[d] itself, which must be present.
- ret_h (h = 1, 5, 21, 63, 252): ln(close[d] / close[d-h]). close[d] must be present. A missing close[d-h] is
  replaced by the last close before it when that is at most FILL_DAYS older; otherwise the feature is missing.
- vol_h (h = 21, 63, 252): standard deviation of the daily log returns ln(close[k] / close[k-1]) (both closes
  present) over k = d-h+1 .. d, from at least 2/3 of the window.
- log_dv_21: ln(1 + median dollar volume over days d-20 .. d), from at least 11 present values.
- v_lag1 .. v_lag4: v on days d-1 .. d-4, missing counted as 0. With vsum_1 = v[d] these are v on the last 5 days.
- mkt_1, mkt_5, mkt_21: the market's log return over days d-h+1 .. d (sum of `market`); every day must be present.

Medians are upper medians (the (k // 2)-th of k sorted present values), as correlation2 takes them.
`complete[j, d]` is True where every feature is finite. `liquid_mask` is the point-in-time training-row filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VSUM_H = (1, 5, 21, 63, 252)
RET_H = (1, 5, 21, 63, 252)
VOL_H = (21, 63, 252)
DV_H = 21
V_LAGS = (1, 2, 3, 4)
MKT_H = (1, 5, 21)
FILL_DAYS = 5

STOCK_FEATURES = (
    *(f"vsum_{h}" for h in VSUM_H),
    *(f"ret_{h}" for h in RET_H),
    *(f"vol_{h}" for h in VOL_H),
    f"log_dv_{DV_H}",
    *(f"v_lag{k}" for k in V_LAGS),
)
MARKET_FEATURES = tuple(f"mkt_{h}" for h in MKT_H)
FEATURES = STOCK_FEATURES + MARKET_FEATURES  # column order of every feature matrix
HISTORY = max(*VSUM_H, *RET_H, *VOL_H)  # days of history the longest window needs


@dataclass
class Features:
    stock: np.ndarray  # float32 [len(STOCK_FEATURES), N, T]
    market: np.ndarray  # float32 [len(MARKET_FEATURES), T]; the same for every stock
    complete: np.ndarray  # bool [N, T]: every feature finite

    @property
    def names(self) -> tuple[str, ...]:
        return FEATURES

    def rows(self, j: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Feature matrix float32 [len(j), len(FEATURES)] for the (stock j, day d) pairs."""
        j, d = np.asarray(j), np.asarray(d)
        x = np.empty((j.size, len(FEATURES)), dtype=np.float32)
        ns = self.stock.shape[0]
        for k in range(ns):
            x[:, k] = self.stock[k][j, d]
        for k in range(self.market.shape[0]):
            x[:, ns + k] = self.market[k][d]
        return x

    def panel(self, name: str) -> np.ndarray:
        """One feature as [N, T] (market features broadcast to every stock)."""
        if name in STOCK_FEATURES:
            return self.stock[STOCK_FEATURES.index(name)]
        m = self.market[MARKET_FEATURES.index(name)]
        return np.broadcast_to(m, self.complete.shape)


def window_sum(x: np.ndarray, h: int) -> np.ndarray:
    """out[..., d] = x[..., d-h+1] + ... + x[..., d] (float64 running sums); NaN for d < h-1. `x` has no NaN.

    out[..., d] depends on x[..., :d+1] only.
    """
    c = np.cumsum(x, axis=-1, dtype=np.float64)
    out = np.full(c.shape, np.nan)
    t = c.shape[-1]
    if h <= t:
        out[..., h - 1] = c[..., h - 1]
        out[..., h:] = c[..., h:] - c[..., : t - h]
    return out


def window_count(mask: np.ndarray, h: int) -> np.ndarray:
    """Number of True values over days d-h+1 .. d (int32); -1 for d < h-1."""
    c = np.cumsum(mask, axis=-1, dtype=np.int32)
    out = np.full(c.shape, -1, dtype=np.int32)
    t = c.shape[-1]
    if h <= t:
        out[..., h - 1] = c[..., h - 1]
        out[..., h:] = c[..., h:] - c[..., : t - h]
    return out


def window_median(x: np.ndarray, h: int, min_count: int, chunk_rows: int = 256) -> np.ndarray:
    """Upper median of the present (finite) values of x over days d-h+1 .. d, float32 [N, T].

    NaN with fewer than `min_count` present values, and for d < h-1.
    """
    n, t = x.shape
    out = np.full((n, t), np.nan, dtype=np.float32)
    if h > t:
        return out
    for a in range(0, n, chunk_rows):
        blk = np.array(x[a : a + chunk_rows], dtype=np.float32)
        blk[~np.isfinite(blk)] = np.nan
        w = np.lib.stride_tricks.sliding_window_view(blk, h, axis=1)  # [rows, t-h+1, h]
        k = h - np.isnan(w).sum(axis=-1)
        s = np.sort(w, axis=-1)  # NaN sort last
        med = np.take_along_axis(s, (k // 2)[..., None], axis=-1)[..., 0]
        med[k < min_count] = np.nan
        out[a : a + chunk_rows, h - 1 :] = med
    return out


def present_close(close: np.ndarray) -> np.ndarray:
    """Closes as float64 with missing or non-positive values as NaN."""
    c = np.array(close, dtype=np.float64)
    c[~(np.isfinite(c) & (c > 0))] = np.nan
    return c


def fill_forward(c: np.ndarray, limit: int) -> np.ndarray:
    """Each missing value replaced by the last present value at most `limit` days before it (past values only)."""
    t = c.shape[1]
    idx = np.where(np.isfinite(c), np.arange(t, dtype=np.int64)[None, :], -1)
    np.maximum.accumulate(idx, axis=1, out=idx)
    ok = (idx >= 0) & (np.arange(t)[None, :] - idx <= limit)
    out = np.take_along_axis(c, np.maximum(idx, 0), axis=1)
    out[~ok] = np.nan
    return out


def lag(x: np.ndarray, k: int, fill=np.nan) -> np.ndarray:
    """out[..., d] = x[..., d-k]; `fill` for d < k."""
    out = np.full(x.shape, fill, dtype=x.dtype)
    if k < x.shape[-1]:
        out[..., k:] = x[..., : x.shape[-1] - k]
    return out


def compute_features(v: np.ndarray, close: np.ndarray, dollar_volume: np.ndarray, market: np.ndarray) -> Features:
    """The feature panel for every stock and day of the given [N, T] arrays (and market [T])."""
    n, t = np.shape(v)
    stock = np.empty((len(STOCK_FEATURES), n, t), dtype=np.float32)

    def put(name: str, x: np.ndarray) -> None:
        stock[STOCK_FEATURES.index(name)] = x

    v64 = np.array(v, dtype=np.float64)
    v_present = np.isfinite(v64)
    v0 = np.where(v_present, v64, 0.0)
    del v64
    for h in VSUM_H:
        s = window_sum(v0, h) / np.sqrt(h)
        if h == 1:
            s[~v_present] = np.nan  # v[d] itself must be present
        put(f"vsum_{h}", s)
    for k in V_LAGS:
        put(f"v_lag{k}", lag(v0, k))
    del v0

    c = present_close(close)
    filled = fill_forward(c, FILL_DAYS)
    with np.errstate(divide="ignore", invalid="ignore"):
        logc, logf = np.log(c), np.log(filled)
        for h in RET_H:
            put(f"ret_{h}", logc - lag(logf, h))  # NaN unless close[d] is present
        del logf, filled
        r = logc - lag(logc, 1)  # daily log return, both closes present
    del logc, c
    r_present = np.isfinite(r)
    r0 = np.where(r_present, r, 0.0)
    del r
    for h in VOL_H:
        k = window_count(r_present, h)
        s1, s2 = window_sum(r0, h), window_sum(r0 * r0, h)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = s1 / k
            var = np.maximum(s2 / k - mean * mean, 0.0)
        sd = np.sqrt(var)
        sd[k < -(-2 * h // 3)] = np.nan
        put(f"vol_{h}", sd)
        del s1, s2, mean, var, sd, k
    del r0, r_present

    med = window_median(dollar_volume, DV_H, min_count=DV_H // 2 + 1)
    put(f"log_dv_{DV_H}", np.log1p(np.maximum(med, 0.0)))  # NaN stays NaN
    del med

    m = np.asarray(market, dtype=np.float64)
    m_present = np.isfinite(m)
    m0 = np.where(m_present, m, 0.0)
    mk = np.empty((len(MARKET_FEATURES), t), dtype=np.float32)
    for i, h in enumerate(MKT_H):
        s = window_sum(m0, h)
        s[window_count(m_present, h) < h] = np.nan
        mk[i] = s

    complete = np.isfinite(mk).all(axis=0)[None, :] & np.ones((n, 1), dtype=bool)
    for k in range(stock.shape[0]):
        complete &= np.isfinite(stock[k])
    return Features(stock=stock, market=mk, complete=complete)


def liquid_mask(close: np.ndarray, dollar_volume: np.ndarray, min_price: float = 5.0, min_dollar_volume: float = 5e6,
                window: int = 63) -> np.ndarray:
    """bool [N, T]: close[d] >= min_price and the upper median of dollar volume over days d-window+1 .. d is
    >= min_dollar_volume, from at least half the window present. Uses data through day d only.

    The median test needs no sort: the (k // 2)-th smallest of k values is >= X exactly when at most k // 2 are < X.
    """
    c = np.asarray(close, dtype=np.float32)
    dv = np.asarray(dollar_volume, dtype=np.float32)
    dv_present = np.isfinite(dv)
    k = window_count(dv_present, window)
    below = window_count(dv_present & (dv < np.float32(min_dollar_volume)), window)
    with np.errstate(invalid="ignore"):
        price_ok = np.isfinite(c) & (c >= np.float32(min_price))
    return price_ok & (k >= -(-window // 2)) & (below <= k // 2)
