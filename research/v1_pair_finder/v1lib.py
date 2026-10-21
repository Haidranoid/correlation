"""v1's pair finder (correlation-original, 2025), re-implemented exactly and as matrix products.

v1's method, from its code (app/services/correlation_analysis_service.rb#find_best_correlation, app/models/stock_price.rb):
- % change over `window` of the stock's own trading rows, every row (overlapping): (close[i] - close[i-w]) / close[i-w] * 100.
- Pairing at offset k: A's change dated t with B's change dated exactly t + k calendar days (no data if that is not
  one of B's trading days).
- Offsets 0 .. the largest multiple of 50 below (overlap span - window) in calendar days, every one of them tested.
- Pearson r on raw % changes (no market adjustment), over the whole overlapping history.
- Kept ("statistically relevant") if |r| >= 0.3 (and r^2 >= 0.09, the same thing) with at least 10 points.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import numpy as np

import c2eval

R_KEEP = 0.3
N_MIN = 10
WINDOWS_RUN = (7, 14, 21, 30, 40)     # windows v1 was actually run with (saved rows; 20 is folded into 21)
WINDOWS_README = (60, 90, 180)        # further windows its README advertises (30, 60, 90, 180)
WINDOWS = WINDOWS_RUN + WINDOWS_README


# ---------------------------------------------------------------- exact per-pair v1 (for checks)

def v1_changes(dates, closes, w):
    """v1's StockPrice.calculate_percentage_changes: dict date -> % change over w rows."""
    ok = np.isfinite(closes) & (closes != 0)
    d, c = np.asarray(dates)[ok], np.asarray(closes, dtype=np.float64)[ok]
    if len(c) <= w:
        return {}
    ch = (c[w:] - c[:-w]) / c[:-w] * 100.0
    return dict(zip(d[w:].tolist(), ch.tolist()))


def v1_max_offset(dA, dB, w1, w2):
    """find_actual_max_viable_offset: the largest multiple of 50 with max(minA+w1, minB+w2) < min(maxA, maxB) - k."""
    start = max(dA.min() + w1, dB.min() + w2)
    end = min(dA.max(), dB.max())
    k = 0
    while start < end - (k + 50):
        k += 50
    return k if start < end - k else -1


def v1_corr(chA, chB, k):
    """align_data_points + the inline Pearson: (r, n) or (None, n)."""
    xs, ys = [], []
    for t, x in chA.items():
        y = chB.get(t + k)
        if y is not None:
            xs.append(x)
            ys.append(y)
    n = len(xs)
    if n < 2:
        return None, n
    x, y = np.array(xs), np.array(ys)
    x, y = x - x.mean(), y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return (None if den == 0 else float((x * y).sum() / den)), n


# ---------------------------------------------------------------- data

def load_panel(export=c2eval.EXPORT_OLD):
    ex = c2eval.load_export(export, arrays=('close', 'dollar_volume'))
    return ex


MAX_MISSING = 0.01


def select_universe(ex, s, e, size=500, min_price=5.0):
    """Stocks listed throughout [s, e) (a close on its first day) with at most 1% of closes missing, median close >= $5
    and ranked by median dollar volume over the last 252 days before e. Uses only data before e. Returns sorted row
    indices. Missing closes are carried forward by ffill() (v1 would skip the row)."""
    close = np.asarray(ex['close'][:, s:e])
    fin = np.isfinite(close) & (close > 0)
    full = fin[:, 0] & ((~fin).mean(axis=1) <= MAX_MISSING)
    tail = slice(max(0, e - s - 252), e - s)
    with np.errstate(invalid='ignore'):
        medp = np.nanmedian(close[:, tail], axis=1)
        dv = np.asarray(ex['dollar_volume'][:, s:e])[:, tail]
        meddv = np.nanmedian(np.where(np.isfinite(dv), dv, np.nan), axis=1)
    ok = full & (medp >= min_price) & np.isfinite(meddv)
    idx = np.where(ok)[0]
    idx = idx[np.argsort(-meddv[idx], kind='stable')][:size]
    return np.sort(idx)


def ffill(close):
    """[N, L] closes with each missing value replaced by the last earlier close (leading NaNs stay NaN). Causal."""
    c = np.where(np.isfinite(close) & (close > 0), close, np.nan)
    idx = np.where(np.isfinite(c), np.arange(c.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = c[np.arange(c.shape[0])[:, None], idx]
    return out


def pct_changes(close, w):
    """[N, L] -> [N, L] % change over w rows, NaN for the first w columns (v1's overlapping changes)."""
    out = np.full(close.shape, np.nan, np.float64)
    out[:, w:] = (close[:, w:] - close[:, :-w]) / close[:, :-w] * 100.0
    return out


def flip_null_prices(close, seed):
    """The sign-flip null adapted to overlapping windows: every daily return is multiplied by a random +-1 that is the
    same for every stock on that day (as in the census), then the price paths are rebuilt so the w-row % changes keep
    v1's overlap. Returns new [N, L] closes."""
    rng = np.random.default_rng(seed)
    ret = close[:, 1:] / close[:, :-1] - 1.0
    s = rng.choice([-1.0, 1.0], size=ret.shape[1])
    ret = np.clip(ret * s[None, :], -0.95, None)
    out = np.empty_like(close)
    out[:, 0] = close[:, 0]
    out[:, 1:] = close[:, :1] * np.cumprod(1.0 + ret, axis=1)
    return out


# ---------------------------------------------------------------- the search as matrix products

def offset_pairs(cal, valid_x, valid_y, k):
    """Index pairs (i, j) with cal[j] == cal[i] + k, valid_x[i] and valid_y[j] (calendar days matched exactly)."""
    pos = np.searchsorted(cal, cal + k)
    pos_c = np.minimum(pos, len(cal) - 1)
    hit = (pos < len(cal)) & (cal[pos_c] == cal + k) & valid_x
    i = np.where(hit)[0]
    j = pos_c[i]
    keep = valid_y[j]
    return i[keep], j[keep]


def corr_block(X, Y):
    """Pearson r of every row of X [N, n] with every row of Y [M, n]: float32 data and matmul, float64 sums (agrees
    with a float64 computation to ~4e-7). NaN where a row is constant or has a NaN."""
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    Xc = X - X.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    Yc = Y - Y.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    nx = np.sqrt(np.einsum('ij,ij->i', Xc, Xc, dtype=np.float64)).astype(np.float32)
    ny = np.sqrt(np.einsum('ij,ij->i', Yc, Yc, dtype=np.float64)).astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        Xc /= nx[:, None]
        Yc /= ny[:, None]
        r = Xc @ Yc.T
    r[~np.isfinite(r)] = np.nan
    return r


def max_offset(cal_span, w):
    """v1's offset range for a period whose closes span `cal_span` calendar days (both stocks cover it fully)."""
    k = 0
    while w < cal_span - (k + 50):
        k += 50
    return k


def keep_mask(r, n):
    m = np.abs(r) >= R_KEEP
    if n < N_MIN:
        m[:] = False
    np.fill_diagonal(m, False)          # v1 refuses A == B
    return m
