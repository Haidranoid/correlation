"""Time-series momentum on 23 asset-class funds (experiments/+1.2-trend-following.md): the published rule of
Moskowitz, Ooi & Pedersen (2012) on Yahoo's dividend-adjusted fund closes.

Rule (fixed in the plan before results):
  - month-ends = the last trading day of each calendar month in the union calendar of the funds' adjusted closes;
  - a fund is active at month-end m if it has an adjusted close at month-end m and at month-end m-12;
  - signal = sign of the 12-month excess return, log(P_m / P_{m-12}) - sum log(1 + rf_d) over the days in between;
  - sigma = ex-ante annualized volatility of daily excess returns, exponentially weighted with center of mass 60 days
    (delta = 60/61), weights normalized over the fund's own history, demeaned by the weighted mean, x 252;
  - weight = sign x (10% / sigma) / N_active, set at the close of month-end m and held to the close of month-end m+1;
  - monthly position return = w x (R_fund - rf_month) (buy and hold within the month); the account earns rf on its
    capital plus the overlay;
  - costs: c x sum |w_new - w_drifted| at each month-end (w_drifted = last month's weights after the month's moves),
    charged to the month the new weights are held.

Everything is monthly. Point in time: weights set at month-end m use closes and T-bill rates through day m only.

    import common; P = common.build('2025-06-30')     # the walk-forward data; '2026-10-19' only in unseen.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data')
import pubdata  # noqa: E402

WORK = Path.home() / '.local/share/correlation2/experiments/trend_following'

KINDS = {
    'Equities': ('SPY', 'IWM', 'EFA', 'EEM', 'EWJ', 'VGK', 'FXI'),
    'Bonds': ('TLT', 'IEF', 'LQD', 'HYG', 'TIP', 'EMB'),
    'Commodities': ('GLD', 'SLV', 'USO', 'UNG', 'DBC', 'DBA'),
    'Currencies': ('UUP', 'FXE', 'FXY'),
    'Real estate': ('VNQ',),
}
TICKERS = tuple(t for v in KINDS.values() for t in v)
KIND_OF = {t: k for k, v in KINDS.items() for t in v}
assert sorted(TICKERS) == sorted(pubdata.ETF_TICKERS)

TARGET_VOL = 0.10
COM = 60.0
ANN = 252
LOOKBACK = 12
COST_BPS = 2.0
END_MAIN = '2025-06-30'     # last close used before the unseen months
END_UNSEEN = '2026-10-19'   # the unseen months end here (data runs to 2026-10-20; that day is not used)
SEED = 20261021


def ewma_sigma(xr):
    """xr [T, F] daily excess returns (NaN = no return). Annualized EW volatility [T, F] through each day, center of
    mass COM days, weights (1-d) d^i normalized over the available history, demeaned by the EW mean."""
    d = COM / (COM + 1.0)
    T, F = xr.shape
    s0 = np.zeros(F); s1 = np.zeros(F); s2 = np.zeros(F)
    out = np.full((T, F), np.nan)
    for t in range(T):
        x = xr[t]
        ok = np.isfinite(x)
        s0[ok] = d * s0[ok] + 1.0
        s1[ok] = d * s1[ok] + x[ok]
        s2[ok] = d * s2[ok] + x[ok] ** 2
        with np.errstate(all='ignore'):
            m = s1 / s0
            v = s2 / s0 - m ** 2
        out[t] = np.where(s0 > 0, np.sqrt(np.maximum(v, 0.0) * ANN), np.nan)
    return out


def load_daily(end_iso):
    """Union-calendar dates through end_iso, prices [T, F] in TICKERS order, rf [T], daily simple returns [T, F]."""
    dates, px = pubdata.etf_adjclose(TICKERS)
    keep = dates <= np.datetime64(end_iso)
    dates = dates[keep]
    P = np.stack([px[t][keep] for t in TICKERS], 1)
    rf = pubdata.rf_daily(dates)
    r = np.full(P.shape, np.nan)
    with np.errstate(all='ignore'):
        r[1:] = P[1:] / P[:-1] - 1.0
    return dates, P, rf, r


def month_ends(dates):
    """Indices of the last date of each calendar month present in dates."""
    ym = dates.astype('datetime64[M]')
    last = np.nonzero(np.r_[ym[1:] != ym[:-1], True])[0]
    return last


def build(end_iso=END_MAIN):
    """Run the rule on data through end_iso. Returns a dict of monthly arrays. Months are RETURN months: entry k is
    held from the close of month-end k-1's date to the close of month-end k's date; 'label' is that month (YYYY-MM).
    The final month is partial if end_iso is not a month-end."""
    dates, P, rf, r = load_daily(end_iso)
    T, F = P.shape
    xr = r - rf[:, None]
    sig = ewma_sigma(xr)
    me = month_ends(dates)
    K = len(me)
    lrf = np.log1p(rf)
    clrf = np.r_[0.0, np.cumsum(lrf)]  # clrf[t+1] = sum lrf[0..t]

    # Signals and weights at each month-end k (used for return month k+1).
    sign = np.zeros((K, F)); active = np.zeros((K, F), bool); sigma_m = np.full((K, F), np.nan)
    xs12 = np.full((K, F), np.nan)
    for k in range(LOOKBACK, K):
        a, b = me[k - LOOKBACK], me[k]
        ok = np.isfinite(P[a]) & np.isfinite(P[b]) & np.isfinite(sig[b]) & (sig[b] > 0)
        with np.errstate(all='ignore'):
            x = np.log(P[b] / P[a]) - (clrf[b + 1] - clrf[a + 1])
        xs12[k] = np.where(ok, x, np.nan)
        active[k] = ok
        sign[k] = np.where(ok, np.sign(x), 0.0)
        sigma_m[k] = np.where(ok, sig[b], np.nan)
    n_active = active.sum(1)
    with np.errstate(all='ignore'):
        wabs = np.where(active, TARGET_VOL / sigma_m / np.maximum(n_active, 1)[:, None], 0.0)
    w = sign * wabs

    # Monthly fund returns and T-bill returns for return month k+1 (between me[k] and me[k+1]).
    R = np.full((K, F), np.nan); rfm = np.full(K, np.nan)
    for k in range(K - 1):
        a, b = me[k], me[k + 1]
        with np.errstate(all='ignore'):
            R[k + 1] = P[b] / P[a] - 1.0
        rfm[k + 1] = np.expm1(clrf[b + 1] - clrf[a + 1])

    # Shift so index k = return month k: weights held during month k were set at me[k-1].
    W = np.zeros((K, F)); W[1:] = w[:-1]
    WA = np.zeros((K, F)); WA[1:] = wabs[:-1]
    S = np.zeros((K, F)); S[1:] = sign[:-1]
    ACT = np.zeros((K, F), bool); ACT[1:] = active[:-1]
    XR = R - rfm[:, None]
    held = ACT & np.isfinite(XR)
    assert (held == ACT).all(), 'an active fund lacks a return over its holding month'
    contrib = np.where(ACT, W * np.nan_to_num(XR), 0.0)
    overlay = contrib.sum(1)
    total = rfm + overlay

    # Turnover at me[k-1] into month k's weights: |W[k] - drifted W[k-1]|.
    turn_f = np.zeros((K, F))
    prev_drift = np.zeros(F)
    for k in range(1, K):
        turn_f[k] = np.abs(W[k] - prev_drift)
        g = 1.0 + total[k]
        prev_drift = np.where(ACT[k], W[k] * (1.0 + np.nan_to_num(R[k])) / g, 0.0)
    labels = dates[me].astype('datetime64[M]')
    return {
        'dates': dates, 'me': me, 'me_date': dates[me], 'label': labels, 'P': P, 'rf': rf, 'sig_daily': sig,
        'R': R, 'rfm': rfm, 'XR': XR, 'W': W, 'WA': WA, 'S': S, 'ACT': ACT, 'n_active': ACT.sum(1),
        'contrib': contrib, 'overlay_gross': overlay, 'turn_f': turn_f, 'turnover': turn_f.sum(1),
        'gross_exposure': np.abs(W).sum(1), 'xs12_at': xs12, 'sigma_at': sigma_m, 'sign_at': sign,
    }


def net(Pd, bps=COST_BPS):
    """Monthly overlay (excess) return net of bps per side, and the per-fund contribution net of each fund's cost."""
    c = bps / 1e4
    cf = Pd['contrib'] - c * Pd['turn_f']
    return Pd['overlay_gross'] - c * Pd['turnover'], cf


def window(Pd, start_ym, end_ym):
    """Boolean mask over return months with start_ym <= label <= end_ym (strings 'YYYY-MM')."""
    lab = Pd['label']
    return (lab >= np.datetime64(start_ym, 'M')) & (lab <= np.datetime64(end_ym, 'M'))


def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 else float('nan')


def nw_t(x, lags=6):
    """Newey-West t of the mean with Bartlett weights."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    e = x - x.mean()
    v = e @ e / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / np.sqrt(v / n))


def sharpe(x):
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(12))


def max_drawdown(x):
    """Largest peak-to-trough fall of the compounded series (1 + x)."""
    c = np.cumprod(1.0 + np.asarray(x, float))
    c = np.r_[1.0, c]
    peak = np.maximum.accumulate(c)
    return float((c / peak - 1.0).min())


def cagr(x, axis=-1):
    x = np.asarray(x, float)
    n = x.shape[axis]
    return np.exp(np.log1p(x).sum(axis) * 12.0 / n) - 1.0


def calendar_month_join(labels, dates, values):
    """Align a monthly series keyed by calendar month-end dates (pubdata) to our month labels: float [len(labels)]."""
    ym = np.asarray(dates, 'datetime64[D]').astype('datetime64[M]')
    pos = np.searchsorted(ym, labels)
    ok = (pos < len(ym)) & (ym[np.minimum(pos, len(ym) - 1)] == labels)
    out = np.full(len(labels), np.nan)
    out[ok] = np.asarray(values, float)[pos[ok]]
    return out
