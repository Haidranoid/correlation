"""Cost-aware trading (experiments/-5.6-cost-aware-trading.md): shared data, calendars and accounts.

Everything here copies stockai's conventions (read-only, nothing imported from stockai):
  - test years Y = 2010..2023 run [Y-06-28, (Y+1)-06-28): h0 = first export day on/after Y-06-28, t = first on/after
    (Y+1)-06-28 (stockai/stockai/h30/targets.py year_spec; judge main.rs walk-forward);
  - the year's universe is the grader's liquid_universe at h0 (window 756, >= 90% finite census values, upper median
    close >= $5 and dollar volume >= $5M over the 252 days before h0), fixed for the year (targets.judge_universe);
  - the grader's decision days are h0-1 .. t-2, rebalancing every 5 (k % 5 == 0); the 700 graded weeks are those
    rebalance days whose 5-day window ends by the next year's first decision (targets.non_overlapping) and inside the
    development columns (cutoff = index of 2024-06-28 minus 1);
  - a day's return is close[d+1]/close[d]-1 in float32 when both closes are finite and close[d] > 0, else 0
    (holdout.rs `ret`).
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)
WORK = c2eval.WORK / 'cost_aware_trading'
WORK.mkdir(parents=True, exist_ok=True)
SA = Path('/home/bhoult/programming/correlation/stockai')
SCORES = {
    'lgbm_s4': SA / 'runs/1w/jitter/lgbm_s4.npy',      # what produced $19,052 (median of the 5 random tie-breaks)
    'lgbm': SA / 'runs/1w/lgbm/scores.npy',            # ties by ticker order ($13,953)
    'reversal': SA / 'runs/1w/rules/reversal.npy',      # minus the 5-day log return
    'lgbm_daily': SA / 'runs/poc/baselines/lgbm_ret/scores.npy',  # next-day actual-return LightGBM
}
for _s in range(1, 6):
    SCORES[f'lgbm_s{_s}'] = SA / f'runs/1w/jitter/lgbm_s{_s}.npy'

YEARS = list(range(2010, 2024))
SEL_YEARS = list(range(2010, 2017))
TEST_YEARS = list(range(2017, 2024))
H = 5
WINDOW, LOOKBACK, MIN_PRICE, MIN_DV = 756, 252, 5.0, 5e6
COST = 20.0           # bps per stock trade (primary)
SPY_COST = 1.0        # bps per SPY trade
ROUND_TRIP = 2 * COST / 1e4

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def load():
    E = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close', 'dollar_volume', 'v'))
    return E


def first_on_or_after(days, iso):
    return c2eval.day_index(days, iso)


def cutoff(days):
    """stockai's development cutoff: columns [0, cutoff) may be read; cutoff - 1 is 2024-06-26."""
    return first_on_or_after(days, '2024-06-28') - 1


def judge_universe(E, h0):
    """targets.judge_universe (holdout.rs liquid_universe), adjusted closes for the $5 test (the Stooq export)."""
    v, close, dv = E['v'], E['close'], E['dollar_volume']
    w0 = max(h0 - WINDOW, 0)
    n_days = h0 - w0
    tail = min(n_days, LOOKBACK)
    vv = np.asarray(v[:, w0:h0], dtype=np.float32)
    enough = np.isfinite(vv).sum(axis=1) * 10 >= n_days * 9
    cc = np.asarray(close[:, h0 - tail:h0], dtype=np.float32)
    dd = np.asarray(dv[:, h0 - tail:h0], dtype=np.float32)

    def upper_median(x):
        k = np.isfinite(x).sum(axis=1)
        s = np.sort(np.where(np.isfinite(x), x, np.inf), axis=1)
        idx = np.minimum(k // 2, x.shape[1] - 1)
        m = np.take_along_axis(s, idx[:, None], axis=1)[:, 0]
        return np.where(k > 0, m, np.float32(0.0))

    ok = enough & (upper_median(cc) >= np.float32(MIN_PRICE)) & (upper_median(dd) >= np.float32(MIN_DV))
    return np.flatnonzero(ok)


def year_specs(E, h=H):
    """Per test year: h0, t, universe, the grader's rebalance days and the graded decision days (stockai's)."""
    days = E['days']
    cut = cutoff(days)
    specs = []
    for y in YEARS:
        h0 = first_on_or_after(days, f'{y}-06-28')
        t = first_on_or_after(days, f'{y + 1}-06-28')
        reb = np.arange(h0 - 1, t - 1, h)
        graded = reb + h <= cut - 1 + 0  # window's last close must be a development column (< cutoff)
        # targets.year_spec: graded = reb + h <= cutoff - 1 with cutoff = dev.cutoff (number of readable columns)
        specs.append(dict(year=y, h0=h0, t=t, universe=judge_universe(E, h0), reb=reb, decisions=reb[graded]))
    for k, s in enumerate(specs):  # targets.non_overlapping
        if k + 1 < len(specs):
            nxt = specs[k + 1]['decisions'][0]
            s['decisions'] = s['decisions'][s['decisions'] + h <= nxt]
    return specs


def day_returns(close, d):
    """holdout.rs ret(j, d) for every stock: float32 close[d+1]/close[d]-1, 0 where either is missing."""
    a = np.asarray(close[:, d], dtype=np.float32)
    b = np.asarray(close[:, d + 1], dtype=np.float32)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (b / a - np.float32(1.0)).astype(np.float64)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0)
    return np.where(ok, r, 0.0)


class Returns:
    """Daily returns ret[d] (the day after decision day d) for columns d in [d_lo, d_hi), as float64 [n_days, N];
    optionally with pubdata.div_fix added to the daily log return from 2022-07-01 on."""

    def __init__(self, E, d_lo, d_hi, div_fix=False):
        self.d_lo, self.d_hi = d_lo, d_hi
        close = E['close']
        out = np.empty((d_hi - d_lo, close.shape[0]), dtype=np.float64)
        for i, d in enumerate(range(d_lo, d_hi)):
            out[i] = day_returns(close, d)
        if div_fix:
            import pubdata
            fx = pubdata.div_fix('old')
            start = first_on_or_after(E['days'], '2022-07-01')
            for i, d in enumerate(range(d_lo, d_hi)):
                if d + 1 >= start:
                    f = np.asarray(fx[:, d + 1], dtype=np.float64)
                    nz = f != 0
                    if nz.any():
                        out[i, nz] = (1.0 + out[i, nz]) * np.exp(f[nz]) - 1.0
        self.r = out

    def __call__(self, d):
        return self.r[d - self.d_lo]


def spy_returns(E):
    """SPY's daily simple return from Yahoo's adjusted close on export days: spy[d] = adj[d+1]/adj[d]-1 (NaN-safe:
    0 where missing)."""
    import pubdata
    dates, cols = pubdata.etf_adjclose(['SPY'])
    adj = pubdata.on_days(dates, cols['SPY'], E['days'])
    r = np.zeros(len(adj))
    ok = np.isfinite(adj[:-1]) & np.isfinite(adj[1:])
    r[:-1][ok] = adj[1:][ok] / adj[:-1][ok] - 1.0
    return r, adj


def save_json(obj, name):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    return p
