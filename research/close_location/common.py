"""Close location (experiments/+2.5-close-location.md): shared constants and helpers.

Prices: high, low and close for CLV come from the unrepaired store (stocks by E['ids'], twin funds by tw['fund']);
every return comes from the judge's export closes (repaired, actual returns). Everything is point in time.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import numpy as np  # noqa: E402
import c2eval  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
BIG = c2eval.WORK / 'close_location'
SRC = {'old': (c2eval.EXPORT_OLD, c2eval.STORE_OLD), 'new': (c2eval.EXPORT_NEW, c2eval.STORE_NEW)}
FIRST = '2008-01-01'     # CLV is computed from the first trading day of 2008 only
WIN, MIN_OK = 5, 4       # P = mean of the last 5 daily (CLV stock - CLV twin), needing >= 4 of the 5
MAX_RANGE = 0.5          # bad print: (H - L) / C > 0.5
LIQ_W, MED_W = 756, 252  # the judge's liquid rule
TOP, TOP_W = 500, 63     # the 500 largest: 63-day median dollar volume
MIN_NAMES = 100          # a week with fewer ranked names is skipped (all sets, including the 500 largest)
COST_BPS = 5.0           # per side, on traded notional
OLD_END = '2025-06-27'   # last day of the old export; unseen holds are formed on or after it

PERIODS = [('2008-2011', '2008-01-01', '2012-01-01'), ('2012-2015', '2012-01-01', '2016-01-01'),
           ('2016-2019', '2016-01-01', '2020-01-01'), ('2020-2022', '2020-01-01', '2023-01-01'),
           ('2023-2025-06', '2023-01-01', '2025-06-28')]
TEST1 = ('2008-01-01', '2020-01-01')
TEST3 = ('2020-01-01', '2025-06-28')


def clv(H, L, C):
    """CLV = ((C - L) - (H - C)) / (H - L) and the per-rule bad-print masks. A day is missing when any of H, L, C is
    missing or <= 0, (H - L) / C > 0.5, H < C, L > C, or H = L."""
    with np.errstate(all='ignore'):
        miss = ~(np.isfinite(H) & np.isfinite(L) & np.isfinite(C) & (H > 0) & (L > 0) & (C > 0))
        rng = (H - L) / C > MAX_RANGE
        hc = H < C
        lc = L > C
        eq = H == L
        bad = miss | rng | hc | lc | eq
        x = ((C - L) - (H - C)) / (H - L)
    x = np.where(bad, np.nan, x).astype(np.float32)
    return x, {'missing': miss, 'range>0.5': rng & ~miss, 'H<C': hc & ~miss, 'L>C': lc & ~miss, 'H=L': eq & ~miss}


def liquid(E, t0):
    """The judge's liquid rule (stockai/judge/src/holdout.rs liquid_universe) at day t0, using data through t0, as
    research/twin_gap_fade and research/analog_days apply it: v finite on >= 90% of the 756 days ending at t0, and
    median close >= $5 and median dollar volume >= $5M over the 252 days ending at t0."""
    w0 = max(t0 + 1 - LIQ_W, 0)
    v = np.asarray(E['v'][:, w0:t0 + 1])
    ok = np.isfinite(v).sum(1) * 10 >= (t0 + 1 - w0) * 9
    a = max(t0 + 1 - MED_W, 0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mp = np.nanmedian(np.asarray(E['close'][:, a:t0 + 1]), 1)
        md = np.nanmedian(np.asarray(E['dollar_volume'][:, a:t0 + 1]), 1)
    return ok & (np.nan_to_num(mp) >= 5.0) & (np.nan_to_num(md) >= 5e6)


def rank01(x):
    """Cross-sectional ranks scaled to [0, 1], ties averaged."""
    n = len(x)
    order = np.argsort(x, kind='mergesort')
    xs = x[order]
    r = np.empty(n, np.float64)
    # average rank within runs of equal values
    starts = np.r_[0, np.nonzero(np.diff(xs))[0] + 1]
    ends = np.r_[starts[1:], n]
    avg = (starts + ends - 1) / 2.0
    r[order] = np.repeat(avg, ends - starts)
    return r / (n - 1) if n > 1 else r * 0


def tstat(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float('nan')
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def load_rows(tag):
    z = np.load(BIG / f'rows_{tag}.npz')
    return {k: z[k] for k in z.files}
