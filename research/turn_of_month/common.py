"""Turn of the month (experiments/-9.6-turn-of-month.md): shared settings, data and calendar.

Rules are fixed in the .md ("Rules" and "Decisions made while building it"). Data:
  - SPY's daily total return: Yahoo's adjusted closes (pubdata.etf_adjclose), close to close, on the export's days;
  - cash: pubdata.rf_daily (DGS3MO / 252);
  - the equal-weight market: the export's `market` (equal-weight mean daily LOG return of the liquid stocks).
TOM days: the last trading day of each month (position -1) and the first three of the next (+1, +2, +3), from the
export's trading calendar. The old export ends 2025-06-27, which is not June 2025's last trading day (2025-06-30), so
month ends are found on the new export's calendar, whose first 5,911 days equal the old one's (checked in check.py).

    import common as C
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402
import pubdata  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = Path.home() / '.local/share/correlation2/experiments/turn_of_month'
WORK.mkdir(parents=True, exist_ok=True)

N_RANDOM = 10_000
SEED = 20261021
COST = 1e-4          # 1 bp per side
DPY = 252
POSITIONS = (-1, 1, 2, 3)

# Walk-forward span and its periods (calendar dates, inclusive)
SPAN = ('2006-01-01', '2025-06-30')
PERIODS = {
    '2006-2025-06': ('2006-01-01', '2025-06-30'),
    '2006-2019': ('2006-01-01', '2019-12-31'),
    '2020-2025-06': ('2020-01-01', '2025-06-30'),
}
BLOCKS = {
    '2006-2009': ('2006-01-01', '2009-12-31'),
    '2010-2014': ('2010-01-01', '2014-12-31'),
    '2015-2019': ('2015-01-01', '2019-12-31'),
    '2020-2025-06': ('2020-01-01', '2025-06-30'),
}
UNSEEN = ('2025-06-30', '2026-10-19')


def dnum(iso):
    return int((np.datetime64(iso) - np.datetime64('1970-01-01')).astype(int))


def ymd(days):
    """int days since 1970 -> (year, month) int arrays."""
    dt = np.asarray(days).astype('datetime64[D]')
    y = dt.astype('datetime64[Y]').astype(int) + 1970
    m = dt.astype('datetime64[M]').astype(int) % 12 + 1
    return y, m


def tom_position(days, full_days):
    """Position of each day in `days` relative to the turn of the month, computed on the calendar `full_days` (a
    superset that continues past the end of `days`): -k for the k-th last trading day of its month (k = 1..), +k for
    the k-th trading day of its month. Returns (pos_from_start [1..], pos_from_end [-1..], known_start, known_end):
    the calendar's first month may begin before it (known_start False there) and its last month may continue after
    it (known_end False there), so no label is given from an unknown side."""
    full_days = np.asarray(full_days)
    y, m = ymd(full_days)
    key = y * 12 + m
    T = len(full_days)
    start = np.empty(T, int)
    end = np.empty(T, int)
    ks = np.ones(T, bool)
    ke = np.ones(T, bool)
    i = 0
    while i < T:
        j = i
        while j + 1 < T and key[j + 1] == key[i]:
            j += 1
        n = j - i + 1
        start[i:j + 1] = np.arange(1, n + 1)
        end[i:j + 1] = -np.arange(n, 0, -1)
        if i == 0:
            ks[i:j + 1] = False
        if j == T - 1:
            ke[i:j + 1] = False
        i = j + 1
    pos = np.searchsorted(full_days, days)
    assert (full_days[pos] == np.asarray(days)).all()
    return start[pos], end[pos], ks[pos], ke[pos]


def tom_label(days, full_days):
    """TOM label per day: -1 (last trading day of the month), +1, +2, +3 (first three), 0 otherwise."""
    s, e, ks, ke = tom_position(days, full_days)
    lab = np.zeros(len(days), int)
    first = (s <= 3) & ks
    lab[first] = s[first]
    is_last = (e == -1) & ke
    assert not np.any(is_last & first)
    lab[is_last] = -1
    return lab


def spy_total(days):
    """SPY's daily simple total return on export days, close to close from Yahoo's adjusted closes: r[d] =
    P[d] / P[d-1] - 1 with d-1 the previous export day. Also the aligned prices."""
    yd, px = pubdata.etf_adjclose(['SPY'])
    P = pubdata.on_days(yd, px['SPY'], days)
    r = np.full(len(days), np.nan)
    r[1:] = P[1:] / P[:-1] - 1.0
    return r, P, yd


def span_mask(days, lo, hi):
    return (np.asarray(days) >= dnum(lo)) & (np.asarray(days) <= dnum(hi))


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float(d), float(se), float(d / se)
