"""SPY total return (Yahoo adjusted close, research/published_data/pubdata.py) over a judge period.

The judge's (and c2's) benchmark convention (stockai/judge/src/holdout.rs, `bench`): from the close of the trading day
before the holdout's first day to the close of its last day. Run with python3 -I -B; this module adds its own paths.
"""
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data')
import numpy as np  # noqa: E402
import pubdata  # noqa: E402

_DATES, _ADJ = pubdata.etf_adjclose(['SPY'])
_ADJ = _ADJ['SPY']
_OK = np.isfinite(_ADJ)
_DATES, _ADJ = _DATES[_OK], _ADJ[_OK]


def spy_pct(first_day, last_day):
    """SPY total return in percent from the close of the trading day before first_day to the close of last_day
    (ISO strings). Both days must be Yahoo trading days (checked)."""
    a = np.searchsorted(_DATES, np.datetime64(first_day))
    b = np.searchsorted(_DATES, np.datetime64(last_day))
    assert _DATES[a] == np.datetime64(first_day), (first_day, _DATES[a])
    assert _DATES[b] == np.datetime64(last_day), (last_day, _DATES[b])
    return 100.0 * (_ADJ[b] / _ADJ[a - 1] - 1.0)


def spy_pct_between_closes(start_close_day, end_close_day):
    """SPY total return in percent from the close of start_close_day to the close of end_close_day."""
    a = np.searchsorted(_DATES, np.datetime64(start_close_day))
    b = np.searchsorted(_DATES, np.datetime64(end_close_day))
    assert _DATES[a] == np.datetime64(start_close_day) and _DATES[b] == np.datetime64(end_close_day)
    return 100.0 * (_ADJ[b] / _ADJ[a] - 1.0)
