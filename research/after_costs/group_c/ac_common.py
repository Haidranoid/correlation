"""Shared helpers for group_c's after-cost numbers (read-only on every experiment's outputs).

SPY total return: Yahoo's adjusted close via research/published_data/pubdata.py's etf_adjclose(['SPY']).
Judge years: the judge's benchmark window, close of the trading day before first_day to the close of last_day
(stockai/judge/src/holdout.rs, `bench`: var.days[h0 - 1] .. var.days[t - 1]).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data')
import c2eval  # noqa: E402
import pubdata  # noqa: E402

WORK = Path.home() / '.local/share/correlation2/experiments'
OUT = Path('/home/bhoult/programming/correlation/research/after_costs/group_c')

_spy = None


def spy():
    global _spy
    if _spy is None:
        d, c = pubdata.etf_adjclose(['SPY'])
        x = c['SPY']
        ok = np.isfinite(x)
        _spy = (d[ok], x[ok])
    return _spy


def spy_close_on_or_before(date):
    d, x = spy()
    k = int(np.searchsorted(d, np.datetime64(date, 'D'), side='right')) - 1
    assert k >= 0
    return d[k], x[k]


def spy_close_before(date):
    """SPY's adjusted close on the last Yahoo trading day strictly before date."""
    d, x = spy()
    k = int(np.searchsorted(d, np.datetime64(date, 'D'), side='left')) - 1
    assert k >= 0
    return d[k], x[k]


def spy_ret(d0, d1):
    """Total return (decimal) from SPY's close on (or before) d0 to its close on (or before) d1."""
    _, a = spy_close_on_or_before(d0)
    _, b = spy_close_on_or_before(d1)
    return b / a - 1.0


def spy_judge_year(first_day, last_day):
    """SPY total return over a judge year: close before first_day -> close of last_day (decimal)."""
    _, a = spy_close_before(first_day)
    _, b = spy_close_on_or_before(last_day)
    return b / a - 1.0


def spy_calendar_year(y, first=None, last=None):
    """SPY total return over calendar year y (close of the last day of y-1 to the last close of y), optionally cut to
    start at the close before `first` and end at the close on/before `last`."""
    a_date = first if first is not None else f'{y}-01-01'
    b_date = last if last is not None else f'{y}-12-31'
    _, a = spy_close_before(a_date)
    _, b = spy_close_on_or_before(b_date)
    return b / a - 1.0


def judge_rows(path, cost_bps=5.0):
    """Per graded year of a judge walk-forward JSON: dates, the account at 0 and at cost_bps, the median random at 0,
    the judge's own spy_pct (store closes) and SPY's Yahoo total return over the same window (percent)."""
    js = json.loads(Path(path).read_text())
    rows = []
    for y in js['years']:
        by = {float(c['cost_bps']): c for c in y['costs']}
        c0, ck = by[0.0], by[float(cost_bps)]
        rows.append({'start': y['start'], 'end': y['end'], 'first_day': y['first_day'], 'last_day': y['last_day'],
                     'acct0': c0['final_pct'], 'med0': c0['random_median_pct'], 'acct': ck['final_pct'],
                     'med': ck['random_median_pct'], 'judge_spy': y['spy_pct'],
                     'spy': 100 * spy_judge_year(y['first_day'], y['last_day'])})
    return js, rows


def summarize(rows, key='acct'):
    a = np.array([r[key] for r in rows])
    s = np.array([r['spy'] for r in rows])
    return {'n': len(rows), 'acct_mean': float(a.mean()), 'spy_mean': float(s.mean()), 'after_cost': float((a - s).mean()),
            'change0': float(np.mean([r['acct0'] - r['med0'] for r in rows]))}


def fmt(x):
    return f'{x:+.1f}'
