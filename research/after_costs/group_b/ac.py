"""group_b after-cost helpers: SPY total return (Yahoo adjusted close) over a judge year, and the judge account's
yearly return at 5 bps a side minus SPY's, averaged over the graded years.

    import sys; sys.path.insert(0, '/home/bhoult/programming/correlation/research/after_costs/group_b'); import ac
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data')
import c2eval  # noqa: E402
import pubdata  # noqa: E402

_D, _P = pubdata.etf_adjclose(['SPY'])
SPY_DATES, SPY_ADJ = _D, _P['SPY']
assert np.isfinite(SPY_ADJ).all()


def spy_px(day, before=False):
    """SPY adjusted close on `day` (ISO), or on the last Yahoo trading day strictly before it if before=True."""
    d = np.datetime64(day)
    if before:
        k = int(np.searchsorted(SPY_DATES, d)) - 1
    else:
        k = int(np.searchsorted(SPY_DATES, d))
        assert SPY_DATES[k] == d, (day, SPY_DATES[k])
    return float(SPY_ADJ[k]), str(SPY_DATES[k])


def spy_ret(first_day, last_day):
    """SPY total return (decimal) from the close before first_day to the close of last_day (the judge's window)."""
    a, _ = spy_px(first_day, before=True)
    b, _ = spy_px(last_day)
    return b / a - 1


def spy_ret_between(start_close_day, end_close_day):
    """SPY total return from the close of start_close_day to the close of end_close_day (both trading days)."""
    a, _ = spy_px(start_close_day)
    b, _ = spy_px(end_close_day)
    return b / a - 1


def judge_rows(path, cost_bps=5.0):
    """Per graded year: (first_day, last_day, strategy % at cost, judge's own spy_pct, SPY adj total %)."""
    js = json.load(open(path))
    rows = []
    for y in js['years']:
        c = [x for x in y['costs'] if abs(x['cost_bps'] - cost_bps) < 1e-9]
        assert len(c) == 1, (path, [x['cost_bps'] for x in y['costs']])
        rows.append((y['first_day'], y['last_day'], c[0]['final_pct'], y.get('spy_pct'),
                     100 * spy_ret(y['first_day'], y['last_day'])))
    return rows


def judge_after_cost(path, cost_bps=5.0):
    rows = judge_rows(path, cost_bps)
    strat = np.array([r[2] for r in rows])
    spy = np.array([r[4] for r in rows])
    js = json.load(open(path))
    cor0 = c2eval.change_over_random(js, 0)['change']
    return {'path': str(path), 'n': len(rows), 'first': rows[0][0], 'last': rows[-1][1],
            'strat5': float(strat.mean()), 'spy': float(spy.mean()), 'after_cost': float((strat - spy).mean()),
            'cor0': cor0, 'judge_spy': float(np.mean([r[3] for r in rows])), 'rows': rows}
