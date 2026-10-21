"""The pieces the tests and the money test share: the export, the universe on the twins' calendar, and the daily
similarity panels (point in time).

Universe at calendar day t0 (export index 252 + 21k): a close on >= 90% of days t0-755..t0, median close >= $5 and
median dollar volume >= $5M over the finite values of t0-251..t0, minus c2eval.excluded(E, tw) at that rebalance.
It holds until the next calendar day.

Similarity known at day d: the pair (last year's filing -> this year's) whose fiscal year is the expected latest one
at d (common.epoch_of) and whose date (common.avail_day) is on or before d. So a pair for fiscal year y counts from its
date until May 31 of y+2, when next year's would be expected.
"""
import json
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import (DATA_FRAC, DATA_W, MAP, MED_W, MIN_DV, MIN_PRICE, WORK, c2eval, ffill, np)  # noqa: E402

MEASURES = ('sim', 'sim_1', 'sim_1A', 'sim_7', 'sim_all')


def universe(E, tw, cache=True):
    """[K, N] bool on tw['rebal'], and [K, N] median dollar volume over t0-251..t0 (NaN where none)."""
    key = WORK / f"universe_{E['meta']['fingerprint']}.npz"
    if cache and key.exists():
        z = np.load(key)
        return z['uni'], z['mdv']
    rebal = tw['rebal']
    ex = c2eval.excluded(E, tw)
    K, N = len(rebal), len(E['tickers'])
    uni = np.zeros((K, N), bool)
    mdv = np.full((K, N), np.nan, np.float32)
    close, dv = E['close'], E['dollar_volume']
    for k, t0 in enumerate(rebal):
        if t0 < DATA_W - 1:
            continue
        c = np.asarray(close[:, t0 - DATA_W + 1:t0 + 1], np.float64)
        cov = np.isfinite(c).mean(1) >= DATA_FRAC
        with np.errstate(all='ignore'):
            mp = np.nanmedian(c[:, -MED_W:], 1)
            md = np.nanmedian(np.asarray(dv[:, t0 - MED_W + 1:t0 + 1], np.float64), 1)
        mdv[k] = md
        uni[k] = cov & (np.nan_to_num(mp) >= MIN_PRICE) & (np.nan_to_num(md) >= MIN_DV) & ~ex[k]
    np.savez(key, uni=uni, mdv=mdv)
    return uni, mdv


def pairs():
    z = np.load(WORK / 'pairs.npz')
    return {k: z[k] for k in z.files}


def cik_rows():
    m = json.loads(MAP.read_text())
    return {int(c): [int(r) for r in rows] for c, rows in m.items()}


def sim_panel(E, P, measure='sim', rule='june'):
    """[N, T] float32: the measure of the pair known at each day (NaN where none). rule='april': the sensitivity's
    dates (common.avail_day), each pair counting until March 31 of FY+2."""
    days = E['days']
    N, T = len(E['tickers']), len(days)
    out = np.full((N, T), np.nan, np.float32)
    rows_of = cik_rows()
    mmdd = '06-01' if rule == 'june' else '04-01'
    epoch0 = lambda y: int((np.datetime64(f'{y}-{mmdd}') - np.datetime64('1970-01-01')).astype(int))  # noqa: E731
    avail = P['avail'] if rule == 'june' else P['avail_april']
    order = np.argsort(P['fy'], kind='stable')
    for i in order:
        v = P[measure][i]
        if not np.isfinite(v):
            continue
        rows = rows_of.get(int(P['cik'][i]))
        if not rows:
            continue
        fy = int(P['fy'][i])
        a = int(np.searchsorted(days, avail[i]))
        b = int(np.searchsorted(days, epoch0(fy + 2)))
        if a >= b:
            continue
        out[np.array(rows)[:, None], np.arange(a, b)[None, :]] = v
    return out


def load(export=c2eval.EXPORT_OLD):
    E = c2eval.load_export(export, arrays=('close', 'dollar_volume', 'market'))
    tw = c2eval.twins(E)
    uni, mdv = universe(E, tw)
    return E, tw, uni, mdv


def price_paths(E):
    """Forward-filled closes [N, T] float64 for buy-and-hold returns (a missing day earns 0)."""
    return ffill(E['close'])
