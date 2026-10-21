"""Shared code for the survivorship stress test of experiments/+18.6-monthly-top10-picks.md.

numpy-only copies of stockai's logic (stockai/stockai/h30/{targets,metrics,evaluate}.py and the grader's account in
stockai/judge/src/holdout.rs), so nothing under stockai/ is run or touched. Each function says what it copies.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data')
import c2eval  # noqa: E402

ROOT = Path('/home/bhoult/programming/correlation')
STOCKAI = ROOT / 'stockai'
RUNS = STOCKAI / 'runs/30d'
JUDGE_DIR = STOCKAI / 'results/30d/judge'
OUT = HERE / 'out'
BIG = Path.home() / '.local/share/correlation2/experiments/survivorship_stress'

H = 21
TOP = 10
COMPARABLE = 20
FILL_DAYS = 5
UNIVERSE_WINDOW = 756
LOOKBACK = 252
MIN_PRICE = np.float32(5.0)
MIN_DV = np.float32(5e6)
VOL_WINDOW, VOL_MIN = 63, 42
YEARS = list(range(2010, 2024))
COSTS = (0, 5, 20, 50)


def to_day(iso: str) -> int:
    return int((np.datetime64(iso) - np.datetime64('1970-01-01')).astype(int))


def iso(day) -> str:
    return c2eval.iso(day)


def load():
    """The export stockai used (through 2025-06-27): close, dollar_volume, v as memmaps; days, tickers, ids."""
    return c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close', 'dollar_volume', 'v'))


def dev_cutoff(days) -> int:
    """stockai.baselines.common.dev_cutoff: columns [0, cutoff) are development (the locked year starts 2024-06-28)."""
    h0 = int(np.searchsorted(days, to_day('2024-06-28'), side='left'))
    return h0 - 1


def year_bounds(days, year):
    """h0 (first day >= year-06-28) and t (first day >= (year+1)-06-28), as targets.year_spec and holdout.rs."""
    h0 = int(np.searchsorted(days, to_day(f'{year}-06-28'), side='left'))
    t = int(np.searchsorted(days, to_day(f'{year + 1}-06-28'), side='left'))
    return h0, t


def _upper_median(x):
    """targets.judge_universe's upper median: the (k // 2)-th smallest of the k finite values per row, 0 when none."""
    k = np.isfinite(x).sum(axis=1)
    s = np.sort(np.where(np.isfinite(x), x, np.inf), axis=1)
    idx = np.minimum(k // 2, x.shape[1] - 1)
    m = np.take_along_axis(s, idx[:, None], axis=1)[:, 0]
    return np.where(k > 0, m, np.float32(0.0))


def judge_universe(E, h0, window=UNIVERSE_WINDOW):
    """targets.judge_universe (= holdout.rs liquid_universe) on adjusted closes: candidates for a period starting at h0,
    from days [h0-window, h0) only."""
    w0 = max(h0 - window, 0)
    n_days = h0 - w0
    tail = min(n_days, LOOKBACK)
    out = []
    step = 2000
    N = E['close'].shape[0]
    for a in range(0, N, step):
        b = min(N, a + step)
        vv = np.asarray(E['v'][a:b, w0:h0], dtype=np.float32)
        enough = np.isfinite(vv).sum(axis=1) * 10 >= n_days * 9
        cc = np.asarray(E['close'][a:b, h0 - tail:h0], dtype=np.float32)
        dv = np.asarray(E['dollar_volume'][a:b, h0 - tail:h0], dtype=np.float32)
        ok = enough & (_upper_median(cc) >= MIN_PRICE) & (_upper_median(dv) >= MIN_DV)
        out.append(a + np.flatnonzero(ok))
    return np.concatenate(out)


def fwd_col(close, rows, d, h=H, fill=FILL_DAYS):
    """targets.forward_returns at one decision column d for the given rows: close[d+h]/close[d]-1, a missing end close
    replaced by the nearest earlier close within min(fill, h-1) days (never close[d]); NaN if close[d] is missing or
    not positive."""
    c0 = np.asarray(close[rows, d], dtype=np.float64)
    blk = np.asarray(close[rows, d + h - min(fill, h - 1):d + h + 1], dtype=np.float64)  # columns d+h-fill .. d+h
    end = blk[:, -1].copy()
    for k in range(1, min(fill, h - 1) + 1):
        gap = ~np.isfinite(end)
        end[gap] = blk[gap, -1 - k]
    with np.errstate(divide='ignore', invalid='ignore'):
        f = end / c0 - 1.0
    f[~(np.isfinite(c0) & (c0 > 0))] = np.nan
    f[~np.isfinite(f)] = np.nan
    return f


def vol_col(close, rows, d, window=VOL_WINDOW, min_obs=VOL_MIN):
    """metrics.trailing_vol at column d: sd (ddof 1) of the daily log returns ending at d over window days, from at
    least min_obs returns with both closes present."""
    lo = max(d - window, 0)
    c = np.asarray(close[rows, lo:d + 1], dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.log(c[:, 1:] / c[:, :-1])
    if d - window < 0:  # the first column has no return; the cumsum version then counts window returns from col 1
        pass
    ok = np.isfinite(r)
    k = ok.sum(axis=1)
    x = np.where(ok, r, 0.0)
    s1, s2 = x.sum(axis=1), (x * x).sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        var = (s2 - s1 * s1 / k) / (k - 1)
    out = np.sqrt(np.where(var > 0, var, np.nan))
    out[k < min_obs] = np.nan
    return out


def ratio_col(close, rows, d):
    """metrics.past_year_ratio at column d: close[d]/close[d-252] (NaN if either is missing)."""
    if d < 252:
        return np.full(len(rows), np.nan)
    a = np.asarray(close[rows, d], dtype=np.float64)
    b = np.asarray(close[rows, d - 252], dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = a / b
    r[~np.isfinite(r)] = np.nan
    return r


def med_dv_col(dv, rows, d, window=63):
    """Median dollar volume over days d-window+1 .. d (finite values only); NaN if none."""
    x = np.asarray(dv[rows, d - window + 1:d + 1], dtype=np.float64)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.nanmedian(x, axis=1)


def features_col(E, rows, d):
    """The hazard model's raw regressors at column d for the given rows (dict of arrays)."""
    close = E['close']
    ratio = ratio_col(close, rows, d)
    c = np.asarray(close[rows, d], dtype=np.float64)
    mdv = med_dv_col(E['dollar_volume'], rows, d)
    with np.errstate(divide='ignore', invalid='ignore'):
        return {
            'ret12': ratio - 1.0,
            'down50': np.where(np.isfinite(ratio), (ratio < 0.5).astype(float), 0.0),
            'vol63': vol_col(close, rows, d),
            'logp': np.where(c > 0, np.log(c), np.nan),
            'logdv': np.where(mdv > 0, np.log(mdv), np.nan),
        }


CONT = ('ret12', 'vol63', 'logp', 'logdv')


def design(feat, model, drop=()):
    """Standardized design matrix [n, 1 + p] for a fitted model dict (means/sds from the fit sample); a missing
    continuous regressor is set to its mean (z = 0)."""
    cols = [np.ones(len(feat['ret12']))]
    for name in model['names']:
        if name == 'down50':
            cols.append(feat['down50'])
        else:
            z = (feat[name] - model['mean'][name]) / model['sd'][name]
            cols.append(np.where(np.isfinite(z), z, 0.0))
    return np.column_stack(cols)


def predict_p(feat, model):
    X = design(feat, model)
    eta = X @ np.asarray(model['beta'])
    p = 1.0 / (1.0 + np.exp(-eta))
    return np.clip(p, 1e-6, 1 - 1e-6)


def monthly_hazard(p, m):
    return 1.0 - (1.0 - p) ** (1.0 / m)


def pct_rank(x):
    """metrics.pct_rank: (rank + 0.5) / n, ties broken by position."""
    order = np.argsort(x, kind='stable')
    r = np.empty(x.size)
    r[order] = np.arange(x.size)
    return (r + 0.5) / x.size


def comparable_parts(f, w, vv, rr, k=COMPARABLE):
    """metrics.comparable_excess, returning its parts: [(i, w_i, neighbour indices)] for every pick it counts, so the
    excess is sum w_i (f_i - mean f[nb]) / TOP. None when fewer than k candidates (the day's value is NaN)."""
    ok = np.isfinite(vv) & np.isfinite(rr)
    cand = np.flatnonzero(ok & (w == 0))
    if cand.size < k:
        return None
    pv, pr = np.full(f.size, np.nan), np.full(f.size, np.nan)
    idx = np.flatnonzero(ok)
    pv[idx], pr[idx] = pct_rank(vv[idx]), pct_rank(rr[idx])
    parts = []
    for i in np.flatnonzero((w > 0) & ok):
        nb = cand[np.argsort((pv[cand] - pv[i]) ** 2 + (pr[cand] - pr[i]) ** 2, kind='stable')[:k]]
        parts.append((int(i), float(w[i]), nb))
    return parts


def tstat(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2 or x.std(ddof=1) == 0:
        return float('nan')
    return float(x.mean() / x.std(ddof=1) * math.sqrt(x.size))


def graded_days(E, cutoff):
    """targets.year_spec + non_overlapping for 2010-2023 at H = 21: per year (year, h0, t, universe, graded days)."""
    specs = []
    for y in YEARS:
        h0, t = year_bounds(E['days'], y)
        reb = np.arange(h0 - 1, t - 1, H)
        graded = reb[reb + H <= cutoff - 1]
        specs.append([y, h0, t, judge_universe(E, h0), graded])
    for k in range(len(specs)):
        if k + 1 < len(specs):
            nxt = specs[k + 1][4][0]
            specs[k][4] = specs[k][4][specs[k][4] + H <= nxt]
    return specs


def grader_picks(scores, cand, d):
    """holdout.rs: the top TOP candidates with a positive score, by descending score (stable: ties to the lower
    candidate position). scores: [N, T] (float64 for the jitter files). Returns export rows."""
    s = np.asarray(scores[cand, d], dtype=np.float64)
    pos = np.flatnonzero(s > 0)
    order = pos[np.argsort(-s[pos], kind='stable')]
    return cand[order[:TOP]]


def daily_rets_f32(close, rows, d0, d1):
    """holdout.rs ret(j, d) for d in [d0, d1): (close[d+1]/close[d] - 1) in float32, 0 where either is missing or
    close[d] <= 0. Returns float64 [len(rows), d1-d0]."""
    c = np.asarray(close[rows, d0:d1 + 1], dtype=np.float32)
    a, b = c[:, :-1], c[:, 1:]
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (b / a - np.float32(1.0)).astype(np.float32)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0)
    return np.where(ok, r, np.float32(0.0)).astype(np.float64)


def grader_year(E, scores, year, cand=None):
    """The grader's account structure for one walk-forward year (holdout.rs, delay 0, rebalance 21): candidates, days
    h0-1 .. t-2, blocks of up to 21 account days each starting at a rebalance day, the picks of each block, and each
    held pick's daily returns. Returns a dict."""
    days = E['days']
    h0, t = year_bounds(days, year)
    if cand is None:
        cand = judge_universe(E, h0)
    acct = np.arange(h0 - 1, t - 1)  # decision days d; the account holds over d+1
    blocks = []
    for s in range(0, acct.size, H):
        ds = acct[s:s + H]
        picks = grader_picks(scores, cand, int(ds[0]))
        rets = daily_rets_f32(E['close'], picks, int(ds[0]), int(ds[-1]) + 1) if picks.size else np.zeros((0, ds.size))
        blocks.append({'d': int(ds[0]), 'len': int(ds.size), 'picks': picks, 'rets': rets})
    return {'year': year, 'h0': h0, 't': t, 'cand': cand, 'blocks': blocks}


def grader_value(gy, cost_bps, delta=None):
    """holdout.rs simulate() over one year, $1 start: daily equal weight 1/TOP over the held picks, turnover cost per
    side at each rebalance. With delta (list, one per block), each block's growth factor is increased by delta[b]
    (the haircut). Returns the year's growth factor."""
    w = 1.0 / TOP
    value, prev = 1.0, set()
    for b, blk in enumerate(gy['blocks']):
        now = set(blk['picks'].tolist())
        kept = len(now & prev)
        turnover = w * ((len(now) - kept) + (len(prev) - kept))
        gross = w * blk['rets'].sum(axis=0) if blk['rets'].size else np.zeros(blk['len'])
        g = 1.0
        for k in range(blk['len']):
            g *= 1.0 + gross[k] - (turnover * cost_bps / 1e4 if k == 0 else 0.0)
        if delta is not None:
            g += delta[b]
        value *= g
        prev = now
    return value


def block_pick_returns(blk):
    """Each held pick's compounded grader return over its block."""
    if not blk['rets'].size:
        return np.zeros(0)
    return np.prod(1.0 + blk['rets'], axis=1) - 1.0


def spy_store(store_path=c2eval.STORE_OLD):
    """SPY's close series (dates as days since 1970, close) from a Stooq store (the grader's benchmark)."""
    st = c2eval.load_store(store_path)
    i = next(s['id'] for s in st['syms'] if s['ticker'] == 'SPY')
    a, b = st['offs'][i], st['offs'][i + 1]
    return np.asarray(st['date'][a:b]), np.asarray(st['close'][a:b], dtype=np.float64)


def spy_years(E, dates, close):
    """Per walk-forward year: SPY close at days[t-1] / close at days[h0-1] - 1 (holdout.rs bench)."""
    out = []
    for y in YEARS:
        h0, t = year_bounds(E['days'], y)
        p = lambda day: close[np.searchsorted(dates, day)] if dates[np.searchsorted(dates, day)] == day else np.nan  # noqa: E731
        out.append(p(E['days'][t - 1]) / p(E['days'][h0 - 1]) - 1.0)
    return np.array(out)


def spy_tr_years(E):
    """SPY total return (Yahoo adjusted close, pubdata.etf_adjclose) over the same spans."""
    import pubdata
    dates, d = pubdata.etf_adjclose(['SPY'])
    dd = pubdata.to_days(dates)
    x = np.asarray(d['SPY'], dtype=np.float64)
    ok = np.isfinite(x)
    dd, x = dd[ok], x[ok]
    return spy_years(E, dd, x)


def save_json(obj, name):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float))
