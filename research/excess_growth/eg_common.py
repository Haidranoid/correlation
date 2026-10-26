"""Excess growth (experiments/untested-excess-growth.md): the fixed rules as constants, the universe, the portfolio
engine, gamma*, the random-walk resampling and the statistics. numpy only; run with system python3 -I -B.

Everything at a decision close t0 uses data through t0 only. Trades fill at the next close (delay 1) unless a
reported delay-0 version says otherwise.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data')
import c2eval  # noqa: E402
import pubdata  # noqa: E402

WORK = c2eval.WORK / 'excess_growth'
WORK.mkdir(parents=True, exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
STORES = {'old': c2eval.STORE_OLD, 'new': c2eval.STORE_NEW}

# ---- the plan's rules ------------------------------------------------------------------------------------------
N_UNIV = 500          # the 500 most-traded
WIN = 252             # gamma* window: daily log returns over t0-251..t0
MIN_RET = 227         # at least 227 valid returns in the window
DV_WIN = 63           # median dollar volume over t0-62..t0
COST = 5e-4           # 5 bps a side
COST20 = 20e-4        # reported
MONTH, QUARTER = 21, 63
N_BOOT, BLOCK = 10_000, 21
NW_LAGS, NW_LAGS2 = 21, 63
N_SIM = 200           # random-walk draws per period
N_SUB, SUB_K = 1000, 20
DEEP = 0.5            # S1: a close at least 50% below the purchase price
S2_END = 0.2          # S2: ends the period at 20% of its purchase price relative to the index
S2_L = (-0.30, -0.55, -1.00)
S2_K = tuple(range(0, 11))
SEED = 20261025
# Holm across P1 and P2 at the published-strategy bar (t >= 2 one-sided, family 0.0228), as edited before the snapshot
HOLM = (0.0114, 0.0228)
P_T3 = 0.00135
DEV = (2008, 2019)
CHECK = (2020, 2025)
SECTOR = ('XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY')

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


# ---- data ------------------------------------------------------------------------------------------------------
def load(tag, export=None, store=None, div=True):
    """Export (close, dollar volume), twins, the exclusion flags, div_fix (zeros if div=False or no panel), and
    each stock's last finite close. export/store override the paths (forward mode)."""
    path = Path(export) if export else EXPORTS[tag]
    if export:
        c2eval.STORE_FOR[str(path)] = Path(store)
    E = c2eval.load_export(path, arrays=('close', 'dollar_volume'))
    tw = c2eval.twins(E)
    excl = c2eval.excluded(E, tw)
    if div and tag in ('old', 'new') and not export:
        dfix = pubdata.div_fix(tag)
        assert dfix.shape == E['close'].shape
    else:
        dfix = None
    fin = np.isfinite(np.asarray(E['close']))
    T = fin.shape[1]
    last = np.where(fin.any(1), T - 1 - np.argmax(fin[:, ::-1], 1), -1)
    del fin
    return {'tag': tag, 'E': E, 'tw': tw, 'excl': excl, 'div': dfix, 'last': last, 'days': E['days'],
            'iso': np.array([c2eval.iso(d) for d in E['days']])}


def year_end_index(D, year):
    """Index of the export's last day in a calendar year."""
    y = np.array([int(s[:4]) for s in D['iso']])
    k = np.nonzero(y == year)[0]
    return int(k[-1]) if len(k) else None


def periods(D, decisions, last_end=None):
    """decisions: export indices t0. Period p: purchase at s = t0 + 1, ends at the next period's purchase (or last_end,
    default the export's last day)."""
    T = len(D['days'])
    out = []
    for k, t0 in enumerate(decisions):
        s = t0 + 1
        end = decisions[k + 1] + 1 if k + 1 < len(decisions) else (last_end if last_end is not None else T - 1)
        out.append({'t0': int(t0), 's': int(s), 'end': int(end), 'n': int(end - s),
                    'label': D['iso'][s][:4] if k + 1 < len(decisions) else D['iso'][s][:4]})
    return out


def window_logret(D, rows, t0, close=None, div=None):
    """[K, 252] total daily log returns over t0-251..t0 (NaN where either close is missing)."""
    close = D['E']['close'] if close is None else close
    div = D['div'] if div is None else div
    c = np.asarray(close[rows, t0 - WIN:t0 + 1], np.float64)
    with np.errstate(all='ignore'):
        r = np.log(c[:, 1:] / c[:, :-1])
    if div is not False and div is not None:
        r = r + np.asarray(div[rows, t0 - WIN + 1:t0 + 1], np.float64)
    r[~np.isfinite(r)] = np.nan
    return r


def select(D, t0):
    """The 500 at decision close t0: not excluded at the latest twin day <= t0, a close on t0, >= 227 valid log
    returns over t0-251..t0, positive 63-day median dollar volume; the 500 with the highest 63-day median dollar volume.
    Returns row indices, most traded first, and the count of eligible stocks."""
    E = D['E']
    close, dv = E['close'], E['dollar_volume']
    k = c2eval.rebal_index(D['tw'], t0)
    excl = D['excl'][k]
    allr = np.arange(close.shape[0])
    lr = window_logret(D, allr, t0)
    nval = np.isfinite(lr).sum(1)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mdv = np.nanmedian(np.asarray(dv[:, t0 - DV_WIN + 1:t0 + 1], np.float64), 1)
    ok = (~excl) & np.isfinite(np.asarray(close[:, t0], np.float64)) & (nval >= MIN_RET) & (np.nan_to_num(mdv) > 0)
    order = np.argsort(-np.where(ok, mdv, -np.inf), kind='stable')
    n = int(min(N_UNIV, ok.sum()))
    return order[:n], int(ok.sum())


def gamma_star(X):
    """Daily gamma* = 1/2 (mean_i sigma_i^2 - w' S w), w = 1/K, from window log returns X [K, W] (each row demeaned
    over its valid days, missing set to 0, covariance with 1/W)."""
    Z = demean_fill(X)
    W = Z.shape[1]
    s2 = (Z ** 2).sum(1) / W
    m = Z.mean(0)
    return 0.5 * (s2.mean() - (m ** 2).sum() / W)


def gamma_star_batch(X, idx):
    """gamma* for each subset idx [B, k] of window returns X [K, W] (already demeaned and filled)."""
    W = X.shape[1]
    s2 = (X ** 2).sum(1) / W
    m = X[idx].mean(1)                                 # [B, W]
    return 0.5 * (s2[idx].mean(1) - (m ** 2).sum(1) / W)


def demean_fill(X):
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mu = np.nanmean(X, 1, keepdims=True)
    Z = X - mu
    Z[~np.isfinite(Z)] = 0.0
    return Z


def holding(D, rows, p, close=None, dv=None, div=None, last=None):
    """Holding-period data for rows over period p (offsets 0..n; offset 0 = the purchase day s):
    R [K, n+1] simple total returns (forward-filled prices, so a missing close returns 0 and its move lands on the next
    valid close; div_fix added on valid closes), trade [K, n+1] (close present and dollar volume > 0), ends [K]
    (offset of the stock's last close if its data ends inside the period, else n + 1), and price [K, n+1] (ffilled
    close, for S1)."""
    E = D['E']
    close = E['close'] if close is None else close
    dv = E['dollar_volume'] if dv is None else dv
    div = D['div'] if div is None else div
    last = D['last'] if last is None else last
    t0, s, end = p['t0'], p['s'], p['end']
    n = end - s
    c = np.asarray(close[rows, t0:end + 1], np.float64)          # columns t0..end
    v = np.asarray(dv[rows, t0:end + 1], np.float64)
    valid = np.isfinite(c)
    assert valid[:, 0].all()
    P = c.copy()
    for j in range(1, P.shape[1]):
        bad = ~np.isfinite(P[:, j])
        P[bad, j] = P[bad, j - 1]
    with np.errstate(all='ignore'):
        lr = np.log(P[:, 1:] / P[:, :-1])
    if div is not False and div is not None:
        dd = np.asarray(div[rows, s:end + 1], np.float64)
        lr = lr + np.where(valid[:, 1:], np.nan_to_num(dd), 0.0)
    assert np.isfinite(lr).all()
    R = np.expm1(lr)                                             # offsets 0..n
    trade = valid[:, 1:] & (np.nan_to_num(v[:, 1:]) > 0)
    ends = np.full(len(rows), n + 1, int)
    lv = last[rows]
    m = lv < end
    ends[m] = np.maximum(1, lv[m] - s)
    return {'R': R, 'trade': trade, 'ends': ends, 'price': P[:, 1:], 'n': n, 'n_end': int(m.sum())}


# ---- the engine ------------------------------------------------------------------------------------------------
def fills_for(freq, n, delay):
    """Fill offsets (1..n-1) for a rebalancing frequency: 'd' daily, 'm' every 21 days, 'q' every 63, None = hold."""
    if freq is None:
        return ()
    if freq == 'd':
        return tuple(range(1, n))
    step = MONTH if freq == 'm' else QUARTER
    return tuple(f for f in range(step + delay, n, step))


def start_weights(R0, delay):
    """Purchase weights at offset 0: delay 1 -> proportional to (1 + r on the purchase day); delay 0 -> 1/K.
    R0 [B, K]."""
    if delay == 0:
        return np.full(R0.shape, 1.0 / R0.shape[1])
    w = 1.0 + R0
    return w / w.sum(1, keepdims=True)


def simulate(n, ret_at, w0, fills=(), delay=1, trade_at=None, ends=None):
    """One period for a batch of B portfolios of K stocks each, value 1 at the close of offset 0 (after the purchase).
    ret_at(d) -> [B, K] simple returns on offset d (1..n); trade_at(d) -> [B, K] bool (None = all tradeable);
    ends [B, K] offsets of forced sales (n + 1 = none). At a fill offset the tradeable alive stocks are set to their
    targets (equal at the previous close drifted by today's return under delay 1; exactly equal under delay 0) scaled
    to the value not held in untradeable stocks. A forced sale spreads its proceeds over the other holdings in
    proportion to their value. Returns gross [B, n], turnover [B, n] (sum |trade| / value at that close, both sides,
    excluding the purchase and the end-of-period trade) and the end weights [B, K]."""
    B, K = w0.shape
    h = np.array(w0, np.float64)
    alive = np.ones((B, K), bool)
    gross = np.empty((B, n))
    turn = np.zeros((B, n))
    V = h.sum(1)
    fills = set(int(f) for f in fills if 1 <= f < n)
    for d in range(1, n + 1):
        r = ret_at(d)
        h = h * (1.0 + r)
        Vn = h.sum(1)
        gross[:, d - 1] = Vn / V - 1.0
        if ends is not None:
            m = alive & (ends == d)
            if m.any():
                p = np.where(m, h, 0.0).sum(1)
                h = np.where(m, 0.0, h)
                rest = Vn - p
                ok = rest > 0
                h = h * np.where(ok, Vn / np.where(ok, rest, 1.0), 1.0)[:, None]
                turn[:, d - 1] += 2.0 * p / Vn
                alive &= ~m
        if d in fills:
            tr = alive if trade_at is None else (alive & trade_at(d))
            keep = alive & ~tr
            tau = np.where(tr, (1.0 + r) if delay == 1 else 1.0, 0.0)
            ts = tau.sum(1, keepdims=True)
            room = Vn - np.where(keep, h, 0.0).sum(1)
            new = np.where(tr, tau / np.where(ts > 0, ts, 1.0) * room[:, None], h)
            turn[:, d - 1] += np.abs(new - h).sum(1) / Vn
            h = new
        V = Vn
    return gross, turn, h / V[:, None]


def run_period(H, kind, delay, idx=None):
    """kind: 'B' (hold) or a frequency 'd', 'm', 'q'. H from holding(); idx [B, k] subsets (None = all stocks, B = 1).
    Returns gross, turn, w0, wend (each [B, ...])."""
    R, trade, ends, n = H['R'], H['trade'], H['ends'], H['n']
    if idx is None:
        idx = np.arange(R.shape[0])[None, :]
    w0 = start_weights(R[idx, 0], delay)
    freq = None if kind == 'B' else kind
    g, t, we = simulate(n, lambda d: R[idx, d], w0, fills_for(freq, n, delay), delay,
                        trade_at=lambda d: trade[idx, d], ends=ends[idx])
    return g, t, w0, we


def chain(per, first_from_cash=True):
    """Chain one portfolio across periods. per: list of dicts with rows (global ids), gross [n], turn [n], w0 [K],
    wend [K], days (export indices s+1..end). Adds the purchase from cash to the first day and the reconstitution
    turnover (sum |w_new - w_old| over the union of stocks) to each period's last day. Returns days, gross, turn and
    the period index of each day."""
    days, gross, turn, pid = [], [], [], []
    for k, p in enumerate(per):
        t = p['turn'].copy()
        if k == 0 and first_from_cash:
            t[0] += 1.0
        if k + 1 < len(per):
            q = per[k + 1]
            old = dict(zip(p['rows'].tolist(), p['wend'].tolist()))
            new = dict(zip(q['rows'].tolist(), q['w0'].tolist()))
            t[-1] += sum(abs(new.get(j, 0.0) - old.get(j, 0.0)) for j in set(old) | set(new))
        days.append(p['days']); gross.append(p['gross']); turn.append(t); pid.append(np.full(len(t), k))
    return np.concatenate(days), np.concatenate(gross), np.concatenate(turn), np.concatenate(pid)


def net(gross, turn, cost):
    return (1.0 + gross) * (1.0 - cost * turn) - 1.0


# ---- statistics ------------------------------------------------------------------------------------------------
def nw_t(x, lags=NW_LAGS):
    """Newey-West (Bartlett) t of the mean of x."""
    x = np.asarray(x, np.float64)
    T = len(x)
    e = x - x.mean()
    v = e @ e / T
    for l in range(1, lags + 1):
        v += 2.0 * (1.0 - l / (lags + 1.0)) * (e[l:] @ e[:-l]) / T
    return float(x.mean() / np.sqrt(v / T))


def boot_p(x, n=N_BOOT, block=BLOCK, seed=SEED):
    """Circular block bootstrap of whole dates (21-day blocks), centred at the mean. One-sided p that the mean is
    > 0: (1 + #{mean* - mean >= mean}) / (1 + n). Also the 5..95% interval of mean*."""
    x = np.asarray(x, np.float64)
    T = len(x)
    rng = np.random.default_rng(seed)
    xx = np.concatenate([x, x[:block]])
    cs = np.concatenate([[0.0], np.cumsum(xx)])
    nb = int(np.ceil(T / block))
    last = T - block * (nb - 1)
    st = rng.integers(0, T, size=(n, nb))
    s = (cs[st[:, :-1] + block] - cs[st[:, :-1]]).sum(1) + (cs[st[:, -1] + last] - cs[st[:, -1]])
    m = s / T
    mu = x.mean()
    p = (1 + np.sum(m - mu >= mu)) / (1 + n)
    return float(p), float(np.percentile(m, 5)), float(np.percentile(m, 95))


def normal_p(t):
    """One-sided upper p of a standard normal."""
    from math import erfc, sqrt
    return 0.5 * erfc(t / sqrt(2.0))


def test_series(d):
    """The primary statistics of a daily log-difference series: per year (x 252), NW t (21, 63), bootstrap p, the
    larger p."""
    t21, t63 = nw_t(d, NW_LAGS), nw_t(d, NW_LAGS2)
    pb, lo, hi = boot_p(d)
    pn = normal_p(t21)
    return {'days': len(d), 'per_year': 252 * float(np.mean(d)), 'total': float(np.sum(d)), 't_nw21': t21,
            't_nw63': t63, 'p_nw': pn, 'p_boot': pb, 'p': max(pn, pb), 'boot5': 252 * lo, 'boot95': 252 * hi}


def holm(ps, levels=HOLM):
    """Holm across len(ps) one-sided p's with the plan's sorted thresholds; returns pass flags in input order."""
    order = np.argsort(ps)
    out = [False] * len(ps)
    for rank, i in enumerate(order):
        if ps[i] < levels[rank]:
            out[i] = True
        else:
            break
    return out


def fmt(x, nd=2):
    return f'{x:+.{nd}f}'
