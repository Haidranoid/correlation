"""Short volume: the weekly panel shared by train.py and tests.py.

Decision days t: export indices idx(2010-01-04) + 5k (the plan's fixed calendar), with close t+5 inside the export.
Outcome: the market-relative log return close t -> close t+5 = log(c[t+5] / c[t]) - sum(market[t+1..t+5]) (market =
the export's equal-weight mean daily log return of the liquid stocks); the book uses the simple return c[t+5]/c[t]-1.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np, c2eval = C.np, C.c2eval

SIGNALS = ('G', 'A', 'G_lag', 'A_lag')


def load(tag, signals=SIGNALS):
    E = c2eval.load_export(C.EXPORTS[tag], arrays=('close', 'market'))
    days = E['days']
    N, T = E['close'].shape
    dd = C.decision_days(days, T)
    close = E['close']
    c0 = np.asarray(close[:, dd], np.float64)
    c5 = np.asarray(close[:, dd + C.H], np.float64)
    m = np.asarray(E['market'], np.float64)
    cm = np.concatenate([[0.0], np.cumsum(np.nan_to_num(m))])
    mk = cm[dd + C.H + 1] - cm[dd + 1]
    with np.errstate(all='ignore'):
        y = np.log(c5 / c0) - mk[None, :]
        r = c5 / c0 - 1.0
    y[~np.isfinite(y)] = np.nan
    r[~np.isfinite(r)] = np.nan
    z = np.load(C.WORK / tag / 'universe.npz')
    uni, big, rebal = z['uni'], z['big'], z['rebal']
    ks = np.array([C.kidx(rebal, t) for t in dd])
    U = np.stack([uni[k] if k >= 0 else np.zeros(N, bool) for k in ks], 1)
    B = np.stack([big[k] if k >= 0 else np.zeros(N, bool) for k in ks], 1)
    S = {}
    for s in signals:
        X = np.load(C.WORK / tag / f'{s}.npy', mmap_mode='r')
        S[s] = np.asarray(X[:, dd], np.float64)
    return {'days': days, 'dd': dd, 'y': y, 'r': r, 'U': U, 'B': B, 'S': S, 'market': m, 'E': E}


def market_check(E):
    """Is market[d] day d's return? Correlation of market with the cross-sectional mean log return of day d and d+1."""
    c = np.asarray(E['close'][:, 1000:2500], np.float64)   # 2005-2011, before the signal's tests
    with np.errstate(all='ignore'):
        lr = np.log(c[:, 1:] / c[:, :-1])
    lr[~np.isfinite(lr) | (np.abs(lr) > 0.5)] = np.nan
    mx = np.nanmedian(lr, 0)
    m = np.asarray(E['market'][1000:2500], np.float64)[1:]
    ok = np.isfinite(mx) & np.isfinite(m)
    same = np.corrcoef(mx[ok], m[ok])[0, 1]
    ok2 = np.isfinite(mx[:-1]) & np.isfinite(m[1:])
    nxt = np.corrcoef(mx[:-1][ok2], m[1:][ok2])[0, 1]
    return float(same), float(nxt)


def ics(W, sig, sign=1.0, members='U', mask=None):
    """Weekly rank IC of sign x signal with the outcome, on the decision days in mask; NaN where < 50 names."""
    X = W['S'][sig]
    M = W[members]
    out = np.full(len(W['dd']), np.nan)
    nn = np.zeros(len(W['dd']), int)
    for i in range(len(W['dd'])):
        if mask is not None and not mask[i]:
            continue
        ok = M[:, i] & np.isfinite(X[:, i]) & np.isfinite(W['y'][:, i])
        nn[i] = ok.sum()
        if ok.sum() >= C.MIN_NAMES:
            out[i] = C.spearman(sign * X[ok, i], W['y'][ok, i])
    return out, nn


def book(W, sig, sign=1.0, members='U', mask=None, cost_bps=C.COST_BPS):
    """Weekly decile long-short on sign x signal: long the top 10%, short the bottom 10% (ranked among members with a
    finite signal), equal weight, simple returns close t -> t+5 (members without a close at t+5 dropped from the
    means). Turnover = sum |w_new - w_old| per side (weights 1/m), cost = c x (turnover long + turnover short).
    Returns gross, cost, net (fractions per week; NaN outside mask or with < 50 names) and per-side counts."""
    X = W['S'][sig]
    M = W[members]
    n = len(W['dd'])
    gross = np.full(n, np.nan)
    cost = np.full(n, np.nan)
    turn = np.full(n, np.nan)
    wl_prev, ws_prev = {}, {}
    sizes = np.zeros(n, int)
    for i in range(n):
        ok = M[:, i] & np.isfinite(X[:, i])
        idx = np.nonzero(ok)[0]
        if len(idx) < C.MIN_NAMES:
            wl_prev, ws_prev = {}, {}
            continue
        s = sign * X[idx, i]
        o = idx[np.argsort(s, kind='stable')]
        m = int(round(C.DECILE * len(idx)))
        lo, hi = o[:m], o[-m:]
        wl = {j: 1.0 / m for j in hi}
        ws = {j: 1.0 / m for j in lo}
        tl = sum(abs(wl.get(j, 0) - wl_prev.get(j, 0)) for j in set(wl) | set(wl_prev)) if wl_prev else 1.0
        ts = sum(abs(ws.get(j, 0) - ws_prev.get(j, 0)) for j in set(ws) | set(ws_prev)) if ws_prev else 1.0
        wl_prev, ws_prev = wl, ws
        if mask is not None and not mask[i]:
            continue
        rl, rs = W['r'][hi, i], W['r'][lo, i]
        if np.isfinite(rl).sum() == 0 or np.isfinite(rs).sum() == 0:
            continue
        gross[i] = np.nanmean(rl) - np.nanmean(rs)
        cost[i] = cost_bps * 1e-4 * (tl + ts)
        turn[i] = (tl + ts) / 2
        sizes[i] = m
    return {'gross': gross, 'cost': cost, 'net': gross - cost, 'turnover': turn, 'm': sizes}
