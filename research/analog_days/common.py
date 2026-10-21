"""Analog days (experiments/-11.5-analog-days.md): shared pieces. The rules are the .md's "Rules" plus its
"Decisions made while building it".

    import common as C; P = C.build('old')     # funds, state, analogs and daily forecasts for one store/export
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)
WORK = c2eval.WORK / 'analog_days'
WORK.mkdir(parents=True, exist_ok=True)

TICK = ['SPY', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
SECT = TICK[1:]
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
STORES = {'old': c2eval.STORE_OLD, 'new': c2eval.STORE_NEW}

ZWIN = 252          # z-score window (days d-251..d)
K = 50              # analogs
GAP = 5             # analogs at least 5 trading days apart
H = 5               # outcome and hold length; candidates a <= t - 5
STEP = 5            # portfolio calendar: export day indices divisible by 5
NLEG = 3            # funds per leg
COST_BPS = 2.0      # per side per fund on turnover
BAD = 0.25          # |daily simple move| beyond this would be a bad print (none found)
LAST_OLD = 5910     # old export's last index (2025-06-27)


def fund_closes(tag):
    """[10, T] float64 closes of SPY + the 9 sectors on the export's days, with bad prints (none) set missing and
    then missing closes after each fund's first close forward-filled (only 2011-02-17). Returns (days, closes, notes)."""
    E = c2eval.load_export(EXPORTS[tag], arrays=())
    S = c2eval.load_store(STORES[tag])
    ids = []
    for t in TICK:
        m = [s for s in S['syms'] if s['ticker'] == t and s['kind'] == 'etf']
        assert len(m) == 1, (t, m)
        ids.append(m[0]['id'])
    c = c2eval.store_panel(S, ids, E['days'], 'close').astype(np.float64)
    notes = []
    r = c[:, 1:] / c[:, :-1] - 1
    bad = np.argwhere(np.abs(r) > BAD)
    for i, d in bad:
        notes.append(f'bad print treated as missing: {TICK[i]} {c2eval.iso(E["days"][d + 1])} ({r[i, d]:+.3f})')
        c[i, d + 1] = np.nan
    for i in range(len(TICK)):
        first = int(np.argmax(np.isfinite(c[i])))
        for d in range(first + 1, c.shape[1]):
            if not np.isfinite(c[i, d]):
                notes.append(f'forward-filled: {TICK[i]} {c2eval.iso(E["days"][d])}')
                c[i, d] = c[i, d - 1]
    return E['days'], c, notes


def features(c):
    """[T, 21] raw state: 9 sectors' 5-day log return minus SPY's, 9 sectors' 21-day minus SPY's, SPY 5-day, SPY 21-day,
    SPY 21-day realized vol (sd, ddof 1, of daily log returns d-20..d)."""
    L = np.log(c)
    T = c.shape[1]
    r5 = np.full((10, T), np.nan)
    r21 = np.full((10, T), np.nan)
    r5[:, 5:] = L[:, 5:] - L[:, :-5]
    r21[:, 21:] = L[:, 21:] - L[:, :-21]
    d1 = np.full(T, np.nan)
    d1[1:] = L[0, 1:] - L[0, :-1]
    vol = np.full(T, np.nan)
    for d in range(21, T):
        w = d1[d - 20:d + 1]
        if np.isfinite(w).all():
            vol[d] = w.std(ddof=1)
    X = np.column_stack([(r5[1:] - r5[0]).T, (r21[1:] - r21[0]).T, r5[0], r21[0], vol])
    return X


def zscore(X, w=ZWIN):
    """Each column z-scored with its own mean and sd (ddof 1) over d-w+1..d, all w present; NaN otherwise."""
    T, F = X.shape
    Z = np.full((T, F), np.nan)
    for d in range(w - 1, T):
        win = X[d - w + 1:d + 1]
        if not np.isfinite(win).all():
            continue
        Z[d] = (X[d] - win.mean(0)) / win.std(0, ddof=1)
    return Z


def outcomes(c):
    """[T, 9] each sector's log return over a+1..a+5 minus SPY's (NaN where a+5 is past the end)."""
    L = np.log(c)
    T = c.shape[1]
    O = np.full((T, 9), np.nan)
    O[:T - H] = ((L[1:, H:] - L[1:, :-H]) - (L[0, H:] - L[0, :-H])).T
    return O


def find_analogs(Z, ok, k=K, gap=GAP, lag=H):
    """[T, k] analog day indices (-1 = no forecast). Candidates a <= t-lag with a complete state, nearest first
    (Euclidean; ties: earlier day first), accepted when >= gap days from every accepted one; exactly k needed."""
    T = len(Z)
    idx = np.full((T, k), -1, np.int32)
    cand = np.nonzero(ok)[0]
    blocked = np.zeros(T + 2 * gap, bool)
    for t in np.nonzero(ok)[0]:
        n = int(np.searchsorted(cand, t - lag, side='right'))
        if n < k:
            continue
        C = cand[:n]
        d2 = ((Z[C] - Z[t]) ** 2).sum(1)
        order = C[np.argsort(d2, kind='stable')]
        acc = []
        for a in order:
            if blocked[a]:
                continue
            acc.append(a)
            blocked[max(a - gap + 1, 0):a + gap] = True
            if len(acc) == k:
                break
        for a in acc:
            blocked[max(a - gap + 1, 0):a + gap] = False
        if len(acc) == k:
            idx[t] = acc
    return idx


def build(tag, cache=True):
    """Funds, state, analogs and forecasts for one tag. Cached in WORK/state_<tag>.npz."""
    p = WORK / f'state_{tag}.npz'
    if cache and p.exists():
        z = np.load(p, allow_pickle=True)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    days, c, notes = fund_closes(tag)
    X = features(c)
    Z = zscore(X)
    ok = np.isfinite(Z).all(1)
    O = outcomes(c)
    idx = find_analogs(Z, ok)
    has = idx[:, 0] >= 0
    F = np.full((len(days), 9), np.nan)
    F[has] = O[idx[has]].mean(1)
    assert np.isfinite(F[has]).all()
    out = dict(days=days, c=c, X=X, Z=Z, ok=ok, O=O, idx=idx, F=F, notes=np.array(notes, dtype=object))
    np.savez(p, **out)
    print(f'{tag}: built in {time.time() - t0:.0f}s; first complete state {c2eval.iso(days[np.argmax(ok)])}, '
          f'first forecast {c2eval.iso(days[np.argmax(has)])}, forecasts {has.sum()}', flush=True)
    return out


# ---------------------------------------------------------------------------------------------------------------
# The weekly long-short

def legs(score):
    """+1/3 for the 3 highest scores, -1/3 for the 3 lowest (stable order: earlier fund first among ties)."""
    order = np.argsort(-score, kind='stable')
    w = np.zeros(9)
    w[order[:NLEG]] = 1.0 / NLEG
    w[order[-NLEG:]] = -1.0 / NLEG
    return w


def weekly(P, forms, signal):
    """Run a strategy over consecutive formation days `forms` (continuous book). signal: [T, 9] scores (higher =
    long). Returns dict of arrays per week: gross, cost, net, turnover, w [W, 9]."""
    c = P['c'][1:]          # sectors
    W = len(forms)
    gross = np.zeros(W); cost = np.zeros(W); turn = np.zeros(W)
    ws = np.zeros((W, 9))
    prev = np.zeros(9)
    for i, f in enumerate(forms):
        w = legs(signal[f])
        g = c[:, f + H] / c[:, f]
        turn[i] = np.abs(w - prev).sum()
        cost[i] = COST_BPS * 1e-4 * turn[i]
        gross[i] = float((w * (g - 1)).sum())
        ws[i] = w
        prev = w * g
    return dict(gross=gross, cost=cost, net=gross - cost, turnover=turn, w=ws)


def random_weekly(P, forms, n=10_000, seed=20261021, chunk=2_500):
    """n random strategies over `forms`: each week 3 random sectors long, 3 different ones short, costs on actual
    turnover. Returns gross [n, W], net [n, W] (float32)."""
    c = P['c'][1:]
    rng = np.random.default_rng(seed)
    W = len(forms)
    G = np.zeros((n, W), np.float32)
    N = np.zeros((n, W), np.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        m = b - a
        prev = np.zeros((m, 9))
        for i, f in enumerate(forms):
            perm = rng.random((m, 9)).argsort(1)
            w = np.zeros((m, 9))
            np.put_along_axis(w, perm[:, :NLEG], 1.0 / NLEG, 1)
            np.put_along_axis(w, perm[:, NLEG:2 * NLEG], -1.0 / NLEG, 1)
            g = c[:, f + H] / c[:, f]
            gr = (w * (g - 1)).sum(1)
            G[a:b, i] = gr
            N[a:b, i] = gr - COST_BPS * 1e-4 * np.abs(w - prev).sum(1)
            prev = w * g
    return G, N


def signals(P):
    """[T, 9] momentum (21-day log return) and reversal (minus 5-day log return) scores for the sectors."""
    L = np.log(P['c'][1:])
    T = L.shape[1]
    mom = np.full((T, 9), np.nan)
    rev = np.full((T, 9), np.nan)
    mom[21:] = (L[:, 21:] - L[:, :-21]).T
    rev[5:] = -(L[:, 5:] - L[:, :-5]).T
    return mom, rev


def first_formation(has, last):
    """The first calendar formation (index divisible by STEP) from which every day through `last` has a forecast.
    Early on the pool is small and the greedy spacing sometimes can't reach 50 (2007-06-13..2007-07-09 on the old
    store), so the book starts after the last such day."""
    miss = np.nonzero(~has[:last + 1])[0]
    f = int(miss.max()) + 1
    return f + (-f) % STEP


def tstat(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float('nan'), float('nan'), len(x)
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)


def year_of(days, d):
    return int(c2eval.iso(days[d])[:4])
