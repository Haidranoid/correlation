"""Free timing: time a monthly rebalancer's trades within a 5-day window with short-term signals, against executing
the same trades on a random window day. Rules: experiments/+0.7-free-timing.md (plan and "Decisions").

    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python3 -I research/free_timing/free_timing.py build        # old export
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python3 -I research/free_timing/free_timing.py build-unseen # once, at the end

`build` writes the trade table (one row per planned trade, with its prices and each signal's timing) to
research/free_timing/trades_old.npz; `report.py` turns it into the tables in the .md.
"""
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = '/home/bhoult/programming/correlation'
MOM_SAVED = f'{REPO}/stockai/runs/poc/baselines/momentum_12_1.npy'
LGBM_SAVED = f'{REPO}/stockai/runs/poc/baselines/lgbm_ret/scores.npy'
W = 5            # window days k = 0..4; END = close of day 5
BAD_OPEN = 0.25  # an open this far (log) from both the previous and the same day's close is a bad print
P1_N, P1_MIN_TRADE = 100, 0.001
P2_N = 20


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def momentum_scores(close, skip=21, lookback=252):
    """stockai.baselines.momentum.momentum_scores, copied (stockai's venv is off limits): ln(close[d-skip]/close[d-lookback])."""
    c = np.asarray(close, dtype=np.float64)
    n, t = c.shape
    out = np.full((n, t), np.nan)
    old = c[:, : t - lookback]
    new = c[:, lookback - skip: t - skip]
    ok = np.isfinite(old) & np.isfinite(new) & (old > 0) & (new > 0)
    tail = out[:, lookback:]
    tail[ok] = np.log(new[ok] / old[ok])
    return out


def liquid_set(close, dv, d0, fund_like_k):
    """Judge's liquid rule, point in time at d0: >= 90% finite closes over d0-251..d0, median close >= $5 and median
    dollar volume >= $5M over those days; minus fund-like stocks."""
    c = np.asarray(close[:, d0 - 251:d0 + 1], np.float64)
    v = np.asarray(dv[:, d0 - 251:d0 + 1], np.float64)
    with np.errstate(all='ignore'):
        ok = np.isfinite(c).mean(1) >= 0.9
        ok &= np.nan_to_num(np.nanmedian(c, 1)) >= 5.0
        ok &= np.nan_to_num(np.nanmedian(v, 1)) >= 5e6
    return ok & ~fund_like_k


def last_close(close, j, d_lo, d_hi):
    """Last finite close of row j in days d_lo..d_hi (inclusive), or NaN."""
    seg = np.asarray(close[j, d_lo:d_hi + 1], np.float64)
    f = np.nonzero(np.isfinite(seg))[0]
    return seg[f[-1]] if len(f) else np.nan


def growth(close, j, d_prev, d0):
    """Gross return of row j from the close of d_prev to the close of d0 (last finite closes); 1 if unknown."""
    a = last_close(close, j, max(d_prev - 5, 0), d_prev)
    b = last_close(close, j, d_prev, d0)
    g = b / a if np.isfinite(a) and np.isfinite(b) and a > 0 else 1.0
    return g


def plan_trades(E, tw, mom, ks, eval_from):
    """Simulate P1 and P2 over twin-calendar rebalances ks (ascending indices into tw['rebal']), returning the trades
    of rebalances whose index is >= eval_from: list of (k, d0, row, side, size, portfolio)."""
    close, dv = E['close'], E['dollar_volume']
    rebal = tw['rebal']
    trades = []
    p1 = {}          # row -> weight after the last rebalance (fractions of portfolio value; the rest is cash)
    p2 = []          # rows held (each at 1/20 after the last rebalance)
    d_prev = None
    for k in ks:
        d0 = int(rebal[k])
        fl = tw['fund_like'][k]
        liq = liquid_set(close, dv, d0, fl)
        # ---- P1: top 100 by median dollar volume over d0-62..d0, close >= $5 on d0, not fund-like
        c0 = np.asarray(close[:, d0], np.float64)
        v63 = np.asarray(dv[:, d0 - 62:d0 + 1], np.float64)
        with np.errstate(all='ignore'):
            ok = (np.isfinite(v63).sum(1) >= 57) & (np.nan_to_num(c0) >= 5.0) & ~fl
            med = np.where(ok, np.nanmedian(v63, 1), -np.inf)
        members = np.argsort(-med, kind='stable')[:P1_N]
        members = members[np.isfinite(med[members])]
        # drift since the last rebalance
        if d_prev is not None and p1:
            val = {j: w * growth(close, j, d_prev, d0) for j, w in p1.items()}
            cash = 1.0 - sum(p1.values())
            tot = sum(val.values()) + cash
            cur = {j: v / tot for j, v in val.items()}
        else:
            cur = {}
        target = {int(j): 1.0 / P1_N for j in members}
        new = dict(cur)
        for j in set(cur) | set(target):
            dw = target.get(j, 0.0) - cur.get(j, 0.0)
            if abs(dw) >= P1_MIN_TRADE:
                if d_prev is not None and k >= eval_from:
                    trades.append((k, d0, j, 1 if dw > 0 else -1, abs(dw), 1))
                new[j] = target.get(j, 0.0)
        p1 = {j: w for j, w in new.items() if w > 0}
        # ---- P2: top 20 positive 12-1 momentum among the liquid set
        s = np.where(liq & np.isfinite(mom[:, d0]), mom[:, d0], -np.inf)
        order = np.argsort(-s, kind='stable')[:P2_N]
        picks = [int(j) for j in order if s[j] > 0]
        if d_prev is not None and k >= eval_from:
            g = {j: growth(close, j, d_prev, d0) for j in p2}
            gp = sum(g.values()) / P2_N + (1.0 - len(p2) / P2_N)
            for j in p2:
                if j not in picks:
                    trades.append((k, d0, j, -1, g[j] / P2_N / gp, 2))
            for j in picks:
                if j not in p2:
                    trades.append((k, d0, j, 1, 1.0 / P2_N, 2))
        p2 = picks
        d_prev = d0
    return trades


def opens_panel(store, E):
    """Store opens aligned to the export, with bad prints (> BAD_OPEN log from both neighbouring closes) set NaN."""
    o = c2eval.store_panel(store, E['ids'], E['days'], 'open')
    c = np.asarray(E['close'])
    pc = np.full_like(c, np.nan)
    pc[:, 1:] = c[:, :-1]
    with np.errstate(all='ignore'):
        a = np.abs(np.log(o / pc))
        b = np.abs(np.log(o / c))
    far_prev = ~(a <= BAD_OPEN)   # NaN counts as far
    far_same = ~(b <= BAD_OPEN)
    bad = np.isfinite(o) & far_prev & far_same
    o[bad] = np.nan
    o[~(o > 0)] = np.nan
    return o, bad


def spy_series(store, E):
    sid = [s['id'] for s in store['syms'] if s['ticker'] == 'SPY' and s['kind'] == 'etf']
    assert len(sid) == 1, sid
    oc = c2eval.store_panel(store, sid, E['days'], 'open')[0].astype(np.float64)
    cc = c2eval.store_panel(store, sid, E['days'], 'close')[0].astype(np.float64)
    return sid[0], oc, cc


def thirds_flags(S, liq_rows, d, rows, high_is_buy):
    """Buy- and sell-favourable flags of `rows` on day d, from the thirds of S[liq_rows, d]."""
    ref = S[liq_rows, d]
    ref = ref[np.isfinite(ref)]
    v = S[rows, d].astype(np.float64)
    if len(ref) < 30:
        z = np.zeros(len(rows), bool)
        return z, z, np.isfinite(v), np.nan, np.nan
    lo, hi = np.quantile(ref.astype(np.float64), [1 / 3, 2 / 3])
    with np.errstate(invalid='ignore'):
        low, high = v <= lo, v >= hi
    if high_is_buy:
        return high, low, np.isfinite(v), lo, hi
    return low, high, np.isfinite(v), lo, hi


def build(which):
    t_start = time.time()
    if which == 'old':
        E = c2eval.load_export(c2eval.EXPORT_OLD)
        store = c2eval.load_store(c2eval.STORE_OLD)
        start, end = '2008-01-01', None
    else:
        E = c2eval.load_export(c2eval.EXPORT_NEW)
        store = c2eval.load_store(c2eval.STORE_NEW)
        start, end = '2025-06-30', None
    E['close'] = np.asarray(E['close'])
    E['dollar_volume'] = np.asarray(E['dollar_volume'])
    N, T = E['close'].shape
    log(which, 'export', N, T, c2eval.iso(E['days'][0]), c2eval.iso(E['days'][-1]))
    tw = c2eval.twins(E)
    log('twins', len(tw['rebal']), 'refreshes')
    rebal = tw['rebal']
    d_start = c2eval.day_index(E['days'], start)
    eval_ks = [k for k in range(len(rebal)) if rebal[k] >= d_start and rebal[k] + W < T]
    eval_from = eval_ks[0]
    ks = list(range(max(eval_from - 3, 0), eval_ks[-1] + 1))
    log('evaluated rebalances', len(eval_ks), c2eval.iso(E['days'][rebal[eval_ks[0]]]), '..',
        c2eval.iso(E['days'][rebal[eval_ks[-1]]]))

    # P2's scores: stockai's rule, recomputed; equal to the saved file wherever it has a value (old export).
    mom = momentum_scores(E['close'])
    if which == 'old':
        saved = np.load(MOM_SAVED, mmap_mode='r')
        assert saved.shape == (8503, 5911), saved.shape
        sv = np.asarray(saved)
        f = np.isfinite(sv)
        assert np.array_equal(np.isfinite(mom[:, :f.any(0).nonzero()[0][-1] + 1]), f[:, :f.any(0).nonzero()[0][-1] + 1])
        assert np.allclose(mom[f], sv[f], rtol=0, atol=1e-12), 'momentum recomputation differs from the saved file'
        log('momentum: recomputed equals saved on', int(f.sum()), 'values; saved file ends',
            c2eval.iso(E['days'][f.any(0).nonzero()[0][-1]]))
        del sv, f
    mom = mom.astype(np.float32)

    trades = plan_trades(E, tw, mom, ks, eval_from)
    tr = np.array([t[:5] for t in trades], np.float64)
    k_ = tr[:, 0].astype(int); d0 = tr[:, 1].astype(int); row = tr[:, 2].astype(int)
    side = tr[:, 3].astype(int); size = tr[:, 4]; port = np.array([t[5] for t in trades], int)
    log('trades', len(trades), 'P1', int((port == 1).sum()), 'P2', int((port == 2).sum()))

    # ---- prices
    close = E['close']
    opens, bad = opens_panel(store, E)
    log('bad opens set missing:', int(bad.sum()))
    sid, spy_o, spy_c = spy_series(store, E)
    n = len(row)
    END = np.asarray(close[row, d0 + W], np.float64)
    spy_END = spy_c[d0 + W]
    px = {'open': np.full((n, W), np.nan), 'close': np.full((n, W), np.nan)}
    spx = {'open': np.full((n, W), np.nan), 'close': np.full((n, W), np.nan)}
    open_eq_close = np.zeros((n, W), bool)
    bad_hit = np.zeros(n, bool)
    for k in range(W):
        px['open'][:, k] = opens[row, d0 + k + 1]
        px['close'][:, k] = np.asarray(close[row, d0 + k], np.float64)
        spx['open'][:, k] = spy_o[d0 + k + 1]
        spx['close'][:, k] = spy_c[d0 + k]
        open_eq_close[:, k] = opens[row, d0 + k + 1] == np.asarray(close[row, d0 + k + 1])
        bad_hit |= bad[row, d0 + k + 1]

    # ---- signals: [N, T] panels, then each trade's favourable flags on window days 0..4
    log('S1: twin residual')
    S1 = c2eval.twin_residual(E, tw, 5, scaled=True)
    log('S2: 5-day reversal')
    c64 = np.asarray(close, np.float64)
    S2 = np.full((N, T), np.nan, np.float32)
    with np.errstate(all='ignore'):
        S2[:, 5:] = np.log(c64[:, 5:] / c64[:, :-5])
    del c64
    mk = np.concatenate([[0.0], np.cumsum(np.nan_to_num(np.asarray(E['market'], np.float64)))])
    m5 = np.full(T, np.nan)
    m5[5:] = mk[6:] - mk[1:-5]          # sum of market[d-4..d]
    S2 -= m5[None, :].astype(np.float32)
    sigs = {'S1': (S1, False), 'S2': (S2, False)}
    if which == 'old':
        S3 = np.load(LGBM_SAVED, mmap_mode='r')
        assert S3.shape == (8503, 5911), S3.shape
        sigs['S3'] = (S3, True)

    out = dict(k=k_, d0=d0, row=row, side=side, size=size, port=port, day=E['days'][d0],
               END=END, spy_END=spy_END, px_open=px['open'], px_close=px['close'],
               spy_open=spx['open'], spy_close=spx['close'], open_eq_close=open_eq_close, bad_open_hit=bad_hit,
               tickers=E['tickers'][row], spy_id=sid)
    liq_cache = {}
    for name, (S, high_is_buy) in sigs.items():
        buy = np.zeros((n, W), bool); sell = np.zeros((n, W), bool); fin = np.zeros((n, W), bool)
        for kk in np.unique(k_):
            m = k_ == kk
            dd = int(rebal[kk])
            if kk not in liq_cache:
                liq_cache[kk] = np.nonzero(liquid_set(close, E['dollar_volume'], dd, tw['fund_like'][kk]))[0]
            liq = liq_cache[kk]
            for w in range(W):
                b, s, f, lo, hi = thirds_flags(S, liq, dd + w, row[m], high_is_buy)
                buy[m, w], sell[m, w], fin[m, w] = b, s, f
        out[f'{name}_buy'] = buy; out[f'{name}_sell'] = sell; out[f'{name}_fin'] = fin
        log(name, 'day-0 coverage', round(float(fin[:, 0].mean()), 3))
    out['liq_size'] = np.array([len(liq_cache[kk]) for kk in k_])
    path = os.path.join(HERE, f'trades_{which}.npz')
    np.savez_compressed(path, **out)
    log('wrote', path, f'[{time.time() - t_start:.0f}s]')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'build'
    if cmd == 'build':
        build('old')
    elif cmd == 'build-unseen':
        build('new')
    else:
        raise SystemExit(f'unknown command {cmd}')
