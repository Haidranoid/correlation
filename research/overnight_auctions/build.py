"""Overnight auctions: build the signal and the legs for one export (rules: experiments/+15.7-overnight-auctions.md
and out/decisions.txt).

    python3 -I build.py old|new      # -> WORK/panels_<tag>.npz, WORK/S_<tag>.npy, WORK/Sraw_<tag>.npy

Signal for day t (formed before the 15:50 closing-auction cutoff: opens through t, closes through t-1):
  o_i(d) = log(open_d / close_{d-1}) from the store; missing if a price is missing or <= 0, |o| > 0.5, d before the
  first day, or open_d == close_{d-1} (stale open). Uses only open_d and close_{d-1}.
  x_i(d) = o_i(d) - beta_i * o_twin(d), twin and beta from rebalance k = kidx(rebal, t-1) for all d = t-20..t.
  S_i(t) = mean of the present x_i(d), d = t-20..t, >= 17 present. S_raw: the same mean of o_i.
Legs on exit day d = t+1: overnight o(d) (same rule), intraday log(close_d / open_d) (|.| <= 0.5), ok = both valid
and not open = close = high = low; stale = open_d == close_{d-1} with everything else valid (for the sensitivity).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np = C.np
c2eval = C.c2eval

START = {'old': C.FIRST, 'new': '2025-01-01'}   # first overnight day used (new: only the unseen months are graded)


def store_panels(S, ids, days, c0):
    """open, close, high, low [len(ids), T - c0] float32 on export days c0.."""
    d = days[c0:]
    return {f: c2eval.store_panel(S, ids, d, f) for f in ('open', 'close', 'high', 'low')}


def legs(P, first_col):
    """From store panels (columns = export days c0..): overnight (signal rule), intraday, ok (outcome rule), stale.
    Column 0 has no overnight. first_col: panel column of the first overnight day allowed."""
    o, c, h, lo = P['open'], P['close'], P['high'], P['low']
    n, Tc = o.shape
    on = np.full((n, Tc), np.nan, np.float32)
    intra = np.full((n, Tc), np.nan, np.float32)
    ok = np.zeros((n, Tc), bool)
    stale = np.zeros((n, Tc), bool)
    with np.errstate(all='ignore'):
        o1, c1, cp = o[:, 1:].astype(np.float64), c[:, 1:].astype(np.float64), c[:, :-1].astype(np.float64)
        prices = np.isfinite(o1) & (o1 > 0) & np.isfinite(cp) & (cp > 0)
        a = np.log(o1 / cp)
        same = o[:, 1:] == c[:, :-1]
        sig = prices & (np.abs(a) <= C.MAX_LOG) & ~same
        sig[:, :max(first_col - 1, 0)] = False
        on[:, 1:] = np.where(sig, a, np.nan)
        b = np.log(c1 / o1)
        iv = np.isfinite(c1) & (c1 > 0) & np.isfinite(o1) & (o1 > 0) & (np.abs(b) <= C.MAX_LOG)
        intra[:, 1:] = np.where(iv, b, np.nan)
        norange = (o[:, 1:] == c[:, 1:]) & (h[:, 1:] == lo[:, 1:]) & (o[:, 1:] == h[:, 1:])
        ok[:, 1:] = sig & iv & ~norange
        st = prices & same & iv & ~norange
        st[:, :max(first_col - 1, 0)] = False
        stale[:, 1:] = st
    return on, intra, ok, stale


def signal(on_s, on_f, tw, uni, c0, T, ks):
    """S and S_raw [N, T] float32 (NaN outside the universe-with-a-twin) for signal days of blocks ks.
    Block k covers signal days t = rebal[k]+1 .. rebal[k+1] (the days with kidx(rebal, t-1) == k)."""
    N = on_s.shape[0]
    rebal = tw['rebal']
    K = len(rebal)
    S = np.full((N, T), np.nan, np.float32)
    Sr = np.full((N, T), np.nan, np.float32)
    for k in ks:
        ta = int(rebal[k]) + 1
        tb = int(rebal[k + 1]) if k + 1 < K else T - 1
        ta = max(ta, c0 + C.H_MIN)        # earlier windows can't hold 17 overnights (days before the panels are missing)
        if tb < ta:
            continue
        assert int(rebal[k]) <= ta - 1    # timing: the rebalance's inputs end at a close no later than t-1
        si = np.nonzero(uni[k] & (tw['fund_row'][k] >= 0))[0]
        if len(si) == 0:
            continue
        fj = tw['fund_row'][k, si].astype(np.int64)
        b = tw['beta'][k, si].astype(np.float64)
        w0, w1 = ta - (C.H - 1) - c0, tb + 1 - c0          # panel columns of days ta-20 .. tb
        pad = max(0, -w0)                                   # days before the panels count as missing
        os_ = np.concatenate([np.full((len(si), pad), np.nan), on_s[si, max(w0, 0):w1].astype(np.float64)], 1)
        of_ = np.concatenate([np.full((len(si), pad), np.nan), on_f[fj, max(w0, 0):w1].astype(np.float64)], 1)
        x = os_ - b[:, None] * of_
        for arr, out in ((x, S), (os_, Sr)):
            f = np.isfinite(arr)
            z = np.zeros((len(si), 1))
            cs = np.concatenate([z, np.cumsum(np.where(f, arr, 0.0), 1)], 1)
            cn = np.concatenate([z, np.cumsum(f, 1)], 1)
            j = np.arange(tb - ta + 1)                    # window for t = ta + j is columns j .. j + H - 1
            n = cn[:, j + C.H] - cn[:, j]
            s = cs[:, j + C.H] - cs[:, j]
            with np.errstate(all='ignore'):
                m = s / n
            m[n < C.H_MIN] = np.nan
            out[np.ix_(si, np.arange(ta, tb + 1))] = m.astype(np.float32)
    return S, Sr


def spy_beta(close, spy_close, rebal, ks, W=252, min_obs=227):
    """[K, N] float32: OLS slope of each stock's daily log return on SPY's over the W days ending at rebal[k]."""
    N = close.shape[0]
    out = np.full((len(rebal), N), np.nan, np.float32)
    for k in ks:
        t0 = int(rebal[k])
        if t0 < W:
            continue
        c = np.asarray(close[:, t0 - W:t0 + 1], np.float64)
        sc = spy_close[t0 - W:t0 + 1].astype(np.float64)
        with np.errstate(all='ignore'):
            r = np.log(c[:, 1:] / c[:, :-1])
            m = np.log(sc[1:] / sc[:-1])
        r[~np.isfinite(r) | (np.abs(r) > C.MAX_LOG)] = np.nan
        m[~np.isfinite(m) | (np.abs(m) > C.MAX_LOG)] = np.nan
        M = np.isfinite(r) & np.isfinite(m)[None, :]
        n = M.sum(1)
        mm = np.where(M, m[None, :], 0.0)
        rr = np.where(M, r, 0.0)
        with np.errstate(all='ignore'):
            mx, my = mm.sum(1) / n, rr.sum(1) / n
            cov = (mm * rr).sum(1) / n - mx * my
            var = (mm * mm).sum(1) / n - mx * mx
            bt = cov / var
        bt[(n < min_obs) | ~np.isfinite(bt)] = np.nan
        out[k] = bt
    return out


def blocks_needed(rebal, c0, T):
    """Blocks whose signal days reach past the panels' first usable day."""
    K = len(rebal)
    ks = []
    for k in range(K):
        tb = int(rebal[k + 1]) if k + 1 < K else T - 1
        if tb >= c0 + C.H_MIN:
            ks.append(k)
    return ks


def build(tag):
    C.log('build', tag)
    E = c2eval.load_export(C.EXPORTS[tag], arrays=('close', 'dollar_volume'))
    days = E['days']
    N, T = E['close'].shape
    tw = c2eval.twins(E)
    tw.pop('F')
    rebal = tw['rebal']
    uni, big = C.universe(E, tw)
    C.log('universe done')
    first_day = c2eval.day_index(days, START[tag])
    c0 = first_day - 1                      # the previous close of the first overnight
    ks = blocks_needed(rebal, c0, T)
    S_ = c2eval.load_store(C.STORES[tag])
    Ps = store_panels(S_, E['ids'], days, c0)
    on_s, in_s, ok_s, st_s = legs(Ps, first_day - c0)
    # store close vs export close on the panels (the judge's returns use the export's)
    ce = np.asarray(E['close'][:, c0:], np.float32)
    both = np.isfinite(ce) & np.isfinite(Ps['close'])
    with np.errstate(all='ignore'):
        mis = both & (np.abs(np.log(Ps['close'] / ce)) > 1e-6)
    C.log('store close vs export close: both finite', int(both.sum()), 'differ', int(mis.sum()))
    del Ps, ce, both
    Pf = store_panels(S_, tw['fund'], days, c0)
    on_f, in_f, ok_f, st_f = legs(Pf, first_day - c0)
    del Pf
    spy_row = int(np.nonzero(tw['fund_ticker'] == 'SPY')[0][0])
    spy_full = c2eval.store_panel(S_, [int(tw['fund'][spy_row])], days, 'close')[0]
    C.log('legs done; stock overnights valid', int(np.isfinite(on_s).sum()), 'stale', int(st_s.sum()),
          'fund overnights valid', int(np.isfinite(on_f).sum()))
    S, Sr = signal(on_s, on_f, tw, uni, c0, T, ks)
    C.log('signal done; finite S', int(np.isfinite(S).sum()))
    bspy = spy_beta(E['close'], spy_full, rebal, ks)
    np.save(C.WORK / f'S_{tag}.npy', S)
    np.save(C.WORK / f'Sraw_{tag}.npy', Sr)
    np.savez(C.WORK / f'panels_{tag}.npz', days=days, c0=c0, first_day=first_day, rebal=rebal, uni=uni, big=big,
             fund_row=tw['fund_row'], beta=tw['beta'], bspy=bspy, spy_row=spy_row,
             on_s=on_s, in_s=in_s, ok_s=ok_s, st_s=st_s, on_f=on_f, in_f=in_f, ok_f=ok_f, st_f=st_f,
             mismatch=int(mis.sum()))
    C.log('saved')


def load(tag):
    z = np.load(C.WORK / f'panels_{tag}.npz')
    P = {k: z[k] for k in z.files}
    P['S'] = np.load(C.WORK / f'S_{tag}.npy')
    P['Sraw'] = np.load(C.WORK / f'Sraw_{tag}.npy')
    P['c0'] = int(P['c0']); P['first_day'] = int(P['first_day']); P['spy_row'] = int(P['spy_row'])
    P['tag'] = tag
    return P


if __name__ == '__main__':
    build(sys.argv[1])
