"""Build the per-day ranked sets for the twin gap fade (see experiments/-16.2-twin-gap-fade.md).

For every signal day t from 2008 on and every universe member, one row with: the scaled own gap z (u / trailing sd),
the raw gap g, the Akbas signal a (yesterday's overnight minus intraday), the stock's and its twin's open-to-close
simple returns, beta, the twin, the day's volume ratio, the 500-largest flag and the intraday bad-print flag.
Also writes the judge's side-check scores (the Akbas signal at the close).

    python3 -I build.py old|new     # rows -> ~/.local/share/correlation2/experiments/twin_gap_fade/rows_<tag>.npz
"""
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import numpy as np  # noqa: E402
import c2eval  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
BIG = c2eval.WORK / 'twin_gap_fade'
SRC = {'old': (c2eval.EXPORT_OLD, c2eval.STORE_OLD), 'new': (c2eval.EXPORT_NEW, c2eval.STORE_NEW)}
FIRST = '2008-01-01'
BAD = 0.5            # |overnight| or |intraday| log return beyond this = bad print, day missing
SD_W, SD_MIN = 252, 189
LIQ_W, MED_W = 756, 252
TOP, TOP_W = 500, 63
VOL_W = 60
LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def legs(o, c):
    """overnight[:, t] = log(open_t / close_{t-1}), intraday[:, t] = log(close_t / open_t) (float32, NaN where
    missing), and the bad-print flags of each."""
    with np.errstate(all='ignore'):
        on = np.full(o.shape, np.nan, np.float32)
        on[:, 1:] = np.log(o[:, 1:] / c[:, :-1])
        idr = np.log(c / o).astype(np.float32)
    on[~np.isfinite(on)] = np.nan
    idr[~np.isfinite(idr)] = np.nan
    with np.errstate(invalid='ignore'):
        bad_on = np.abs(on) > BAD
        bad_id = np.abs(idr) > BAD
    return on, idr, bad_on, bad_id


def W(A, rows, a, b):
    """A[rows, a:b] as float64, slicing the columns first (a view) so only the window is copied."""
    return np.asarray(A[:, a:b][rows], np.float64)


def liquid(E, t0):
    """The judge's liquid rule (holdout.rs liquid_universe) at day t0, using data through t0."""
    w0 = max(t0 + 1 - LIQ_W, 0)
    v = np.asarray(E['v'][:, w0:t0 + 1])
    ok = np.isfinite(v).sum(1) * 10 >= (t0 + 1 - w0) * 9
    a = max(t0 + 1 - MED_W, 0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mp = np.nanmedian(np.asarray(E['close'][:, a:t0 + 1]), 1)
        md = np.nanmedian(np.asarray(E['dollar_volume'][:, a:t0 + 1]), 1)
    return ok & (np.nan_to_num(mp) >= 5.0) & (np.nan_to_num(md) >= 5e6)


def main(tag):
    t_start = time.time()
    exp, sto = SRC[tag]
    E = c2eval.load_export(exp, arrays=('close', 'dollar_volume', 'v'))
    days = E['days']
    N, T = len(E['ids']), len(days)
    say(f'{tag}: export {N} x {T}, {c2eval.iso(days[0])}..{c2eval.iso(days[-1])}')
    tw = c2eval.twins(E)
    F = tw.pop('F')
    del F  # the twin funds' prices are read from the store below (opens and closes)
    excl = c2eval.excluded(E, tw)
    S = c2eval.load_store(sto)
    so = c2eval.store_panel(S, E['ids'], days, 'open')
    sc = c2eval.store_panel(S, E['ids'], days, 'close')
    sv = c2eval.store_panel(S, E['ids'], days, 'volume')
    fo = c2eval.store_panel(S, tw['fund'], days, 'open')
    fc = c2eval.store_panel(S, tw['fund'], days, 'close')
    say(f'panels loaded ({time.time() - t_start:.0f}s); funds {len(tw["fund"])}')
    on_s, id_s, bon_s, bid_s = legs(so, sc)
    del so, sc
    on_f, id_f, bon_f, bid_f = legs(fo, fc)
    del fo, fc
    bad_s = bon_s | bid_s
    bad_f = bon_f | bid_f
    # masked: the day is missing if either leg is a bad print
    on_sm = np.where(bad_s, np.nan, on_s).astype(np.float32)
    id_sm = np.where(bad_s, np.nan, id_s).astype(np.float32)
    on_fm = np.where(bad_f, np.nan, on_f).astype(np.float32)
    id_fm = np.where(bad_f, np.nan, id_f).astype(np.float32)
    t_first = c2eval.day_index(days, FIRST)
    say('bad prints from 2008: stock days with |overnight|>0.5', int(bon_s[:, t_first:].sum()), ', |intraday|>0.5',
        int(bid_s[:, t_first:].sum()), '; fund days', int(bon_f[:, t_first:].sum()), int(bid_f[:, t_first:].sum()))

    rebal = tw['rebal']
    K = len(rebal)
    cols = {k: [] for k in ('t', 'i', 'f', 'beta', 'z', 'g', 'a', 'rs', 'rf', 'vr', 'in500', 'in100', 'idbad')}
    scores = np.full((N, T), np.nan, np.float32)
    blocks = []
    dv = E['dollar_volume']
    for k in range(K):
        t0 = int(rebal[k])
        ta = t0 + 1
        tb = int(rebal[k + 1]) if k + 1 < K else T - 1
        if tb < t_first or ta > T - 1:
            continue
        ta_eff = max(ta, t_first)
        L = liquid(E, t0)
        U = L & (tw['fund_row'][k] >= 0) & ~excl[k]
        si = np.nonzero(U)[0]
        fj = tw['fund_row'][k, si].astype(np.int64)
        b = tw['beta'][k, si].astype(np.float64)
        # 500 largest by 63-day median dollar volume at t0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mdv = np.nan_to_num(np.nanmedian(np.asarray(dv[si, t0 + 1 - TOP_W:t0 + 1]), 1))
        in500 = np.zeros(len(si), bool)
        in500[np.argsort(-mdv, kind='stable')[:TOP]] = True
        in100 = np.zeros(len(si), bool)  # post hoc diagnostic: the 100 most traded (spreads of 1-3 bps)
        in100[np.argsort(-mdv, kind='stable')[:100]] = True
        # trailing u history with the current twin and beta: days ta-SD_W .. tb
        c0 = ta - SD_W
        uh = W(on_sm, si, c0, tb + 1) - b[:, None] * W(on_fm, fj, c0, tb + 1)
        fin = np.isfinite(uh)
        x = np.where(fin, uh, 0.0)
        z0 = np.zeros((len(si), 1))
        cs1 = np.concatenate([z0, np.cumsum(x, 1)], 1)
        cs2 = np.concatenate([z0, np.cumsum(x * x, 1)], 1)
        cn = np.concatenate([z0, np.cumsum(fin, 1)], 1)
        tt = np.arange(ta_eff, tb + 1)
        j = tt - ta  # window for day t is hist positions [t - SD_W - c0, t - c0) = [j, j + SD_W)
        n = cn[:, j + SD_W] - cn[:, j]
        s1 = cs1[:, j + SD_W] - cs1[:, j]
        s2 = cs2[:, j + SD_W] - cs2[:, j]
        with np.errstate(all='ignore'):
            var = (s2 - s1 * s1 / n) / (n - 1)
            sd = np.sqrt(np.maximum(var, 0))
        sd[(n < SD_MIN) | ~(sd > 0)] = np.nan
        # today's legs
        e0, e1 = ta_eff, tb + 1
        bon_t = bon_s[:, e0:e1][si]
        g = W(on_s, si, e0, e1)
        g[bon_t] = np.nan
        Gf = W(on_fm, fj, e0, e1)
        u = g - b[:, None] * Gf
        z = u / sd
        a = W(on_sm, si, e0 - 1, e1 - 1) - W(id_sm, si, e0 - 1, e1 - 1)
        ids = W(id_s, si, e0, e1)
        idbad = bid_s[:, e0:e1][si] & ~bon_t
        rs = np.expm1(np.where(idbad, np.clip(ids, -BAD, BAD), ids))
        rs[bon_t] = np.nan
        rf = np.expm1(W(id_fm, fj, e0, e1))
        svb = W(sv, si, e0 - VOL_W, e1)
        vw = np.lib.stride_tricks.sliding_window_view(svb[:, :-1], VOL_W, axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            vmed = np.nanmedian(vw, 2)
        with np.errstate(all='ignore'):
            vr = svb[:, VOL_W:] / vmed
        ok = np.isfinite(z) & np.isfinite(g) & np.isfinite(a) & np.isfinite(rs) & np.isfinite(rf)
        r, c = np.nonzero(ok)
        cols['t'].append(tt[c].astype(np.int32))
        cols['i'].append(si[r].astype(np.int32))
        cols['f'].append(fj[r].astype(np.int32))
        cols['beta'].append(b[r].astype(np.float32))
        cols['z'].append(z[r, c].astype(np.float32))
        cols['g'].append(g[r, c].astype(np.float32))
        cols['a'].append(a[r, c].astype(np.float32))
        cols['rs'].append(rs[r, c].astype(np.float64))
        cols['rf'].append(rf[r, c].astype(np.float64))
        cols['vr'].append(vr[r, c].astype(np.float32))
        cols['in500'].append(in500[r])
        cols['in100'].append(in100[r])
        cols['idbad'].append(idbad[r, c])
        okh = ok & ~idbad
        blocks.append(dict(k=k, t0=t0, day=c2eval.iso(days[t0]), universe=int(len(si)), liquid=int(L.sum()),
                           ranked_median=float(np.median(okh.sum(0))), ranked500_median=float(np.median((okh & in500[:, None]).sum(0))),
                           sd_ok=float(np.isfinite(sd).mean()), funds=int(len(np.unique(fj))), beta_median=float(np.median(b)),
                           corr_median=float(np.median(tw['corr'][k, si]))))
        # judge side check: Akbas signal at close d, for the universe in force for day d+1 (rebal_index(d) = k)
        dd = np.arange(max(t0, t_first), min(tb, T))  # closes d with rebal_index(d) = k: t0 .. rebal[k+1]-1
        if k + 1 >= K:
            dd = np.arange(max(t0, t_first), T)
        scores[np.ix_(si, dd)] = W(on_sm, si, dd[0], dd[-1] + 1) - W(id_sm, si, dd[0], dd[-1] + 1)
    rows = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.lexsort((rows['i'], rows['t']))
    rows = {k: v[order] for k, v in rows.items()}
    BIG.mkdir(parents=True, exist_ok=True)
    np.savez(BIG / f'rows_{tag}.npz', days=days, **rows)
    p = c2eval.save_scores(scores, f'twin_gap_fade/akbas_close_{tag}')
    say(f'rows {len(rows["t"])}, idbad rows {int(rows["idbad"].sum())}; scores -> {p}')
    # coverage by year
    by = {}
    for bl in blocks:
        by.setdefault(bl['day'][:4], []).append(bl)
    say('year  rebalances  universe  ranked/day  ranked500/day  distinct twins  median beta  median twin corr')
    for y, bl in sorted(by.items()):
        say(f'{y}  {len(bl):>3}  {np.mean([x["universe"] for x in bl]):>7.0f}  {np.mean([x["ranked_median"] for x in bl]):>7.0f}  '
            f'{np.mean([x["ranked500_median"] for x in bl]):>6.0f}  {np.mean([x["funds"] for x in bl]):>5.0f}  '
            f'{np.mean([x["beta_median"] for x in bl]):.2f}  {np.mean([x["corr_median"] for x in bl]):.2f}')
    OUT.mkdir(exist_ok=True)
    (OUT / f'build_{tag}.txt').write_text('\n'.join(LOG) + '\n')
    (OUT / f'build_{tag}_blocks.json').write_text(json.dumps(blocks, indent=0))
    say(f'done in {time.time() - t_start:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1])
