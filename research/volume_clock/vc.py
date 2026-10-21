"""Volume clock (experiments/-7.7-volume-clock.md): build the event table for one export.

    systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 -- \
        env OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python3 -I vc.py old|new

Rules are those of the .md ("Rules" and "Decisions made while building it"). One event per kept prediction:
stock j, spike day D (export index), predicted centre c, entry e = c - 6, exit c + 1. Saved to
~/.local/share/correlation2/experiments/volume_clock/events_<tag>.npz with the outside-window baseline arrays.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402

HERE = Path(__file__).resolve().parent
BIG = c2eval.WORK / 'volume_clock'
BIG.mkdir(parents=True, exist_ok=True)
SRC = {'old': (c2eval.EXPORT_OLD, c2eval.STORE_OLD), 'new': (c2eval.EXPORT_NEW, c2eval.STORE_NEW)}

VR_W, VR_MIN, SPIKE, RUN = 60, 40, 4.0, 5
YEAR_CAL = 364          # calendar days
RECUR_TD = 5            # recurrence tolerance, trading days
DEDUP_TD = 10           # centres within 10 trading days: keep the earliest
PRE, POST = 6, 1        # buy at close c-6, sell at close c+1
HIT_TD = 3
RANK_TD = 504           # past two years for the ranking
MED_W, DATA_W, DATA_FRAC = 252, 756, 0.9
MIN_PRICE, MIN_DV = 5.0, 5e6
GRID = 21


def roll_median_prev(a, w=VR_W, minobs=VR_MIN, chunk=128):
    """out[:, d] = median of the finite values of a[:, d-w:d] if >= minobs, else NaN (sort-based, float32)."""
    N, T = a.shape
    out = np.full((N, T), np.nan, np.float32)
    for d0 in range(w, T, chunk):
        d1 = min(T, d0 + chunk)
        blk = np.lib.stride_tricks.sliding_window_view(a[:, d0 - w:d1 - 1], w, axis=1)  # [N, d1-d0, w]
        s = np.sort(blk, axis=2)  # NaN last
        n = np.isfinite(s).sum(2)
        lo = np.clip((n - 1) // 2, 0, w - 1)
        hi = np.clip(n // 2, 0, w - 1)
        med = 0.5 * (np.take_along_axis(s, lo[..., None], 2)[..., 0] + np.take_along_axis(s, hi[..., None], 2)[..., 0])
        med[n < minobs] = np.nan
        out[:, d0:d1] = med
    return out


def csum(x, dtype):
    """[N, T+1] cumulative sum along days with a leading zero column."""
    out = np.zeros((x.shape[0], x.shape[1] + 1), dtype)
    np.cumsum(x, axis=1, dtype=dtype, out=out[:, 1:])
    return out


def nearest_day(days, target):
    """Index of the export day nearest in calendar days to each target (tie -> earlier); -1 if outside the export."""
    k = np.searchsorted(days, target)
    lo = np.clip(k - 1, 0, len(days) - 1)
    hi = np.clip(k, 0, len(days) - 1)
    pick = np.where(np.abs(days[hi] - target) < np.abs(target - days[lo]), hi, lo)
    pick = np.where((target < days[0]) | (target > days[-1]), -1, pick)
    return pick


def universe_at(close, dv, finite_cs, j, e):
    """[len(j)] bool: the judge-like liquid set at day e (see the .md). close/dv are [N, T] arrays."""
    out = np.zeros(len(j), bool)
    ok = e >= DATA_W - 1
    for s in range(0, len(j), 20000):
        jj, ee, oo = j[s:s + 20000], e[s:s + 20000], ok[s:s + 20000]
        res = np.zeros(len(jj), bool)
        idx = np.where(oo)[0]
        if len(idx):
            J, Ee = jj[idx], ee[idx]
            cols = Ee[:, None] + np.arange(-MED_W + 1, 1)[None, :]
            pc = np.asarray(close[J[:, None], cols], np.float64)
            pv = np.asarray(dv[J[:, None], cols], np.float64)
            with np.errstate(all='ignore'):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    mp = np.nanmedian(pc, 1)
                    mv = np.nanmedian(pv, 1)
            nfin = finite_cs[J, Ee + 1] - finite_cs[J, Ee + 1 - DATA_W]
            cfin = np.isfinite(np.asarray(close[J, Ee]))
            res[idx] = cfin & (mp >= MIN_PRICE) & (mv >= MIN_DV) & (nfin >= DATA_FRAC * DATA_W)
        out[s:s + 20000] = res
    return out


def fund_like_at(tw, j, e):
    k = np.searchsorted(tw['rebal'], e, side='right') - 1
    out = np.zeros(len(j), bool)
    ok = k >= 0
    out[ok] = tw['fund_like'][k[ok], j[ok]]
    return out


def build(tag):
    exp, sto = SRC[tag]
    E = c2eval.load_export(exp, arrays=('close', 'dollar_volume', 'market'))
    days = E['days'].astype(np.int64)
    N, T = len(E['ids']), len(days)
    print(f'{tag}: {N} stocks x {T} days, {c2eval.iso(days[0])}..{c2eval.iso(days[-1])}', flush=True)
    S = c2eval.load_store(sto)
    vol = c2eval.store_panel(S, E['ids'], E['days'], 'volume')
    del S
    vol[~(vol > 0)] = np.nan
    print('rolling volume median', flush=True)
    vmed = roll_median_prev(vol)
    with np.errstate(all='ignore'):
        vr = (vol / vmed).astype(np.float32)
    del vol, vmed
    raw = vr >= SPIKE  # NaN -> False
    prev = np.zeros_like(raw)
    for k in range(1, RUN + 1):
        prev[:, k:] |= raw[:, :-k]
    counted = raw & ~prev
    del prev
    print(f'raw spikes {raw.sum()}, counted {counted.sum()}', flush=True)
    cs_cnt = csum(counted, np.int32)
    cs_raw = csum(raw, np.int32)
    cs_vr = csum(np.where(counted, vr, 0.0), np.float64)

    # --- the clock
    sj, sd = np.nonzero(counted)
    fwd = nearest_day(days, days[sd] + YEAR_CAL)
    back = nearest_day(days, days[sd] - YEAR_CAL)
    keep = (fwd >= 0) & (back >= 0)
    a = np.clip(back - RECUR_TD, 0, T - 1)
    b = np.clip(back + RECUR_TD, 0, T - 1)
    recur = (cs_cnt[sj, b + 1] - cs_cnt[sj, a]) > 0
    n_proj = int((fwd >= 0).sum())
    keep &= recur
    pj, pD, pc = sj[keep], sd[keep], fwd[keep]
    print(f'projections inside the export {n_proj}, with recurrence {len(pj)}', flush=True)
    order = np.lexsort((pD, pc, pj))
    pj, pD, pc = pj[order], pD[order], pc[order]
    kept = np.zeros(len(pj), bool)
    last_j, last_c = -1, -10 ** 9
    for i in range(len(pj)):
        if pj[i] != last_j:
            last_j, last_c = pj[i], -10 ** 9
        if pc[i] - last_c > DEDUP_TD:
            kept[i] = True
            last_c = pc[i]
    pj, pD, pc = pj[kept], pD[kept], pc[kept]
    print(f'after the 10-day rule {len(pj)}', flush=True)

    # --- events: entry e = c - 6
    e = pc - PRE
    ok = e >= 0
    pj, pD, pc, e = pj[ok], pD[ok], pc[ok], e[ok]
    close = E['close']
    dv = E['dollar_volume']
    cl = np.asarray(close, np.float32)
    finite_cs = csum(np.isfinite(cl), np.int32)
    print('universe at entry', flush=True)
    univ = universe_at(cl, dv, finite_cs, pj, e)
    tw = c2eval.twins(E)
    fl = fund_like_at(tw, pj, e)
    # hit: a counted (raw) spike within +-3 trading days of c; NaN if c+3 is past the export
    hv = pc + HIT_TD <= T - 1
    a = np.clip(pc - HIT_TD, 0, T - 1)
    b = np.clip(pc + HIT_TD, 0, T - 1)
    hit = np.where(hv, ((cs_cnt[pj, b + 1] - cs_cnt[pj, a]) > 0).astype(np.float32), np.nan)
    hit_raw = np.where(hv, ((cs_raw[pj, b + 1] - cs_raw[pj, a]) > 0).astype(np.float32), np.nan)
    # Diagnostics added after seeing tests 1-2 (they enter no test): a counted spike within +-10 trading days, a day
    # with volume ratio >= 2 within +-3, and the offset of the counted spike nearest to c within +-10.
    hv10 = pc + 10 <= T - 1
    a10, b10 = np.clip(pc - 10, 0, T - 1), np.clip(pc + 10, 0, T - 1)
    hit10 = np.where(hv10, ((cs_cnt[pj, b10 + 1] - cs_cnt[pj, a10]) > 0).astype(np.float32), np.nan)
    win = pc[:, None] + np.arange(-HIT_TD, HIT_TD + 1)[None, :]
    vol2 = np.where(hv, (vr[pj[:, None], np.clip(win, 0, T - 1)] >= 2).any(1).astype(np.float32), np.nan)
    offs = np.arange(-10, 11)
    win10 = np.clip(pc[:, None] + offs[None, :], 0, T - 1)
    cw = counted[pj[:, None], win10]
    dist = np.where(cw, np.abs(offs)[None, :] * 2 + (offs[None, :] > 0), 999)  # nearest; ties -> earlier
    near_off = np.where(cw.any(1) & hv10, offs[np.argmin(dist, 1)], 99).astype(np.int8)
    # return: ln(close[c+1]/close[c-6]) - sum market[c-5..c+1]
    mk = np.asarray(E['market'], np.float64)
    mcs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(mk))])
    mnan = np.concatenate([[0], np.cumsum(~np.isfinite(mk))])
    x = pc + POST
    rv = x <= T - 1
    xc = np.minimum(x, T - 1)
    with np.errstate(all='ignore'):
        r_stock = np.log(cl[pj, xc].astype(np.float64) / cl[pj, e].astype(np.float64))
        r_mkt = mcs[xc + 1] - mcs[e + 1]
        bad_m = (mnan[xc + 1] - mnan[e + 1]) > 0
        ret = r_stock - r_mkt
    ret[~rv | bad_m | ~np.isfinite(ret)] = np.nan
    # Diagnostic added after seeing test 2: simple return minus the market's compounded return over the same days
    with np.errstate(all='ignore'):
        ret_simple = cl[pj, xc].astype(np.float64) / cl[pj, e].astype(np.float64) - 1 - (np.exp(r_mkt) - 1)
    ret_simple[~np.isfinite(ret)] = np.nan
    # ranking: mean ratio on counted spikes in e-503..e
    a = np.clip(e - RANK_TD + 1, 0, T - 1)
    n_sp = cs_cnt[pj, e + 1] - cs_cnt[pj, a]
    with np.errstate(all='ignore'):
        rank = (cs_vr[pj, e + 1] - cs_vr[pj, a]) / n_sp
    dates = days.astype('datetime64[D]')
    cdate = dates[pc]
    month_day = np.array([str(s)[5:] for s in cdate])
    late_june = (month_day >= '06-22') & (month_day <= '06-30')
    ey = np.array([int(str(s)[:4]) for s in dates[e]])
    print(f'events {len(pj)}, in universe and not fund-like {(univ & ~fl).sum()}', flush=True)

    # --- outside-window baseline and chance hit rate, per stock and period (stock-level, weighted later)
    print('outside baseline', flush=True)
    grid = np.arange(DATA_W - 1, T, GRID)
    gj = np.repeat(np.arange(N), len(grid))
    ge = np.tile(grid, N)
    gu = universe_at(cl, dv, finite_cs, gj, ge).reshape(N, len(grid))
    gfl = fund_like_at(tw, gj, ge).reshape(N, len(grid))
    gk = np.searchsorted(grid, np.arange(T), side='right') - 1
    U = np.zeros((N, T), bool)
    okd = gk >= 0
    U[:, okd] = (gu & ~gfl)[:, gk[okd]]
    del gj, ge
    # blocked start days: [d, d+7] overlaps [c-6, c+1]  <=>  d in [c-13, c+1]
    blk = np.zeros((N, T + 1), np.int32)
    np.add.at(blk, (pj, np.clip(pc - PRE - 7, 0, T)), 1)
    np.add.at(blk, (pj, np.clip(pc + POST + 1, 0, T)), -1)
    blocked = np.cumsum(blk, 1)[:, :T] > 0
    del blk
    r7 = np.full((N, T), np.nan, np.float32)
    with np.errstate(all='ignore'):
        r7[:, :T - 7] = (np.log(cl[:, 7:].astype(np.float64) / cl[:, :T - 7]) - (mcs[None, 8:T + 1] - mcs[None, 1:T - 6])).astype(np.float32)
    mbad = (mnan[8:T + 1] - mnan[1:T - 6]) > 0
    r7[:, :T - 7][:, mbad] = np.nan
    r7s = np.full((N, T), np.nan, np.float32)  # simple version (diagnostic added after seeing test 2)
    with np.errstate(all='ignore'):
        r7s[:, :T - 7] = (cl[:, 7:].astype(np.float64) / cl[:, :T - 7] - np.exp(mcs[None, 8:T + 1] - mcs[None, 1:T - 6])).astype(np.float32)
    r7s[~np.isfinite(r7)] = np.nan
    near = np.zeros((N, T), bool)
    lo = np.clip(np.arange(T) - HIT_TD, 0, T)
    hi = np.clip(np.arange(T) + HIT_TD + 1, 0, T)
    near[:] = (cs_cnt[:, hi] - cs_cnt[:, lo]) > 0
    cs2 = csum(vr >= 2, np.int32)  # diagnostic added after seeing tests 1-2: chance of a >= 2x day within +-3
    near2 = (cs2[:, hi] - cs2[:, lo]) > 0
    del cs2
    periods = period_bounds(days, tag)
    base = {}
    for name, (d0, d1) in periods.items():
        inP = np.zeros(T, bool); inP[d0:d1] = True
        mo = U & inP[None, :] & ~blocked & np.isfinite(r7)
        n_out = mo.sum(1)
        s_out = np.where(mo, r7, 0).sum(1, dtype=np.float64)
        mu = U & inP[None, :]
        n_u = mu.sum(1)
        s_near = (near & mu).sum(1)
        s_out_s = np.where(mo, r7s, 0).sum(1, dtype=np.float64)
        s_near2 = (near2 & mu).sum(1)
        base[name] = np.stack([n_out, s_out, n_u, s_near, s_out_s, s_near2]).astype(np.float64)
    np.savez(BIG / f'events_{tag}.npz', j=pj, D=pD, c=pc, e=e, univ=univ, fund_like=fl, hit=hit, hit_raw=hit_raw,
             ret=ret, rank=rank, ret_simple=ret_simple, hit10=hit10, vol2=vol2, near_off=near_off, n_spikes=n_sp, late_june=late_june, entry_year=ey, vr_D=vr[pj, pD],
             days=days, tickers=E['tickers'], ids=E['ids'],
             **{f'base_{k}': v for k, v in base.items()}, period_names=np.array(list(periods)),
             period_bounds=np.array([periods[k] for k in periods]))
    print('saved', BIG / f'events_{tag}.npz', flush=True)


def period_bounds(days, tag):
    """Entry-day index ranges [d0, d1) of the test periods."""
    def ix(iso):
        return c2eval.day_index(days, iso)
    if tag == 'old':
        return {'2006-2019': (ix('2006-01-01'), ix('2020-01-01')), '2020-2025': (ix('2020-01-01'), len(days))}
    return {'unseen': (ix('2025-06-30'), len(days))}


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else 'old')
