"""Silent-peer drift: panels, news days, co-movers and the echo E for one export. No forward returns are computed here.

    systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 -- \
        env OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python3 -I build.py old|new

Rules: experiments/-8.2-silent-peer-drift.md ("Rules" and "Decisions made while building it").
Adapted from research/bystanders/common.py and events.py (copied, not imported), with these differences:
  - market-relative move = daily log return - the export's `market` (equal-weight mean log return of the liquid stocks);
  - universe = the judge-like liquid set (>= 90% finite closes over the last 252 days, median close >= $5 and median
    dollar volume >= $5M over them) minus c2eval.twins fund-like stocks, refreshed on the twins calendar (every 21 days);
  - NO solo filter: every news day is kept, with its co-movers' echo E.

Output (~/.local/share/correlation2/experiments/silent_peer_drift/):
  panels_<tag>.npz   x (float32 [N,T]), sd60, vr, dvmed, U-calendar (rebal, liq), msimple (equal-weight simple market)
  events_<tag>.npz   one row per news day: A, d, xA, sdA, vrA, dvA, n_co, sum_b, sum_xB, E, mean_corr, n_part
                     and one row per quiet big move (the no-news contrast): qA, qd, qx, qsd, qvr
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402

BIG = Path.home() / '.local/share/correlation2/experiments/silent_peer_drift'
BIG.mkdir(parents=True, exist_ok=True)
SRC = {'old': (c2eval.EXPORT_OLD, c2eval.STORE_OLD), 'new': (c2eval.EXPORT_NEW, c2eval.STORE_NEW)}

CORR_DAYS, MIN_PAIR_OBS, TOP, MIN_CORR = 252, 200, 20, 0.3
VOL_X, MOVE_SD, GAP = 4.0, 3.0, 5
QUIET_VR = 1.5           # no-news contrast: same-size move on volume <= 1.5x its 60-day median
PART_VR = 2.0            # a co-mover with its own volume >= 2x counts as a participant (context only)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def roll_nanmedian(a, w=60, minobs=40, chunk=96):
    """out[:, d] = nanmedian(a[:, d-w:d]) where >= minobs finite, else NaN (copied from bystanders/common.py)."""
    N, T = a.shape
    out = np.full((N, T), np.nan, np.float32)
    for d0 in range(w, T, chunk):
        d1 = min(T, d0 + chunk)
        blk = np.lib.stride_tricks.sliding_window_view(a[:, d0 - w:d1 - 1], w, axis=1)
        cnt = np.isfinite(blk).sum(2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            med = np.nanmedian(blk, axis=2)
        med[cnt < minobs] = np.nan
        out[:, d0:d1] = med
    return out


def roll_nanstd(a, w=60, minobs=40):
    """out[:, d] = sd of a[:, d-w:d] (ddof 1) where >= minobs finite (copied from bystanders/common.py)."""
    N, T = a.shape
    f = np.isfinite(a)
    a0 = np.where(f, a, 0).astype(np.float64)
    c1 = np.zeros((N, T + 1)); c2 = np.zeros((N, T + 1)); cn = np.zeros((N, T + 1))
    c1[:, 1:] = np.cumsum(a0, 1); c2[:, 1:] = np.cumsum(a0 * a0, 1); cn[:, 1:] = np.cumsum(f, 1)
    del a0, f
    out = np.full((N, T), np.nan, np.float32)
    d = np.arange(w, T)
    s1 = c1[:, d] - c1[:, d - w]; s2 = c2[:, d] - c2[:, d - w]; n = cn[:, d] - cn[:, d - w]
    with np.errstate(all='ignore'):
        var = (s2 - s1 * s1 / n) / (n - 1)
        sd = np.sqrt(np.maximum(var, 0))
    sd[n < minobs] = np.nan
    out[:, w:] = sd
    return out


def liquid_set(close, dv, d0, fund_like_k):
    """Judge-like liquid rule at d0 (as research/free_timing): >= 90% finite closes over d0-251..d0, median close
    >= $5 and median dollar volume >= $5M over those days; minus fund-like stocks."""
    c = np.asarray(close[:, d0 - 251:d0 + 1], np.float64)
    v = np.asarray(dv[:, d0 - 251:d0 + 1], np.float64)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ok = np.isfinite(c).mean(1) >= 0.9
        ok &= np.nan_to_num(np.nanmedian(c, 1)) >= 5.0
        ok &= np.nan_to_num(np.nanmedian(v, 1)) >= 5e6
    return ok & ~fund_like_k


def build_panels(tag):
    exp, sto = SRC[tag]
    E = c2eval.load_export(exp, arrays=('close', 'dollar_volume', 'market'))
    days = E['days']
    N, T = len(E['ids']), len(days)
    log(tag, 'N', N, 'T', T)
    c = np.asarray(E['close'], dtype=np.float32)
    mkt = np.asarray(E['market'], dtype=np.float64)
    lr = np.full((N, T), np.nan, np.float32)
    with np.errstate(all='ignore'):
        lr[:, 1:] = np.log(c[:, 1:] / c[:, :-1])
    lr[~np.isfinite(lr)] = np.nan
    x = (lr - mkt[None, :].astype(np.float32)).astype(np.float32)
    log('sd60')
    sd60 = roll_nanstd(x)
    log('twins (cached)')
    tw = c2eval.twins(E)
    rebal = tw['rebal']
    fl = tw['fund_like']
    log('liquid set on the twins calendar')
    dv = np.asarray(E['dollar_volume'], dtype=np.float32)
    liq = np.zeros((len(rebal), N), bool)
    for k, t0 in enumerate(rebal):
        liq[k] = liquid_set(c, dv, int(t0), fl[k])
    log('liquid set sizes: median', int(np.median(liq.sum(1))), 'min', int(liq.sum(1).min()), 'max', int(liq.sum(1).max()))
    log('dollar-volume median (60 days)')
    dvp = np.where(dv > 0, dv, np.nan).astype(np.float32)
    dvmed = roll_nanmedian(dvp)
    del dvp
    log('volume from the store')
    S = c2eval.load_store(sto)
    vol = c2eval.store_panel(S, E['ids'], days, 'volume')
    vol = np.where(vol > 0, vol, np.nan).astype(np.float32)
    vmed = roll_nanmedian(vol)
    with np.errstate(all='ignore'):
        vr = (vol / vmed).astype(np.float32)
    del vol, vmed
    # equal-weight simple-return market over the universe (for the plan's long-short hedge), per day
    kd = np.searchsorted(rebal, np.arange(T), side='right') - 1
    sr = np.full((N, T), np.nan, np.float32)
    with np.errstate(all='ignore'):
        sr[:, 1:] = c[:, 1:] / c[:, :-1] - 1
    sr[~np.isfinite(sr)] = np.nan
    msimple = np.full(T, np.nan)
    for d in range(T):
        if kd[d] < 0:
            continue
        u = liq[kd[d]] & np.isfinite(sr[:, d])
        if u.sum() >= 50:
            msimple[d] = float(np.mean(sr[u, d], dtype=np.float64))
    del sr
    log('saving panels')
    np.save(BIG / f'x_{tag}.npy', x)
    np.save(BIG / f'sd60_{tag}.npy', sd60)
    np.save(BIG / f'vr_{tag}.npy', vr)
    np.save(BIG / f'dvmed_{tag}.npy', dvmed)
    np.savez(BIG / f'univ_{tag}.npz', rebal=rebal, liq=liq, msimple=msimple, days=days, ids=E['ids'], tickers=E['tickers'],
             market=mkt)
    log('panels done')


def load_panels(tag):
    P = {n: np.load(BIG / f'{n}_{tag}.npy', mmap_mode='r') for n in ('x', 'sd60', 'vr', 'dvmed')}
    z = np.load(BIG / f'univ_{tag}.npz')
    P.update({k: z[k] for k in z.files})
    return P


def build_events(tag):
    P = load_panels(tag)
    x = np.asarray(P['x']); sd60 = np.asarray(P['sd60']); vr = np.asarray(P['vr']); dvmed = np.asarray(P['dvmed'])
    rebal, liq, days = P['rebal'], P['liq'], P['days']
    N, T = x.shape
    kd = np.searchsorted(rebal, np.arange(T), side='right') - 1
    with np.errstate(invalid='ignore'):
        big = np.abs(x) >= MOVE_SD * sd60
        spike = (vr >= VOL_X) & big
        quiet = (vr <= QUIET_VR) & big
    prev_spike = np.zeros_like(spike)
    prev_quiet = np.zeros_like(quiet)
    for k in range(1, GAP + 1):
        prev_spike[:, k:] |= spike[:, :-k]
        prev_quiet[:, k:] |= quiet[:, :-k]
    ok_any = np.isfinite(x) & np.isfinite(sd60) & (sd60 > 0)
    rows = []
    qrows = []
    t_start = max(CORR_DAYS + 1, int(rebal[0]))
    log('events from', c2eval.iso(days[t_start]))
    for d in range(t_start, T):
        k = kd[d]
        U = liq[k] & np.isfinite(x[:, d])
        # quiet big moves (no-news contrast): same size, normal volume, no news day in d-5..d-1, first in 5 days
        qa = np.where(quiet[:, d] & U & ok_any[:, d] & ~prev_spike[:, d] & ~prev_quiet[:, d])[0]
        for a in qa:
            qrows.append((a, d, x[a, d], sd60[a, d], vr[a, d]))
        A_idx = np.where(spike[:, d] & U & ok_any[:, d] & ~prev_spike[:, d])[0]
        if len(A_idx) == 0:
            continue
        pool = np.where(U)[0]
        W = x[pool, d - CORR_DAYS:d]
        Wa = x[A_idx, d - CORR_DAYS:d]
        Mp = np.isfinite(W).astype(np.float32); W0 = np.nan_to_num(W)
        Ma = np.isfinite(Wa).astype(np.float32); A0 = np.nan_to_num(Wa)
        n = Ma @ Mp.T
        sab = A0 @ W0.T
        sa = A0 @ Mp.T; sa2 = (A0 * A0) @ Mp.T
        sb = Ma @ W0.T; sb2 = Ma @ (W0 * W0).T
        with np.errstate(all='ignore'):
            cov = sab - sa * sb / n
            va = sa2 - sa * sa / n
            vb = sb2 - sb * sb / n
            cor = cov / np.sqrt(va * vb)
            beta = cov / va  # slope of B on A
        cor[~np.isfinite(cor) | (n < MIN_PAIR_OBS)] = -9
        xd = x[pool, d]
        vrd = vr[pool, d]
        for i, a in enumerate(A_idx):
            ck = cor[i].copy()
            ck[pool == a] = -9
            top = np.argsort(-ck)[:TOP]
            top = top[ck[top] >= MIN_CORR]
            n_co = len(top)
            xa = float(x[a, d])
            if n_co:
                sum_b = float(beta[i, top].sum())
                sum_xb = float(xd[top].astype(np.float64).sum())
                Ech = sum_xb / (sum_b * xa) if sum_b * xa != 0 else np.nan
                mcor = float(cor[i, top].mean())
                npart = int(np.sum(vrd[top] >= PART_VR))
            else:
                sum_b = sum_xb = Ech = mcor = np.nan
                npart = 0
            rows.append((a, d, xa, sd60[a, d], vr[a, d], dvmed[a, d], n_co, sum_b, sum_xb, Ech, mcor, npart))
        if d % 500 == 0:
            log(c2eval.iso(days[d]), 'news rows', len(rows), 'quiet rows', len(qrows))
    ev = np.array(rows, dtype=[('A', 'i4'), ('d', 'i4'), ('xA', 'f8'), ('sdA', 'f8'), ('vrA', 'f8'), ('dvA', 'f8'),
                                ('n_co', 'i2'), ('sum_b', 'f8'), ('sum_xB', 'f8'), ('E', 'f8'), ('mean_corr', 'f8'),
                                ('n_part', 'i2')])
    q = np.array(qrows, dtype=[('A', 'i4'), ('d', 'i4'), ('x', 'f8'), ('sd', 'f8'), ('vr', 'f8')])
    np.savez(BIG / f'events_{tag}.npz', **{k: ev[k] for k in ev.dtype.names},
             **{'q' + k: q[k] for k in q.dtype.names}, days=days)
    log('news days', len(ev), 'with co-movers', int((ev['n_co'] > 0).sum()), 'quiet moves', len(q))


if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else 'old'
    what = sys.argv[2] if len(sys.argv) > 2 else 'all'
    if what in ('all', 'panels'):
        build_panels(tag)
    if what in ('all', 'events'):
        build_events(tag)
