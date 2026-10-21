"""Bystanders: shared data panels.

Builds, for one judge export (old or new) and its price store:
  r      simple close-to-close return (export closes)
  U      universe: not a fund, close[d-1] >= $5, 60-day median dollar volume (d-60..d-1) >= $5M, r[d] finite
  m      equal-weight market (mean r over U), with its per-day sum and count so one stock can be left out
  x      market-relative return r - m
  sd60   sd of x over d-60..d-1 (>= 40 obs)
  vr     share volume (store) / its median over d-60..d-1 (>= 40 positive obs)
  spike  vr >= 4 and |x| >= 3 sd60 (any stock, any day)
  opens  store opens (unrepaired), for the next-open entry check
Cached under ~/.local/share/correlation2/experiments/bystanders/panels_<tag>/.
"""
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402

BIG = Path.home() / '.local/share/correlation2/experiments/bystanders'
BIG.mkdir(parents=True, exist_ok=True)

SRC = {'old': (c2eval.EXPORT_OLD, c2eval.STORE_OLD), 'new': (c2eval.EXPORT_NEW, c2eval.STORE_NEW)}


def store_first_dates(store_path):
    """id -> first date (days since 1970) from the store's symbols.tsv."""
    out = {}
    for l in list(open(Path(store_path) / 'symbols.tsv'))[1:]:
        f = l.rstrip('\n').split('\t')
        out[int(f[0])] = int((np.datetime64(f[4]) - np.datetime64('1970-01-01')).astype(int))
    return out


def _rets(p):
    r = np.full_like(p, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        r[:, 1:] = p[:, 1:] / p[:, :-1] - 1
    r[~np.isfinite(r)] = np.nan
    return r


def _pair_stats(X, Y):
    """Pairwise-complete correlation and slope of X-rows on Y-rows (X: [n,T], Y: [m,T], NaN = missing)."""
    Mx = np.isfinite(X).astype(np.float32); X0 = np.nan_to_num(X).astype(np.float32)
    My = np.isfinite(Y).astype(np.float32); Y0 = np.nan_to_num(Y).astype(np.float32)
    n = Mx @ My.T
    sxy = X0 @ Y0.T
    sx = X0 @ My.T; sx2 = (X0 * X0) @ My.T
    sy = Mx @ Y0.T; sy2 = Mx @ (Y0 * Y0).T
    with np.errstate(invalid='ignore', divide='ignore'):
        cov = sxy - sx * sy / n
        vx = sx2 - sx * sx / n
        vy = sy2 - sy * sy / n
        r = cov / np.sqrt(vx * vy)
        beta = cov / vy  # slope of X on Y
    r[n < 250] = 0; beta[n < 250] = 0
    return np.nan_to_num(r), np.nan_to_num(beta)


def fund_flags(E, S, store_path, liquid_ever):
    """Funds filed as stocks (see the .md 'Decisions'):
    (ii) |corr| >= 0.95 with a store ETF whose price history starts earlier, or
    (iii)|corr| >= 0.95 with another export stock whose history starts earlier and slope outside [0.85, 1.15]
         (a leveraged or inverse fund on that stock; share classes, slope ~1, are kept),
    each measured over the full shared history OR over the export's last 504 days (>= 250 shared days).
    The first version also had (i) |corr| >= 0.95 with >= 5 ETFs; on the new export that flagged NVDA, TSLA, AMD,
    AVGO, MSTR and PLTR (now tracked by many single-stock ETFs), so it was replaced by the recent-window check.
    Daily returns clipped to +-50%. Only stocks ever in the liquid universe are checked."""
    first = store_first_dates(store_path)
    days = E['days']
    etf_ids = [s['id'] for s in S['syms'] if s['kind'] == 'etf']
    ec = c2eval.store_panel(S, etf_ids, days, 'close')
    X = np.clip(_rets(np.asarray(E['close'], dtype=np.float32)), -0.5, 0.5)
    Y = np.clip(_rets(ec), -0.5, 0.5)
    idx = np.where(liquid_ever)[0]
    Xl = X[idx]
    fx = np.array([first[int(E['ids'][j])] for j in idx])
    fe = np.array([first[i] for i in etf_ids])
    etk = [S['syms'][i]['ticker'] for i in etf_ids]
    flag_l = np.zeros(len(idx), bool)
    why = {}
    T = X.shape[1]
    for label, sl in (('full', slice(0, T)), ('last 504 days', slice(T - 504, T))):
        r, b = _pair_stats(Xl[:, sl], Y[:, sl])
        ar = np.abs(r)
        f2 = ((ar >= 0.95) & (fe[None, :] < fx[:, None])).any(1)
        r2, b2 = _pair_stats(Xl[:, sl], Xl[:, sl])
        np.fill_diagonal(r2, 0)
        ar2 = np.abs(r2)
        f3 = ((ar2 >= 0.95) & (fx[None, :] < fx[:, None]) & ((b2 < 0.85) | (b2 > 1.15))).any(1)
        for k in np.where((f2 | f3) & ~flag_l)[0]:
            j = idx[k]
            if f2[k]:
                cand = np.where((ar[k] >= 0.95) & (fe < fx[k]))[0]
                t = cand[np.argmax(ar[k, cand])]
                why[E['tickers'][j]] = f"older ETF {etk[t]} r={r[k, t]:+.2f} ({label})"
            else:
                cand = np.where((ar2[k] >= 0.95) & (fx < fx[k]) & ((b2[k] < 0.85) | (b2[k] > 1.15)))[0]
                t = cand[np.argmax(ar2[k, cand])]
                why[E['tickers'][j]] = f"older stock {E['tickers'][idx[t]]} r={r2[k, t]:+.2f} slope={b2[k, t]:+.2f} ({label})"
        flag_l |= f2 | f3
    flag = np.zeros(len(E['ids']), bool)
    flag[idx] = flag_l
    return flag, why


def _roll_nanmedian(a, w=60, minobs=40, chunk=96):
    """out[:, d] = nanmedian(a[:, d-w:d]) where >= minobs finite, else NaN."""
    N, T = a.shape
    out = np.full((N, T), np.nan, np.float32)
    for d0 in range(w, T, chunk):
        d1 = min(T, d0 + chunk)
        blk = np.lib.stride_tricks.sliding_window_view(a[:, d0 - w:d1 - 1], w, axis=1)  # [N, d1-d0, w]
        cnt = np.isfinite(blk).sum(2)
        with np.errstate(all='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                med = np.nanmedian(blk, axis=2)
        med[cnt < minobs] = np.nan
        out[:, d0:d1] = med
    return out


def _roll_nanstd(a, w=60, minobs=40):
    N, T = a.shape
    f = np.isfinite(a)
    a0 = np.where(f, a, 0).astype(np.float64)
    c1 = np.zeros((N, T + 1)); c2 = np.zeros((N, T + 1)); cn = np.zeros((N, T + 1))
    c1[:, 1:] = np.cumsum(a0, 1); c2[:, 1:] = np.cumsum(a0 * a0, 1); cn[:, 1:] = np.cumsum(f, 1)
    out = np.full((N, T), np.nan, np.float32)
    d = np.arange(w, T)
    s1 = c1[:, d] - c1[:, d - w]; s2 = c2[:, d] - c2[:, d - w]; n = cn[:, d] - cn[:, d - w]
    with np.errstate(all='ignore'):
        var = (s2 - s1 * s1 / n) / (n - 1)
        sd = np.sqrt(np.maximum(var, 0))
    sd[n < minobs] = np.nan
    out[:, w:] = sd
    return out


def build_panels(tag):
    exp, sto = SRC[tag]
    out = BIG / f'panels_{tag}'
    out.mkdir(exist_ok=True)
    E = c2eval.load_export(exp, arrays=('close', 'dollar_volume'))
    S = c2eval.load_store(sto)
    days = E['days']
    N, T = len(E['ids']), len(days)
    c = np.asarray(E['close'], dtype=np.float32)
    dv = np.asarray(E['dollar_volume'], dtype=np.float32)
    dv = np.where(dv > 0, dv, np.nan).astype(np.float32)
    print('rolling dollar-volume median', flush=True)
    dvmed = _roll_nanmedian(dv)
    prev_c = np.full_like(c, np.nan); prev_c[:, 1:] = c[:, :-1]
    with np.errstate(invalid='ignore'):
        liq = (prev_c >= 5) & (dvmed >= 5e6)
    liquid_ever = liq.any(1)
    print('fund screen', flush=True)
    fund, why = fund_flags(E, S, sto, liquid_ever)
    (out / 'funds.json').write_text(json.dumps(why, indent=1, sort_keys=True))
    print(f'funds dropped: {fund.sum()}', flush=True)
    r = _rets(c)
    U = liq & ~fund[:, None] & np.isfinite(r)
    rU = np.where(U, r, 0.0).astype(np.float64)
    msum = rU.sum(0)
    mcnt = U.sum(0).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        m = msum / mcnt
    m[mcnt < 50] = np.nan
    x = (r - m[None, :].astype(np.float32)).astype(np.float32)
    print('rolling sd', flush=True)
    sd60 = _roll_nanstd(x)
    print('volume', flush=True)
    vol = c2eval.store_panel(S, E['ids'], days, 'volume')
    vol = np.where(vol > 0, vol, np.nan).astype(np.float32)
    vmed = _roll_nanmedian(vol)
    with np.errstate(invalid='ignore', divide='ignore'):
        vr = (vol / vmed).astype(np.float32)
        spike = (vr >= 4) & (np.abs(x) >= 3 * sd60)
    op = c2eval.store_panel(S, E['ids'], days, 'open')
    # market open-to-close on each day (for next-open entry), over U
    with np.errstate(invalid='ignore', divide='ignore'):
        oc = c / op - 1
    okoc = U & np.isfinite(oc) & (op > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        moc = np.where(okoc, oc, 0).sum(0) / okoc.sum(0)
    for name, arr in (('r', r), ('U', U), ('x', x), ('sd60', sd60), ('vr', vr), ('spike', spike), ('open', op),
                      ('close', c), ('dvmed', dvmed)):
        np.save(out / f'{name}.npy', arr)
    np.savez(out / 'market.npz', m=m, msum=msum, mcnt=mcnt, moc=moc, days=days, ids=E['ids'], tickers=E['tickers'],
             fund=fund)
    print('done', out, flush=True)


def load_panels(tag, names=('r', 'U', 'x', 'sd60', 'vr', 'spike', 'open', 'close')):
    p = BIG / f'panels_{tag}'
    P = {n: np.load(p / f'{n}.npy', mmap_mode='r') for n in names}
    mk = np.load(p / 'market.npz', allow_pickle=False)
    P.update({k: mk[k] for k in mk.files})
    return P


def expiry_days(days):
    """Indices of quarterly expiry days (third Friday of Mar/Jun/Sep/Dec, previous trading day if a holiday) and
    the Russell reconstitution day (last Friday of June, or the Friday before if that is the 29th or 30th)."""
    dt = days.astype('datetime64[D]')
    s = set()
    years = range(int(str(dt[0])[:4]), int(str(dt[-1])[:4]) + 1)
    for y in years:
        for mth in (3, 6, 9, 12):
            first = np.datetime64(f'{y}-{mth:02d}-01')
            wd = (first.astype(int) - 4) % 7  # 0 = Monday (1970-01-01 was a Thursday)
            fri1 = first + ((4 - wd) % 7)
            tgt = fri1 + 14
            k = np.searchsorted(dt, tgt, side='right') - 1
            if 0 <= k < len(dt) and (tgt - dt[k]).astype(int) < 7:
                s.add(int(k))
        last = np.datetime64(f'{y}-06-30')
        wd = (last.astype(int) - 4) % 7
        lf = last - ((wd - 4) % 7)
        if int(str(lf)[8:10]) >= 29:
            lf = lf - 7
        k = np.searchsorted(dt, lf, side='right') - 1
        if 0 <= k < len(dt) and (lf - dt[k]).astype(int) < 7:
            s.add(int(k))
    return np.array(sorted(s), dtype=np.int64)


if __name__ == '__main__':
    build_panels(sys.argv[1] if len(sys.argv) > 1 else 'old')
