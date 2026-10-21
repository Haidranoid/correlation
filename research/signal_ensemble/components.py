"""Raw component values, [N, T] float32, each at day d from data through the close of d (higher = expected to rise,
except vol/corr, which the composite step turns into the low-correlation rank).

    python3 -I components.py old|new rev5|mom12_1|twin5|lowcorr|check

  rev5     minus the 5-day market-relative log return: -(ln(close[d] / close[d-5]) - sum of market[d-4..d]).
  mom12_1  stockai.baselines.momentum.momentum_scores (copied): ln(close[d-21] / close[d-252]).
  twin5    minus c2eval.twin_residual(E, tw, 5) (scaled), on the export as it is.
  lowcorr  vol (sd of daily log returns over d-251..d, >= 120) and corr (correlation of overlapping 3-day log returns
           with the export's market, 3-day returns ending d-1259..d, >= 750), copied from
           research/betting_against_correlation/panels.py. Written as vol.npy and corr.npy.
  check    compares mom12_1 with stockai's momentum_12_1.npy and vol/corr with the BAC panels (old export only).
The LightGBM component is read directly from stockai's scores file in composite.py.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import time
from common import c2eval, np, EXPORTS, wdir, log, VOL_W, VOL_MIN, COR_W, COR_MIN, MOM_STOCKAI


def save(tag, name, X):
    np.save(wdir(tag) / f'{name}.npy', np.ascontiguousarray(X, dtype=np.float32))
    fin = np.isfinite(X)
    log(f'{tag} {name}: {int(fin.sum()):,} finite values; first day with a value '
        f'{int(np.argmax(fin.any(0)))}')


def momentum_scores(close, skip=21, lookback=252):
    """Copied from stockai/stockai/baselines/momentum.py: out[j, d] = ln(close[j, d - skip] / close[j, d - lookback])."""
    c = np.asarray(close, dtype=np.float64)
    n, t = c.shape
    out = np.full((n, t), np.nan)
    if lookback < t:
        old = c[:, : t - lookback]
        new = c[:, lookback - skip: t - skip]
        ok = np.isfinite(old) & np.isfinite(new) & (old > 0) & (new > 0)
        tail = out[:, lookback:]
        tail[ok] = np.log(new[ok] / old[ok])
    return out


def rolling(a, mask, w):
    """Copied from research/betting_against_correlation/panels.py: window sums over d-w+1..d of a (where mask) and of
    mask, via cumsums along time."""
    n, T = a.shape
    ca = np.zeros((n, T + 1)); ca[:, 1:] = np.cumsum(np.where(mask, a, 0.0), 1)
    cm = np.zeros((n, T + 1)); cm[:, 1:] = np.cumsum(mask, 1)
    lo = np.maximum(np.arange(T) - w + 1, 0)
    hi = np.arange(T) + 1
    return ca[:, hi] - ca[:, lo], cm[:, hi] - cm[:, lo]


def lowcorr(tag):
    """vol and corr exactly as research/betting_against_correlation/panels.py builds them."""
    t_start = time.time()
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'market'))
    N, T = E['close'].shape
    mkt = np.asarray(E['market'], np.float64)
    m3 = np.full(T, np.nan); m3[2:] = mkt[2:] + mkt[1:-1] + mkt[:-2]
    vol = np.full((N, T), np.nan, np.float32)
    corr = np.full((N, T), np.nan, np.float32)
    CH = 1000
    for a in range(0, N, CH):
        b = min(N, a + CH)
        c = np.asarray(E['close'][a:b], np.float64)
        lr = np.full(c.shape, np.nan)
        with np.errstate(all='ignore'):
            lr[:, 1:] = np.log(c[:, 1:] / c[:, :-1])
        ok = np.isfinite(lr)
        s1, n = rolling(lr, ok, VOL_W)
        s2, _ = rolling(lr * lr, ok, VOL_W)
        with np.errstate(all='ignore'):
            v = np.sqrt(np.maximum(s2 - s1 ** 2 / n, 0) / (n - 1))
        v[n < VOL_MIN] = np.nan
        vol[a:b] = v
        x3 = np.full(c.shape, np.nan)
        with np.errstate(all='ignore'):
            x3[:, 3:] = np.log(c[:, 3:] / c[:, :-3])
        ok3 = np.isfinite(x3) & np.isfinite(m3)[None, :]
        M3 = np.broadcast_to(np.nan_to_num(m3), c.shape)
        sx, n3 = rolling(x3, ok3, COR_W)
        sm, _ = rolling(M3, ok3, COR_W)
        sxx, _ = rolling(x3 * x3, ok3, COR_W)
        smm, _ = rolling(M3 * M3, ok3, COR_W)
        sxm, _ = rolling(x3 * M3, ok3, COR_W)
        with np.errstate(all='ignore'):
            cov = sxm - sx * sm / n3
            vx = sxx - sx * sx / n3
            vm = smm - sm * sm / n3
            r = cov / np.sqrt(vx * vm)
        r[(n3 < COR_MIN) | ~np.isfinite(r)] = np.nan
        corr[a:b] = r
        log(f'  rows {a}..{b} done {time.time() - t_start:.0f}s')
    save(tag, 'vol', vol)
    save(tag, 'corr', corr)


def main(tag, what):
    if what == 'rev5':
        E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'market'))
        c = np.asarray(E['close'], np.float64)
        mkt = np.asarray(E['market'], np.float64)
        m5 = np.full(len(mkt), np.nan)
        m5[4:] = mkt[4:] + mkt[3:-1] + mkt[2:-2] + mkt[1:-3] + mkt[:-4]
        out = np.full(c.shape, np.nan)
        with np.errstate(all='ignore'):
            out[:, 5:] = -(np.log(c[:, 5:] / c[:, :-5]) - m5[None, 5:])
        out[~np.isfinite(out)] = np.nan
        save(tag, 'rev5', out)
    elif what == 'mom12_1':
        E = c2eval.load_export(EXPORTS[tag], arrays=('close',))
        save(tag, 'mom12_1', momentum_scores(E['close'], 21, 252))
    elif what == 'twin5':
        E = c2eval.load_export(EXPORTS[tag], arrays=('close',))
        tw = c2eval.twins(E)
        r = c2eval.twin_residual(E, tw, 5)
        save(tag, 'twin5', -r)
        # market-wide gap days: windows spanning them may lose a return for most stocks (counted, not filled)
        n = np.isfinite(np.asarray(E['close'])).sum(0)
        gaps = [c2eval.iso(E['days'][d]) for d in range(10, len(n)) if n[d] < 0.5 * np.median(n[d - 10:d])]
        log(f'{tag}: market-wide gap days (fewer than half the usual closes): {gaps}')
    elif what == 'lowcorr':
        lowcorr(tag)
    elif what == 'check':
        assert tag == 'old'
        mine = np.load(wdir(tag) / 'mom12_1.npy', mmap_mode='r')
        ref = np.load(MOM_STOCKAI, mmap_mode='r')
        T = ref.shape[1]
        a = np.asarray(mine[:, :T], np.float64)
        b = np.asarray(ref, np.float64)
        both = np.isfinite(a) & np.isfinite(b)
        only_a = np.isfinite(a) & ~np.isfinite(b)
        only_b = ~np.isfinite(a) & np.isfinite(b)
        lastb = int(np.nonzero(np.isfinite(b).any(0))[0][-1])
        E = c2eval.load_export(EXPORTS[tag], arrays=())
        log(f'momentum vs stockai: {int(both.sum()):,} both finite; max |diff| {np.abs(a - b)[both].max():.3g} '
            f'(float32 storage); finite only here {int(only_a.sum()):,} (of which after stockai\'s last column '
            f'{c2eval.iso(E["days"][lastb])}: {int(only_a[:, lastb + 1:].sum()):,}); finite only in stockai\'s {int(only_b.sum()):,}')
        del a, b
        z = np.load(Path_bac(), mmap_mode='r')
        for nm in ('vol', 'corr'):
            x = np.load(wdir(tag) / f'{nm}.npy', mmap_mode='r')
            y = z[nm]
            fx, fy = np.isfinite(x), np.isfinite(y)
            d = np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))[fx & fy]
            log(f'{nm} vs BAC panels: same finite cells {bool((fx == fy).all())}; max |diff| {d.max():.3g}')


def Path_bac():
    from pathlib import Path
    return Path.home() / '.local/share/correlation2/experiments/betting_against_correlation/panels_old.npz'


if __name__ == '__main__':
    main(_sys.argv[1], _sys.argv[2])
