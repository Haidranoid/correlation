"""Build the 22 price, volume and range base features as [T, N] float32 panels (the 4 twin features are in
build_twin.py).

    systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -- env OMP_NUM_THREADS=4 \
        OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 -I build_features.py old|new

Sources: returns, volatility, beta, skew, highs of closes and dollar volume from the judge's export (repaired closes);
open, high, low, close and volume from the unrepaired store, with the close-location bad-print rules:
a day's range is missing when (H - L)/C > 0.5, H < C, L > C or H = L (store close C). Opens only from 2008, and only
when L <= O <= H on a valid day; gap and intraday returns beyond +-0.5 (log) are treated as bad prints.
"""
import sys

import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/formula_mining')
from common import (FEATURES, OPEN_FROM, SRC, c2eval, feat_dir, log, roll_beta_corr, roll_max, roll_mean, roll_skew,
                    roll_std)

PRICE_FEATS = [f for f in FEATURES if not f.startswith('tw')]
CHUNK = 600


def main(tag):
    exp, sto = SRC[tag]
    E = c2eval.load_export(exp, arrays=('close', 'dollar_volume', 'market'))
    days = E['days']
    N, T = len(E['ids']), len(days)
    log(tag, 'N', N, 'T', T)
    S = c2eval.load_store(sto)
    mkt = np.asarray(E['market'], np.float64)
    open_from = c2eval.day_index(days, OPEN_FROM)
    out = {f: np.lib.format.open_memmap(feat_dir(tag) / f'{f}.npy', mode='w+', dtype=np.float32, shape=(T, N))
           for f in PRICE_FEATS}
    for j0 in range(0, N, CHUNK):
        j1 = min(N, j0 + CHUNK)
        res = {}
        C = np.asarray(E['close'][j0:j1], np.float64)
        C[~(C > 0)] = np.nan
        with np.errstate(all='ignore'):
            lC = np.log(C)
        r = np.full(C.shape, np.nan)
        r[:, 1:] = lC[:, 1:] - lC[:, :-1]
        for h in (1, 5, 21, 63, 126, 252):
            x = np.full(C.shape, np.nan)
            x[:, h:] = lC[:, h:] - lC[:, :-h]
            res[f'ret{h}'] = x
        x = np.full(C.shape, np.nan)
        x[:, 252:] = lC[:, 231:T - 21] - lC[:, :T - 252]
        res['mom12_1'] = x
        res['vol21'] = roll_std(r, 21, 17)
        res['vol63'] = roll_std(r, 63, 51)
        res['maxret21'] = roll_max(r, 21, 17)
        res['skew63'] = roll_skew(r, 63, 51)
        res['beta252'], res['mcorr252'] = roll_beta_corr(r, mkt, 252, 202)
        hi = roll_max(lC, 252, 202)
        res['dist_high252'] = lC - hi
        dv = np.asarray(E['dollar_volume'][j0:j1], np.float64)
        dv[~(dv > 0)] = np.nan
        res['dvol21'] = roll_mean(dv, 21, 13)
        del dv, hi, r, lC, C
        ids = E['ids'][j0:j1]
        O = c2eval.store_panel(S, ids, days, 'open').astype(np.float64)
        H = c2eval.store_panel(S, ids, days, 'high').astype(np.float64)
        L = c2eval.store_panel(S, ids, days, 'low').astype(np.float64)
        SC = c2eval.store_panel(S, ids, days, 'close').astype(np.float64)
        V = c2eval.store_panel(S, ids, days, 'volume').astype(np.float64)
        with np.errstate(all='ignore'):
            ok = (np.isfinite(H) & np.isfinite(L) & (SC > 0) & ((H - L) / SC <= 0.5) & (H >= SC) & (L <= SC) & (H != L))
            hl = np.where(ok, (H - L) / SC, np.nan)
            clv = np.where(ok, ((SC - L) - (H - SC)) / (H - L), np.nan)
            ook = ok & (O > 0) & (O >= L) & (O <= H)
            ook[:, :open_from] = False
            gap = np.full(SC.shape, np.nan)
            gap[:, 1:] = np.where(ook[:, 1:] & ok[:, :-1], np.log(O[:, 1:] / SC[:, :-1]), np.nan)
            intra = np.where(ook, np.log(SC / O), np.nan)
        gap[np.abs(gap) > 0.5] = np.nan
        intra[np.abs(intra) > 0.5] = np.nan
        res['hl5'] = roll_mean(hl, 5, 3)
        res['hl21'] = roll_mean(hl, 21, 13)
        res['clv5'] = roll_mean(clv, 5, 3)
        res['gap5'] = roll_mean(gap, 5, 3)
        res['intra5'] = roll_mean(intra, 5, 3)
        V[~(V > 0)] = np.nan
        with np.errstate(all='ignore'):
            res['vr5_60'] = roll_mean(V, 5, 3) / roll_mean(V, 60, 36)
            res['vr21_252'] = roll_mean(V, 21, 13) / roll_mean(V, 252, 152)
        del O, H, L, SC, V, ok, hl, clv, ook, gap, intra
        for f in PRICE_FEATS:
            x = res[f]
            x[~np.isfinite(x)] = np.nan
            out[f][:, j0:j1] = x.T.astype(np.float32)
        del res
        log(f'rows {j0}..{j1}')
    for f in PRICE_FEATS:
        out[f].flush()
        a = out[f]
        fin = np.isfinite(a[-1]).mean()
        log(f'{f}: finite share on the last day {fin:.2f}')
    log('done')


if __name__ == '__main__':
    main(sys.argv[1])
