"""Step 1: daily panels for the phantom-portfolio experiment, cached under BIG/<tag>/.

    OMP_NUM_THREADS=8 python3 -I pp_panels.py old|new

returns, open->close returns, raw volume / its 63-day median, the factor universe, the equal-weight market,
six price-based style characteristics (as of each close) and their long-short tercile factor returns, and
label ETFs' returns. Everything on day d uses data through the close of day d; factor membership for day d
uses characteristics as of d-1.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pp_common import (EXPORTS, MIN_DV, MIN_PRICE, VOL_NORMAL_WIN, big, c2eval, rolling_median_prev,  # noqa: E402
                       rolling_sum)

LABEL_ETFS = ['SPY', 'IWD', 'IWF', 'IWN', 'IWO', 'IWM', 'MTUM', 'QUAL', 'USMV', 'VLUE', 'TLT', 'XLE', 'USO', 'GLD',
              'GDX', 'FXI', 'KWEB', 'IBB', 'XBI', 'VNQ', 'XLU', 'XLF', 'KRE', 'XHB', 'ITB', 'SMH', 'ICLN', 'TAN',
              'ARKK', 'XRT', 'XLK', 'XLV', 'XLP', 'XLY', 'XLI', 'XLB', 'EEM', 'UUP', 'JETS', 'URA', 'LIT', 'MJ',
              'BITO', 'HYG']
CHARS = ['mom12_1', 'rev1m', 'vol60', 'beta', 'dollar_volume', 'price']


def main(tag):
    t0 = time.time()
    out = big(tag)
    export, store_path = EXPORTS[tag]
    E = c2eval.load_export(export)
    close = np.asarray(E['close'], dtype=np.float32)
    dv = np.asarray(E['dollar_volume'], dtype=np.float32)
    days = E['days']
    n, t = close.shape
    print(f'{tag}: {n} stocks x {t} days, {c2eval.iso(days[0])}..{c2eval.iso(days[-1])}', flush=True)

    with np.errstate(all='ignore'):
        ret = np.full((n, t), np.nan, np.float32)
        ok = np.isfinite(close[:, 1:]) & np.isfinite(close[:, :-1]) & (close[:, :-1] > 0)
        ret[:, 1:] = np.where(ok, close[:, 1:] / close[:, :-1] - 1, np.nan)

    store = c2eval.load_store(store_path)
    ids = E['ids']
    # The export's ids are the store's ids; check the tickers line up.
    assert all(store['syms'][i]['ticker'] == tk for i, tk in zip(ids, E['tickers'])), 'export ids != store ids'
    vol = c2eval.store_panel(store, ids, days, 'volume')
    opn = c2eval.store_panel(store, ids, days, 'open')
    sclose = c2eval.store_panel(store, ids, days, 'close')
    with np.errstate(all='ignore'):
        intra = np.where(np.isfinite(opn) & np.isfinite(sclose) & (opn > 0), sclose / opn - 1, np.nan).astype(np.float32)
    print(f'  store panels {time.time() - t0:.0f}s', flush=True)

    vol_norm = rolling_median_prev(vol, VOL_NORMAL_WIN, 40)
    with np.errstate(all='ignore'):
        volratio = np.where(vol_norm > 0, vol / vol_norm, np.nan).astype(np.float32)
    del vol_norm
    meddv = rolling_median_prev(dv, 63, 40)  # median dollar volume over d-63..d-1
    print(f'  rolling medians {time.time() - t0:.0f}s', flush=True)

    # Factor / market universe for day d: close(d-1) >= $5, median dollar volume over d-63..d-1 >= $5M, return on d.
    prev_close = np.full_like(close, np.nan)
    prev_close[:, 1:] = close[:, :-1]
    elig = (prev_close >= MIN_PRICE) & (meddv >= MIN_DV) & np.isfinite(ret)
    cnt = elig.sum(0)
    M = np.where(cnt > 0, np.where(elig, ret, 0).sum(0, dtype=np.float64) / np.maximum(cnt, 1), np.nan)
    M[cnt < 100] = np.nan
    ei = elig & np.isfinite(intra)
    ci = ei.sum(0)
    M_intra = np.where(ci >= 100, np.where(ei, intra, 0).sum(0, dtype=np.float64) / np.maximum(ci, 1), np.nan)

    # Characteristics as of the close of day d.
    chars = np.full((len(CHARS), n, t), np.nan, np.float32)
    with np.errstate(all='ignore'):
        chars[0, :, 252:] = close[:, 231:t - 21] / close[:, :t - 252] - 1          # 12-1 momentum
        chars[1, :, 21:] = close[:, 21:] / close[:, :t - 21] - 1                    # 1-month reversal
        s1, k1 = rolling_sum(ret, 60)
        s2, _ = rolling_sum(ret * ret, 60)
        var = (s2 - s1 * s1 / np.maximum(k1, 1)) / np.maximum(k1 - 1, 1)
        chars[2] = np.where(k1 >= 40, np.sqrt(np.maximum(var, 0)), np.nan)          # 60-day volatility
        del s1, s2, var
        Mf = np.where(np.isfinite(M), M, np.nan)
        pair = np.isfinite(ret) & np.isfinite(Mf)[None, :]
        r0 = np.where(pair, ret, np.nan)
        m0 = np.where(pair, Mf[None, :], np.nan)
        sr, kk = rolling_sum(r0, 252)
        sm, _ = rolling_sum(m0, 252)
        srm, _ = rolling_sum(r0 * m0, 252)
        smm, _ = rolling_sum(m0 * m0, 252)
        kf = np.maximum(kk, 1)
        cov = srm - sr * sm / kf
        vm = smm - sm * sm / kf
        chars[3] = np.where((kk >= 200) & (vm > 0), cov / np.where(vm > 0, vm, 1), np.nan)  # market beta
        del sr, sm, srm, smm, cov, vm, r0, m0, pair
        chars[4] = np.log(meddv)                                                    # dollar volume (size stand-in)
        chars[5] = np.log(close)                                                    # price level
    print(f'  characteristics {time.time() - t0:.0f}s', flush=True)

    # Long-short tercile factors: membership on day d from characteristics as of d-1 among the eligible stocks.
    F = np.full((len(CHARS), t), np.nan)
    F_intra = np.full((len(CHARS), t), np.nan)
    for k in range(len(CHARS)):
        c = np.full((n, t), np.nan, np.float32)
        c[:, 1:] = chars[k][:, :-1]
        c[~elig] = np.nan
        q = np.nanquantile(c, [1 / 3, 2 / 3], axis=0)
        top = c >= q[1][None, :]
        bot = c <= q[0][None, :]
        for arr, dest in ((ret, F), (intra, F_intra)):
            ft = top & np.isfinite(arr)
            fb = bot & np.isfinite(arr)
            nt, nb = ft.sum(0), fb.sum(0)
            mt = np.where(ft, arr, 0).sum(0, dtype=np.float64) / np.maximum(nt, 1)
            mb = np.where(fb, arr, 0).sum(0, dtype=np.float64) / np.maximum(nb, 1)
            dest[k] = np.where((nt >= 30) & (nb >= 30), mt - mb, np.nan)
        del c, top, bot
        print(f'  factor {CHARS[k]} {time.time() - t0:.0f}s', flush=True)

    # Label ETFs (returns aligned to the export's days).
    sym = {s['ticker']: s['id'] for s in store['syms'] if s['kind'] == 'etf'}
    have = [x for x in LABEL_ETFS if x in sym]
    ec = c2eval.store_panel(store, [sym[x] for x in have], days, 'close')
    with np.errstate(all='ignore'):
        etf_ret = np.full(ec.shape, np.nan, np.float32)
        etf_ret[:, 1:] = np.where(np.isfinite(ec[:, 1:]) & np.isfinite(ec[:, :-1]) & (ec[:, :-1] > 0), ec[:, 1:] / ec[:, :-1] - 1, np.nan)

    np.save(out / 'ret.npy', ret)
    np.save(out / 'intra.npy', intra)
    np.save(out / 'volratio.npy', volratio)
    np.save(out / 'meddv.npy', meddv)
    np.save(out / 'chars.npy', chars)
    np.savez(out / 'factors.npz', M=M, M_intra=M_intra, F=F, F_intra=F_intra, chars=np.array(CHARS), days=days,
             elig_count=cnt, etf_ret=etf_ret, etf_tickers=np.array(have), tickers=E['tickers'], ids=ids)
    print(f'done {time.time() - t0:.0f}s; market days {np.isfinite(M).sum()}, factor days {np.isfinite(F).all(0).sum()}, '
          f'intraday factor days {np.isfinite(F_intra).all(0).sum()}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
