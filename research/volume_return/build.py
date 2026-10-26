"""Panels and data checks (plan: "Data and timing", "V, detrended volume", "The proxy and the thirds", "Universes",
"Checks before any result").

    build.py old|new|fwd  -> BIG/<tag>/R.npy, V.npy, cal.npz; out/build_<tag>.txt

R: simple daily returns of the export's closes (|R| > 0.5 missing). V: ln share volume (dollar volume / close) minus
its mean over the valid days of t-200..t-1 (>= 160), broken prints (< 1% of the window's median) missing and unused.
On each twins calendar day t0 (export days 252 + 21k), over the 252 days through t0: the universes U_W ($5, $250k,
>= 90% closes and >= 90% positive volume), U_W1 ($1), U_J ($5, $5M, >= 90% closes), the 500 largest of U_J by 63-day
median dollar volume; minus c2eval.excluded. G = the percentile of the 252-day median dollar volume in each universe.
Reported proxies: Amihud, idiosyncratic volatility, Corwin-Schultz (store highs and lows, from 2008).
"""
import sys
import time

sys.path.insert(0, '/home/bhoult/programming/correlation/research/volume_return')
import numpy as np  # noqa: E402

import common as C  # noqa: E402

c2eval = C.c2eval


def main(tag):
    t_start = time.time()
    rep = C.OUT / f'build_{tag}.txt'
    rep.write_text('')
    E = C.export(tag)
    days, tickers = E['days'], E['tickers']
    N, T = E['close'].shape
    d = C.tagdir(tag)
    years = np.array([int(c2eval.iso(x)[:4]) for x in days])
    C.log(rep, f'build {tag}: {N} stocks x {T} days, {c2eval.iso(days[0])}..{c2eval.iso(days[-1])}, '
               f'fingerprint {E["meta"]["fingerprint"]}')

    # ---- returns and volume
    R = C.simple_returns(E['close'])
    np.save(d / 'R.npy', R)
    lnsv = C.share_volume(E['close'], E['dollar_volume'])
    V, brk = C.detrended_volume(lnsv)
    np.save(d / 'V.npy', V)
    np.save(d / 'broken.npy', brk)
    C.log(rep, f'returns and V built ({time.time() - t_start:.0f} s)')

    # ---- calendar: universes, G, proxies
    tw = c2eval.twins(E, cache=(tag != 'fwd'))
    excl = c2eval.excluded(E, tw)
    rebal = tw['rebal']
    K = len(rebal)
    m = C.market_simple(tag)
    close = E['close']
    dv = E['dollar_volume']
    out = {k: np.full((K, N), np.nan, np.float32) for k in
           ('medC', 'medDV', 'medDV63', 'fclose', 'fvol', 'amihud', 'idio', 'GW', 'GW1', 'GJ', 'GT')}
    for k in ('UW', 'UW1', 'UJ', 'TOP'):
        out[k] = np.zeros((K, N), bool)
    for k, t0 in enumerate(rebal):
        a = t0 - C.CAL_W + 1
        c = np.asarray(close[:, a:t0 + 1], np.float64)
        x = np.asarray(dv[:, a:t0 + 1], np.float64)
        sv_ok = np.isfinite(lnsv[:, a:t0 + 1])
        c_ok = np.isfinite(c) & (c > 0)
        x[~(np.isfinite(x) & (x > 0))] = np.nan
        c[~c_ok] = np.nan
        with np.errstate(all='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                medC = np.nanmedian(c, 1)
                medDV = np.nanmedian(x, 1)
                medDV63 = np.nanmedian(x[:, -C.TOP_W:], 1)
        fc = c_ok.mean(1)
        fv = sv_ok.mean(1)
        r = np.asarray(R[:, a:t0 + 1], np.float64)
        # Amihud: mean |R| / dollar volume (x 1e6) over valid days, >= 200
        ok = np.isfinite(r) & np.isfinite(x)
        n_ok = ok.sum(1)
        ami = np.where(ok, np.abs(np.nan_to_num(r)) / np.where(ok, x, 1.0), 0.0).sum(1) / np.maximum(n_ok, 1) * 1e6
        ami[n_ok < 200] = np.nan
        # idiosyncratic volatility: sd of R - beta * market over valid days (>= 200), beta from the same window
        mm = m[a:t0 + 1][None, :]
        ok2 = np.isfinite(r) & np.isfinite(mm)
        n2 = ok2.sum(1)
        rz = np.where(ok2, r, 0.0)
        mz = np.where(ok2, mm, 0.0)
        with np.errstate(all='ignore'):
            rb = rz.sum(1) / n2
            mb = mz.sum(1) / n2
            rc = np.where(ok2, r - rb[:, None], 0.0)
            mc = np.where(ok2, mm - mb[:, None], 0.0)
            beta = (rc * mc).sum(1) / (mc * mc).sum(1)
            res = np.where(ok2, rc - beta[:, None] * mc, 0.0)
            idio = np.sqrt((res * res).sum(1) / n2)
        idio[n2 < 200] = np.nan
        base = (fc >= C.CAL_FRAC) & np.isfinite(medC) & np.isfinite(medDV) & ~excl[k] & np.isfinite(close[:, t0])
        uw = base & (fv >= C.CAL_FRAC) & (medC >= C.FLOOR_PRICE) & (medDV >= C.FLOOR_DV_WIDE)
        uw1 = base & (fv >= C.CAL_FRAC) & (medC >= C.FLOOR_PRICE_LOW) & (medDV >= C.FLOOR_DV_WIDE)
        uj = base & (medC >= C.FLOOR_PRICE) & (medDV >= C.FLOOR_DV_JUDGE)
        top = np.zeros(N, bool)
        cand = np.nonzero(uj & np.isfinite(medDV63))[0]
        if len(cand):
            order = cand[np.argsort(-medDV63[cand], kind='mergesort')]
            top[order[:C.TOP]] = True
        for name, u, gname in (('UW', uw, 'GW'), ('UW1', uw1, 'GW1'), ('UJ', uj, 'GJ'), ('TOP', top, 'GT')):
            out[name][k] = u
            idx = np.nonzero(u)[0]
            out[gname][k, idx] = C.pct_rank(medDV[idx])
        out['medC'][k], out['medDV'][k], out['medDV63'][k] = medC, medDV, medDV63
        out['fclose'][k], out['fvol'][k], out['amihud'][k], out['idio'][k] = fc, fv, ami, idio
    C.log(rep, f'calendar built: {K} calendar days ({time.time() - t_start:.0f} s)')
    np.savez(d / 'cal.npz', rebal=rebal, excl=excl, **out)

    # ---- data checks
    C.log(rep, '', '== Universe sizes (median over the calendar days in each year) ==',
          'year | U_W ($5, $250k) | U_W1 ($1) | U_J ($5, $5M) | 500 largest | U_W: low-third median DV ($M) | '
          'U_W high-third median DV ($M) | U_J low-third median DV ($M)')
    ry = np.array([years[t] for t in rebal])
    for y in range(int(ry.min()), int(ry.max()) + 1):
        ks = np.nonzero(ry == y)[0]
        if not len(ks):
            continue
        sizes = [int(np.median(out[n][ks].sum(1))) for n in ('UW', 'UW1', 'UJ', 'TOP')]
        k = ks[0]
        lo = np.nanmedian(out['medDV'][k][out['UW'][k] & (out['GW'][k] < 1 / 3)]) / 1e6 if out['UW'][k].any() else np.nan
        hi = np.nanmedian(out['medDV'][k][out['UW'][k] & (out['GW'][k] >= 2 / 3)]) / 1e6 if out['UW'][k].any() else np.nan
        lj = np.nanmedian(out['medDV'][k][out['UJ'][k] & (out['GJ'][k] < 1 / 3)]) / 1e6 if out['UJ'][k].any() else np.nan
        C.log(rep, f'{y} | {sizes[0]} | {sizes[1]} | {sizes[2]} | {sizes[3]} | {lo:.2f} | {hi:.1f} | {lj:.1f}')

    C.log(rep, '', '== Broken volume prints (sv < 1% of the 200-day median): all stock-days, and U_W members ==')
    member = np.zeros(N, bool)
    for y in np.unique(years):
        cols = np.nonzero(years == y)[0]
        nb_all = int(brk[:, cols].sum())
        nb_uw = 0
        for t in cols[::1]:
            k = C.cal_index(rebal, t)
            if k >= 0:
                nb_uw += int((brk[:, t] & out['UW'][k]).sum())
        C.log(rep, f'{y}: {nb_all} stock-days ({nb_uw} in U_W)')
    top_days = np.argsort(-brk.sum(0))[:5]
    C.log(rep, 'days with the most broken prints: ' + ', '.join(f'{c2eval.iso(days[t])} ({int(brk[:, t].sum())})'
                                                              for t in top_days))
    nclip = int((np.abs(V) >= C.V_CLIP).sum())
    C.log(rep, f'V clipped at +-{C.V_CLIP}: {nclip} of {int(np.isfinite(V).sum())} finite stock-days')

    C.log(rep, '', '== Coverage in U_W: share of member stock-days with finite R and V, by year ==')
    for y in np.unique(years):
        cols = np.nonzero(years == y)[0]
        tot = okc = 0
        for t in cols:
            k = C.cal_index(rebal, t)
            if k < 0:
                continue
            u = out['UW'][k]
            tot += int(u.sum())
            okc += int((u & np.isfinite(R[:, t]) & np.isfinite(V[:, t])).sum())
        if tot:
            C.log(rep, f'{y}: {okc / tot:.3f} of {tot}')

    t11 = c2eval.day_index(days, '2011-02-17')
    if c2eval.iso(days[t11]) == '2011-02-17':
        C.log(rep, '', f'2011-02-17: {int(np.isfinite(np.asarray(close[:, t11])).sum())} closes '
                       f'(the day before: {int(np.isfinite(np.asarray(close[:, t11 - 1])).sum())})')

    C.log(rep, '', '== Splits: share volume (dollar volume / close), 20-day medians after / before ==',
          'ticker | split day | ratio | after / before | |ln| < 0.5 ln(ratio)?')
    row = {t: j for j, t in enumerate(tickers)}
    fails = 0
    for tk, day, ratio in C.SPLITS:
        j = row.get(tk)
        t = c2eval.day_index(days, day)
        if j is None or t >= T or c2eval.iso(days[t]) != day:
            C.log(rep, f'{tk} | {day} | {ratio} | not in this export')
            continue
        b = np.nanmedian(np.exp(lnsv[j, t - 20:t]))
        a_ = np.nanmedian(np.exp(lnsv[j, t:t + 20]))
        ok = abs(np.log(a_ / b)) < 0.5 * np.log(ratio)
        fails += int(not ok)
        C.log(rep, f'{tk} | {day} | {ratio} | {a_ / b:.2f} | {"ok" if ok else "FAIL"}')
    C.log(rep, f'{len(C.SPLITS) - fails} of {len(C.SPLITS)} checked splits show no step in share volume')

    # ---- Corwin-Schultz spread from the store's highs and lows, from 2008 (close-location's bad-print rules)
    del lnsv
    S = c2eval.load_store(C.STORE[tag])
    H = c2eval.store_panel(S, E['ids'], days, 'high')
    L = c2eval.store_panel(S, E['ids'], days, 'low')
    Cs = c2eval.store_panel(S, E['ids'], days, 'close')
    del S
    first08 = c2eval.day_index(days, '2008-01-01')
    with np.errstate(all='ignore'):
        good = (np.isfinite(H) & np.isfinite(L) & np.isfinite(Cs) & (H > 0) & (L > 0) & (Cs > 0)
                & ((H - L) / Cs <= 0.5) & (H >= Cs) & (L <= Cs) & (H > L))
    del Cs
    good[:, :first08] = False
    H[~good] = np.nan
    L[~good] = np.nan
    spr = np.full((N, T), np.nan, np.float32)    # spr[:, t]: the estimate from the two days (t-1, t)
    k3 = 3 - 2 * np.sqrt(2)
    for a in range(0, N, 1000):
        h = H[a:a + 1000].astype(np.float64)
        lo = L[a:a + 1000].astype(np.float64)
        with np.errstate(all='ignore'):
            lh2 = np.log(h / lo) ** 2
            beta_cs = lh2[:, 1:] + lh2[:, :-1]
            gam = np.log(np.maximum(h[:, 1:], h[:, :-1]) / np.minimum(lo[:, 1:], lo[:, :-1])) ** 2
            alpha = (np.sqrt(2 * beta_cs) - np.sqrt(beta_cs)) / k3 - np.sqrt(gam / k3)
            sp = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        sp = np.where(np.isfinite(sp), np.maximum(sp, 0.0), np.nan)
        spr[a:a + 1000, 1:] = sp
    del H, L, good
    cs = np.full((K, N), np.nan, np.float32)
    for k, t0 in enumerate(rebal):
        w = spr[:, t0 - C.CAL_W + 1:t0 + 1]
        n = np.isfinite(w).sum(1)
        with np.errstate(all='ignore'):
            v = np.nansum(w, 1) / n
        v[n < 126] = np.nan
        cs[k] = v
    out['cs'] = cs
    del spr
    np.savez(d / 'cal.npz', rebal=rebal, excl=excl, **out)
    C.log(rep, f'Corwin-Schultz built ({time.time() - t_start:.0f} s)')

    if tag == 'new':
        C.log(rep, '', '== Overlap: the new export against the old over the old export\'s days ==')
        Eo = C.export('old')
        To = len(Eo['days'])
        assert np.array_equal(Eo['days'], days[:To]), 'day calendars differ'
        rown = {t: j for j, t in enumerate(tickers)}
        common = [(jo, rown[t]) for jo, t in enumerate(Eo['tickers']) if t in rown]
        jo = np.array([a for a, _ in common])
        jn = np.array([b for _, b in common])
        bad_c = bad_d = 0
        maxrel = 0.0
        for a in range(0, len(jo), 500):
            co = np.asarray(Eo['close'][jo[a:a + 500]], np.float64)
            cn = np.asarray(close[jn[a:a + 500], :To], np.float64)
            do = np.asarray(Eo['dollar_volume'][jo[a:a + 500]], np.float64)
            dn = np.asarray(dv[jn[a:a + 500], :To], np.float64)
            same_nan_c = np.isnan(co) == np.isnan(cn)
            same_nan_d = np.isnan(do) == np.isnan(dn)
            with np.errstate(all='ignore'):
                rc = np.abs(cn / co - 1)
                rd = np.abs(dn / do - 1)
            bad_c += int((~same_nan_c).sum() + (np.nan_to_num(rc) > 1e-6).sum())
            bad_d += int((~same_nan_d).sum() + (np.nan_to_num(rd) > 1e-6).sum())
            maxrel = max(maxrel, float(np.nanmax(rc)) if np.isfinite(rc).any() else 0.0)
        C.log(rep, f'{len(common)} tickers in both of {len(Eo["tickers"])} old; stock-days differing (NaN pattern or '
                   f'> 1e-6 relative): closes {bad_c}, dollar volume {bad_d}; largest close difference {maxrel:.2e}')
    C.log(rep, '', f'done in {time.time() - t_start:.0f} s')


if __name__ == '__main__':
    main(sys.argv[1])
