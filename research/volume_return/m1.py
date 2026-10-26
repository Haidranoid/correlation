"""M1 (primary) and its reported items 8-14 (plan: "M1 (primary)", "Reported (not pass/fail)").

    m1.py old   -> out/m1_old.txt, out/m1_old.json; BIG/old/m1_series.npz; BIG/old/scores_*.npy (the judge's inputs)
    m1.py new   -> the same for the new export (unseen.py reads the 16 unseen months from it)
    m1.py fwd   -> forward.py

U_J (judge-like liquid set); thirds by the percentile of the 252-day median dollar volume. Features at the close of t:
R(t), V(t), winsorized at the 1st/99th percentiles within each third on each day. Target R(t+2) (bought at the close of
t+1, the judge's delay=1). Daily OLS within each third: full [1, R, V, V*R], base [1, R, V]. Coefficients on calendar
day t0: means of the daily estimates over origins s with s + 2 <= t0 (asserted), expanding, >= 252 origins.
Scores (no intercepts): P_full = b R + v V + c V R; P_base = b' R + v' V. Books: daily, top and bottom 10% of U_J,
equal weight, bought at the close of t+1, sold at the close of t+2; 5 bps a side on turnover. M1 = mean of
D = book(P_full) - book(P_base), gross, 2008-01-02..2019-12-31; t from a circular 21-day block bootstrap.
"""
import json
import sys
import time

sys.path.insert(0, '/home/bhoult/programming/correlation/research/volume_return')
import numpy as np  # noqa: E402

import common as C  # noqa: E402

c2eval = C.c2eval


def wins_in(x, idx):
    """Winsorize x[idx] at its 1st/99th percentiles (in place on a copy); returns the clipped values."""
    v = x[idx]
    lo, hi = np.percentile(v, [C.WIN_LO, C.WIN_HI])
    return np.clip(v, lo, hi)


def features(R, V, cal, uname, gname, t, rw=None, vw=None):
    """Members of the universe on day t with finite features; their third and winsorized R and V (within third)."""
    k = C.cal_index(cal['rebal'], t)
    if k < 0:
        return None
    u = cal[uname][k]
    r = np.asarray(R[:, t], np.float64) if rw is None else rw
    v = np.asarray(V[:, t], np.float64) if vw is None else vw
    g = C.thirds_of(cal[gname][k].astype(np.float64))
    ok = u & np.isfinite(r) & np.isfinite(v) & (g >= 0)
    rows = np.nonzero(ok)[0]
    if len(rows) == 0:
        return None
    gr = g[rows]
    rr = np.empty(len(rows))
    vv = np.empty(len(rows))
    for q in (0, 1, 2):
        j = np.nonzero(gr == q)[0]
        if len(j) >= 3:
            rr[j] = np.clip(r[rows[j]], *np.percentile(r[rows[j]], [C.WIN_LO, C.WIN_HI]))
            vv[j] = np.clip(v[rows[j]], *np.percentile(v[rows[j]], [C.WIN_LO, C.WIN_HI]))
        else:
            rr[j], vv[j] = r[rows[j]], v[rows[j]]
    return {'k': k, 'rows': rows, 'g': gr, 'r': rr, 'v': vv}


def daily_fits(F, Y, T, first, last):
    """Daily OLS within each third: full [1, r, v, v r] and base [1, r, v] of Y[:, t] (target panel) on the features."""
    cf = np.full((T, 3, 4), np.nan)
    cb = np.full((T, 3, 3), np.nan)
    nn = np.zeros((T, 3), int)
    for t in range(first, last + 1):
        f = F(t)
        if f is None:
            continue
        y = Y(t)[f['rows']]
        for q in (0, 1, 2):
            j = np.nonzero((f['g'] == q) & np.isfinite(y))[0]
            nn[t, q] = len(j)
            if len(j) < C.M1_MIN_GROUP:
                continue
            r, v = f['r'][j], f['v'][j]
            one = np.ones(len(j))
            cf[t, q] = C.ols(np.column_stack([one, r, v, v * r]), y[j])
            cb[t, q] = C.ols(np.column_stack([one, r, v]), y[j])
    return cf, cb, nn


def calendar_coefs(daily, rebal, lag):
    """Expanding means of the daily coefficients over origins s with s + lag <= t0, per calendar day and third."""
    K = len(rebal)
    T, G, P = daily.shape
    out = np.full((K, G, P), np.nan)
    ok = np.all(np.isfinite(daily), axis=2)                     # [T, G]
    cs = np.concatenate([np.zeros((1, G, P)), np.cumsum(np.where(ok[..., None], daily, 0.0), 0)])
    cn = np.concatenate([np.zeros((1, G)), np.cumsum(ok, 0)])
    for k, t0 in enumerate(rebal):
        last_origin = t0 - lag                                  # s + lag <= t0
        if last_origin < 0:
            continue
        n = cn[last_origin + 1]
        assert last_origin + lag <= t0                          # the eligibility rule
        m = cs[last_origin + 1] / np.maximum(n, 1)[:, None]
        m[n < C.M1_MIN_ORIGINS] = np.nan
        out[k] = m
    return out


def scores_from(F, coefs, T, first, N, full=True):
    """Score panel [N, T] float32: P_full (or P_base) from the coefficients in force on each day."""
    S = np.full((N, T), np.nan, np.float32)
    for t in range(first, T):
        f = F(t)
        if f is None:
            continue
        c = coefs[f['k']]                                      # [3, P]
        cg = c[f['g']]
        if full:
            s = cg[:, 1] * f['r'] + cg[:, 2] * f['v'] + cg[:, 3] * f['v'] * f['r']
        else:
            s = cg[:, 1] * f['r'] + cg[:, 2] * f['v']
        S[f['rows'], t] = s
    return S


def hold_returns(R, t, delay, H):
    """Per-stock buy-and-hold return from the close of t + delay over the next H days (missing days count as 0;
    NaN if fewer than H - 1 of H valid when H > 1, or the one day missing when H = 1)."""
    seg = np.asarray(R[:, t + delay + 1:t + delay + 1 + H], np.float64)
    valid = np.isfinite(seg).sum(1)
    hr = np.prod(1 + np.nan_to_num(seg), 1) - 1
    need = H if H == 1 else H - 1
    hr[valid < need] = np.nan
    return hr


def book(S, R, decisions, delay, H, cells=None):
    """Decile long-short book on score panel S over decision days; returns per-decision arrays."""
    N, T = S.shape
    out = {k: np.full(len(decisions), np.nan) for k in ('gross', 'to_l', 'to_s', 'long', 'short', 'n')}
    legs = []
    wl_old = np.zeros(N)
    ws_old = np.zeros(N)
    prev_hr = None
    for j, t in enumerate(decisions):
        s = np.asarray(S[:, t], np.float64)
        ok = np.isfinite(s) if cells is None else (np.isfinite(s) & cells(t))
        rows = np.nonzero(ok)[0]
        n = len(rows)
        hr = hold_returns(R, t, delay, H)
        # drift the previous book over its holding period
        if prev_hr is not None:
            g = 1 + np.nan_to_num(prev_hr)
            wl_d = wl_old * g
            ws_d = ws_old * g
            wl_d = wl_d / wl_d.sum() if wl_d.sum() > 0 else wl_d
            ws_d = ws_d / ws_d.sum() if ws_d.sum() > 0 else ws_d
        else:
            wl_d = np.zeros(N)
            ws_d = np.zeros(N)
        wl = np.zeros(N)
        ws = np.zeros(N)
        if n >= 20:
            m = int(round(0.1 * n))
            order = rows[np.argsort(-s[rows], kind='mergesort')]   # ties by stock row
            L, Sh = order[:m], order[::-1][:m]
            wl[L] = 1.0 / m
            ws[Sh] = 1.0 / m
            lr, sr = hr[L], hr[Sh]
            if np.isfinite(lr).any() and np.isfinite(sr).any():
                out['long'][j] = np.nanmean(lr)
                out['short'][j] = np.nanmean(sr)
                out['gross'][j] = out['long'][j] - out['short'][j]
            legs.append((L, Sh))
        else:
            legs.append((np.array([], int), np.array([], int)))
        out['to_l'][j] = np.abs(wl - wl_d).sum()
        out['to_s'][j] = np.abs(ws - ws_d).sum()
        out['n'][j] = n
        wl_old, ws_old, prev_hr = wl, ws, hr
    out['legs'] = legs
    return out


def net(b, cost):
    return b['gross'] - cost * (b['to_l'] + b['to_s'])


def overlap(b1, b2):
    sh = []
    for (l1, s1), (l2, s2) in zip(b1['legs'], b2['legs']):
        if len(l1) and len(l2):
            sh.append(0.5 * (len(np.intersect1d(l1, l2)) / len(l1) + len(np.intersect1d(s1, s2)) / len(s1)))
    return float(np.mean(sh)) if sh else float('nan')


def series_stats(x, draws=C.DRAWS, seed=C.SEED):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return {'est': float(np.mean(x)) if len(x) else float('nan'), 'n': len(x)}
    b = C.circular_block_means(x, draws=draws, seed=seed)
    s = C.boot_summary(float(x.mean()), b)
    s['nw_t'] = C.nw_t(x)
    s['n'] = len(x)
    return s


def fmt(s, scale=1e4, unit=' bps'):
    if 'sd' not in s:
        return f"{s['est'] * scale:+.2f}{unit} (n {s['n']})"
    return (f"{s['est'] * scale:+.2f}{unit} a day (90% {s['lo90'] * scale:+.2f}..{s['hi90'] * scale:+.2f}), bootstrap t "
            f"{s['t']:+.2f}, p {s['p']:.4f} (percentile p {s['p_boot']:.4f}), Newey-West t {s['nw_t']:+.2f}, "
            f"{s['n']} days")


def run_model(R, V, cal, uname, gname, Ylag, lag, delay, H, decisions, s0, weekly=False, Rw=None, Vw=None, Rfeat=None):
    """Fit, score and book one model family (full and base); returns fits, scores and books."""
    N, T = R.shape
    rebal = cal['rebal']
    first = int(rebal[0])
    if weekly:
        F = lambda t: features(R, V, cal, uname, gname, t, rw=Rw[:, t], vw=Vw[:, t])  # noqa: E731
    elif Rfeat is not None:     # the signal reads Rfeat (dividend check: ex-dates missing); targets and books read R
        F = lambda t: features(Rfeat, V, cal, uname, gname, t)  # noqa: E731
    else:
        F = lambda t: features(R, V, cal, uname, gname, t)  # noqa: E731
    cf, cb, nn = daily_fits(F, Ylag, T, first, T - 1 - lag)
    kf = calendar_coefs(cf, rebal, lag)
    kb = calendar_coefs(cb, rebal, lag)
    Pf = scores_from(F, kf, T, s0, N, True)
    Pb = scores_from(F, kb, T, s0, N, False)
    cells = np.isfinite(Pf) & np.isfinite(Pb)
    Pf[~cells] = np.nan
    Pb[~cells] = np.nan
    bf = book(Pf, R, decisions, delay, H)
    bb = book(Pb, R, decisions, delay, H)
    fo = []
    for q in range(3):
        okq = np.all(np.isfinite(cf[:, q]), 1)
        fo.append(int(np.nonzero(okq)[0][0]) if okq.any() else -1)
    return {'cf': cf, 'cb': cb, 'nn': nn, 'kf': kf, 'kb': kb, 'Pf': Pf, 'Pb': Pb, 'bf': bf, 'bb': bb,
            'first_origin': fo}


def main(tag):
    t_start = time.time()
    rep = C.OUT / f'm1_{tag}.txt'
    rep.write_text('')
    E = C.export(tag, arrays=())
    days = E['days']
    R, V, cal = C.load_panels(tag)
    R = np.array(R)          # in memory: the daily loops read columns
    V = np.array(V)
    N, T = R.shape
    rebal = cal['rebal']
    first = int(rebal[0])
    s0 = c2eval.day_index(days, C.SCORE_START)
    big = C.tagdir(tag)
    J = {'tag': tag}
    yrs = np.array([int(c2eval.iso(x)[:4]) for x in days])

    def per(lbl):
        if lbl == 'dev':
            return C.day_range(days, *C.DEV)
        if lbl == 'check':
            return C.day_range(days, *C.CHECK)
        if lbl == 'unseen':
            return c2eval.day_index(days, C.UNSEEN_START), T - 1
        return c2eval.day_index(days, C.os.environ.get('C2_FWD_START', '2026-10-20')), T - 1

    periods = ['dev', 'check'] if tag == 'old' else (['unseen'] if tag == 'new' else ['fwd'])

    Y2 = lambda t: np.asarray(R[:, t + 2], np.float64)  # noqa: E731
    Y1 = lambda t: np.asarray(R[:, t + 1], np.float64)  # noqa: E731
    dec = np.arange(s0, T - 2)                               # daily decisions, target t + 2 inside the data
    dsel = {p: (dec >= per(p)[0]) & (dec <= per(p)[1]) & (dec + 2 <= per(p)[1] if p == 'check' else True)
            for p in periods}

    # ---- the primary
    M = run_model(R, V, cal, 'UJ', 'GJ', Y2, 2, 1, 1, dec, s0)
    C.log(rep, f'fits done ({time.time() - t_start:.0f} s); first origin with >= {C.M1_MIN_GROUP} stocks per third: '
               + ', '.join(c2eval.iso(days[x]) if x >= 0 else 'none' for x in M['first_origin']))
    k08 = C.cal_index(rebal, s0)
    n_orig = [int(np.sum(np.all(np.isfinite(M['cf'][:rebal[k08] - 2 + 1, q]), 1))) for q in range(3)]
    C.log(rep, f'first fit in use on {C.SCORE_START}: calendar day {c2eval.iso(days[rebal[k08]])}, '
               f'eligible origins per third {n_orig}')
    np.save(big / 'scores_full.npy', M['Pf'])
    np.save(big / 'scores_base.npy', M['Pb'])
    D = M['bf']['gross'] - M['bb']['gross']
    Dn = net(M['bf'], C.COST) - net(M['bb'], C.COST)
    # plain reversal on the same cells
    negR = np.where(np.isfinite(M['Pf']), -R, np.nan).astype(np.float32)
    np.save(big / 'scores_negR.npy', negR)
    brv = book(negR, R, dec, 1, 1)
    C.log(rep, f'books done ({time.time() - t_start:.0f} s)')
    del negR
    M['Pf'] = M['Pb'] = None

    for p in periods:
        sel = dsel[p]
        C.log(rep, '', f'== M1 ({"primary, " if p == "dev" else ""}{p}): D = book(P_full) - book(P_base), '
                       f'{c2eval.iso(days[dec[sel][0]])}..{c2eval.iso(days[dec[sel][-1]])} decisions ==')
        sg = series_stats(D[sel])
        sn = series_stats(Dn[sel], seed=C.SEED + 1)
        J[f'M1_{p}'] = {'gross': sg, 'net5': sn}
        C.log(rep, f'gross D: {fmt(sg)}', f'net D at 5 bps a side: {fmt(sn)}')
        C.log(rep, 'book | gross bps/day (t) | net 5 bps | net 20 bps | turnover a side a day | break-even bps a side '
                   '| long vs short leg (bps)')
        for nm, b in (('P_full', M['bf']), ('P_base', M['bb']), ('plain reversal -R', brv)):
            g = b['gross'][sel]
            to = (b['to_l'] + b['to_s'])[sel]
            s1 = series_stats(g, draws=2000)
            n5 = np.nanmean(net(b, 5e-4)[sel])
            n20 = np.nanmean(net(b, 20e-4)[sel])
            be = np.nanmean(g) / np.nanmean(to) * 1e4
            C.log(rep, f"{nm} | {np.nanmean(g) * 1e4:+.2f} ({s1.get('t', np.nan):+.2f}) | {n5 * 1e4:+.2f} | "
                       f"{n20 * 1e4:+.2f} | {np.nanmean(to) / 2:.2f} | {be:.2f} | "
                       f"{np.nanmean(b['long'][sel]) * 1e4:+.2f} / {np.nanmean(b['short'][sel]) * 1e4:+.2f}")
            J[f'book_{nm}_{p}'] = {'gross': float(np.nanmean(g)), 'gross_t': s1.get('t'), 'net5': float(n5),
                                   'net20': float(n20), 'turnover_side': float(np.nanmean(to) / 2), 'breakeven': float(be)}
        for nm, x in (('P_full - plain reversal', M['bf']['gross'] - brv['gross']),
                      ('P_base - plain reversal', M['bb']['gross'] - brv['gross'])):
            s2 = series_stats(x[sel], draws=2000, seed=C.SEED + 2)
            C.log(rep, f'{nm}, gross: {fmt(s2)}')
            J[f'{nm}_{p}'] = s2
        C.log(rep, f'names shared by the P_full and P_base books: {overlap({"legs": [M["bf"]["legs"][i] for i in np.nonzero(sel)[0]]}, {"legs": [M["bb"]["legs"][i] for i in np.nonzero(sel)[0]]}):.3f}; '
                   f'with plain reversal: {overlap({"legs": [M["bf"]["legs"][i] for i in np.nonzero(sel)[0]]}, {"legs": [brv["legs"][i] for i in np.nonzero(sel)[0]]}):.3f}')
        C.log(rep, f'stocks ranked a day: median {np.nanmedian(M["bf"]["n"][sel]):.0f}')
        if p in ('dev', 'check'):
            C.log(rep, 'by year: year | days | D gross bps/day (t) | D net | P_full gross | P_base gross | -R gross')
            for y in np.unique(yrs[dec[sel]]):
                yy = sel & (yrs[dec] == y)
                d_ = D[yy]
                tt = np.nanmean(d_) / (np.nanstd(d_, ddof=1) / np.sqrt(np.isfinite(d_).sum()))
                C.log(rep, f'{y} | {int(yy.sum())} | {np.nanmean(d_) * 1e4:+.2f} ({tt:+.2f}) | '
                           f'{np.nanmean(Dn[yy]) * 1e4:+.2f} | {np.nanmean(M["bf"]["gross"][yy]) * 1e4:+.2f} | '
                           f'{np.nanmean(M["bb"]["gross"][yy]) * 1e4:+.2f} | {np.nanmean(brv["gross"][yy]) * 1e4:+.2f}')

    if tag == 'fwd':            # forward.py needs only the primary's scores (saved above)
        json.dump(J, open(C.OUT / f'm1_{tag}.json', 'w'), indent=1, default=lambda o: None)
        return

    # ---- dividends (coordinator's note, decided before any result): check period and unseen months only
    if tag in ('old', 'new'):
        import pubdata
        fix = np.asarray(pubdata.div_fix(tag), np.float64)
        Rfix = ((1 + R.astype(np.float64)) * np.exp(fix) - 1).astype(np.float32)
        del fix
        Rfix[np.abs(Rfix) > C.R_BAD] = np.nan
        Yf = lambda t: np.asarray(Rfix[:, t + 2], np.float64)  # noqa: E731
        MF = run_model(Rfix, V, cal, 'UJ', 'GJ', Yf, 2, 1, 1, dec, s0)
        DF = MF['bf']['gross'] - MF['bb']['gross']
        del MF, Rfix
        import r1 as _r1
        exm = _r1.exdate_mask(tag, R.shape)
        Rx = np.where(exm, R, np.nan).astype(np.float32)
        del exm
        MX = run_model(R, V, cal, 'UJ', 'GJ', Y2, 2, 1, 1, dec, s0, Rfeat=Rx)
        DX = MX['bf']['gross'] - MX['bb']['gross']
        del MX, Rx
        if tag == 'old':        # the store lacks dividends before about 2011-02 too (coordinator's note)
            for lbl, sel in (('2008-2019', dsel['dev']), ('2008-2010', dsel['dev'] & (yrs[dec] <= 2010))):
                sx = series_stats(DX[sel], draws=2000, seed=C.SEED + 11)
                s_ = series_stats(D[sel], draws=2000, seed=C.SEED + 11)
                C.log(rep, '', f'== Dividends, {lbl}: M1 with ex-dates dropped from the signal ==',
                      f'ex-dates dropped: gross D {fmt(sx)}', f'as run: gross D {fmt(s_)}')
                J[f'div_exdrop_{lbl}'] = {'exdrop': sx, 'as_run': s_}
        for p in periods:
            if p == 'dev':
                continue
            sel = dsel[p]
            sf = series_stats(DF[sel], draws=2000, seed=C.SEED + 10)
            sx = series_stats(DX[sel], draws=2000, seed=C.SEED + 11)
            C.log(rep, '', f'== Dividends, {p}: M1 with div_fix applied to every return; with ex-dates dropped from '
                           f'the signal ==', f'div_fix: gross D {fmt(sf)}', f'ex-dates dropped: gross D {fmt(sx)}')
            J[f'div_{p}'] = {'fix': sf, 'exdrop': sx}

    # ---- item 10: the fitted coefficients
    C.log(rep, '', '== Item 10: coefficients in force (b on R, v on V, c on V*R), by third, first calendar day of '
                   'each year ==')
    for y in sorted(set(yrs[rebal[rebal >= s0]])):
        k = int(np.nonzero(yrs[rebal] == y)[0][0])
        c = M['kf'][k]
        C.log(rep, f'{y}: ' + '; '.join(f'third {q}: b {c[q, 1]:+.4f}, v {c[q, 2]:+.5f}, c {c[q, 3]:+.4f}' for q in range(3)))
    if tag == 'old':
        a, b = C.day_range(days, *C.DEV)
        C.log(rep, 'daily full-model estimates over 2008-2019 origins (mean, Newey-West t): third | b | v | c')
        for q in range(3):
            x = M['cf'][a:b + 1, q]
            C.log(rep, f'{q} | ' + ' | '.join(f'{np.nanmean(x[:, i]):+.5f} ({C.nw_t(x[:, i]):+.2f})' for i in (1, 2, 3)))

    # ---- item 11: the paper's own timing (delay 0, target R(t+1))
    M0 = run_model(R, V, cal, 'UJ', 'GJ', Y1, 1, 0, 1, dec, s0)
    np.save(big / 'scores_full_d0.npy', M0['Pf'])
    D0 = M0['bf']['gross'] - M0['bb']['gross']
    D0n = net(M0['bf'], C.COST) - net(M0['bb'], C.COST)
    for p in periods:
        sel = dsel[p]
        s_ = series_stats(D0[sel], draws=2000, seed=C.SEED + 3)
        C.log(rep, '', f'== Item 11, the paper\'s timing (fit on R(t+1), trade at the close of t), {p} ==',
              f'gross D: {fmt(s_)}; net D at 5 bps: {np.nanmean(D0n[sel]) * 1e4:+.2f} bps a day; '
              f'P_full book gross {np.nanmean(M0["bf"]["gross"][sel]) * 1e4:+.2f}')
        J[f'item11_{p}'] = s_
    del M0

    # ---- item 12: weekly
    Rw = np.full((N, T), np.nan, np.float32)
    Vw = np.full((N, T), np.nan, np.float32)
    for t in range(4, T):
        seg = np.asarray(R[:, t - 4:t + 1], np.float64)
        ok = np.isfinite(seg).sum(1)
        x = np.prod(1 + np.nan_to_num(seg), 1) - 1
        x[ok < 4] = np.nan
        Rw[:, t] = x
        vs = np.asarray(V[:, t - 4:t + 1], np.float64)
        okv = np.isfinite(vs).sum(1)
        with np.errstate(all='ignore'):
            y = np.nansum(vs, 1) / okv
        y[okv < 4] = np.nan
        Vw[:, t] = y
    Yw = lambda t: hold_returns(R, t, 1, 5)  # noqa: E731
    decw = np.arange(s0, T - 6, 5)
    MW = run_model(R, V, cal, 'UJ', 'GJ', Yw, 6, 1, 5, decw, s0, weekly=True, Rw=Rw, Vw=Vw)
    np.save(big / 'scores_week.npy', MW['Pf'])
    neg5 = np.where(np.isfinite(MW['Pf']), -Rw, np.nan).astype(np.float32)
    np.save(big / 'scores_neg5.npy', neg5)
    brw = book(neg5, R, decw, 1, 5)
    DW = MW['bf']['gross'] - MW['bb']['gross']
    for p in periods:
        lo, hi = per(p)
        sel = (decw >= lo) & (decw + 6 <= hi if p != 'dev' else decw <= hi)
        sw = series_stats(DW[sel], draws=2000, seed=C.SEED + 4)
        sw2 = series_stats((MW['bf']['gross'] - brw['gross'])[sel], draws=2000, seed=C.SEED + 5)
        C.log(rep, '', f'== Item 12, weekly (decisions every 5 days, held close t+1 .. close t+6), {p} ==',
              f'D = weekly P_full - its base, gross: {fmt(sw, unit=" bps a week").replace(" a day", "")}',
              f'weekly P_full - 5-day reversal, gross: {fmt(sw2, unit=" bps a week").replace(" a day", "")}',
              f'net D at 5 bps: {np.nanmean((net(MW["bf"], C.COST) - net(MW["bb"], C.COST))[sel]) * 1e4:+.2f} bps a week; '
              f'P_full book gross {np.nanmean(MW["bf"]["gross"][sel]) * 1e4:+.2f}, net 5 bps '
              f'{np.nanmean(net(MW["bf"], C.COST)[sel]) * 1e4:+.2f}; 5-day reversal gross '
              f'{np.nanmean(brw["gross"][sel]) * 1e4:+.2f}')
        J[f'item12_{p}'] = {'D': sw, 'vs_rev5': sw2}
    del MW, Rw, Vw

    # ---- item 14: the 500 largest
    MT = run_model(R, V, cal, 'TOP', 'GT', Y2, 2, 1, 1, dec, s0)
    DT = MT['bf']['gross'] - MT['bb']['gross']
    for p in periods:
        sel = dsel[p]
        s_ = series_stats(DT[sel], draws=2000, seed=C.SEED + 6)
        C.log(rep, '', f'== Item 14, the 500 largest, {p} ==', f'gross D: {fmt(s_)}; net D 5 bps: '
              f'{np.nanmean((net(MT["bf"], C.COST) - net(MT["bb"], C.COST))[sel]) * 1e4:+.2f}')
        J[f'item14_{p}'] = s_
    del MT

    # ---- item 13: per-stock coefficients (own last 504 days at lag 2, >= 400 pairs, no shrinkage)
    Pf_s = np.full((N, T), np.nan, np.float32)
    Pb_s = np.full((N, T), np.nan, np.float32)
    for k, t0 in enumerate(rebal):
        t1 = rebal[k + 1] if k + 1 < len(rebal) else T
        if t1 <= s0:
            continue
        a = max(0, t0 - 2 - 503)
        ss = np.arange(a, t0 - 2 + 1)                           # origins s with s + 2 <= t0
        assert ss[-1] + 2 <= t0
        u = np.nonzero(cal['UJ'][k])[0]
        r = np.asarray(R[:, ss][u], np.float64)
        v = np.asarray(V[:, ss][u], np.float64)
        y = np.asarray(R[:, ss + 2][u], np.float64)
        ok = np.isfinite(r) & np.isfinite(v) & np.isfinite(y)
        keep = ok.sum(1) >= 400
        u, r, v, y, ok = u[keep], r[keep], v[keep], y[keep], ok[keep]
        r, v, y = (np.where(ok, z, 0.0) for z in (r, v, y))
        Xf = [ok.astype(float), r, v, v * r]
        for full, P, out in ((True, 4, Pf_s), (False, 3, Pb_s)):
            X = Xf[:P]
            A = np.empty((len(u), P, P))
            bv = np.empty((len(u), P))
            for i in range(P):
                for j in range(P):
                    A[:, i, j] = (X[i] * X[j]).sum(1)
                bv[:, i] = (X[i] * y).sum(1)
            with np.errstate(all='ignore'):
                det = np.linalg.det(A)
            good = np.isfinite(det) & (np.abs(det) > 0)
            coef = np.full((len(u), P), np.nan)
            if good.any():
                coef[good] = np.linalg.solve(A[good], bv[good][..., None])[..., 0]
            tt = np.arange(max(t0, s0), t1)
            rt = np.asarray(R[:, tt][u], np.float64)
            vt = np.asarray(V[:, tt][u], np.float64)
            sc = coef[:, 1:2] * rt + coef[:, 2:3] * vt
            if full:
                sc = sc + coef[:, 3:4] * vt * rt
            out[np.ix_(u, tt)] = sc
    cells = np.isfinite(Pf_s) & np.isfinite(Pb_s)
    Pf_s[~cells] = np.nan
    Pb_s[~cells] = np.nan
    bsf = book(Pf_s, R, dec, 1, 1)
    bsb = book(Pb_s, R, dec, 1, 1)
    DS = bsf['gross'] - bsb['gross']
    for p in periods:
        sel = dsel[p]
        s_ = series_stats(DS[sel], draws=2000, seed=C.SEED + 7)
        C.log(rep, '', f'== Item 13, per-stock coefficients, {p} ==', f'gross D: {fmt(s_)}; P_full book gross '
              f'{np.nanmean(bsf["gross"][sel]) * 1e4:+.2f}; stocks ranked a day median {np.nanmedian(bsf["n"][sel]):.0f}')
        J[f'item13_{p}'] = s_
    del Pf_s, Pb_s

    # ---- item 8: daily Fama-MacBeth of R(t+1) and R(t+2) on [1, R, V, G, GR, GV, VR, GVR] in U_W and U_J
    C.log(rep, '', '== Item 8: daily cross-sections, coefficient on G*V*R (expected negative) and on V*R ==')
    for un, gn in (('UW', 'GW'), ('UJ', 'GJ')):
        coefs = {1: np.full((T, 8), np.nan), 2: np.full((T, 8), np.nan)}
        for t in range(first, T - 2):
            k = C.cal_index(rebal, t)
            u = cal[un][k]
            r = np.asarray(R[:, t], np.float64)
            v = np.asarray(V[:, t], np.float64)
            g = cal[gn][k].astype(np.float64)
            ok = u & np.isfinite(r) & np.isfinite(v) & np.isfinite(g)
            rows = np.nonzero(ok)[0]
            if len(rows) < 200:
                continue
            rr = wins_in(r, rows)
            vv = wins_in(v, rows)
            gg = g[rows]
            X = np.column_stack([np.ones(len(rows)), rr, vv, gg, gg * rr, gg * vv, vv * rr, gg * vv * rr])
            for lag in (1, 2):
                y = np.asarray(R[rows, t + lag], np.float64)
                f = np.isfinite(y)
                if f.sum() >= 200:
                    coefs[lag][t] = C.ols(X[f], y[f])
        for lag in (1, 2):
            for p in periods:
                lo, hi = per(p)
                x7 = coefs[lag][lo:min(hi, T - 1 - lag) + 1, 7]
                x6 = coefs[lag][lo:min(hi, T - 1 - lag) + 1, 6]
                s7 = series_stats(x7, draws=2000, seed=C.SEED + 8)
                s6 = series_stats(x6, draws=2000, seed=C.SEED + 9)
                C.log(rep, f'{un}, target R(t+{lag}), {p}: G*V*R {fmt(s7, scale=1, unit="").replace(" a day", "")}; '
                           f'V*R {s6["est"]:+.4f} (t {s6.get("t", np.nan):+.2f})')
                J[f'item8_{un}_lag{lag}_{p}'] = {'GVR': s7, 'VR': s6}

    np.savez(big / 'm1_series.npz', dec=dec, D=D, Dn=Dn, Dgross_full=M['bf']['gross'], Dgross_base=M['bb']['gross'],
             negR=brv['gross'], D0=D0, DS=DS, DT=DT, DW=DW, decw=decw)
    json.dump(J, open(C.OUT / f'm1_{tag}.json', 'w'), indent=1, default=lambda o: None)
    C.log(rep, '', f'done in {time.time() - t_start:.0f} s')


if __name__ == '__main__':
    main(sys.argv[1])
