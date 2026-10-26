"""P1 and P2 (the primaries), the controlled books (the gate), the decomposition R0-R5, spanning, Fama-MacBeth variants,
how much of ILLIQ is left, robustness, the survivorship haircut and the square-root rule's costs.

    python3 -I -B tests.py old   -> out/results_old.txt, out/results_old.json, WORK/series_old.npz
    python3 -I -B tests.py new   -> out/results_new.txt, ... (the unseen months, once, at the end)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (c2eval, pubdata, np, WORK, OUT, EXPORTS, H, DELAY, COST_BPS, NW_LAGS, MIN_UNIV, FIFTH,  # noqa: E402
                    L_HAIRCUT, DEV, CHECK, PRE, UNSEEN, DIV_FROM, ALPHA, FORWARD_FROM, log, ols, mean_t, pct_rank, winsor,
                    block_boot_p, holm, norm_sf, in_period, dumps)

REGS = ('illiq', 'dv252', 'sig252', 'beta', 'rev', 'mom')
SPECS = {'ILLIQ alone': ('illiq',), 'ILLIQ + DV252 + SIG252': ('illiq', 'dv252', 'sig252'),
         'All controls (the test)': REGS, 'All controls but ILLIQ': REGS[1:]}
FACS = ('Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD')
UNI = {'P1': 'samp_j', 'P2': 'samp_w'}
UNAME = {'P1': 'U_J (judge-like liquid set)', 'P2': 'U_W (no dollar-volume floor)'}


def factor_periods(days, rebal, T):
    """[K, 6] FF5 + UMD compounded over each formation's holding days t+2..t+22 (NaN if any day missing)."""
    d5, f5 = pubdata.ff5_daily()
    du, fu = pubdata.umd_daily()
    cols = [pubdata.on_days(d5, f5[c], days) for c in FACS[:5]] + [pubdata.on_days(du, fu['UMD'], days)]
    out = np.full((len(rebal), 6), np.nan)
    for k, t in enumerate(rebal):
        a, b = t + DELAY, t + DELAY + H
        if b > T - 1:
            continue
        for i, c in enumerate(cols):
            x = c[a + 1:b + 1]
            if np.isfinite(x).all():
                out[k, i] = np.prod(1 + x) - 1
    return out


def adjusted(fwd, haz, k, j, on):
    """Forward returns of rows j at formation k, with the survivorship haircut where `on` (bool [len(j)])."""
    f = fwd[k][j].astype(float)
    if haz is None:
        return f
    h = haz[k][j].astype(float)
    h = np.where(np.isfinite(h), h, 0.0)
    return np.where(on, (1 - h) * f + h * L_HAIRCUT, f)


def transform(Z, k, j, raw, ill_key):
    if not raw:
        X = {r: pct_rank(Z[ill_key if r == 'illiq' else r][k][j]) for r in REGS}
        return X
    il = Z[ill_key][k][j].astype(float)
    X = {'illiq': il / il.mean()}
    X['dv252'] = winsor(np.log(Z['dv252'][k][j].astype(float)))
    for r in ('sig252', 'beta', 'rev', 'mom'):
        X[r] = winsor(Z[r][k][j].astype(float))
    return X


def fm(Z, samp, fwd, raw=False, ill_key='illiq', haz=None, specs=SPECS, sub=None):
    """Fama-MacBeth at every complete formation: OLS of the market-relative forward return on each spec's regressors
    (percentile ranks within the sample, or Amihud's ILLIQMA with winsorized raw controls). Returns {spec: [K, p]},
    [K] sample sizes. sub: optional extra [K, N] mask (e.g. the 500 largest)."""
    K, N = samp.shape
    res = {s: np.full((K, len(r)), np.nan) for s, r in specs.items()}
    nobs = np.zeros(K, int)
    for k in range(K):
        if not Z['complete'][k]:
            continue
        m = samp[k] & np.isfinite(fwd[k]) & np.isfinite(Z[ill_key][k])
        if sub is not None:
            m &= sub[k]
        if m.sum() < MIN_UNIV:
            continue
        j = np.nonzero(m)[0]
        y = adjusted(fwd, haz, k, j, np.ones(len(j), bool))
        y = y - y.mean()
        X = transform(Z, k, j, raw, ill_key)
        nobs[k] = len(j)
        for s, regs in specs.items():
            A = np.column_stack([np.ones(len(j))] + [X[r] for r in regs])
            b, *_ = np.linalg.lstsq(A, y, rcond=None)
            res[s][k] = b[1:]
    return res, nobs


def residual(Z, k, j, kind):
    """log ILLIQ's residual across rows j on the plan's surface: 'both' 1, x, y, x^2, y^2, xy; 'dv' 1, x, x^2;
    'sig' 1, y, y^2 (x = log DV252, y = log SIG252)."""
    li = np.log(Z['illiq'][k][j].astype(float))
    x = np.log(Z['dv252'][k][j].astype(float))
    y = np.log(Z['sig252'][k][j].astype(float))
    cols = {'both': [x, y, x * x, y * y, x * y], 'dv': [x, x * x], 'sig': [y, y * y]}[kind]
    A = np.column_stack([np.ones(len(j))] + cols)
    b, *_ = np.linalg.lstsq(A, li, rcond=None)
    return li - A @ b


def signal(Z, k, j, kind):
    if kind == 'raw':
        return Z['illiq'][k][j].astype(float)
    if kind == 'illiq77':
        return Z['illiq77'][k][j].astype(float)
    if kind == 'dv':
        return -Z['dv252'][k][j].astype(float)        # low dollar volume is long
    if kind == 'sig':
        return Z['sig252'][k][j].astype(float)
    if kind == 'resid':
        return residual(Z, k, j, 'both')
    if kind == 'resid_dv':
        return residual(Z, k, j, 'dv')
    if kind == 'resid_sig':
        return residual(Z, k, j, 'sig')
    raise ValueError(kind)


def book(Z, samp, fwd, kind, q=FIFTH, weights='ew', cells=False, haz=None, haz_long_only=False, sub=None):
    """Long the top q of the signal, short the bottom q, bought at the close of t+1 and held to t+22, prices drifting.
    Costs: 5 bps x turnover (both legs), and the square-root rule per stock (max(5, 5 sqrt(ILLIQ / the 500 largest's median))).
    cells: the 25-cell sort (DV252 fifths, SIG252 fifths within, ILLIQ thirds within each cell). Returns [K] series."""
    K, N = samp.shape
    keys = ('gross', 'net', 'net77', 'long', 'short', 'mkt', 'turnover', 'n_leg', 'ldv_l', 'ldv_s', 'lsig_l', 'lsig_s',
            'cost77_bps')
    out = {c: np.full(K, np.nan) for c in keys}
    wl_prev, ws_prev = np.zeros(N), np.zeros(N)
    cost_prev = np.full(N, COST_BPS)
    prev_ok = False
    ref = Z['s1_ref500']
    for k in range(K):
        m = samp[k] & np.isfinite(fwd[k])
        if sub is not None:
            m &= sub[k]
        if m.sum() < MIN_UNIV or not Z['complete'][k]:
            prev_ok = False
            wl_prev[:] = 0
            ws_prev[:] = 0
            continue
        j = np.nonzero(m)[0]
        sgl = signal(Z, k, j, kind)
        if cells:
            qd = pct_rank(Z['dv252'][k][j])
            gd = np.minimum((qd * 5).astype(int), 4)
            lo, hi = [], []
            for a in range(5):
                ja = np.nonzero(gd == a)[0]
                qs = pct_rank(Z['sig252'][k][j[ja]])
                gs = np.minimum((qs * 5).astype(int), 4)
                for b in range(5):
                    jb = ja[gs == b]
                    if len(jb) < 3:
                        continue
                    o = jb[np.argsort(sgl[jb], kind='mergesort')]
                    n3 = len(o) // 3
                    lo.append(o[:n3])
                    hi.append(o[len(o) - n3:])
            lo, hi = np.concatenate(lo), np.concatenate(hi)
        else:
            nq = int(round(q * len(j)))
            o = np.argsort(sgl, kind='mergesort')
            lo, hi = o[:nq], o[len(o) - nq:]
        L, S = j[hi], j[lo]           # long the high signal
        wl, ws = np.zeros(N), np.zeros(N)
        if weights == 'dv':
            a = Z['dv252'][k][L].astype(float)
            b = Z['dv252'][k][S].astype(float)
            wl[L], ws[S] = a / a.sum(), b / b.sum()
        else:
            wl[L], ws[S] = 1.0 / len(L), 1.0 / len(S)
        rel = Z['illiq'][k].astype(float) / ref[k]
        cost_now = np.where(np.isfinite(rel), np.maximum(COST_BPS, COST_BPS * np.sqrt(np.maximum(rel, 0))), cost_prev)
        dl = np.abs(wl - wl_prev) if prev_ok else wl.copy()
        dsh = np.abs(ws - ws_prev) if prev_ok else ws.copy()
        to = dl.sum() + dsh.sum()
        c77 = ((dl + dsh) * cost_now).sum() * 1e-4
        fl = adjusted(fwd, haz, k, L, np.ones(len(L), bool))
        fs = adjusted(fwd, haz, k, S, np.zeros(len(S), bool) if haz_long_only else np.ones(len(S), bool))
        rl = float((wl[L] * fl).sum())
        rs = float((ws[S] * fs).sum())
        out['long'][k], out['short'][k] = rl, rs
        out['mkt'][k] = float(np.mean(fwd[k][j]))
        out['gross'][k] = rl - rs
        out['turnover'][k] = to
        out['net'][k] = rl - rs - COST_BPS * 1e-4 * to
        out['net77'][k] = rl - rs - c77
        out['cost77_bps'][k] = c77 / to * 1e4 if to > 0 else np.nan
        out['n_leg'][k] = len(L)
        out['ldv_l'][k] = np.log(Z['dv252'][k][L].astype(float)).mean()
        out['ldv_s'][k] = np.log(Z['dv252'][k][S].astype(float)).mean()
        out['lsig_l'][k] = np.log(Z['sig252'][k][L].astype(float)).mean()
        out['lsig_s'][k] = np.log(Z['sig252'][k][S].astype(float)).mean()
        f = np.nan_to_num(fwd[k])
        a = wl * (1 + f)
        b = ws * (1 + f)
        wl_prev, ws_prev = a / a.sum(), b / b.sum()
        cost_prev = cost_now
        prev_ok = True
    return out


def stats(x, Fk, mask):
    """Mean, plain t, NW t, n, Sharpe and the FF5 + UMD alpha (NW t) of a [K] series over mask."""
    m, t, tnw, n = mean_t(x[mask])
    y = x[mask]
    Xf = Fk[mask]
    ok = np.isfinite(y) & np.isfinite(Xf).all(1)
    if ok.sum() >= 12:
        b, tt, na, _ = ols(y[ok], Xf[ok], nw_lags=NW_LAGS)
        a, atn = b[0], tt[0]
    else:
        a = atn = float('nan')
        na = int(ok.sum())
    sd = np.nanstd(y, ddof=1)
    return dict(mean=m, t=t, tnw=tnw, n=n, alpha=a, atn=atn, na=na, sharpe=m / sd * np.sqrt(12) if sd > 0 else np.nan)


def r2_left(Z, samp, mask):
    """Mean cross-sectional R^2 of ILLIQ's rank on DV252's and SIG252's ranks, and of log ILLIQ on the quadratic."""
    a, b = [], []
    for k in np.nonzero(mask)[0]:
        j = np.nonzero(samp[k])[0]
        if len(j) < MIN_UNIV:
            continue
        ri = pct_rank(Z['illiq'][k][j])
        A = np.column_stack([np.ones(len(j)), pct_rank(Z['dv252'][k][j]), pct_rank(Z['sig252'][k][j])])
        bb, *_ = np.linalg.lstsq(A, ri, rcond=None)
        e = ri - A @ bb
        a.append(1 - e.var() / ri.var())
        li = np.log(Z['illiq'][k][j].astype(float))
        e2 = residual(Z, k, j, 'both')
        b.append(1 - e2.var() / li.var())
    if not b:
        return (float('nan'),) * 4
    return float(np.mean(a)), float(np.mean(b)), float(np.min(b)), float(np.max(b))


def pct(x, d=3):
    return f'{100 * x:+.{d}f}'


def fm_line(name, coef, mask, i=0):
    m, t, tnw, n = mean_t(coef[mask, i])
    return f'| {name} | {n} | {pct(m)} | {tnw:+.2f} ({t:+.2f}) |'


def run(tag):
    Z = dict(np.load(WORK / f'form_{tag}.npz', allow_pickle=False))
    E = c2eval.load_export(EXPORTS[tag], arrays=())
    days = E['days']
    T = len(days)
    rebal = Z['rebal']
    K = len(rebal)
    first = Z['first_ret']
    years = np.array([int(s[:4]) for s in first])
    Fk = factor_periods(days, rebal, T)
    comp = Z['complete']
    per = {'dev': in_period(first, *DEV) & comp, 'check': in_period(first, *CHECK) & comp,
           'pre': in_period(first, *PRE) & comp, 'unseen': in_period(first, *UNSEEN) & comp,
           'forward': in_period(first, FORWARD_FROM, '9999') & comp}
    per['dev_ex0809'] = per['dev'] & ~np.isin(years, (2008, 2009))
    per['all'] = (per['dev'] | per['check'] | per['pre']) & comp
    jan = Z['s1_jan'] == 1
    fwd = {'div': Z['fwdd'], 'raw': Z['fwd']}
    R, L, J = {}, [], {}
    P = L.append
    series = {}
    main_p = {'new': 'unseen', 'fwd': 'forward'}.get(tag, 'dev')
    P(f'# Amihud illiquidity: tests ({tag} export)\n')
    P(f'Formations: {K} grid days; complete {int(comp.sum())}; development {int(per["dev"].sum())}, check '
      f'{int(per["check"].sum())}, 2006-2007 {int(per["pre"].sum())}, unseen {int(per["unseen"].sum())}. Forward '
      f'returns: close of t+1 to close of t+22, dividend-corrected from {DIV_FROM} (identical before).\n')

    # ---- Fama-MacBeth and books for both universes ----
    for pk, sk in UNI.items():
        samp = Z[sk]
        for v in ('div', 'raw'):
            R[(pk, 'fm', v)], R[(pk, 'nobs', v)] = fm(Z, samp, fwd[v])
        R[(pk, 'fm_raw', 'div')], _ = fm(Z, samp, fwd['div'], raw=True, specs={'All controls (the test)': REGS})
        R[(pk, 'fm77', 'div')], _ = fm(Z, samp, fwd['div'], ill_key='illiq77', specs={'All controls (the test)': REGS})
        R[(pk, 'fm_haz', 'div')], _ = fm(Z, samp, fwd['div'], haz=Z['haz'], specs={'All controls (the test)': REGS})
        if pk == 'P1':
            R[(pk, 'fm_big', 'div')], _ = fm(Z, samp, fwd['div'], sub=Z['big'], specs={'All controls (the test)': REGS})
        for kind in ('resid', 'raw', 'dv', 'sig', 'resid_dv', 'resid_sig'):
            R[(pk, 'book_' + kind, 'div')] = book(Z, samp, fwd['div'], kind)
        R[(pk, 'book_resid', 'raw')] = book(Z, samp, fwd['raw'], 'resid')
        R[(pk, 'book_cells', 'div')] = book(Z, samp, fwd['div'], 'raw', cells=True)
        R[(pk, 'book_resid10', 'div')] = book(Z, samp, fwd['div'], 'resid', q=0.1)
        R[(pk, 'book_residdv', 'div')] = book(Z, samp, fwd['div'], 'resid', weights='dv')
        R[(pk, 'book_resid_haz', 'div')] = book(Z, samp, fwd['div'], 'resid', haz=Z['haz'])
        R[(pk, 'book_resid_hazL', 'div')] = book(Z, samp, fwd['div'], 'resid', haz=Z['haz'], haz_long_only=True)
        R[(pk, 'book_raw_haz', 'div')] = book(Z, samp, fwd['div'], 'raw', haz=Z['haz'])
        R[(pk, 'book_raw_hazL', 'div')] = book(Z, samp, fwd['div'], 'raw', haz=Z['haz'], haz_long_only=True)
        if pk == 'P1':
            R[(pk, 'book_resid_big', 'div')] = book(Z, samp, fwd['div'], 'resid', sub=Z['big'])
        log(pk, 'done')
    # the gate book on the unscreened 63-day ILLIQ: the residual of log ILLIQ77 on the same surface
    Z77 = dict(Z)
    Z77['illiq'] = Z['illiq77']
    for pk, sk in UNI.items():
        R[(pk, 'book_resid77', 'div')] = book(Z77, Z[sk], fwd['div'], 'resid')

    # ---- the primaries ----
    P('## P1 and P2 (primary): ILLIQ\'s Fama-MacBeth coefficient with all controls\n')
    P('Percentile ranks; a coefficient is the 21-day return gap (%) between the most and the least illiquid stock, the '
      'other ranks held fixed. t: Newey-West (4 lags), plain in brackets. Bootstrap: moving blocks of 3 formations, '
      '10,000 draws, one-sided.\n')
    P('| Primary | Period | Formations | Mean stocks | Coefficient, % | t NW (plain) | p (normal, one-sided) | '
      'Bootstrap p, 3-period blocks | Bootstrap p, iid | Controlled book net, % a period (t NW) |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    pv = {}
    for pk in UNI:
        coef = R[(pk, 'fm', 'div')]['All controls (the test)'][:, 0]
        nob = R[(pk, 'nobs', 'div')]
        for pn in ([main_p] if tag != 'old' else ['dev', 'check', 'pre', 'dev_ex0809', 'all']):
            mk = per[pn]
            m, t, tnw, n = mean_t(coef[mk])
            p = norm_sf(tnw)
            b3, ci3 = block_boot_p(coef[mk], block=3)
            b1, _ = block_boot_p(coef[mk], block=1)
            g = stats(R[(pk, 'book_resid', 'div')]['net'], Fk, mk)
            P(f'| {pk} ({UNAME[pk]}) | {pn} | {n} | {np.mean(nob[mk]):.0f} | {pct(m)} | {tnw:+.2f} ({t:+.2f}) | {p:.4f} | '
              f'{b3:.4f} | {b1:.4f} | {pct(g["mean"])} ({g["tnw"]:+.2f}) |')
            J[f'{pk}_{pn}'] = dict(coef=m, t=t, tnw=tnw, n=n, p=p, boot3=b3, boot3_ci=ci3, boot1=b1, gate=g)
            if pn == main_p:
                pv[pk] = p
            series[f'{pk}_coef'] = coef
    H_ = holm(pv)
    P('')
    P(f'**Holm at a one-sided family level of {ALPHA:.4f} (t = 2):**')
    for pk in UNI:
        p, thr, ok = H_[pk]
        jj = J[f'{pk}_{main_p}']
        boot_ok = np.isfinite(jj['boot3']) and jj['boot3'] < thr
        gate_ok = np.isfinite(jj['gate']['mean']) and jj['gate']['mean'] > 0
        verdict = ok and boot_ok and gate_ok
        J[f'{pk}_pass'] = dict(p=p, threshold=thr, holm=ok, boot_ok=bool(boot_ok), gate_ok=bool(gate_ok), passes=bool(verdict),
                               t3=bool(jj['tnw'] >= 3))
        P(f'- {pk}: p {p:.4f} against {thr:.4f}: Holm {"passes" if ok else "fails"}; bootstrap {jj["boot3"]:.4f} '
          f'{"below" if boot_ok else "not below"}; gate (controlled book net mean > 0): {"yes" if gate_ok else "no"}. '
          f'**{"PASS" if verdict else "FAIL"}**. (t ≥ 3 reading: {"would pass the t" if jj["tnw"] >= 3 else "fails"}.)')
    P('')

    # ---- by year ----
    P('## By year (first return day): ILLIQ\'s controlled coefficient and the controlled book\n')
    P('| Year | Formations | P1 coefficient, % | P1 book net, % | P2 coefficient, % | P2 book net, % |')
    P('|---|---|---|---|---|---|')
    for y in np.unique(years[per['all'] | per['unseen'] | per['forward']]):
        mk = (years == y) & comp
        if not mk.any():
            continue
        cells = []
        for pk in UNI:
            c = R[(pk, 'fm', 'div')]['All controls (the test)'][mk, 0]
            b = R[(pk, 'book_resid', 'div')]['net'][mk]
            cells += [pct(np.nanmean(c), 2), pct(np.nanmean(b), 2)]
        P(f'| {y} | {int(mk.sum())} | ' + ' | '.join(cells) + ' |')
    P('')

    # ---- Fama-MacBeth specifications ----
    P('## Fama-MacBeth specifications\n')
    for pk in UNI:
        for pn in ([main_p] if tag != 'old' else ['dev', 'check']):
            mk = per[pn]
            P(f'**{pk}, {pn}** (rank coefficients, % per 21 days; NW t, plain t):\n')
            P('| Specification | ' + ' | '.join(REGS) + ' |')
            P('|---|' + '---|' * len(REGS))
            for s, regs in SPECS.items():
                c = R[(pk, 'fm', 'div')][s]
                cells = []
                for r in REGS:
                    if r in regs:
                        m, t, tnw, n = mean_t(c[mk, regs.index(r)])
                        cells.append(f'{pct(m, 2)} ({tnw:+.2f}; {t:+.2f})')
                    else:
                        cells.append('')
                P(f'| {s} | ' + ' | '.join(cells) + ' |')
            c = R[(pk, 'fm_raw', 'div')]['All controls (the test)']
            cells = []
            for i, r in enumerate(REGS):
                m, t, tnw, n = mean_t(c[mk, i])
                cells.append(f'{(100 * m):+.3f} ({tnw:+.2f}; {t:+.2f})')
            P('| Amihud\'s form: ILLIQMA, raw winsorized controls (log DV252) | ' + ' | '.join(cells) + ' |')
            P('')
            J[f'{pk}_{pn}_fm'] = {s: [mean_t(R[(pk, 'fm', 'div')][s][mk, i]) for i in range(len(r))] for s, r in SPECS.items()}
            J[f'{pk}_{pn}_fm_raw'] = [mean_t(c[mk, i]) for i in range(len(REGS))]

    # ---- how much of ILLIQ is left ----
    P('## How much of ILLIQ is left to test\n')
    for pk, sk in UNI.items():
        a, b, bmin, bmax = r2_left(Z, Z[sk], per[main_p])
        J[f'{pk}_r2'] = dict(rank=a, log_quadratic=b, log_quadratic_min=bmin, log_quadratic_max=bmax)
        P(f'- {pk}: mean R² of ILLIQ\'s rank on DV252\'s and SIG252\'s ranks {a:.3f}; of log ILLIQ on the quadratic '
          f'surface {b:.3f} (range over formations {bmin:.3f}..{bmax:.3f}).')
    P('')

    # ---- books ----
    names = [('book_resid', 'Controlled book (the gate): residual on the quadratic surface'),
             ('book_raw', 'R0: raw ILLIQ (the published sort)'), ('book_dv', 'R1: low minus high DV252'),
             ('book_sig', 'R2: high minus low SIG252'), ('book_resid_dv', 'R3: ILLIQ beyond DV252 only'),
             ('book_resid_sig', 'R4: ILLIQ beyond SIG252 only'), ('book_cells', 'R5: 25 cells, ILLIQ thirds'),
             ('book_resid10', 'Controlled, tenths'), ('book_residdv', 'Controlled, legs weighted by DV252'),
             ('book_resid77', 'Controlled, on the unscreened 63-day ILLIQ'),
             ('book_resid_haz', 'Controlled, haircut on every stock (k 1, L -55%)'),
             ('book_resid_hazL', 'Controlled, haircut on the long leg only'),
             ('book_raw_haz', 'R0, haircut on every stock'), ('book_raw_hazL', 'R0, haircut on the long leg only'),
             ('book_resid_big', 'Controlled, the 500 largest (P1 only)')]
    for pk in UNI:
        for pn in ([main_p] if tag != 'old' else ['dev', 'check', 'pre', 'dev_ex0809']):
            mk = per[pn]
            P(f'## Books, {pk} ({UNAME[pk]}), {pn}\n')
            P('| Book | Periods | Gross, % | Net 5 bps, % (t NW; plain) | Net, square-root costs, % (t NW) | square-root cost, bps a side | '
              'Turnover | Stocks a leg | FF5+UMD alpha, % (t NW) | Sharpe | Long - short: mean log DV252 / log SIG252 |')
            P('|---|---|---|---|---|---|---|---|---|---|---|')
            for key, nm in names:
                b = R.get((pk, key, 'div'))
                if b is None:
                    continue
                s5 = stats(b['net'], Fk, mk)
                s77 = stats(b['net77'], Fk, mk)
                g = np.nanmean(b['gross'][mk])
                P(f'| {nm} | {s5["n"]} | {pct(g)} | {pct(s5["mean"])} ({s5["tnw"]:+.2f}; {s5["t"]:+.2f}) | '
                  f'{pct(s77["mean"])} ({s77["tnw"]:+.2f}) | {np.nanmean(b["cost77_bps"][mk]):.1f} | '
                  f'{np.nanmean(b["turnover"][mk]):.2f} | {np.nanmean(b["n_leg"][mk]):.0f} | {pct(s5["alpha"])} '
                  f'({s5["atn"]:+.2f}) | {s5["sharpe"]:+.2f} | '
                  f'{np.nanmean(b["ldv_l"][mk] - b["ldv_s"][mk]):+.2f} / {np.nanmean(b["lsig_l"][mk] - b["lsig_s"][mk]):+.2f} |')
                J[f'{pk}_{pn}_{key}'] = dict(net5=s5, net77=s77, gross=g, turnover=float(np.nanmean(b['turnover'][mk])),
                                            cost77=float(np.nanmean(b['cost77_bps'][mk])))
                if pn == main_p:
                    for c, x in b.items():
                        series[f'{pk}_{key}_{c}'] = x
            if pk == 'P1' or True:
                bl = R[(pk, 'book_resid', 'div')]
                P(f'\nLegs of the controlled book (mean over periods): long {pct(np.nanmean(bl["long"][mk]))}%, short '
                  f'{pct(np.nanmean(bl["short"][mk]))}%, all sample stocks {pct(np.nanmean(bl["mkt"][mk]))}%.')
                br = R[(pk, 'book_raw', 'div')]
                P(f'Legs of R0: long (illiquid) {pct(np.nanmean(br["long"][mk]))}%, short (liquid) '
                  f'{pct(np.nanmean(br["short"][mk]))}%.\n')
            # raw (no dividend fix) for periods touching 2022-07
            if pn in ('check', 'unseen', 'forward'):
                bb = R[(pk, 'book_resid', 'raw')]
                s5 = stats(bb['net'], Fk, mk)
                c = R[(pk, 'fm', 'raw')]['All controls (the test)'][:, 0]
                m, t, tnw, n = mean_t(c[mk])
                P(f'Without the dividend correction: controlled book net {pct(s5["mean"])}% (t {s5["tnw"]:+.2f}); '
                  f'coefficient {pct(m)} (t {tnw:+.2f}).\n')
            # spanning regression of the controlled book on R1 and R2
            y = R[(pk, 'book_resid', 'div')]['net'][mk]
            X = np.column_stack([R[(pk, 'book_dv', 'div')]['net'][mk], R[(pk, 'book_sig', 'div')]['net'][mk]])
            ok = np.isfinite(y) & np.isfinite(X).all(1)
            if ok.sum() > 12:
                b, t, n, r2 = ols(y[ok], X[ok], nw_lags=NW_LAGS)
                P(f'Spanning: controlled book net on R1 and R2 net: intercept {pct(b[0])}% (t NW {t[0]:+.2f}), R1 loading '
                  f'{b[1]:+.3f} (t {t[1]:+.2f}), R2 loading {b[2]:+.3f} (t {t[2]:+.2f}), R² {r2:.3f}, n {n}.')
                J[f'{pk}_{pn}_spanning'] = dict(intercept=b[0], t_int=t[0], b_r1=b[1], t_r1=t[1], b_r2=b[2], t_r2=t[2], r2=r2)
            # robustness of the primary coefficient
            rob = [('the unscreened 63-day ILLIQ in place of this plan\'s', R[(pk, 'fm77', 'div')]['All controls (the test)'], mk),
                   ('Haircut on every stock (k 1, L -55%)', R[(pk, 'fm_haz', 'div')]['All controls (the test)'], mk),
                   ('January periods', R[(pk, 'fm', 'div')]['All controls (the test)'], mk & jan),
                   ('Other periods', R[(pk, 'fm', 'div')]['All controls (the test)'], mk & ~jan)]
            if pk == 'P1':
                rob.append(('The 500 largest', R[(pk, 'fm_big', 'div')]['All controls (the test)'], mk))
            P('\n| Primary coefficient, robustness | Formations | Coefficient, % | t NW (plain) |')
            P('|---|---|---|---|')
            for nm, c, mm in rob:
                P(fm_line(nm, c, mm))
                J[f'{pk}_{pn}_rob_{nm}'] = mean_t(c[mm, 0])
            # the controlled book in January
            bl = R[(pk, 'book_resid', 'div')]['net']
            P(f'\nControlled book net, January periods {pct(np.nanmean(bl[mk & jan]))}% ({int((mk & jan).sum())}), '
              f'other {pct(np.nanmean(bl[mk & ~jan]))}% ({int((mk & ~jan).sum())}).\n')
    # dividend-correction coverage for the later periods
    try:
        cov = set(l.split('\t')[0] for l in (Path('/home/bhoult/programming/correlation/research/published_data/out/'
                                                  'tickers_map.tsv').read_text().splitlines()[1:]))
        tk = Z['tickers']
        inn = np.array([t in cov for t in tk])
        late = in_period(Z['iso'], DIV_FROM, '9999') & comp
        for pk, sk in UNI.items():
            sh = [inn[Z[sk][k]].mean() for k in np.nonzero(late)[0] if Z[sk][k].any()]
            P(f'Dividend-correction coverage, {pk}: share of sample stocks with a div_fix ticker on formations from '
              f'{DIV_FROM}: {np.mean(sh):.3f}.')
            J[f'{pk}_divcov'] = float(np.mean(sh))
    except OSError:
        pass
    txt = '\n'.join(L)
    (OUT / f'results_{tag}.txt').write_text(txt + '\n')
    (OUT / f'results_{tag}.json').write_text(dumps(J))
    np.savez(WORK / f'series_{tag}.npz', first_ret=first, complete=comp, **series)
    print(txt)


if __name__ == '__main__':
    run(sys.argv[1])
