"""Panels: the daily ILLIQ panel (this plan's and the earlier test's), the screens' counts, and at each grid formation the two
universes, ILLIQ, its parts, the Fama-MacBeth controls, forward returns, hazards and S1's market illiquidity.

    python3 -I -B panels.py old|new   -> WORK/illiq_<tag>.npy, WORK/illiq77_<tag>.npy, WORK/form_<tag>.npz,
                                         out/panels_<tag>.txt

At the close of grid day t (export day 252 + 21k):
  ILLIQ  = mean of |r| / DV over the counted days in t-251..t, >= 200 counted (common.ratio_panel's screens)
  DV252  = median dollar volume over the same counted days;  SIG252 = sd (ddof 1) of their daily log returns
  BETA   = OLS slope of daily log returns on SPY's (Yahoo adjusted close) over t-251..t, >= 200 joint days
  REV    = P[t] / P[t-21] - 1;  MOM = P[t-21] / P[t-252] - 1   (forward-filled export closes)
  FWD    = P[t+22] / P[t+1] - 1 (bought at the close of t+1, delay 1);  FWDD = with pubdata.div_fix over t+2..t+22
  HAZ    = the survivorship model's monthly hazard at t
  S1     = mean daily |r| / DV over (t_{k-1}, t_k] (>= 15 counted days) for the previous grid day's 500 largest (and
           U_J); cross-sectional mean after dropping the top 1%
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (c2eval, pubdata, np, WORK, OUT, EXPORTS, H, DELAY, ILLIQ_W, ILLIQ_MIN, ILLIQ_MIN_77,  # noqa: E402
                    BETA_W, BETA_MIN, MOM_FROM, MOM_TO, TRIM, S1_MIN_DAYS, M77, log, universe, ffill, log_returns,
                    ratio_panel, window_mean, amihud_panel_77, surv_model, surv_features, surv_hazard)


def trim_mask(vals, ok):
    """ok minus the stocks whose value is in the top or bottom TRIM of the finite values within ok."""
    out = ok.copy()
    v = vals[ok]
    if len(v) < 10:
        return out
    lo, hi = np.quantile(v, [TRIM, 1 - TRIM])
    out[ok] = (v >= lo) & (v <= hi)
    return out


def build(tag):
    log('build', tag)
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume'))
    days = E['days']
    N, T = E['close'].shape
    iso = np.array([c2eval.iso(d) for d in days])
    years = np.array([int(s[:4]) for s in iso])
    tw = c2eval.twins(E)
    rebal = tw['rebal']
    K = len(rebal)
    uj, uw, big, dv63 = universe(E, tw)
    log('universes built')
    lines = [f'Panels ({tag}): {N} stocks x {T} days, {iso[0]}..{iso[-1]}; {K} grid days']

    # ---- the unscreened 63-day panel and the equality checks ----
    A77 = amihud_panel_77(E['close'], E['dollar_volume'])
    x0 = ratio_panel(E['close'], E['dollar_volume'], screens=False)
    A0, _ = window_mean(x0, ILLIQ_W, ILLIQ_MIN_77)
    del x0
    same = bool(np.array_equal(A0, A77, equal_nan=True))
    lines.append(f'Check: this folder\'s panel with both screens off and the minimum at 63 days equals the earlier test\'s '
                 f'amihud_panel (copied) exactly: {same}')
    del A0
    if tag == 'old' and (M77 / 'amihud_old.npy').exists():
        saved = np.load(M77 / 'amihud_old.npy', mmap_mode='r')
        same2 = bool(saved.shape == A77.shape and np.array_equal(np.asarray(saved), A77, equal_nan=True))
        lines.append(f'Check: it equals the earlier saved panel (amihud_old.npy) exactly: {same2}')
    np.save(WORK / f'illiq77_{tag}.npy', A77)
    log('illiq77 saved; equal', same)

    # ---- this plan's panel ----
    cnt = {}
    x = ratio_panel(E['close'], E['dollar_volume'], screens=True, counts=cnt)
    ILL, nday = window_mean(x, ILLIQ_W, ILLIQ_MIN)
    np.save(WORK / f'illiq_{tag}.npy', ILL)
    log('illiq saved; finite', int(np.isfinite(ILL).sum()))
    bm, bd = cnt.pop('broken_mask'), cnt.pop('bad_mask')
    lines.append(f"Days with a ratio (both closes, DV > 0): {cnt['base_days']:,}; bad prints dropped: "
                 f"{cnt['bad_print_days']:,}; broken-volume candidates by the rolling max: {cnt['broken_candidates']:,}; "
                 f"broken-volume days dropped: {cnt['broken_days']:,}")
    by_y = {y: (int(bd[:, years == y].sum()), int(bm[:, years == y].sum())) for y in np.unique(years)}
    lines.append('By year (bad prints, broken volume): ' + ', '.join(f'{y} {a}/{b}' for y, (a, b) in by_y.items()))
    per_day = bm.sum(0)
    top = np.argsort(-per_day)[:8]
    lines.append('Days with the most broken-volume flags: ' + ', '.join(f'{iso[d]} {int(per_day[d])}' for d in top))
    del bm, bd

    # ---- grid formations ----
    r = log_returns(E['close'])
    P = ffill(E['close'])
    sd_, sp = pubdata.etf_adjclose(['SPY'])
    spy = pubdata.on_days(sd_, sp['SPY'], days)
    s = np.full(T, np.nan)
    with np.errstate(all='ignore'):
        s[1:] = np.log(spy[1:] / spy[:-1])
    rf = pubdata.rf_daily(days)
    if tag in ('old', 'new'):
        dfix = pubdata.div_fix(tag)
        assert dfix.shape == (N, T), dfix.shape
    else:   # a forward download has no dividend correction: its forward returns are price-only
        dfix = np.zeros((N, T), np.float32)
    model = surv_model()
    names = ('illiq', 'illiq77', 'dv252', 'sig252', 'beta', 'rev', 'mom', 'fwd', 'fwdd', 'haz', 'nday')
    F = {k: np.full((K, N), np.nan, np.float32) for k in names}
    complete = np.zeros(K, bool)
    first_ret = np.array([iso[t + DELAY + 1] if t + DELAY + 1 < T else '9999' for t in rebal])
    s1 = {k: np.full(K, np.nan) for k in ('milliq500', 'milliq_uj', 'n500', 'n_uj', 'spy_fwd', 'rf_fwd', 'logrv',
                                          'jan', 'ref500')}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for k, t in enumerate(rebal):
            if not uw[k].any():
                continue
            j = np.nonzero(uw[k])[0]
            w = slice(t - ILLIQ_W + 1, t + 1)
            xs = x[j, w]
            cm = np.isfinite(xs)
            F['illiq'][k, j] = ILL[j, t]
            F['illiq77'][k, j] = A77[j, t]
            F['nday'][k, j] = nday[j, t]
            dvw = np.where(cm, np.asarray(E['dollar_volume'][j, w], np.float64), np.nan)
            d252 = np.nanmedian(dvw, 1)
            rw = np.where(cm, r[j, w], np.nan)
            sg = np.nanstd(rw, 1, ddof=1)
            okn = nday[j, t] >= ILLIQ_MIN
            F['dv252'][k, j] = np.where(okn, d252, np.nan)
            F['sig252'][k, j] = np.where(okn, sg, np.nan)
            # beta vs SPY (copied from research/max_lottery/panels.py)
            wb = slice(t - BETA_W + 1, t + 1)
            rb = r[j, wb]
            sb = s[wb][None, :]
            M = np.isfinite(rb) & np.isfinite(sb)
            n = M.sum(1)
            with np.errstate(all='ignore'):
                xm = np.where(M, sb, 0).sum(1) / n
                ym = np.where(M, rb, 0).sum(1) / n
                cov = np.where(M, (rb - ym[:, None]) * (sb - xm[:, None]), 0).sum(1)
                var = np.where(M, (sb - xm[:, None]) ** 2, 0).sum(1)
                be = cov / var
            be[n < BETA_MIN] = np.nan
            F['beta'][k, j] = be
            with np.errstate(all='ignore'):
                F['rev'][k, j] = P[j, t] / P[j, t - H] - 1
                F['mom'][k, j] = P[j, t - MOM_TO] / P[j, t - MOM_FROM] - 1
            F['haz'][k, j] = surv_hazard(surv_features(E, j, t), model)
            a, b = t + DELAY, t + DELAY + H
            if b <= T - 1:
                complete[k] = True
                with np.errstate(all='ignore'):
                    g = P[j, b] / P[j, a]
                    F['fwd'][k, j] = g - 1
                    F['fwdd'][k, j] = g * np.exp(np.asarray(dfix[j, a + 1:b + 1], np.float64).sum(1)) - 1
                    s1['spy_fwd'][k] = spy[b] / spy[a] - 1
                    s1['rf_fwd'][k] = np.prod(1 + rf[a + 1:b + 1]) - 1
                    s1['jan'][k] = float(sum(iso[d][5:7] == '01' for d in range(a + 1, b + 1)) >= 11)
            # S1: this period's market illiquidity, on the previous grid day's 500 largest (and U_J)
            s1['logrv'][k] = np.log(np.nansum(s[t - H + 1:t + 1] ** 2))
            bb = big[k] & np.isfinite(ILL[:, t])
            s1['ref500'][k] = float(np.nanmedian(ILL[bb, t])) if bb.any() else np.nan
            if k > 0:
                p0 = rebal[k - 1]
                for key, mem in (('500', big[k - 1]), ('_uj', uj[k - 1])):
                    jj = np.nonzero(mem)[0]
                    if len(jj) == 0:
                        continue
                    xp = x[jj, p0 + 1:t + 1]
                    c = np.isfinite(xp).sum(1)
                    mm = np.nanmean(xp, 1)
                    mm = mm[(c >= S1_MIN_DAYS) & np.isfinite(mm)]
                    if len(mm) >= 50:
                        mm = mm[mm <= np.quantile(mm, 1 - TRIM)]
                        s1['milliq' + key][k] = mm.mean()
                        s1['n' + key][k] = len(mm)
            if k % 24 == 0:
                log(iso[t], 'U_W', len(j), 'U_J', int(uj[k].sum()))
    # samples: the universe, every control finite, then Amihud's 1% rule on ILLIQ (no forward return is used)
    regs = ('illiq', 'dv252', 'sig252', 'beta', 'rev', 'mom')
    allf = np.ones((K, N), bool)
    for c in regs:
        allf &= np.isfinite(F[c])
    ok_j, ok_w = uj & allf, uw & allf
    samp_j = np.zeros((K, N), bool)
    samp_w = np.zeros((K, N), bool)
    for k in range(K):
        samp_j[k] = trim_mask(F['illiq'][k], ok_j[k])
        samp_w[k] = trim_mask(F['illiq'][k], ok_w[k])
    np.savez(WORK / f'form_{tag}.npz', rebal=rebal, iso=iso[rebal], first_ret=first_ret, complete=complete,
             uj=uj, uw=uw, big=big, ok_j=ok_j, ok_w=ok_w, samp_j=samp_j, samp_w=samp_w, tickers=E['tickers'],
             **F, **{f's1_{k}': v for k, v in s1.items()})
    log('formations', K, 'complete', int(complete.sum()))
    # ---- descriptive: sizes ----
    lines.append('Grid day | U_J | U_J sample (after controls and the 1% rule) | U_W | U_W sample | 500 largest in the '
                 'U_J sample | median ILLIQ in U_J sample (x 1e6, per $M) | median DV252 U_J / U_W, $M')
    for k in range(0, K, 12):
        if uw[k].any():
            mj = np.nanmedian(F['illiq'][k][samp_j[k]]) * 1e6 if samp_j[k].any() else np.nan
            dj = np.nanmedian(F['dv252'][k][samp_j[k]]) / 1e6 if samp_j[k].any() else np.nan
            dw = np.nanmedian(F['dv252'][k][samp_w[k]]) / 1e6 if samp_w[k].any() else np.nan
            lines.append(f'{iso[rebal[k]]} | {int(uj[k].sum())} | {int(samp_j[k].sum())} | {int(uw[k].sum())} | '
                         f'{int(samp_w[k].sum())} | {int((big[k] & samp_j[k]).sum())} | {mj:.4f} | {dj:.1f} / {dw:.1f}')
    lost = [(int(uw[k].sum()), int(ok_w[k].sum())) for k in range(K) if uw[k].any()]
    lines.append(f'U_W stocks with every control (before the 1% rule) as a share of U_W, median over grid days: '
                 f'{np.median([b / a for a, b in lost]):.3f}')
    (OUT / f'panels_{tag}.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    build(sys.argv[1])
