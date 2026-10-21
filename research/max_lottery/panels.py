"""Panels for the MAX test: the daily MAX panel (for the judge's scores) and, at every formation on the fixed (twins)
calendar, MAX, idiosyncratic volatility, the controls and the next-21-day returns.

    python3 -I -B panels.py old|new   -> WORK/max_daily_<tag>.npy, WORK/form_<tag>.npz, check_ivol_<tag>.txt

At the close of formation day t:
  MAX   = the largest daily simple return over t-20..t (export closes; |log r| > 0.5 missing; >= 15 valid of 21)
  IVOL  = sd (ddof 1) of the residuals of daily excess simple returns on const, Mkt-RF, SMB, HML (French) over t-20..t,
          >= 15 valid days (a day with a missing factor is missing)
  REV   = P[t] / P[t-21] - 1;  MOM = P[t-21] / P[t-252] - 1   (forward-filled export closes)
  BETA  = OLS slope of daily stock log returns on SPY's (Yahoo adjusted close) over t-251..t, >= 200 joint days
  SIZE  = log of the 63-day median dollar volume (the universe's own measure)
  FWD   = P[t+21] / P[t] - 1;  FWDD = the same with the dividend correction (pubdata.div_fix) added to the log return
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (c2eval, pubdata, np, WORK, HERE, EXPORTS, H, MIN_OBS, BETA_W, BETA_MIN, MOM_FROM, MOM_TO,  # noqa: E402
                    log, universe, ffill, log_returns)


def daily_max(r):
    """[N, T] float32 MAX: exp(max log return over d-20..d) - 1, NaN where fewer than MIN_OBS valid returns."""
    N, T = r.shape
    out = np.full((N, T), np.nan, np.float32)
    acc = r[:, H - 1:].copy()
    for i in range(1, H):
        np.fmax(acc, r[:, H - 1 - i:T - i], out=acc)
    fin = np.isfinite(r)
    cs = np.zeros((N, T + 1), np.int32)
    np.cumsum(fin, 1, out=cs[:, 1:])
    cnt = cs[:, H:] - cs[:, :-H]               # column i <-> window ending at d = i + H - 1
    del cs, fin
    with np.errstate(all='ignore'):
        m = np.expm1(acc)
    m[cnt < MIN_OBS] = np.nan
    out[:, H - 1:] = m
    return out


def ivol_at(R, X, fac_ok):
    """R [n, H] daily excess simple returns (NaN = missing), X [H, 4] regressors (const + 3 factors), fac_ok [H] bool.
    Returns sd of residuals (ddof 1) [n], NaN where fewer than MIN_OBS valid days."""
    M = np.isfinite(R) & fac_ok[None, :]
    Y = np.where(M, R, 0.0)
    Mf = M.astype(np.float64)
    XtX = np.einsum('nd,di,dj->nij', Mf, X, X)
    Xty = np.einsum('nd,di->ni', Y, X)
    n = M.sum(1)
    out = np.full(len(R), np.nan)
    ok = n >= MIN_OBS
    if ok.any():
        b = np.linalg.solve(XtX[ok], Xty[ok][:, :, None])[:, :, 0]
        e = (Y[ok] - b @ X.T) * Mf[ok]
        out[ok] = np.sqrt((e * e).sum(1) / (n[ok] - 1))
    return out


def build(tag):
    log('build', tag)
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume'))
    days = E['days']
    N, T = E['close'].shape
    iso = np.array([c2eval.iso(d) for d in days])
    log('N', N, 'T', T)
    tw = c2eval.twins(E)
    rebal = tw['rebal']
    uni, big, dv63 = universe(E, tw)
    log('universe built; sizes median', int(np.median(uni.sum(1)[uni.sum(1) > 0])))

    r = log_returns(E['close'])
    mx = daily_max(r)
    np.save(WORK / f'max_daily_{tag}.npy', mx)
    log('daily MAX saved; finite', int(np.isfinite(mx).sum()))

    fd, f3 = pubdata.ff3_daily()
    mkt = pubdata.on_days(fd, f3['Mkt-RF'], days)
    smb = pubdata.on_days(fd, f3['SMB'], days)
    hml = pubdata.on_days(fd, f3['HML'], days)
    rf = pubdata.on_days(fd, f3['RF'], days)
    sd_, sp = pubdata.etf_adjclose(['SPY'])
    spy = pubdata.on_days(sd_, sp['SPY'], days)
    s = np.full(T, np.nan)
    with np.errstate(all='ignore'):
        s[1:] = np.log(spy[1:] / spy[:-1])
    log('French FF3 days on the export', int(np.isfinite(mkt).sum()), 'last', iso[np.isfinite(mkt)][-1],
        '; Yahoo SPY days', int(np.isfinite(s).sum()))

    P = ffill(E['close'])
    dfix = pubdata.div_fix(tag)
    assert dfix.shape == (N, T), dfix.shape
    K = len(rebal)
    names = ('max', 'ivol', 'beta', 'rev', 'mom', 'size', 'fwd', 'fwdd', 'nobs_ivol')
    F = {k: np.full((K, N), np.nan, np.float32) for k in names}
    complete = np.zeros(K, bool)
    first_ret = np.array([iso[t + 1] if t + 1 < T else '9999' for t in rebal])
    check = []
    rng = np.random.default_rng(0)
    for k, t in enumerate(rebal):
        if not uni[k].any():
            continue
        j = np.nonzero(uni[k])[0]
        F['max'][k, j] = mx[j, t]
        # IVOL
        w = slice(t - H + 1, t + 1)
        R = np.expm1(r[j, w]) - rf[w][None, :]
        fac_ok = np.isfinite(mkt[w]) & np.isfinite(smb[w]) & np.isfinite(hml[w]) & np.isfinite(rf[w])
        X = np.column_stack([np.ones(H), np.nan_to_num(mkt[w]), np.nan_to_num(smb[w]), np.nan_to_num(hml[w])])
        iv = ivol_at(np.where(fac_ok[None, :], R, np.nan), X, fac_ok)
        F['ivol'][k, j] = iv
        F['nobs_ivol'][k, j] = (np.isfinite(R) & fac_ok[None, :]).sum(1)
        # check against a direct lstsq on a few stock-days
        if k % 40 == 0 and np.isfinite(iv).any():
            nv = (np.isfinite(R) & fac_ok[None, :]).sum(1)
            pool_full = np.nonzero(np.isfinite(iv) & (nv == H))[0]
            pool_gap = np.nonzero(np.isfinite(iv) & (nv < H))[0]   # stock-days with missing returns test the masking
            pick = list(rng.choice(pool_full, size=min(3, len(pool_full)), replace=False)) + \
                list(rng.choice(pool_gap, size=min(2, len(pool_gap)), replace=False))
            for jj in pick:
                m = np.isfinite(R[jj]) & fac_ok
                b, *_ = np.linalg.lstsq(X[m], R[jj, m], rcond=None)
                e = R[jj, m] - X[m] @ b
                direct = float(np.std(e, ddof=1))
                check.append((iso[t], str(E['tickers'][j[jj]]), int(m.sum()), direct, float(iv[jj])))
        # beta vs SPY
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
        # reversal, momentum, size
        with np.errstate(all='ignore'):
            F['rev'][k, j] = P[j, t] / P[j, t - H] - 1
            F['mom'][k, j] = P[j, t - MOM_TO] / P[j, t - MOM_FROM] - 1
            F['size'][k, j] = np.log(dv63[k, j])
        # forward returns
        if t + H <= T - 1:
            complete[k] = True
            with np.errstate(all='ignore'):
                g = P[j, t + H] / P[j, t]
                F['fwd'][k, j] = g - 1
                F['fwdd'][k, j] = g * np.exp(np.asarray(dfix[j, t + 1:t + H + 1], np.float64).sum(1)) - 1
    log('formations', K, 'complete', int(complete.sum()))
    np.savez(WORK / f'form_{tag}.npz', rebal=rebal, iso=iso[rebal], first_ret=first_ret, complete=complete, uni=uni,
             big=big, tickers=E['tickers'], **F)
    diff = max(abs(c[3] - c[4]) for c in check)
    lines = [f'IVOL check ({tag}): batched normal-equation IVOL vs np.linalg.lstsq on the same rows, {len(check)} stock-days',
             'date        ticker   n   lstsq_sd        batched_sd      abs diff']
    lines += [f'{c[0]}  {c[1]:<7} {c[2]:>3}  {c[3]:.10f}  {c[4]:.10f}  {abs(c[3] - c[4]):.2e}' for c in check]
    lines.append(f'max abs difference {diff:.3e}')
    (HERE / f'check_ivol_{tag}.txt').write_text('\n'.join(lines) + '\n')
    log('IVOL check: max abs diff', diff)
    # descriptive: coverage per formation
    nu = uni.sum(1)
    for k in range(0, K, 24):
        if nu[k]:
            log(iso[rebal[k]], 'universe', int(nu[k]), 'MAX', int(np.isfinite(F['max'][k]).sum()), 'IVOL',
                int(np.isfinite(F['ivol'][k]).sum()), 'beta', int(np.isfinite(F['beta'][k]).sum()))


if __name__ == '__main__':
    build(sys.argv[1])
