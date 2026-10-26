"""S1 (reported only): does expected market illiquidity raise the next period's market excess return?

    python3 -I -B timeseries.py old|new   -> out/timeseries_<tag>.txt, .json

Market illiquidity L_k = log MILLIQ_k (panels.py: the previous grid day's 500 largest, this period's mean daily
ratio, top 1% dropped). Next period: close of t_k + 1 to close of t_k + 22 (as the books), SPY (Yahoo adjusted)
and the equal-weight mean of the U_J sample, each minus the compounded risk-free rate.
Amihud's form: r_{k+1} on L_k, U_{k+1} (the AR(1) residual, Kendall-corrected, fitted over the period described)
and the January dummy. The trading form: r_{k+1} on D_k = L_k - mean(L_{k-12..k-1}), alone and with log RV_k.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import np, WORK, OUT, NW_LAGS, DEV, CHECK, UNSEEN, FORWARD_FROM, log, ols, in_period, dumps  # noqa: E402


def ar1_kendall(Lv):
    """OLS AR(1) of a series, slope Kendall-corrected (+ (1 + 3 c1) / n), intercept refitted so the residuals have
    mean zero. Returns c0, c1 (corrected), c1 (raw) and the residuals (NaN first)."""
    y, x = Lv[1:], Lv[:-1]
    b, t, n, r2 = ols(y, x)
    c1 = b[1] + (1 + 3 * b[1]) / n
    c0 = float(np.mean(y) - c1 * np.mean(x))
    u = np.full(len(Lv), np.nan)
    u[1:] = y - c0 - c1 * x
    return c0, c1, b[1], u


def run(tag):
    Z = dict(np.load(WORK / f'form_{tag}.npz', allow_pickle=False))
    K = len(Z['rebal'])
    first = Z['first_ret']
    comp = Z['complete']
    L = np.log(Z['s1_milliq500'])
    Luj = np.log(Z['s1_milliq_uj'])
    ew = np.full(K, np.nan)
    for k in range(K):
        m = Z['samp_j'][k] & np.isfinite(Z['fwdd'][k])
        if comp[k] and m.sum() >= 100:
            ew[k] = float(np.mean(Z['fwdd'][k][m]))
    ex_spy = Z['s1_spy_fwd'] - Z['s1_rf_fwd']
    ex_ew = ew - Z['s1_rf_fwd']
    jan = Z['s1_jan']
    lrv = Z['s1_logrv']
    out, lines = {}, []
    P = lines.append
    P(f'# S1 (reported only): market illiquidity and the next period\'s excess return ({tag})\n')
    P('Coefficients are % per 21-day period per unit of the regressor; Newey-West t (4 lags), plain t in brackets.\n')
    ok = np.isfinite(L)
    P(f'L (log mean ratio, 500 largest): first {Z["iso"][ok][0]}, {int(ok.sum())} periods; its value fell from '
      f'{np.nanmean(L[:36]):.2f} (first 36 periods) to {np.nanmean(L[-36:]):.2f} (last 36).')

    def D_of(Lv):
        D = np.full(K, np.nan)
        for k in range(12, K):
            w = Lv[k - 12:k]
            if np.isfinite(w).all() and np.isfinite(Lv[k]):
                D[k] = Lv[k] - w.mean()
        return D
    periods = {'new': {'unseen': UNSEEN}, 'fwd': {'forward': (FORWARD_FROM, '9999')}}.get(tag, {'dev': DEV, 'check': CHECK})
    for which, Lv in (('500 largest', L), ('U_J', Luj)):
        D = D_of(Lv)
        for pn, (a, b) in periods.items():
            # regressors dated k predict the period starting at t_k + 1, whose first return day is first[k]
            mk = in_period(first, a, b) & comp & np.isfinite(Lv)
            idx = np.nonzero(mk)[0]
            if len(idx) < 12:
                continue
            for mname, r in (('SPY', ex_spy), ('Equal-weight U_J', ex_ew)):
                res = {}
                # Amihud's form, AR(1) fitted over the period's L
                Lp = Lv[idx]
                c0, c1, c1raw, u = ar1_kendall(Lp)
                # y_k = the excess return held after t_k; L_k predicts it; its contemporaneous illiquidity is the next
                # grid period's, so U_(k+1); the January dummy is that of r[k]'s own period
                y = r[idx][:-1]
                jd = jan[idx][:-1]
                # a January dummy with no January period in the sample (the 15 unseen periods) is left out
                has_jan = np.nanstd(jd) > 0
                X = np.column_stack([Lp[:-1], u[1:]] + ([jd] if has_jan else []))
                b_, tn, n, r2 = ols(y, X, nw_lags=NW_LAGS)
                _, tp, _, _ = ols(y, X)
                res['amihud'] = dict(g1=b_[1], t_g1=tn[1], tp_g1=tp[1], g2=b_[2], t_g2=tn[2], tp_g2=tp[2],
                                     g3=(b_[3] if has_jan else np.nan), t_g3=(tn[3] if has_jan else np.nan),
                                     n=n, r2=r2, c1=c1, c1_raw=c1raw)
                # the trading form
                for lab, XX in (('D alone', D[idx][:, None]), ('D + log RV', np.column_stack([D[idx], lrv[idx]])),
                                ('L level alone', Lv[idx][:, None])):
                    b2, t2, n2, r22 = ols(r[idx], XX, nw_lags=NW_LAGS)
                    _, t2p, _, _ = ols(r[idx], XX)
                    res[lab] = dict(b=b2[1], t=t2[1], tp=t2p[1], n=n2, r2=r22,
                                    b_rv=(b2[2] if XX.shape[1] > 1 else np.nan), t_rv=(t2[2] if XX.shape[1] > 1 else np.nan))
                out[f'{which}|{pn}|{mname}'] = res
                am = res['amihud']
                P(f'**{which}, {pn}, {mname}** ({n} periods; AR(1) slope {am["c1_raw"]:.3f}, Kendall-corrected {am["c1"]:.3f}):')
                P(f'- Amihud\'s form: L_k {100 * am["g1"]:+.3f} (t {am["t_g1"]:+.2f}; {am["tp_g1"]:+.2f}), unexpected '
                  f'U_(k+1) {100 * am["g2"]:+.3f} (t {am["t_g2"]:+.2f}; {am["tp_g2"]:+.2f}), January {100 * am["g3"]:+.3f} '
                  f'(t {am["t_g3"]:+.2f}); R² {am["r2"]:.3f}. Amihud predicts L > 0 and U < 0.')
                for lab in ('D alone', 'D + log RV', 'L level alone'):
                    q = res[lab]
                    extra = f'; log RV {100 * q["b_rv"]:+.3f} (t {q["t_rv"]:+.2f})' if np.isfinite(q['b_rv']) else ''
                    P(f'- {lab}: {100 * q["b"]:+.3f} (t {q["t"]:+.2f}; {q["tp"]:+.2f}){extra}; n {q["n"]}.')
                P('')
    txt = '\n'.join(lines)
    (OUT / f'timeseries_{tag}.txt').write_text(txt + '\n')
    (OUT / f'timeseries_{tag}.json').write_text(dumps(out))
    print(txt)
    log('done')


if __name__ == '__main__':
    run(sys.argv[1])
