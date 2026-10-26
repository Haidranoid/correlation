"""B1 (what curvature is made of), the four checks, and B2 (does curvature add to forecasts of SPY and the twin
funds?). Rules in results_B.md, "Decisions" 21-27 and 45-49.

    python -I -B tests.py old       (after curvature.py old)
Writes tests_old.txt, tests_old.json here and WORK/fund_forecasts_old.pkl (for money.py's fund risk scaling).
The functions take a tag so unseen.py can run the same code on the new export.
"""
import json
import pickle
import sys
import warnings

sys.path.insert(0, '/home/bhoult/programming/correlation/research/market_geometry')
import numpy as np  # noqa: E402

import b_common as B  # noqa: E402

c2eval = B.c2eval
HOLD = 22
MIN_TRAIN = 60
SCORE = {'2008-2019': ('2008-01-01', '2019-12-31'), '2020-2025-06': ('2020-01-01', '2025-06-27')}
BASE = ['lrv21', 'lrv63', 'avg_corr', 'absorption']
STATS = ['degconc', 'tridens', 'rho_edge', 'lcc']
MODELS = {
    'm1': BASE,
    'm2': BASE + STATS,
    'm3': BASE + STATS + ['ollivier'],
    'm4': BASE + ['forman'],
    'm_samal': BASE + ['forman_samal'],
    'm_pub_oll': BASE + ['pub_ollivier'],
    'm_pub_for': BASE + ['pub_forman'],
}
COMPARE = [('m2', 'm3', 'key: model 3 vs 2 (Ollivier beyond graph statistics)'),
           ('m1', 'm4', 'model 4 vs 1 (Forman, read as graph statistics)'),
           ('m1', 'm2', 'model 2 vs 1 (plain graph statistics)'),
           ('m1', 'm3', 'model 3 vs 1'),
           ('m1', 'm_samal', "Samal et al.'s Forman vs 1"),
           ('m1', 'm_pub_oll', 'published graph Ollivier vs 1'),
           ('m1', 'm_pub_for', 'published graph Forman vs 1')]
F_BASE = ['f_lrv21', 'f_lrv63', 'g_rho']
F_MODELS = {'m1': F_BASE, 'm2': F_BASE + ['g_logedges'], 'm3': F_BASE + ['g_logedges', 'g_oll_z'],
            'm4': F_BASE + ['g_for_z']}


def log_rv(r):
    return float(np.log(np.sqrt(np.mean(r * r))))


def max_dd(p):
    return float(np.max(1.0 - p / np.maximum.accumulate(p)))


def load(tag):
    warnings.simplefilter('ignore')
    z = np.load(B.WORK / f'curv_{tag}.npz')
    cv = {k: z[k] for k in z.files}
    with open(B.WORK / f'groups_{tag}.pkl', 'rb') as f:
        groups = pickle.load(f)
    days = cv['days']
    rebal = cv['rebal']
    T = len(days)
    P = B.spy_yahoo(days)
    rfd = B.rf(days)
    with np.errstate(all='ignore'):
        lr = np.full(T, np.nan)
        lr[1:] = np.log(P[1:] / P[:-1])
    K = len(rebal)
    M = {'date': np.array([c2eval.iso(days[t]) for t in rebal]), 't0': rebal, 'K': K}
    for nm in ('avg_corr', 'absorption'):
        M[nm] = cv[nm]
    for nm in STATS + ['ollivier', 'forman', 'forman_samal', 'm', 'T', 'deg_max']:
        M[nm] = cv[f'main_{nm}']
    for nm in ('ollivier', 'forman', 'forman_samal', 'm'):
        M[f'pub_{nm}'] = cv[f'pub_{nm}']
    for v in ('shift', 'factor', 'rewire', 'odd', 'even', 'long'):
        for nm in ('ollivier', 'forman', 'forman_samal', 'tridens', 'degconc'):
            M[f'{v}_{nm}'] = cv[f'{v}_{nm}']
    for nm in ('lrv21', 'lrv63', 'y_dir', 'y_risk', 'y_mdd'):
        M[nm] = np.full(K, np.nan)
    for k, t0 in enumerate(rebal):
        t0 = int(t0)
        r21, r63 = lr[t0 - 20:t0 + 1], lr[t0 - 62:t0 + 1]
        if np.isfinite(r21).all():
            M['lrv21'][k] = log_rv(r21)
        if np.isfinite(r63).all():
            M['lrv63'][k] = log_rv(r63)
        if t0 + HOLD <= T - 1:
            rr = lr[t0 + 2:t0 + HOLD + 1]
            if np.isfinite(rr).all() and np.isfinite(rfd[t0 + 2:t0 + HOLD + 1]).all():
                M['y_dir'][k] = float(np.log(P[t0 + HOLD] / P[t0 + 1]) - np.log1p(rfd[t0 + 2:t0 + HOLD + 1]).sum())
                M['y_risk'][k] = log_rv(rr)
                M['y_mdd'][k] = max_dd(P[t0 + 1:t0 + HOLD + 1])
    return M, groups, days, P, rfd, lr


def ols_fit(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def ols_pred(b, X):
    return b[0] + X @ b[1:]


def expanding(M, y, cols, min_train=MIN_TRAIN):
    """Forecast for every month k from an OLS fitted on months j with t0_j + 22 <= t0_k and finite inputs; NaN where
    fewer than min_train rows. Also the historical-mean forecast on the same rows."""
    X = np.column_stack([M[c] for c in cols])
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    t0 = M['t0']
    f = np.full(M['K'], np.nan)
    hm = np.full(M['K'], np.nan)
    for k in range(M['K']):
        if not np.isfinite(X[k]).all():
            continue
        tr = ok & (t0 + HOLD <= t0[k])
        if tr.sum() < min_train:
            continue
        b = ols_fit(X[tr], y[tr])
        f[k] = ols_pred(b, X[k][None, :])[0]
        hm[k] = y[tr].mean()
    return f, hm


def clark_west(y, f1, f2):
    """a_t = (y - f1)^2 - [(y - f2)^2 - (f1 - f2)^2]; with the MSE improvement and the out-of-sample R^2."""
    a = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)
    r = B.mean_test(a)
    imp = B.mean_test((y - f1) ** 2 - (y - f2) ** 2)
    r['mse_improvement'] = imp['mean']
    r['mse_improvement_ci90_nw'] = imp['ci90_nw']
    r['mse_improvement_ci90_boot'] = imp['ci90_boot']
    r['oos_r2_vs_smaller'] = float(1.0 - np.sum((y - f2) ** 2) / np.sum((y - f1) ** 2))
    return r


def b2_spy(M, period_masks):
    out = {}
    fc = {}
    for tgt in ('y_dir', 'y_risk', 'y_mdd'):
        y = M[tgt]
        fc[tgt] = {}
        for nm, cols in MODELS.items():
            fc[tgt][nm] = expanding(M, y, cols)
        out[tgt] = {}
        for pnm, pm in period_masks.items():
            out[tgt][pnm] = {}
            for a, b, lab in COMPARE:
                f1, f2 = fc[tgt][a][0], fc[tgt][b][0]
                m = pm & np.isfinite(y) & np.isfinite(f1) & np.isfinite(f2)
                if m.sum() < 3:
                    out[tgt][pnm][f'{a}->{b}'] = None
                    continue
                r = clark_west(y[m], f1[m], f2[m])
                hm = fc[tgt][b][1][m]
                r['oos_r2_vs_hist_mean_bigger'] = float(1 - np.sum((y[m] - f2[m]) ** 2) / np.sum((y[m] - hm) ** 2))
                r['oos_r2_vs_hist_mean_smaller'] = float(1 - np.sum((y[m] - f1[m]) ** 2) / np.sum((y[m] - hm) ** 2))
                r['first'] = M['date'][m][0]
                r['last'] = M['date'][m][-1]
                r['label'] = lab
                out[tgt][pnm][f'{a}->{b}'] = r
    return out, fc


# ---------------------------------------------------------------------------------------------------------------
# Funds

def fund_panel(M, groups, days, rfd, tag):
    """Rows (k, fund_row) with a group value: inputs at t0, targets over t0+2..t0+22 (NaN if incomplete)."""
    E = c2eval.load_export(B.EXPORTS[tag], arrays=('close',))
    tw = c2eval.twins(E)
    Fc = np.asarray(tw['F']['close'], np.float64)
    with np.errstate(all='ignore'):
        fr = np.full(Fc.shape, np.nan)
        fr[:, 1:] = np.log(Fc[:, 1:] / Fc[:, :-1])
    fr[~np.isfinite(fr) | (np.abs(fr) > 0.5)] = np.nan
    T = len(days)
    K = M['K']
    # each fund's group series over months, for the trailing standardization
    funds = sorted({f for g in groups for f in g})
    col = {f: i for i, f in enumerate(funds)}
    oll = np.full((len(funds), K), np.nan)
    frm = np.full((len(funds), K), np.nan)
    for k, g in enumerate(groups):
        for f, v in g.items():
            oll[col[f], k] = v['ollivier']
            frm[col[f], k] = v['forman']
    rows = []
    for k, g in enumerate(groups):
        t0 = int(M['t0'][k])
        for f, v in g.items():
            row = {'k': k, 'f': f, 'ticker': str(tw['fund_ticker'][f]) if 'fund_ticker' in tw else str(f),
                   'g_rho': v['rho_pairs'], 'g_logedges': float(np.log(v['n_edges'])), 'g_size': v['size'],
                   'g_oll_z': B.trailing_z(oll[col[f]], k), 'g_for_z': B.trailing_z(frm[col[f]], k)}
            r21, r63 = fr[f, t0 - 20:t0 + 1], fr[f, t0 - 62:t0 + 1]
            row['f_lrv21'] = log_rv(r21) if np.isfinite(r21).all() else np.nan
            row['f_lrv63'] = log_rv(r63) if np.isfinite(r63).all() else np.nan
            row['f_rv63'] = float(np.sqrt(np.mean(r63 * r63))) if np.isfinite(r63).all() else np.nan
            for nm in ('y_dir', 'y_risk', 'y_mdd'):
                row[nm] = np.nan
            if t0 + HOLD <= T - 1:
                rr = fr[f, t0 + 2:t0 + HOLD + 1]
                if np.isfinite(rr).all() and np.isfinite(Fc[f, t0 + 1]):
                    pp = Fc[f, t0 + 1:t0 + HOLD + 1]
                    row['y_dir'] = float(np.log(pp[-1] / pp[0]) - np.log1p(rfd[t0 + 2:t0 + HOLD + 1]).sum())
                    row['y_risk'] = log_rv(rr)
                    row['y_mdd'] = max_dd(pp)
            rows.append(row)
    keys = rows[0].keys()
    Pn = {key: np.array([r[key] for r in rows]) for key in keys}
    return Pn


def fund_expanding(Pn, M, tgt, cols, min_months=MIN_TRAIN):
    X = np.column_stack([Pn[c] for c in cols])
    y = Pn[tgt]
    k = Pn['k']
    t0r = M['t0'][k]
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    f = np.full(len(y), np.nan)
    for kk in np.unique(k):
        here = (k == kk) & np.isfinite(X).all(1)
        if not here.any():
            continue
        tr = ok & (t0r + HOLD <= M['t0'][kk])
        if len(np.unique(k[tr])) < min_months:
            continue
        b = ols_fit(X[tr], y[tr])
        f[here] = ols_pred(b, X[here])
    return f


def b2_funds(Pn, M, period_masks):
    out = {}
    fc = {}
    for tgt in ('y_dir', 'y_risk', 'y_mdd'):
        fc[tgt] = {nm: fund_expanding(Pn, M, tgt, cols) for nm, cols in F_MODELS.items()}
        y = Pn[tgt]
        out[tgt] = {}
        for pnm, pm in period_masks.items():
            out[tgt][pnm] = {}
            for a, b in (('m2', 'm3'), ('m1', 'm4'), ('m1', 'm2'), ('m1', 'm3')):
                f1, f2 = fc[tgt][a], fc[tgt][b]
                ok = np.isfinite(y) & np.isfinite(f1) & np.isfinite(f2) & pm[Pn['k']]
                aa = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)
                ks = np.unique(Pn['k'][ok])
                if len(ks) < 10:
                    out[tgt][pnm][f'{a}->{b}'] = None
                    continue
                per_month = np.array([aa[ok & (Pn['k'] == kk)].mean() for kk in ks])
                r = B.mean_test(per_month)
                imp = np.array([((y - f1) ** 2 - (y - f2) ** 2)[ok & (Pn['k'] == kk)].mean() for kk in ks])
                ri = B.mean_test(imp)
                r['mse_improvement'] = ri['mean']
                r['mse_improvement_ci90_nw'] = ri['ci90_nw']
                r['oos_r2_vs_smaller'] = float(1 - np.sum((y[ok] - f2[ok]) ** 2) / np.sum((y[ok] - f1[ok]) ** 2))
                r['fund_months'] = int(ok.sum())
                r['first'] = M['date'][ks[0]]
                r['last'] = M['date'][ks[-1]]
                out[tgt][pnm][f'{a}->{b}'] = r
    return out, fc


# ---------------------------------------------------------------------------------------------------------------
# B1, checks, drawdowns

def r2_resid(y, X):
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    b = ols_fit(X[ok], y[ok])
    e = y[ok] - ols_pred(b, X[ok])
    return {'r2': float(1 - e.var() / y[ok].var()), 'resid_sd_ratio': float(e.std() / y[ok].std()), 'n': int(ok.sum())}


def b1(M):
    out = {}
    series = {'ollivier': 'Ollivier (5% graph)', 'forman': 'augmented Forman (5% graph)',
              'forman_samal': "Samal et al.'s Forman (5% graph)", 'pub_ollivier': 'Ollivier (published graph)',
              'pub_forman': 'augmented Forman (published graph)', 'pub_forman_samal': "Samal et al.'s Forman (published graph)"}
    Xa = np.column_stack([M['avg_corr'], M['absorption'], M['lrv63']])
    Xb = np.column_stack([M[c] for c in STATS])
    m08 = (M['date'] >= '2008-01-01') & (M['date'] <= '2019-12-31')
    for s, lab in series.items():
        y = M[s]
        out[s] = {'label': lab, 'yardsticks_all': r2_resid(y, Xa), 'stats_all': r2_resid(y, Xb),
                  'yardsticks_2008_2019': r2_resid(np.where(m08, y, np.nan), Xa),
                  'stats_2008_2019': r2_resid(np.where(m08, y, np.nan), Xb),
                  'corr_avg_corr': float(np.corrcoef(y[np.isfinite(y)], M['avg_corr'][np.isfinite(y)])[0, 1])}
    return out


def checks(M):
    out = {}
    for s in ('ollivier', 'forman'):
        o = {}
        real = M[s]
        for v in ('shift', 'factor', 'rewire'):
            nul = M[f'{v}_{s}']
            ok = np.isfinite(real) & np.isfinite(nul)
            d = real[ok] - nul[ok]
            r = B.mean_test(d)
            o[v] = {'real_mean': float(real[ok].mean()), 'null_mean': float(nul[ok].mean()), 'diff_mean': r['mean'],
                    't': r['t'], 'share_real_above': float(np.mean(d > 0)), 'n': int(ok.sum()),
                    'corr_real_null': float(np.corrcoef(real[ok], nul[ok])[0, 1])}
        a, b = M[f'odd_{s}'], M[f'even_{s}']
        ok = np.isfinite(a) & np.isfinite(b)
        o['odd_even_corr'] = float(np.corrcoef(a[ok], b[ok])[0, 1])
        o['odd_even_corr_changes'] = float(np.corrcoef(np.diff(a[ok]), np.diff(b[ok]))[0, 1])
        lg = M[f'long_{s}']
        ok = np.isfinite(lg) & np.isfinite(real)
        o['corr_63_252'] = float(np.corrcoef(real[ok], lg[ok])[0, 1])
        out[s] = o
    tr = M['tridens']
    o = {}
    for v in ('rewire',):
        nul = M[f'{v}_tridens']
        o['tridens_real_mean'] = float(np.nanmean(tr))
        o['tridens_rewired_mean'] = float(np.nanmean(nul))
    out['triangles'] = o
    return out


def drawdowns(M, days, P, lo='2003-01-01', hi='2025-06-27', thresh=0.15):
    d0, d1 = c2eval.day_index(days, lo), c2eval.day_index(days, hi)
    p = P[d0:d1 + 1]
    eps = []
    peak_i, peak_v = 0, p[0]
    i = 1
    while i < len(p):
        if p[i] >= peak_v:
            peak_i, peak_v = i, p[i]
            i += 1
            continue
        if 1 - p[i] / peak_v >= thresh:
            j = i
            while j < len(p) and p[j] < peak_v:
                j += 1
            tr = peak_i + int(np.argmin(p[peak_i:j]))
            eps.append((d0 + peak_i, d0 + tr, float(1 - p[tr] / peak_v), d0 + j if j < len(p) else None))
            if j >= len(p):
                break
            peak_i, peak_v = j, p[j]
            i = j + 1
            continue
        i += 1
    t0 = M['t0']
    zs = {s: np.array([B.trailing_z(M[s], k) for k in range(M['K'])]) for s in ('ollivier', 'forman')}
    out = []
    for pk, trg, depth, rec in eps:
        kp = int(np.searchsorted(t0, pk, side='right') - 1)
        kt = int(np.searchsorted(t0, trg, side='right') - 1)
        row = {'peak': c2eval.iso(days[pk]), 'trough': c2eval.iso(days[trg]), 'depth': depth,
               'recovered': c2eval.iso(days[rec]) if rec is not None else None, 'peak_month': M['date'][kp],
               'trough_month': M['date'][kt]}
        for s, sign in (('ollivier', +1), ('forman', -1)):
            z = zs[s]
            row[s] = {'z_peak_m6': float(z[kp - 6]), 'z_peak_m3': float(z[kp - 3]), 'z_peak': float(z[kp]),
                      'z_trough': float(z[kt])}
            first = None
            for k in range(max(0, kp - 12), kt + 1):
                if np.isfinite(z[k]) and sign * z[k] > 1:
                    first = k - kp
                    break
            row[s]['first_beyond_1_months_from_peak'] = first
        out.append(row)
    return out


def spy_check(days, P):
    """Yahoo SPY daily log returns against the store's SPY, 2005-03..2025-06 (reported)."""
    mt = B.mt_common()
    Ps = mt.spy_close('old', days)
    a, b = c2eval.day_index(days, '2005-03-01'), c2eval.day_index(days, '2025-06-27')
    ry = np.diff(np.log(P[a:b + 1]))
    rs = np.diff(np.log(Ps[a:b + 1]))
    ok = np.isfinite(ry) & np.isfinite(rs)
    d = ry[ok] - rs[ok]
    return {'corr': float(np.corrcoef(ry[ok], rs[ok])[0, 1]), 'max_abs_diff': float(np.abs(d).max()),
            'mean_diff_bps_per_day': float(1e4 * d.mean()), 'days': int(ok.sum())}


def period_masks(M, periods):
    return {nm: (M['date'] >= lo) & (M['date'] <= hi) for nm, (lo, hi) in periods.items()}


def run(tag, periods):
    M, groups, days, P, rfd, lr = load(tag)
    res = {}
    pm = period_masks(M, periods)
    res['b2_spy'], fc = b2_spy(M, pm)
    Pn = fund_panel(M, groups, days, rfd, tag)
    res['b2_funds'], ffc = b2_funds(Pn, M, pm)
    return M, groups, days, P, rfd, res, fc, Pn, ffc


def fmt_cw(r):
    if r is None:
        return 'n/a (too few months)'
    return (f"CW t {r['t']:+.2f} (p {r['p']:.4f}, boot p {r['p_boot']:.4f}); MSE improvement {r['mse_improvement']:+.3e} "
            f"(90% NW {r['mse_improvement_ci90_nw'][0]:+.2e}..{r['mse_improvement_ci90_nw'][1]:+.2e}); "
            f"OOS R2 vs smaller {100 * r['oos_r2_vs_smaller']:+.2f}%; n {r['n']}")


def main():
    tag = 'old'
    M, groups, days, P, rfd, res, fc, Pn, ffc = run(tag, SCORE)
    res['b1'] = b1(M)
    res['checks'] = checks(M)
    res['drawdowns'] = drawdowns(M, days, P)
    res['spy_check'] = spy_check(days, P)
    # forecasts for money.py (fund risk scaling): every fund-month with inputs
    with open(B.WORK / f'fund_forecasts_{tag}.pkl', 'wb') as f:
        pickle.dump({'k': Pn['k'], 'f': Pn['f'], 'm3': ffc['y_risk']['m3'], 'm2': ffc['y_risk']['m2'],
                     'rv63': Pn['f_rv63'], 'date': M['date'][Pn['k']]}, f)
    # the monthly series for the explainer page
    res['monthly'] = {k: [None if not np.isfinite(x) else float(x) for x in M[k]]
                      for k in ('ollivier', 'forman', 'avg_corr', 'lrv21', 'y_risk', 'y_dir', 'pub_ollivier',
                                'shift_ollivier', 'factor_ollivier', 'rewire_ollivier')}
    res['monthly']['date'] = M['date'].tolist()
    res['monthly']['f3_risk'] = [None if not np.isfinite(x) else float(x) for x in fc['y_risk']['m3'][0]]
    res['monthly']['f2_risk'] = [None if not np.isfinite(x) else float(x) for x in fc['y_risk']['m2'][0]]
    res['monthly']['f3_dir'] = [None if not np.isfinite(x) else float(x) for x in fc['y_dir']['m3'][0]]
    res['monthly']['f2_dir'] = [None if not np.isfinite(x) else float(x) for x in fc['y_dir']['m2'][0]]
    i0, i1 = c2eval.day_index(days, '2003-01-01'), c2eval.day_index(days, '2025-06-27')
    pp = P[i0:i1 + 1]
    res['spy_drawdown_daily'] = {'date': [c2eval.iso(d) for d in days[i0:i1 + 1:5]],
                                 'dd': (pp / np.maximum.accumulate(pp) - 1)[::5].round(4).tolist()}
    (B.HERE / f'tests_{tag}.json').write_text(json.dumps(res, indent=1, default=float))
    report(res, Pn, M)


def report(res, Pn, M):
    L = []
    p = L.append
    sc = res['spy_check']
    p(f"SPY: Yahoo adjusted close vs the store over 2005-03..2025-06: daily log-return corr {sc['corr']:.5f}, max |diff| "
      f"{sc['max_abs_diff']:.4f}, mean diff {sc['mean_diff_bps_per_day']:+.2f} bps/day ({sc['days']} days)")
    p('\n## B1: what curvature is made of (R^2; residual sd / series sd)')
    for s, r in res['b1'].items():
        p(f"{r['label']}: on avg corr + absorption + SPY 63-day vol R2 {r['yardsticks_all']['r2']:.3f} "
          f"(resid {r['yardsticks_all']['resid_sd_ratio']:.2f}; 2008-2019 R2 {r['yardsticks_2008_2019']['r2']:.3f}); "
          f"on plain graph statistics R2 {r['stats_all']['r2']:.3f} (resid {r['stats_all']['resid_sd_ratio']:.2f}; "
          f"2008-2019 {r['stats_2008_2019']['r2']:.3f}); corr with avg corr {r['corr_avg_corr']:+.2f}")
    p('\n## Checks')
    for s in ('ollivier', 'forman'):
        c = res['checks'][s]
        for v in ('shift', 'factor', 'rewire'):
            x = c[v]
            p(f"{s} real vs {v}: real mean {x['real_mean']:+.4f}, null mean {x['null_mean']:+.4f}, diff {x['diff_mean']:+.4f} "
              f"(t {x['t']:+.1f}), real above null in {100 * x['share_real_above']:.0f}% of {x['n']} months; corr(real, null) {x['corr_real_null']:+.2f}")
        p(f"{s} stability: odd vs even days corr {c['odd_even_corr']:+.3f} (changes {c['odd_even_corr_changes']:+.3f}); "
          f"63-day vs 252-day corr {c['corr_63_252']:+.3f}")
    t = res['checks']['triangles']
    p(f"triangle density T/m: real {t['tridens_real_mean']:.2f}, rewired {t['tridens_rewired_mean']:.2f}")
    p('\n## B2: SPY (Clark-West, expanding OLS)')
    for tgt, lab in (('y_dir', 'direction (excess return)'), ('y_risk', 'risk (log realized vol)'), ('y_mdd', 'max drawdown (reported)')):
        for pnm, d in res['b2_spy'][tgt].items():
            for key, r in d.items():
                p(f"{lab} {pnm} [{r['first']}..{r['last']}] {r['label']}: {fmt_cw(r)}; OOS R2 vs hist mean "
                  f"{100 * r['oos_r2_vs_hist_mean_bigger']:+.1f}% (smaller {100 * r['oos_r2_vs_hist_mean_smaller']:+.1f}%)")
    p('\n## B2: twin funds (pooled panel, Clark-West averaged per month)')
    p(f"panel rows {len(Pn['k'])}, funds {len(np.unique(Pn['f']))}, months with a group {len(np.unique(Pn['k']))}")
    for tgt in ('y_dir', 'y_risk', 'y_mdd'):
        for pnm, d in res['b2_funds'][tgt].items():
            for key, r in d.items():
                extra = '' if r is None else f" [{r['first']}..{r['last']}, {r['fund_months']} fund-months]"
                p(f"funds {tgt} {pnm} {key}{extra}: {fmt_cw(r)}")
    p('\n## Drawdowns of SPY >= 15% (curvature z vs its trailing 60 month-ends; Ollivier crisis side +, Forman -)')
    for r in res['drawdowns']:
        o, f = r['ollivier'], r['forman']
        p(f"{r['peak']} -> {r['trough']} ({100 * r['depth']:.1f}%): Ollivier z at peak-6m {o['z_peak_m6']:+.2f}, -3m {o['z_peak_m3']:+.2f}, "
          f"peak {o['z_peak']:+.2f}, trough {o['z_trough']:+.2f}, first beyond +1: {o['first_beyond_1_months_from_peak']} months from peak; "
          f"Forman z {f['z_peak_m6']:+.2f}, {f['z_peak_m3']:+.2f}, {f['z_peak']:+.2f}, {f['z_trough']:+.2f}, first beyond -1: "
          f"{f['first_beyond_1_months_from_peak']}")
    txt = '\n'.join(L)
    print(txt)
    (B.HERE / 'tests_old.txt').write_text(txt + '\n')


if __name__ == '__main__':
    main()
