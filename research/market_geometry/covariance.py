"""B5: the covariance manifold M. Does the representation matter? Log-Euclidean against Euclidean combinations of
Ledoit-Wolf covariances over 21, 63 and 252 days, judged by the realized variance of the capped minimum-variance
portfolio each forecast builds (MV-LW's 2% cap and solver), on MV-LW's own list of stocks.

    python -I -B covariance.py old       # every month on the old export; the tests over 2006-2019 and 2020-2025-06
    python -I -B covariance.py new       # every month on the new export (run by unseen.py's step, once, at the end)
Writes WORK/cov_<tag>.npz (per month, all 66 + 66 candidates) and, for old, covariance_old.txt / .json here.
"""
import os

os.environ['OMP_NUM_THREADS'] = '1'          # 3 worker processes x 1 BLAS thread = the 3-core budget
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/market_geometry')
import numpy as np  # noqa: E402

import b_common as B  # noqa: E402

c2eval = B.c2eval
TAG = sys.argv[1] if len(sys.argv) > 1 else 'old'
NPROC = 3
GRID = [(a1 / 10, a2 / 10, (10 - a1 - a2) / 10) for a1 in range(11) for a2 in range(11 - a1)]
IDX_LW = GRID.index((0.0, 0.0, 1.0))
HOLD = 22             # bought at the close of t0 + 1, held to the close of t0 + 22
MIN_HIST = 12
G = {}


def setup():
    G['cp'] = B.cp_common()
    G['est'] = B.cp_estimators()
    E = c2eval.load_export(B.EXPORTS[TAG], arrays=('close', 'dollar_volume'))
    tw = c2eval.twins(E)
    T = len(E['days'])
    rebal = tw['rebal']
    ks = [k for k in range(len(rebal)) if rebal[k] >= G['cp'].DATA_W and rebal[k] + HOLD <= T - 1]
    uni, big = G['cp'].universe(E, tw, set(ks))
    G.update(E=E, tw=tw, ks=ks, big=big, T=T)


def eig_fn(S, fn):
    ev, V = np.linalg.eigh(S)
    return (V * fn(ev)) @ V.T


def qlike(Sh, St):
    """tr(Sh^-1 St) - log det(Sh^-1 St) - N."""
    N = Sh.shape[0]
    Y = np.linalg.solve(Sh, St)
    s1, ld_t = np.linalg.slogdet(St)
    s2, ld_h = np.linalg.slogdet(Sh)
    return float(np.trace(Y) - (ld_t - ld_h) - N)


def month(k):
    warnings.simplefilter('ignore')
    cp, est, E = G['cp'], G['est'], G['E']
    close = E['close']
    t0 = int(G['tw']['rebal'][k])
    sel = np.nonzero(G['big'][k])[0]
    R = cp.window_logret(np.asarray(close[sel, t0 - cp.WIN:t0 + 1], np.float64))       # returns t0-251..t0
    okr = np.isfinite(R).sum(1) >= cp.MIN_RET
    sel, R = sel[okr], R[okr]
    n = len(sel)
    S = {w: est.ledoit_wolf_cc(cp.demean_fill(R[:, -w:]))[0] for w in (21, 63, 252)}
    logs = {w: eig_fn(S[w], np.log) for w in (21, 63, 252)}
    A = est.ledoit_wolf_cc(cp.demean_fill(R[:, -84:-21]))[0]                               # 63 days ending t0-21
    # the next 21 days: fixed weights bought at the close of t0+1, simple returns of forward-filled closes
    Pw = cp.ffill_rows(np.asarray(close[sel, t0 - 300:t0 + HOLD + 1], np.float64))[:, 301:]   # closes t0+1..t0+22
    assert Pw.shape[1] == HOLD and np.isfinite(Pw).all(), (k, t0)
    Rf = Pw[:, 1:] / Pw[:, :-1] - 1.0                                                       # days t0+2..t0+22
    St = est.ledoit_wolf_cc(cp.demean_fill(cp.window_logret(np.asarray(close[sel, t0 + 1:t0 + HOLD + 1], np.float64))))[0]

    def evaluate(Sig, w0=None):
        w, info = est.minvar_capped(Sig, cp.CAP, w0=w0)
        pr = w @ Rf
        return w, float(pr @ pr), qlike(Sig, St), max(info['kkt_primal'], info['kkt_dual'])

    out = {'k': k, 't0': t0, 'n': n, 'loss_e': np.empty(len(GRID)), 'loss_le': np.empty(len(GRID)),
           'q_e': np.empty(len(GRID)), 'q_le': np.empty(len(GRID))}
    kkt = 0.0
    wp = None
    for g_i, g in enumerate(GRID):
        Sig = g[0] * S[21] + g[1] * S[63] + g[2] * S[252]
        wp, out['loss_e'][g_i], out['q_e'][g_i], kv = evaluate(Sig, wp)
        kkt = max(kkt, kv)
        if g_i == IDX_LW:
            w_lw = wp.copy()
    wp = None
    for g_i, g in enumerate(GRID):
        if g in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):     # the vertices are the same matrix
            out['loss_le'][g_i], out['q_le'][g_i] = out['loss_e'][g_i], out['q_e'][g_i]
            continue
        Sig = eig_fn(g[0] * logs[21] + g[1] * logs[63] + g[2] * logs[252], np.exp)
        wp, out['loss_le'][g_i], out['q_le'][g_i], kv = evaluate(Sig, wp)
        kkt = max(kkt, kv)
    third = 1.0 / 3.0
    _, out['eq_e_loss'], out['eq_e_q'], _ = evaluate(third * (S[21] + S[63] + S[252]))
    _, out['eq_le_loss'], out['eq_le_q'], _ = evaluate(eig_fn(third * (logs[21] + logs[63] + logs[252]), np.exp))
    # affine-invariant step past B along the geodesic from A: B A^-1 B = B^1/2 (B^1/2 A^-1 B^1/2) B^1/2
    Bh = eig_fn(S[63], np.sqrt)
    Ainv = eig_fn(A, lambda e: 1.0 / e)
    M = Bh @ Ainv @ Bh
    Fa = Bh @ (0.5 * (M + M.T)) @ Bh
    Fa = 0.5 * (Fa + Fa.T)
    ev = np.linalg.eigvalsh(Fa)
    out['aff_mineig'], out['aff_cond'] = float(ev[0]), float(ev[-1] / ev[0]) if ev[0] > 0 else float('inf')
    if ev[0] > 0:
        _, out['aff_loss'], out['aff_q'], _ = evaluate(Fa)
    else:
        out['aff_loss'], out['aff_q'] = float('nan'), float('nan')
    out['kkt_max'] = kkt
    out['w_lw'] = w_lw
    out['sel'] = sel
    return out


def mv_lw_check(res, days):
    """MV-LW's saved weights (research/correlation_portfolios, read only) against LW-252 here, per month."""
    p = c2eval.WORK / 'correlation_portfolios' / f'series_{TAG}.npz'
    if not p.exists():
        return None
    z = np.load(p)
    t0s = list(z['t0'])
    port = list(z['ports']).index('MV-LW')
    diffs = []
    for r in res:
        if r['t0'] not in t0s:
            continue
        i = t0s.index(r['t0'])
        a, b = z['w_offs'][i], z['w_offs'][i + 1]
        idx, w = z['w_idx'][a:b], z['w'][a:b, port]
        if not np.array_equal(idx, r['sel']):
            diffs.append(np.inf)
            continue
        diffs.append(float(np.abs(w - r['w_lw']).max()))
    return {'months': len(diffs), 'max_abs_weight_diff': float(np.max(diffs)) if diffs else float('nan'),
            'lists_differ': int(np.sum(np.isinf(diffs)))}


def choose(t0s, loss):
    """Per month: the grid index with the lowest mean loss over months whose target ended by t0 (ties: first), or
    -1 (equal weights) before 12 such months exist. Also the history count."""
    K = len(t0s)
    ch = np.full(K, -1)
    nh = np.zeros(K, int)
    for a in range(K):
        hist = np.nonzero(t0s + HOLD <= t0s[a])[0]
        nh[a] = len(hist)
        if len(hist) >= MIN_HIST:
            ch[a] = int(np.argmin(loss[hist].mean(0)))
    return ch, nh


def main():
    setup()
    days = G['E']['days']
    B.log(TAG, 'months', len(G['ks']), c2eval.iso(days[G['tw']['rebal'][G['ks'][0]]]), '..',
          c2eval.iso(days[G['tw']['rebal'][G['ks'][-1]]]))
    res = []
    ctx = mp.get_context('fork')
    with ctx.Pool(NPROC) as pool:
        for r in pool.imap(month, G['ks'], chunksize=1):
            res.append(r)
            if len(res) % 20 == 0:
                B.log(f"{c2eval.iso(days[r['t0']])} n {r['n']} kkt {r['kkt_max']:.0e}")
    chk = mv_lw_check(res, days)
    arr = {'t0': np.array([r['t0'] for r in res]), 'n': np.array([r['n'] for r in res]), 'days': days,
           'grid': np.array(GRID)}
    for key in ('loss_e', 'loss_le', 'q_e', 'q_le'):
        arr[key] = np.stack([r[key] for r in res])
    for key in ('eq_e_loss', 'eq_e_q', 'eq_le_loss', 'eq_le_q', 'aff_loss', 'aff_q', 'aff_mineig', 'aff_cond', 'kkt_max'):
        arr[key] = np.array([r[key] for r in res])
    np.savez(B.WORK / f'cov_{TAG}.npz', **arr, mvlw_check=json.dumps(chk))
    B.log('saved', B.WORK / f'cov_{TAG}.npz', 'MV-LW check', chk)
    if TAG == 'old':
        report(arr, chk)


def series(arr):
    """The scored series: chosen-weight losses for both representations, LW-252, the equal-weight fallbacks."""
    t0s = arr['t0']
    ch_e, nh = choose(t0s, arr['loss_e'])
    ch_le, _ = choose(t0s, arr['loss_le'])
    K = len(t0s)
    s = {'t0': t0s, 'nh': nh, 'ch_e': ch_e, 'ch_le': ch_le}
    s['e'] = np.where(ch_e >= 0, arr['loss_e'][np.arange(K), np.maximum(ch_e, 0)], arr['eq_e_loss'])
    s['le'] = np.where(ch_le >= 0, arr['loss_le'][np.arange(K), np.maximum(ch_le, 0)], arr['eq_le_loss'])
    s['qe'] = np.where(ch_e >= 0, arr['q_e'][np.arange(K), np.maximum(ch_e, 0)], arr['eq_e_q'])
    s['qle'] = np.where(ch_le >= 0, arr['q_le'][np.arange(K), np.maximum(ch_le, 0)], arr['eq_le_q'])
    s['lw'] = arr['loss_e'][:, IDX_LW]
    s['qlw'] = arr['q_e'][:, IDX_LW]
    s['aff'] = arr['aff_loss']
    s['qaff'] = arr['aff_q']
    s['scored'] = nh >= MIN_HIST
    return s


def compare(s, m):
    """All comparisons over the months mask m (losses: positive difference = the second named is better)."""
    out = {'n': int(m.sum())}
    out['vol'] = {k: float(np.sqrt(s[k][m].mean() * 12)) for k in ('e', 'le', 'lw', 'aff')}   # annualized from monthly
    out['euc_minus_le'] = B.mean_test(s['e'][m] - s['le'][m])
    out['lw_minus_le'] = B.mean_test(s['lw'][m] - s['le'][m])
    out['lw_minus_e'] = B.mean_test(s['lw'][m] - s['e'][m])
    out['lw_minus_aff'] = B.mean_test(s['lw'][m] - s['aff'][m])
    qm = m & np.isfinite(s['qe']) & np.isfinite(s['qle']) & np.isfinite(s['qlw']) & np.isfinite(s['qaff'])
    out['q_months'] = int(qm.sum())          # QLIKE's target is undefined when a listed stock has no forward returns
    out['q'] = {k: float(np.mean(s[k][qm])) for k in ('qe', 'qle', 'qlw', 'qaff')}
    out['q_euc_minus_le'] = B.mean_test(s['qe'][qm] - s['qle'][qm])
    out['share_le_better'] = float(np.mean(s['le'][m] < s['e'][m]))
    out['share_e_beats_le_by_any'] = float(np.mean(s['e'][m] < s['le'][m]))
    return out


def report(arr, chk):
    days = arr['days']
    s = series(arr)
    dates = np.array([c2eval.iso(days[t]) for t in s['t0']])
    res = {'first_month': dates[0], 'first_scored': dates[s['scored']][0], 'mvlw_check': chk,
           'kkt_max': float(arr['kkt_max'].max()), 'aff_mineig_min': float(np.nanmin(arr['aff_mineig'])),
           'aff_cond_median': float(np.median(arr['aff_cond'])), 'aff_cond_max': float(np.max(arr['aff_cond']))}
    periods = {'2006-2019': ('2006-01-01', '2019-12-31'), '2020-2025-06': ('2020-01-01', '2025-06-27'),
               '2006-2009': ('2006-01-01', '2009-12-31'), '2010-2014': ('2010-01-01', '2014-12-31'),
               '2015-2019': ('2015-01-01', '2019-12-31')}
    for nm, (lo, hi) in periods.items():
        m = s['scored'] & (dates >= lo) & (dates <= hi)
        res[nm] = compare(s, m)
    # the chosen weights over the primary period
    m = s['scored'] & (dates >= '2006-01-01') & (dates <= '2019-12-31')
    for key, nm in (('ch_e', 'euclidean'), ('ch_le', 'log_euclidean')):
        c = s[key][m]
        vals, cnt = np.unique(c, return_counts=True)
        res[f'weights_{nm}'] = {str(tuple(GRID[v])): int(n) for v, n in zip(vals, cnt)}
        res[f'weights_{nm}_mean'] = np.array([GRID[v] for v in c]).mean(0).tolist()
        res[f'weights_{nm}_last'] = list(GRID[s[key][m][-1]])
    res['monthly'] = {'date': dates.tolist(), 'scored': s['scored'].tolist(),
                      'vol_e': np.sqrt(s['e'] * 12).tolist(), 'vol_le': np.sqrt(s['le'] * 12).tolist(),
                      'vol_lw': np.sqrt(s['lw'] * 12).tolist(), 'vol_aff': np.sqrt(np.nan_to_num(s['aff']) * 12).tolist()}
    p = res['2006-2019']['euc_minus_le']
    res['pass'] = bool(p['t'] >= 3)
    (B.HERE / 'covariance_old.json').write_text(json.dumps(res, indent=1))
    L = [f"B5 months {dates[0]} .. {dates[-1]}; scored from {res['first_scored']} (12 months of loss history); "
         f"MV-LW check {chk}; worst KKT {res['kkt_max']:.1e}; affine step min eigenvalue {res['aff_mineig_min']:.2e}, "
         f"condition median {res['aff_cond_median']:.3g} max {res['aff_cond_max']:.3g}"]
    for nm in periods:
        r = res[nm]
        d = r['euc_minus_le']
        L.append(f"\n{nm} ({r['n']} months): annualized realized vol of the MV portfolio: Euclidean {100 * r['vol']['e']:.2f}%, "
                 f"log-Euclidean {100 * r['vol']['le']:.2f}%, LW-252 {100 * r['vol']['lw']:.2f}%, affine step {100 * r['vol']['aff']:.2f}%")
        for key, lab in (('euc_minus_le', 'Euclidean - log-Euclidean'), ('lw_minus_le', 'LW-252 - log-Euclidean'),
                         ('lw_minus_e', 'LW-252 - Euclidean'), ('lw_minus_aff', 'LW-252 - affine step'),
                         ('q_euc_minus_le', 'QLIKE Euclidean - log-Euclidean')):
            x = r[key]
            L.append(f"  {lab}: mean {x['mean']:+.3e} (NW se {x['se']:.2e}), t {x['t']:+.2f}, p {x['p']:.4f}, boot p {x['p_boot']:.4f}, "
                     f"90% NW {x['ci90_nw'][0]:+.2e}..{x['ci90_nw'][1]:+.2e}, boot {x['ci90_boot'][0]:+.2e}..{x['ci90_boot'][1]:+.2e}")
        L.append(f"  QLIKE means ({r['q_months']} months with a defined target): Euclidean {r['q']['qe']:.2f}, log-Euclidean {r['q']['qle']:.2f}, LW-252 {r['q']['qlw']:.2f}, "
                 f"affine {r['q']['qaff']:.2f}; log-Euclidean had the lower realized variance in {100 * r['share_le_better']:.0f}% of months")
    L.append(f"\nChosen weights 2006-2019 (w21, w63, w252): Euclidean mean {np.round(res['weights_euclidean_mean'], 2).tolist()}, "
             f"last {res['weights_euclidean_last']}; log-Euclidean mean {np.round(res['weights_log_euclidean_mean'], 2).tolist()}, "
             f"last {res['weights_log_euclidean_last']}")
    L.append(f"Pass (Euclidean - log-Euclidean, t >= 3, 2006-2019): {'PASS' if res['pass'] else 'FAIL'}")
    txt = '\n'.join(L)
    print(txt)
    (B.HERE / 'covariance_old.txt').write_text(txt + '\n')


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[2] == 'report':     # rebuild the report from the saved months
        z = np.load(B.WORK / f'cov_{TAG}.npz')
        report({k: z[k] for k in z.files}, json.loads(str(z['mvlw_check'])))
    else:
        main()
