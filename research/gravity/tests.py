"""Gravity, step 4: S (the signal test), M (the money test), Romano-Wolf within each stage, Holm across the six, and
everything the plan reports for every candidate.

    python -I -B tests.py old|new|fwd

Daily series (decision day t, from D0 or the period's start, to the third-last day):
- returns of a position decided at the close of t: open-to-open t+1 -> t+2 (primary), the next close t+1 -> t+2, the
  same close t -> t+1 (simple returns from the store's prices, both days valid);
- S: a cross-sectional OLS of the centred rank of the return on the centred ranks of the force, its benchmark's force,
  B's 1-day and 5-day log returns (universe stocks with all five; ranks within that sample; >= 50 stocks), the force's
  coefficient; raw: the same without the benchmark; IC: the rank correlation of force and return;
- M: long the top fifth, short the bottom fifth by force (universe stocks with a force), $1 a side, entered at the
  next open and held one day; gross return, turnover sum |w_t - w_{t-1}|, net = gross - c x turnover.
Statistics per period: circular block bootstrap of dates (21-day blocks, 10,000 draws, the same draws for every
series), t = estimate / bootstrap sd, one-sided normal p; Newey-West (10 lags) beside it. Romano-Wolf step-down
max-t within each stage's candidates for S and for M (net at 5 bps); Holm across the six stage p's at 0.0027, each
also needing t >= 3. Writes BIG/series_<tag>.npz, out/tests_<tag>.json / .txt.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import data  # noqa: E402
import kern  # noqa: E402
np = K.np

VARS = K.variants()
KEYS = [v['key'] for v in VARS]
EXEC = ('oo', 'nc', 'sc')
MIN_N = 50


def periods_for(tag):
    if tag == 'old':
        return K.PERIODS_OLD
    if tag == 'new':
        return {'unseen': K.UNSEEN}
    return {'forward': (K.FWD_START, '2100-01-01')}


def first_day(tag, P):
    if tag == 'old':
        return P['d0']
    if tag == 'new':
        return max(P['d0'], K.day_of(P['days'], K.UNSEEN[0]))
    return max(P['d0'], K.day_of(P['days'], K.FWD_START))


def returns(P):
    lnO, lnC = np.asarray(P['lnO'], np.float64), np.asarray(P['lnC'], np.float64)
    R, T = lnC.shape
    rets = np.full((3, R, T), np.nan)
    rets[0, :, :T - 2] = np.expm1(lnO[:, 2:] - lnO[:, 1:-1])
    rets[1, :, :T - 2] = np.expm1(lnC[:, 2:] - lnC[:, 1:-1])
    rets[2, :, :T - 1] = np.expm1(lnC[:, 1:] - lnC[:, :-1])
    r1 = np.full((R, T), np.nan)
    r5 = np.full((R, T), np.nan)
    r1[:, 1:] = lnC[:, 1:] - lnC[:, :-1]
    r5[:, 5:] = lnC[:, 5:] - lnC[:, :-5]
    return rets, r1, r5


def fm(y, xs, M):
    """Per column: OLS of y on [1, xs...] over rows with M, after re-ranking every variable within that sample.
    Returns the coefficients [T, 1 + k] (NaN where fewer than MIN_N rows)."""
    ry = kern.rank_center(y, M)
    rx = [kern.rank_center(x, M) for x in xs]
    n = M.sum(0)
    X = np.stack([M.astype(np.float64)] + [np.where(M, r, 0.0) for r in rx], -1)     # [R, T, 1 + k]
    Y = np.where(M, ry, 0.0)
    XtX = np.einsum('rtk,rtl->tkl', X, X)
    Xty = np.einsum('rtk,rt->tk', X, Y)
    out = np.full((y.shape[1], X.shape[2]), np.nan)
    ok = n >= MIN_N
    if ok.any():
        out[ok] = np.linalg.solve(XtX[ok], Xty[ok][:, :, None])[:, :, 0]
    return out


def spearman(x, y, M):
    rx = kern.rank_center(x, M)
    ry = kern.rank_center(y, M)
    a, b = np.where(M, rx, 0.0), np.where(M, ry, 0.0)
    with np.errstate(all='ignore'):
        v = (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))
    v[M.sum(0) < MIN_N] = np.nan
    return v


def build_series(tag, fh):
    P = data.load_panel(tag)
    T = P['T']
    ta = first_day(tag, P)
    tb = T - 2
    inU = np.asarray(P['inU'])[:, ta:tb]
    rets, r1, r5 = returns(P)
    rets, r1, r5 = rets[:, :, ta:tb], r1[:, ta:tb], r5[:, ta:tb]
    nd = tb - ta
    d = K.BIG / f'forces_{tag}'
    nV = len(VARS)
    S = np.full((nV, 3, nd), np.nan)
    RAW = np.full((nV, nd), np.nan)
    IC = np.full((nV, nd), np.nan)
    G = np.full((nV, 3, nd), np.nan)
    TO = np.full((nV, nd), np.nan)
    NS = np.zeros((nV, nd), np.int64)
    base = [np.isfinite(r1), np.isfinite(r5)]
    t0 = time.time()
    cache = {}
    for i, v in enumerate(VARS):
        F = np.asarray(np.load(d / f'{v["key"]}.npy', mmap_mode='r')[:, ta:tb], np.float64)
        Fm = inU & np.isfinite(F)
        g, to, ns = kern.book(F, Fm, rets, 0, nd, 5)
        G[i], TO[i], NS[i] = g, to, ns
        y = rets[0]
        Mr = Fm & base[0] & base[1] & np.isfinite(y)
        RAW[i] = fm(y, [F, r1, r5], Mr)[:, 1]
        IC[i] = spearman(F, y, Fm & np.isfinite(y))
        if v['bench'] is not None:
            if v['bench'] not in cache:
                cache.clear()
                cache[v['bench']] = np.asarray(np.load(d / f'{v["bench"]}.npy', mmap_mode='r')[:, ta:tb], np.float64)
            B = cache[v['bench']]
            for e in range(3):
                Ms = Fm & base[0] & base[1] & np.isfinite(B) & np.isfinite(rets[e])
                S[i, e] = fm(rets[e], [F, B, r1, r5], Ms)[:, 1]
        if i % 10 == 0:
            K.log(f'  series {i + 1}/{nV} ({v["key"]}): {time.time() - t0:.0f}s', fh)
    # 5-day reversal's long-short in the same universe (bottom fifth of the 5-day return long)
    Mrev = inU & np.isfinite(r5)
    grev, torev, _ = kern.book(-r5, Mrev, rets, 0, nd, 5)
    days = np.asarray(P['days'])[ta:tb]
    np.savez(K.BIG / f'series_{tag}.npz', S=S, RAW=RAW, IC=IC, G=G, TO=TO, NS=NS, grev=grev, torev=torev, days=days)
    return days


def stats_period(Z, sel, W, fh=None):
    """Everything for one period. Z: the series file; sel: bool over its days."""
    S, RAW, IC, G, TO = Z['S'][:, :, sel], Z['RAW'][:, sel], Z['IC'][:, sel], Z['G'][:, :, sel], Z['TO'][:, sel]
    grev = Z['grev'][:, sel]
    nV = S.shape[0]
    net = {c: G[:, 0] - c * 1e-4 * TO for c in K.COSTS}
    cols = {'S_oo': S[:, 0], 'S_nc': S[:, 1], 'S_sc': S[:, 2], 'raw': RAW, 'ic': IC,
            'M0': net[0.0], 'M5': net[5.0], 'M20': net[20.0],
            'M5_nc': G[:, 1] - 5e-4 * TO, 'M5_sc': G[:, 2] - 5e-4 * TO}
    X = np.concatenate([cols[c].T for c in cols], 1)                 # [n, nV * ncols]
    bm = K.boot_means(W, X)
    out = {'days': int(sel.sum()), 'bm': {}, 'est': {}}
    for j, c in enumerate(cols):
        b = bm[:, j * nV:(j + 1) * nV]
        out['bm'][c] = b
        out['est'][c] = np.nanmean(cols[c], 1)
    per = []
    for i, v in enumerate(VARS):
        rec = {'key': v['key'], 'name': v['name']}
        for c in cols:
            e = out['est'][c][i]
            rec[c] = K.summarize(e, out['bm'][c][:, i]) if np.isfinite(e) else None
            if rec[c] is not None and c in ('S_oo', 'M5'):
                rec[c]['nw_t'] = float(e / K.nw_se(cols[c][i]))
        g0 = np.nanmean(G[i, 0])
        to = np.nanmean(TO[i])
        rec['turnover'] = float(to)
        rec['gross_bps'] = float(g0 * 1e4)
        rec['breakeven_bps'] = float(g0 / (to * 1e-4)) if to > 0 else None
        r5 = net[5.0][i]
        rec['vol_ann'] = float(np.nanstd(r5) * np.sqrt(252))
        rec['maxdd'] = K.max_drawdown(r5)
        rec['net5_ann_pct'] = float(np.nanmean(r5) * 252 * 100)
        try:
            a = K.ols_nw(r5, grev[0])
            rec['alpha_rev'] = {'alpha_bps': a['coef'][0] * 1e4, 't': a['t'][0], 'beta': a['coef'][1], 't_beta': a['t'][1]}
        except Exception:
            rec['alpha_rev'] = None
        if v['bench'] is not None:
            bi = KEYS.index(v['bench'])
            try:
                a = K.ols_nw(r5, G[bi, 0])
                rec['alpha_bench'] = {'alpha_bps': a['coef'][0] * 1e4, 't': a['t'][0], 'beta': a['coef'][1], 't_beta': a['t'][1]}
            except Exception:
                rec['alpha_bench'] = None
        per.append(rec)
    out['per'] = per
    out['rev'] = {'gross_bps': float(np.nanmean(grev[0]) * 1e4), 'turnover': float(np.nanmean(Z['torev'][sel])),
                  'stats': K.summarize(np.nanmean(grev[0]), K.boot_means(W, grev[0])[:, 0])}
    # Romano-Wolf within each stage, for each world and kind
    rw = {}
    for stage in ('G1', 'G2', 'G3'):
        for world in ('real', 'random'):
            for kind in ('cand', 'closes'):
                idx = [i for i, v in enumerate(VARS) if v['stage'] == stage and v['world'] == world and v['kind'] == kind]
                if not idx:
                    continue
                key = f'{stage}_{world}_{kind}'
                rw[key] = {}
                for c in ('S_oo', 'M5'):
                    est = out['est'][c][idx]
                    adj, p, best, t = K.romano_wolf(est, out['bm'][c][:, idx])
                    rw[key][c] = {'p': p, 'best': VARS[idx[best]]['key'], 'best_name': VARS[idx[best]]['name'],
                                  't_best': float(t[best]), 'est_best': float(est[best]),
                                  'adj': {VARS[i]['key']: float(a) for i, a in zip(idx, adj)}}
    out['rw'] = rw
    del out['bm']
    out['est'] = {c: v.tolist() for c, v in out['est'].items()}
    return out


def main(tag):
    fh = open(K.OUT / f'tests_{tag}.txt', 'w')
    if not (K.BIG / f'series_{tag}.npz').exists() or (len(sys.argv) > 2 and sys.argv[2] == 'rebuild'):
        build_series(tag, fh)
    Z = dict(np.load(K.BIG / f'series_{tag}.npz'))
    days = Z['days']
    res = {'tag': tag, 'periods': {}}
    for name, (a, b) in periods_for(tag).items():
        lo = (np.datetime64(a) - np.datetime64('1970-01-01')).astype(int)
        hi = (np.datetime64(b) - np.datetime64('1970-01-01')).astype(int)
        sel = (days >= lo) & (days < hi)
        if sel.sum() < 30:
            continue
        W = K.boot_weights(int(sel.sum()))
        res['periods'][name] = stats_period(Z, sel, W, fh)
        res['periods'][name]['first'] = K.c2eval.iso(days[sel][0])
        res['periods'][name]['last'] = K.c2eval.iso(days[sel][-1])
        del W
    # The six primaries (2008-2019 for old; each period's own for new / fwd, reported)
    main_p = '2008-2019' if tag == 'old' else list(res['periods'])[0]
    rw = res['periods'][main_p]['rw']
    six = []
    for stage in ('G1', 'G2', 'G3'):
        for c in ('S_oo', 'M5'):
            r = rw[f'{stage}_real_cand'][c]
            six.append({'test': f'{stage}-{"S" if c == "S_oo" else "M"}', 'p': r['p'], 't': r['t_best'], 'best': r['best_name'],
                        'est': r['est_best']})
    adj, rej = K.holm([s['p'] for s in six], K.HOLM_FAMILY)
    for s, a, r in zip(six, adj, rej):
        s['holm_p'] = a
        s['pass'] = bool(r and s['t'] >= K.T_BAR)
    verdict = []
    for stage in ('G1', 'G2', 'G3'):
        ok = all(s['pass'] for s in six if s['test'].startswith(stage))
        verdict.append({'stage': stage, 'pass': ok})
        if not ok:
            break
    res['six'] = six
    res['verdict_order'] = verdict
    res['main_period'] = main_p
    K.jdump(res, K.OUT / f'tests_{tag}.json')
    K.log(f'Six primaries ({main_p}; Romano-Wolf within the stage, Holm across the six at {K.HOLM_FAMILY}):', fh)
    for s in six:
        K.log(f'  {s["test"]}: best {s["best"]}: est {s["est"]:+.6f}, t {s["t"]:+.2f}, RW p {s["p"]:.4f}, Holm p {s["holm_p"]:.4f} '
              f'-> {"PASS" if s["pass"] else "fail"}', fh)
    K.log(f'Verdict in order: {verdict}', fh)
    for name, pr in res['periods'].items():
        K.log(f'--- {name} ({pr["first"]}..{pr["last"]}, {pr["days"]} days)', fh)
        for v, rec in zip(VARS, pr['per']):
            if v['world'] != 'real':
                continue
            s = rec['S_oo']
            m = rec['M5']
            K.log(f'  {v["name"]:>34}: S {s["est"] if s else float("nan"):+.5f} (t {s["t"] if s else float("nan"):+.2f}); '
                  f'raw {rec["raw"]["est"]:+.5f} (t {rec["raw"]["t"]:+.2f}); IC {rec["ic"]["est"]:+.5f} (t {rec["ic"]["t"]:+.2f}); '
                  f'M5 {m["est"] * 1e4:+.2f} bps/day (t {m["t"]:+.2f}); gross {rec["gross_bps"]:+.2f} bps, turnover {rec["turnover"]:.2f}', fh)
    fh.close()


if __name__ == '__main__':
    main(sys.argv[1])
