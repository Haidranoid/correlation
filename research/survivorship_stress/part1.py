"""Part 1: how often stocks like the monthly test's candidates vanish, and a logistic hazard model of it.

Universe: stockai's candidates (judge_universe) on data through 2025-06-27 in the older store's export. Each is
looked up in the newer store (stooq-2026-10-19): by ticker if that series is the same company (constant price ratio
over 2025-04-01..2025-06-27), else by a rename (a newer ticker absent from the older store with the same constant
ratio). It disappeared if it has no close after 2025-06-27 or its data ends before 2026-10-01. Rules and decisions:
experiments/+18.6-monthly-top10-picks.md and decisions.md here.

Writes out/part1.json (rates, model, diagnostics) and out/part1_stocks.tsv (one row per universe stock).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import math  # noqa: E402

import numpy as np  # noqa: E402

import common as C  # noqa: E402

WIN0, WIN1 = C.to_day('2025-04-01'), C.to_day('2025-06-27')
END = C.to_day('2025-06-27')
EARLY = C.to_day('2026-10-01')
MIN_COMMON, TOL, SHARE = 40, 0.005, 0.95
MIN_MOVES = 20  # days with a daily move above TOL in the old window (decisions.md)


def ratio_test(new, old):
    """The plan's literal level test on two aligned window series: >= MIN_COMMON common days and >= SHARE of the
    daily ratios new/old within TOL of their median. Returns (passes, common days, median ratio, share)."""
    ok = np.isfinite(new) & np.isfinite(old) & (old > 0) & (new > 0)
    n = int(ok.sum())
    if n < MIN_COMMON:
        return False, n, float('nan'), 0.0
    r = new[ok] / old[ok]
    med = float(np.median(r))
    share = float((np.abs(r / med - 1.0) <= TOL).mean())
    return share >= SHARE, n, med, share


MAX_STEPS = 4


def step_test(new, old):
    """The test used (decisions.md, Later): >= MIN_COMMON common days and at most MAX_STEPS days on which new/old
    moves by more than TOL from the previous common day (adjustment steps at ex-dates, or bad prints).
    Returns (passes, common days, steps, median daily change)."""
    ok = np.isfinite(new) & np.isfinite(old) & (old > 0) & (new > 0)
    n = int(ok.sum())
    if n < MIN_COMMON:
        return False, n, 10 ** 6, float('inf')
    r = new[ok] / old[ok]
    ch = np.abs(r[1:] / r[:-1] - 1.0)
    steps = int((ch > TOL).sum())
    return steps <= MAX_STEPS, n, steps, float(np.median(ch))


def sym_rows(st, i):
    a, b = st['offs'][i], st['offs'][i + 1]
    return np.asarray(st['date'][a:b]), np.asarray(st['close'][a:b], dtype=np.float64)


def status_of(new, found, r):
    """Survived / no close after 2025-06-27 / ended early / vanished for a match (and fill r with the details)."""
    if found is None:
        if r is not None:
            r['how'] = r.get('how') or 'not in newer store'
        return 'vanished'
    nid, ntk, how = found
    dates, cl = sym_rows(new, nid)
    after = dates > END
    last = int(dates[-1])
    if r is not None and r is not {}:
        r.update(new_ticker=ntk, how=how, new_last=C.iso(last))
    if not after.any():
        return 'no close after 2025-06-27'
    if last < EARLY:
        if r:
            i0 = np.searchsorted(dates, END, side='right') - 1  # last close on or before 2025-06-27
            r['ret_to_last'] = float(cl[-1] / cl[i0] - 1) if i0 >= 0 and cl[i0] > 0 else float('nan')
            r['ret_last21'] = float(cl[-1] / cl[max(len(cl) - 22, 0)] - 1)
            r['trading_days_after'] = int(after.sum())
        return 'ended early'
    return 'survived'


def irls(X, y, ridge=0.0, iters=100):
    beta = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], ridge)
    pen[0] = 0.0
    for it in range(iters):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        g = X.T @ (y - p) - pen * beta
        Hm = (X * W[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(Hm, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            return beta, Hm, it + 1, True
    return beta, Hm, iters, False


def main():
    E = C.load()
    T = len(E['days'])
    assert C.iso(E['days'][-1]) == '2025-06-27'
    U = C.judge_universe(E, T)
    d = T - 1
    feat_all = C.features_col(E, U, d)
    c_last = np.asarray(E['close'][U, d], dtype=np.float64)
    alive = np.isfinite(c_last) & (c_last > 0)
    old = C.c2eval.load_store(C.c2eval.STORE_OLD)
    new = C.c2eval.load_store(C.c2eval.STORE_NEW)
    old_tickers = {s['ticker'] for s in old['syms']}
    new_by_ticker = {s['ticker']: s['id'] for s in new['syms'] if s['kind'] != 'test'}
    # trading calendar: the newer store's SPY
    spy_new = new_by_ticker['SPY']
    cal, _ = sym_rows(new, spy_new)
    win = cal[(cal >= WIN0) & (cal <= WIN1)]
    m_days = int(((cal > END) & (cal < EARLY)).sum())
    m = m_days / 21.0
    print(f'universe {U.size}; window days {win.size}; m = {m_days}/21 = {m:.3f}; newer store calendar ends '
          f'{C.iso(cal[-1])}', flush=True)
    old_win = C.c2eval.store_panel(old, E['ids'][U], win).astype(np.float64)
    # rename candidates: newer-store symbols whose ticker is not in the older store, with data in the window
    cand_ids = np.array([s['id'] for s in new['syms'] if s['kind'] != 'test' and s['ticker'] not in old_tickers])
    cand_win = C.c2eval.store_panel(new, cand_ids, win).astype(np.float64)
    has = np.isfinite(cand_win).sum(axis=1) >= MIN_COMMON
    cand_ids, cand_win = cand_ids[has], cand_win[has]
    print(f'rename candidates with >= {MIN_COMMON} window closes: {cand_ids.size}', flush=True)

    rows = []
    for k, j in enumerate(U):
        tk = E['tickers'][j]
        o = old_win[k]
        r = {'ticker': tk, 'new_ticker': '', 'how': '', 'n_old_window': int(np.isfinite(o).sum()),
             'alive_0627': int(alive[k])}
        found, found_lit = None, None  # (newer id, ticker, how) under the step test / the literal level test
        mv = np.diff(np.log(o[np.isfinite(o) & (o > 0)]))
        r['old_moves'] = int((np.abs(mv) > TOL).sum())
        testable = r['n_old_window'] >= MIN_COMMON and r['old_moves'] >= MIN_MOVES
        r['same_after'] = 0
        if tk in new_by_ticker:
            nid = new_by_ticker[tk]
            r['same_after'] = int((sym_rows(new, nid)[0] > END).any())
            nw = C.c2eval.store_panel(new, [nid], win)[0].astype(np.float64)
            ok, n, steps, mch = step_test(nw, o)
            lok, _, med, share = ratio_test(nw, o)
            r.update(same_common=n, same_steps=steps, same_ratio=med, same_share=share, same_literal=int(lok))
            if ok or not testable:
                found = (nid, tk, 'same' if testable else 'same (untestable)')
            else:
                r['how'] = 'ticker reused or broken'
            if lok or not testable:
                found_lit = (nid, tk, 'same')
        if (found is None or found_lit is None) and testable:
            ok_o = np.isfinite(o) & (o > 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                R = cand_win[:, ok_o] / o[ok_o]
            R[~np.isfinite(R) | (R <= 0)] = np.nan
            n = np.isfinite(R).sum(axis=1)
            matches, lit = [], []
            for c in np.flatnonzero(n >= MIN_COMMON):
                okc, _, steps, mch = step_test(cand_win[c], o)
                if okc:
                    matches.append((steps, mch, int(cand_ids[c])))
                lokc, _, _, share = ratio_test(cand_win[c], o)
                if lokc:
                    lit.append((-share, int(cand_ids[c])))
            if found is None and matches:
                matches.sort()
                found = (matches[0][2], new['syms'][matches[0][2]]['ticker'], 'rename')
                r['rename_matches'] = len(matches)
                r['rename_all'] = ','.join(new['syms'][mm[2]]['ticker'] for mm in matches)
            if found_lit is None and lit:
                lit.sort()
                found_lit = (lit[0][1], new['syms'][lit[0][1]]['ticker'], 'rename')
        r['status'] = status_of(new, found, r)
        r['status_literal'] = status_of(new, found_lit, {})
        r['gone'] = int(r['status'] != 'survived')
        r['gone_literal'] = int(r['status_literal'] != 'survived')
        for f in ('ret12', 'down50', 'vol63', 'logp', 'logdv'):
            r[f] = float(feat_all[f][k])
        rows.append(r)

    # the analysed universe: alive (a close) on 2025-06-27
    dead_before = [r['ticker'] for r in rows if not r['alive_0627']]
    sens_with_dead = {'n': len(rows), 'gone': sum(r['gone'] for r in rows)}
    sens_literal = {'n': sum(r['alive_0627'] for r in rows),
                    'gone': sum(r['gone_literal'] for r in rows if r['alive_0627'])}
    rows = [r for r in rows if r['alive_0627']]
    feat = {f: feat_all[f][alive] for f in feat_all}
    y = np.array([r['gone'] for r in rows], dtype=float)
    # raw rates
    ret12 = feat['ret12']
    price = np.exp(feat['logp'])
    buckets = [('below -80%', ret12 < -0.8), ('-80% to -50%', (ret12 >= -0.8) & (ret12 < -0.5)),
               ('-50% to 0', (ret12 >= -0.5) & (ret12 <= 0)), ('above 0', ret12 > 0),
               ('12-month return missing', ~np.isfinite(ret12))]
    def rate(mask):
        n = int(mask.sum()); g = int(y[mask].sum())
        p = g / n if n else float('nan')
        return {'n': n, 'gone': g, 'rate': p, 'monthly': C.monthly_hazard(p, m) if n else float('nan')}
    raw = {'all': rate(np.ones(y.size, bool))}
    for name, mk in buckets:
        raw[name] = rate(mk)
    raw['price below $5'] = rate(price < 5)
    raw['price $5 or more'] = rate(price >= 5)

    # hazard model
    model = {'names': ['ret12', 'down50', 'vol63', 'logp', 'logdv'], 'mean': {}, 'sd': {}, 'm': m, 'm_days': m_days}
    missing = {}
    for f in C.CONT:
        x = feat[f]
        fin = np.isfinite(x)
        missing[f] = int((~fin).sum())
        model['mean'][f] = float(x[fin].mean())
        model['sd'][f] = float(x[fin].std(ddof=0))
    X = C.design(feat, model)
    beta, Hm, its, conv = irls(X, y)
    ridge = 0.0
    if not conv or not np.all(np.isfinite(beta)):
        ridge = 1e-3
        beta, Hm, its, conv = irls(X, y, ridge=ridge)
    cov = np.linalg.inv(Hm)
    se = np.sqrt(np.diag(cov))
    model.update(beta=beta.tolist(), se=se.tolist(), iterations=its, converged=bool(conv), ridge=ridge,
                 missing=missing)
    model2 = {'names': ['ret12', 'down50', 'vol63', 'logdv'], 'mean': model['mean'], 'sd': model['sd'], 'm': m,
              'm_days': m_days}
    b2, H2, it2, c2 = irls(C.design(feat, model2), y)
    model2.update(beta=b2.tolist(), se=np.sqrt(np.diag(np.linalg.inv(H2))).tolist(), converged=bool(c2))
    p = C.predict_p(feat, model)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    p0 = y.mean()
    ll0 = float(np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0)))
    # AUC (Mann-Whitney)
    order = np.argsort(p, kind='stable')
    ranks = np.empty(p.size); ranks[order] = np.arange(1, p.size + 1)
    npos, nneg = y.sum(), (1 - y).sum()
    auc = float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    model.update(loglik=ll, loglik_null=ll0, mcfadden_r2=1 - ll / ll0, auc=auc)
    coefs = []
    for name, b, s in zip(['intercept'] + model['names'], beta, se):
        z = b / s
        coefs.append({'name': name, 'beta': float(b), 'se': float(s), 'z': float(z),
                      'p': float(math.erfc(abs(z) / math.sqrt(2))), 'odds_ratio': float(math.exp(b))})
    # calibration by predicted decile
    q = np.quantile(p, np.linspace(0, 1, 11))
    calib = []
    for a, b in zip(q[:-1], q[1:]):
        mk = (p >= a) & (p <= b) if b == q[-1] else (p >= a) & (p < b)
        calib.append({'p_lo': float(a), 'p_hi': float(b), 'n': int(mk.sum()), 'pred': float(p[mk].mean()),
                      'actual': float(y[mk].mean())})
    for r, pp in zip(rows, p):
        r['p_window'] = float(pp)
        r['h_month'] = float(C.monthly_hazard(pp, m))

    status = {}
    for r in rows:
        status[r['status']] = status.get(r['status'], 0) + 1
    hows = {}
    for r in rows:
        key = f"{r['status']} / {r['how']}"
        hows[key] = hows.get(key, 0) + 1
    early = [r for r in rows if r['status'] == 'ended early']
    er = np.array([r['ret_to_last'] for r in early])
    e21 = np.array([r['ret_last21'] for r in early])
    early_summary = {
        'n': len(early),
        'ret_to_last_mean': float(np.nanmean(er)) if er.size else float('nan'),
        'ret_to_last_median': float(np.nanmedian(er)) if er.size else float('nan'),
        'ret_to_last_quartiles': np.nanquantile(er, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist() if er.size else [],
        'share_below_-50%': float(np.mean(er < -0.5)) if er.size else float('nan'),
        'share_below_-90%': float(np.mean(er < -0.9)) if er.size else float('nan'),
        'share_above_0': float(np.mean(er > 0)) if er.size else float('nan'),
        'last21_mean': float(np.nanmean(e21)) if e21.size else float('nan'),
        'last21_median': float(np.nanmedian(e21)) if e21.size else float('nan'),
    }
    # early enders that had collapsed already (down >50% over the past year at 2025-06-27)
    ec = np.array([r['down50'] for r in early]) == 1
    early_summary['collapsed_n'] = int(ec.sum())
    early_summary['collapsed_ret_mean'] = float(np.nanmean(er[ec])) if ec.any() else float('nan')
    early_summary['not_collapsed_ret_mean'] = float(np.nanmean(er[~ec])) if (~ec).any() else float('nan')
    def rate_of(key):
        g = sum(r[key] for r in rows)
        return {'n': len(rows), 'gone': g, 'rate': g / len(rows), 'monthly': C.monthly_hazard(g / len(rows), m)}
    diag = {
        'universe_candidates': int(U.size), 'no_close_on_2025_06_27': dead_before, 'analysed': len(rows),
        'status': status, 'status_how': hows,
        'renames': sum(r['how'] == 'rename' for r in rows),
        'renames_list': [f"{r['ticker']}>{r['new_ticker']}" for r in rows if r['how'] == 'rename'],
        'ticker_reused_or_broken': [r['ticker'] for r in rows if r['how'] == 'ticker reused or broken'],
        'renamed_while_old_ticker_reused': [r['ticker'] for r in rows if 'same_common' in r and r['how'] == 'rename'],
        'untestable_list': [r['ticker'] for r in rows if r['how'] == 'same (untestable)'],
        'no_same_ticker_close_after': sum(1 for r in rows if not r['same_after']),
        'untestable_same_ticker': sum(1 for r in rows if r['how'] == 'same (untestable)'),
        'multiple_rename_matches': [r['ticker'] + ':' + r['rename_all'] for r in rows if r.get('rename_matches', 0) > 1],
        'same_ticker_failing_literal_level_test': [r['ticker'] for r in rows if r['how'] == 'same' and not r['same_literal']],
        'window_days': int(win.size), 'rename_candidates': int(cand_ids.size),
        'sensitivity_with_dead_before': {**sens_with_dead, 'rate': sens_with_dead['gone'] / sens_with_dead['n']},
        'sensitivity_literal_level_test': {**sens_literal, 'rate': sens_literal['gone'] / sens_literal['n'],
                                           'monthly': C.monthly_hazard(sens_literal['gone'] / sens_literal['n'], m)},
    }
    out = {'raw_rates': raw, 'model': model, 'model_no_logp': model2, 'coefficients': coefs, 'calibration': calib,
           'early_enders': early_summary, 'diagnostics': diag,
           'predicted_h_month': {'mean': float(np.mean([r['h_month'] for r in rows])),
                                 'quantiles': np.quantile([r['h_month'] for r in rows],
                                                          [0.1, 0.5, 0.9, 0.99]).tolist()}}
    C.save_json(out, 'part1.json')
    cols = ['ticker', 'status', 'how', 'new_ticker', 'new_last', 'gone', 'status_literal', 'ret12', 'down50',
            'vol63', 'logp', 'logdv', 'p_window', 'h_month', 'ret_to_last', 'ret_last21', 'trading_days_after',
            'rename_matches', 'rename_all', 'same_common', 'same_steps', 'same_ratio', 'same_share', 'same_literal',
            'n_old_window', 'old_moves']
    with open(C.OUT / 'part1_stocks.tsv', 'w') as fh:
        fh.write('\t'.join(cols) + '\n')
        for r in rows:
            fh.write('\t'.join('' if r.get(c) is None else (f'{r[c]:.6g}' if isinstance(r.get(c), float) else str(r.get(c, '')))
                               for c in cols) + '\n')
    print(json_short(out))


def json_short(out):
    import json
    return json.dumps({k: out[k] for k in ('raw_rates', 'coefficients', 'early_enders', 'diagnostics',
                                           'predicted_h_month')}, indent=1, default=float)


if __name__ == '__main__':
    main()
