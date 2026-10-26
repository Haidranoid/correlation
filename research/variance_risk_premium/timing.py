"""T (secondary): does the VRP forecast add money to timing SPY by its own variance? Also both numbers for the README
index, and the replications' rules.

    python3 -I -B timing.py old|new|fwd        (after forecast.py <tag>)

-3.2's set-up with an earlier timing test's timing: a decision at the close of each calendar date t0, the trade at the close of
t0 + 1, held to the close of t0 + 22; SPY (Yahoo's adjusted close) or cash at 0; exposure in [0, 1], no leverage;
5 bps on every SPY trade, including the first buy and each month's trim of drift.
  VT:  w = min(1, c / s2), s2 = np.var of the 21 daily log returns ending at t0 (-3.2's measure), c = the median of
       s2 over the previous 1,260 trading days (at least 252).
  VRP: w = min(1, max(0, (c / s2) x (mu_hat / mu_bar))), mu_hat = P1's VRP-model forecast at t0, mu_bar = the
       historical-mean forecast at t0 from the same fit. mu_bar <= 0: VT's weight (logged).
Reuses research/market_timing/engine.py (measures, random timing, the 12-month block bootstrap), read-only.
"""
import json
import sys
import time

sys.path.insert(0, '/home/bhoult/programming/correlation/research/variance_risk_premium')
import numpy as np  # noqa: E402

import vrp_common as V  # noqa: E402
import engine as G  # noqa: E402  research/market_timing/engine.py (read-only)

t_start = time.time()
tag = sys.argv[1]
assert tag in V.TAGS
D = V.load(tag)
dates, T = D['dates'], D['T']
fc = dict(np.load(V.WORK / f'fc_{tag}.npz'))
assert np.array_equal(fc['days'], D['days'])
cal = D['cal']
COST = V.COST
HOLD = 21


def s2_series(r):
    out = np.full(T, np.nan)
    for t in range(V.RV_WIN, T):
        out[t] = V.sigma2(r, t)
    return out


def c_at(s2, t0):
    prev = s2[max(0, t0 - V.C_MAX):t0]
    prev = prev[np.isfinite(prev)]
    return float(np.median(prev)) if len(prev) >= V.C_MIN else float('nan')


def decisions():
    if tag == 'old':
        lo, hi = V.DEV[0], V.CONF[1]
    elif tag == 'new':
        lo, hi = V.NEW_FIRST_DECISION, V.UNSEEN[1]
    else:
        lo, hi = V.FWD
    t0s = np.array([int(t) for t in cal if lo <= dates[t] <= hi and t + 1 + HOLD <= T - 1])
    assert len(t0s) and np.all(np.diff(t0s) == HOLD), 'decision dates must be contiguous'
    return t0s


def weights(t0s, s2, mu_hat, mu_bar, mu_a=None):
    """Per decision: s2, c, VT's, the VRP rule's and (optionally) the model-(a) tilt's weight; logs of special cases."""
    rows, log = [], {'mu_bar_nonpositive': [], 'carried': []}
    prev = None
    for t0 in t0s:
        s, c = s2[t0], c_at(s2, t0)
        assert np.isfinite(s) and np.isfinite(c) and s > 0
        wvt = min(1.0, c / s)
        mh, mb = mu_hat[t0], mu_bar[t0]

        def tilt(m):
            if not (np.isfinite(m) and np.isfinite(mb)):
                return None
            if mb <= 0:
                return wvt
            return min(1.0, max(0.0, (c / s) * (m / mb)))
        wv = tilt(mh)
        if wv is None:
            wv = prev[3] if prev is not None else wvt
            log['carried'].append(dates[t0])
        elif mb <= 0:
            log['mu_bar_nonpositive'].append(dates[t0])
        wa = tilt(mu_a[t0]) if mu_a is not None else np.nan
        if wa is None:
            wa = prev[6] if prev is not None else wvt
        row = (s, c, wvt, wv, mh, mb, wa)
        rows.append(row)
        prev = row
    A = np.array(rows)
    return {'s2': A[:, 0], 'c': A[:, 1], 'vt': A[:, 2], 'vrp': A[:, 3], 'mu_hat': A[:, 4], 'mu_bar': A[:, 5], 'a': A[:, 6]}, log


def path(Px, t0s, w, cost, rf=None):
    """Daily returns [M, 21]: buy w of the fund at the close of t0 + 1 (paying cost on |w - drifted weight|), hold to
    the close of t0 + 22; cash at 0, or at the daily risk-free rate if rf is given. Months must be contiguous."""
    out = np.empty((len(t0s), HOLD))
    wd, prev = 0.0, 1.0
    for m, t0 in enumerate(t0s):
        s = t0 + 1
        v = prev * (1.0 - abs(w[m] - wd) * cost)
        rel = Px[s + 1:s + HOLD + 1] / Px[s]
        cash = np.ones(HOLD) if rf is None else np.cumprod(1.0 + rf[s + 1:s + HOLD + 1])
        val = v * ((1 - w[m]) * cash + w[m] * rel)
        out[m] = val / np.concatenate([[prev], val[:-1]]) - 1.0
        prev = val[-1]
        wd = w[m] * rel[-1] / ((1 - w[m]) * cash[-1] + w[m] * rel[-1])
    return out


def turnover(Px, t0s, w):
    wd, n, tot = 0.0, 0, 0.0
    for m, t0 in enumerate(t0s):
        d = abs(w[m] - wd)
        if d > 1e-12:
            n += 1
        tot += d
        rel = Px[t0 + 1 + HOLD] / Px[t0 + 1]
        wd = w[m] * rel / ((1 - w[m]) + w[m] * rel)
    years = len(t0s) * HOLD / 252
    return {'trades': n, 'turnover_per_year': tot / years, 'cost_per_year_pct': 100 * tot * COST / years}


def boot_sharpe(dr_a, dr_b, n=G.C.N_BOOT, block=G.C.BLOCK, seed=G.C.SEED + 1):
    """engine.block_bootstrap_sharpe's paired circular 12-month block bootstrap (copied so it also returns the
    2.5th and 97.5th percentiles; its p is asserted equal to the engine's)."""
    rng = np.random.default_rng(seed)
    M = dr_a.shape[0]
    s1a, s2a = dr_a.sum(1), (dr_a ** 2).sum(1)
    s1b, s2b = dr_b.sum(1), (dr_b ** 2).sum(1)
    nb = int(np.ceil(M / block))
    starts = rng.integers(0, M, size=(n, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % M).reshape(n, nb * block)[:, :M]
    nd = M * dr_a.shape[1]

    def sharpe(s1, s2):
        S1, S2 = s1[idx].sum(1), s2[idx].sum(1)
        mu = S1 / nd
        var = (S2 - nd * mu ** 2) / (nd - 1)
        return mu / np.sqrt(var) * np.sqrt(G.DPY)
    d = sharpe(s1a, s2a) - sharpe(s1b, s2b)
    point = G.measures(dr_a)['sharpe'] - G.measures(dr_b)['sharpe']
    ref = G.block_bootstrap_sharpe(dr_a, dr_b)
    assert abs(ref['p'] - float(np.mean(d <= 0))) < 1e-12
    return {'diff': float(point), 'p': float(np.mean(d <= 0)), 'lo2.5': float(np.percentile(d, 2.5)),
            'hi97.5': float(np.percentile(d, 97.5))}


def after_cost_vs(Px, t0s, w):
    """The file-name number, -3.2's way, as in an earlier timing test: the rule held in Yahoo SPY at 5 bps on every trade, daily returns
    compounded per calendar year (by each day's date), minus SPY's total return over the same days, averaged."""
    dr = path(Px, t0s, w, COST).ravel()
    span = np.arange(t0s[0] + 1, t0s[-1] + 1 + HOLD + 1)
    assert np.isfinite(Px[span]).all()
    spy_d = Px[span[1:]] / Px[span[:-1]] - 1
    yr = np.array([d[:4] for d in dates[span[1:]]])
    res = {}
    for y_ in sorted(set(yr)):
        m = yr == y_
        a, s = 100 * (np.prod(1 + dr[m]) - 1), 100 * (np.prod(1 + spy_d[m]) - 1)
        res[y_] = {'rule5': float(a), 'spy': float(s), 'diff': float(a - s), 'days': int(m.sum())}
    n = len(dr)
    span_diff = 100 * ((np.prod(1 + dr)) ** (252 / n) - (np.prod(1 + spy_d)) ** (252 / n))
    return {'years': res, 'number': float(np.mean([v['diff'] for v in res.values()])), 'span_annualized_diff': float(span_diff),
            'first_close': dates[span[0]], 'last_close': dates[span[-1]]}


def evaluate_rules(Px, t0s, W, m, seed_off=0, rf=None):
    out = {'months': int(m.sum())}
    if m.sum() < 3:
        return out
    rel = G.rel_paths(Px, t0s[m] + 1)
    for nm, w in W.items():
        d5 = path(Px, t0s[m], w[m], COST)
        d0 = path(Px, t0s[m], w[m], 0.0)
        out[nm] = G.measures(d5) | {'exposure': float(np.mean(w[m])), 'annual_0bps': G.measures(d0)['annual'],
                                    'sharpe_0bps': G.measures(d0)['sharpe']}
        if rf is not None:
            out[nm]['cash_rf'] = G.measures(path(Px, t0s[m], w[m], COST, rf))
        if nm != 'bh':
            rd = G.random_timing(w[m], rel, seed=G.C.SEED + seed_off + {'vrp': 0, 'vt': 3, 'a': 7}.get(nm, 11))
            out[nm]['random_median_annual'] = float(np.median(rd['annual']))
            out[nm]['random_p_annual'] = G.random_p(G.measures(d0)['annual'], rd['annual'])
            out[nm]['over_random'] = out[nm]['annual_0bps'] - out[nm]['random_median_annual']
    dv, dt = path(Px, t0s[m], W['vrp'][m], COST), path(Px, t0s[m], W['vt'][m], COST)
    out['boot_vrp_vs_vt'] = boot_sharpe(dv, dt)
    if 'a' in W:
        out['boot_a_vs_vt'] = boot_sharpe(path(Px, t0s[m], W['a'][m], COST), dt)
    out['boot_vrp_vs_bh'] = boot_sharpe(dv, path(Px, t0s[m], np.ones(m.sum()), COST))
    return out


t0s = decisions()
dd = dates[t0s]
res = {'tag': tag, 'first_decision': dd[0], 'last_decision': dd[-1], 'last_hold_end': dates[t0s[-1] + 1 + HOLD], 'months': len(t0s)}
S = V.build(D)
s2 = s2_series(S['r'])
W, wlog = weights(t0s, s2, fc['yhat_vrp'], fc['yhat_mean'], fc['yhat_a'])
res['weight_log'] = wlog
Px = D['spy']
Wd = {'vrp': W['vrp'], 'vt': W['vt'], 'a': W['a'], 'bh': np.ones(len(t0s))}
if tag == 'old':
    periods = {'2008-01..2019-12 (pass window)': (dd >= '2008-01-01') & (dd <= '2019-12-31'),
               '2020-01..2025-05': (dd >= '2020-01-01'),
               '2008-01..2025-05': np.ones(len(dd), bool),
               '2008-2009': (dd >= '2008-01-01') & (dd <= '2009-12-31'), '2010-2014': (dd >= '2010-01-01') & (dd <= '2014-12-31'),
               '2015-2019': (dd >= '2015-01-01') & (dd <= '2019-12-31'),
               '2010-2019 (without 2008-2009)': (dd >= '2010-01-01') & (dd <= '2019-12-31'),
               '2021-2025-05 (without 2020)': (dd >= '2021-01-01')}
else:
    periods = {'unseen months' if tag == 'new' else 'forward': np.ones(len(dd), bool)}
res['periods'] = {k: evaluate_rules(Px, t0s, Wd, m, rf=D['rf']) for k, m in periods.items()}
res['turnover'] = {nm: turnover(Px, t0s, w) for nm, w in Wd.items()}
if tag == 'old':
    pw = res['periods']['2008-01..2019-12 (pass window)']
    res['T_pass'] = bool(pw['vrp']['sharpe'] > pw['vt']['sharpe'] and pw['boot_vrp_vs_vt']['p'] < 0.10)
    ck = res['periods']['2020-01..2025-05']
    res['check_same_sign'] = bool(np.sign(ck['boot_vrp_vs_vt']['diff']) == np.sign(pw['boot_vrp_vs_vt']['diff']))
    whole = res['periods']['2008-01..2025-05']
    res['improvement_over_random'] = whole['vrp']['over_random']
    res['improvement_over_random_p'] = whole['vrp']['random_p_annual']
    # diagnostic only (uses the evaluation period): VT with c scaled so its mean exposure matches the VRP rule's
    lo_, hi_ = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo_ + hi_)
        if np.mean(np.minimum(1, np.exp(mid) * W['c'] / W['s2'])) < np.mean(W['vrp']):
            lo_ = mid
        else:
            hi_ = mid
    w_m = np.minimum(1, np.exp(0.5 * (lo_ + hi_)) * W['c'] / W['s2'])
    res['diag_vt_matched'] = {'scale': float(np.exp(0.5 * (lo_ + hi_))), 'exposure': float(w_m.mean()),
                              'sharpe_vt_matched': G.measures(path(Px, t0s, w_m, COST))['sharpe'],
                              'sharpe_vrp': G.measures(path(Px, t0s, W['vrp'], COST))['sharpe'],
                              'boot_vrp_vs_vt_matched': boot_sharpe(path(Px, t0s, W['vrp'], COST), path(Px, t0s, w_m, COST))}
res['after_cost'] = {nm: after_cost_vs(Px, t0s, w) for nm, w in (('vrp', W['vrp']), ('vt', W['vt']), ('a', W['a']))}
res['file_name_number'] = res['after_cost']['vrp']['number']
res['months_vrp_below_vt'] = int(np.sum(W['vrp'] < W['vt'] - 1e-12))
res['months_vrp_above_vt'] = int(np.sum(W['vrp'] > W['vt'] + 1e-12))
res['months_vrp_zero'] = int(np.sum(W['vrp'] <= 1e-12))
res['months_vt_full'] = int(np.sum(W['vt'] >= 1 - 1e-12))

# ---- Replications: the same rule on QQQ (VXN) and IWM (RVX), against each fund's own VT -----------------------------
res['replications'] = {}
for fund, vol in (('qqq', 'vxn'), ('iwm', 'rvx')):
    if f'{fund}_yhat_vrp' not in fc or not np.isfinite(D[fund][t0s[0] - V.C_MAX:]).all():
        continue
    Pf = D[fund]
    sf = s2_series(V.logret(Pf))
    Wf, lf = weights(t0s, sf, fc[f'{fund}_yhat_vrp'], fc[f'{fund}_yhat_mean'])
    Wfd = {'vrp': Wf['vrp'], 'vt': Wf['vt'], 'bh': np.ones(len(t0s))}
    rr = {'weight_log': lf}
    for k, m in periods.items():
        if k in ('2008-01..2019-12 (pass window)', '2020-01..2025-05', '2008-01..2025-05', 'unseen months', 'forward'):
            rr[k] = evaluate_rules(Pf, t0s, Wfd, m, seed_off=100)
    res['replications'][f'{vol.upper()} with {fund.upper()}'] = rr

np.savez(V.WORK / f'timing_{tag}.npz', t0s=t0s, **{f'w_{k}': v for k, v in W.items()})
assert tag != 'new' or (V.WORK / 'timing_old.npz').exists(), 'run timing.py old first'
if tag in ('new', 'fwd') and (V.WORK / f"timing_{'old' if tag == 'new' else 'new'}.npz").exists():
    prev = 'old' if tag == 'new' else 'new'          # the previous run's exposures must be reproduced exactly
    zo = np.load(V.WORK / f'timing_{prev}.npz')
    t_old = zo['t0s']
    Wo, _ = weights(t_old, s2, fc['yhat_vrp'], fc['yhat_mean'], fc['yhat_a'])
    res['previous_run_match'] = {k: bool(np.array_equal(Wo[k], zo[f'w_{k}'])) for k in ('vt', 'vrp', 'a')}
    if tag == 'new':
        assert all(res['previous_run_match'].values()), res['previous_run_match']
res['seconds'] = round(time.time() - t_start)
with open(V.HERE / f'timing_{tag}.json', 'w') as f:
    json.dump(res, f, indent=1, default=float)
with open(V.HERE / f'months_{tag}.tsv', 'w') as f:
    f.write('date\ts2\tc\tw_vt\tmu_hat\tmu_bar\tw_vrp\tw_a\tspy_next21\n')
    for m, t in enumerate(t0s):
        f.write(f"{dates[t]}\t{W['s2'][m]:.4e}\t{W['c'][m]:.4e}\t{W['vt'][m]:.4f}\t{W['mu_hat'][m]:+.5f}\t{W['mu_bar'][m]:+.5f}\t"
                f"{W['vrp'][m]:.4f}\t{W['a'][m]:.4f}\t{Px[t + 1 + HOLD] / Px[t + 1] - 1:+.4f}\n")

# ---- Report ---------------------------------------------------------------------------------------------------------
L = []
p = L.append
pc = lambda v: f'{100 * v:+.1f}%'  # noqa: E731
p(f"T ({tag}): {res['months']} decisions {res['first_decision']} .. {res['last_decision']} (last hold ends {res['last_hold_end']}); "
  f"each held from the close of t0 + 1 to the close of t0 + 22, 5 bps on every SPY trade, cash at 0.")
p(f"Weight log: {wlog}. VRP rule below VT in {res['months_vrp_below_vt']} months, above in {res['months_vrp_above_vt']}, at 0 in {res['months_vrp_zero']}; "
  f"VT at 100% in {res['months_vt_full']} of {res['months']}")
LAB = {'vrp': 'VRP rule', 'vt': 'VT (variance timing)', 'a': 'Tilt from model (a)', 'bh': 'Buy and hold SPY'}


def table(q, labels=LAB):
    p('| Rule | Annual (5 bps) | Sharpe (5 bps) | Max drawdown | Avg exposure | Annual (0 bps) | Median random (0 bps) | Over random | Random p | Sharpe, cash at RF |')
    p('|---|---|---|---|---|---|---|---|---|---|')
    for nm, lab in labels.items():
        if nm not in q:
            continue
        a = q[nm]
        extra = (f" | {pc(a['random_median_annual'])} | {100 * a['over_random']:+.2f} | {a['random_p_annual']:.3f}"
                 if 'random_median_annual' in a else ' | | |')
        crf = f" | {a['cash_rf']['sharpe']:.3f}" if 'cash_rf' in a else ' |'
        p(f"| {lab} | {pc(a['annual'])} | {a['sharpe']:.3f} | {pc(-a['mdd'])} | {a['exposure']:.0%} | {pc(a['annual_0bps'])}{extra}{crf} |")


for k, q in res['periods'].items():
    if q['months'] < 3:
        continue
    p(f'\n## {k}: {q["months"]} months')
    table(q)
    for nm, lab in (('boot_vrp_vs_vt', 'VRP - VT'), ('boot_a_vs_vt', 'tilt (a) - VT'), ('boot_vrp_vs_bh', 'VRP - buy and hold')):
        b = q[nm]
        p(f"Sharpe {lab}: {b['diff']:+.3f}, bootstrap p {b['p']:.4f} (95%: {b['lo2.5']:+.3f}..{b['hi97.5']:+.3f})")
if tag == 'old':
    p(f"\nT pass (2008-2019: higher Sharpe than VT, bootstrap p < 0.10): {'PASS' if res['T_pass'] else 'FAIL'}; "
      f"2020-2025-05 same sign: {'yes' if res['check_same_sign'] else 'no'}")
    p(f"Improvement over random (VRP rule, 2008-01..2025-05, 0 bps): {100 * res['improvement_over_random']:+.2f} pts/yr, p {res['improvement_over_random_p']:.3f}")
    d = res['diag_vt_matched']
    p(f"Diagnostic (after the fact): VT with c x{d['scale']:.3f}, mean exposure {d['exposure']:.0%} (the VRP rule's): Sharpe {d['sharpe_vt_matched']:.3f} "
      f"against the VRP rule's {d['sharpe_vrp']:.3f}; bootstrap p {d['boot_vrp_vs_vt_matched']['p']:.3f}")
p('\nTurnover: ' + '; '.join(f"{LAB[nm]} {q['trades']} trades, {q['turnover_per_year']:.2f} a year, cost {q['cost_per_year_pct']:.3f}% a year" for nm, q in res['turnover'].items()))
for nm in ('vrp', 'vt', 'a'):
    a = res['after_cost'][nm]
    p(f"\nAfter costs, {LAB[nm]}: yearly return at 5 bps a side minus SPY's, {a['first_close']} .. {a['last_close']}: {a['number']:+.2f} points a year "
      f"(span annualized {a['span_annualized_diff']:+.2f})")
    p(' | '.join(f"{y_} {v['rule5']:+.1f} vs {v['spy']:+.1f} ({v['diff']:+.1f})" for y_, v in a['years'].items()))
p(f"\nFile-name number (VRP rule): {res['file_name_number']:+.1f}")
for nm, rr in res['replications'].items():
    p(f"\nReplication {nm}: weight log {rr['weight_log']}")
    for k, q in rr.items():
        if k == 'weight_log' or q['months'] < 3:
            continue
        b = q['boot_vrp_vs_vt']
        p(f"  {k}: VRP rule {pc(q['vrp']['annual'])} Sharpe {q['vrp']['sharpe']:.3f} ({q['vrp']['exposure']:.0%}) over random {100 * q['vrp']['over_random']:+.2f} "
          f"(p {q['vrp']['random_p_annual']:.3f}); VT {pc(q['vt']['annual'])} Sharpe {q['vt']['sharpe']:.3f} ({q['vt']['exposure']:.0%}); "
          f"buy and hold {pc(q['bh']['annual'])} Sharpe {q['bh']['sharpe']:.3f}; Sharpe VRP - VT {b['diff']:+.3f}, p {b['p']:.3f}")
if 'previous_run_match' in res:
    p(f"\nExposures over the previous run's decisions equal: {res['previous_run_match']}")
p(f'Runtime {res["seconds"]} s')
txt = '\n'.join(L)
print(txt)
(V.HERE / f'timing_{tag}.txt').write_text(txt + '\n')
