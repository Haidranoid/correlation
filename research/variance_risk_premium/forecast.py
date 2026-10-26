"""P1 (primary) and G (gate): does the variance risk premium forecast SPY's excess return over the next 63 trading
days out of sample, better than the historical mean and better than the volatility levels it is built from?

    python3 -I -B forecast.py old|new|fwd [--noise-calibration N]

VRP_t = VIX_t^2 / 12 - 10,000 x sum of SPY's squared daily log returns over t-20..t (monthly percent squared).
Target: SPY's log excess return from the close of t + 1 to the close of t + 64. Expanding-window least squares from
1993-03-02, refit every 21 trading days on -3.2's calendar (first fit 2007-12-03), daily forecasts; a fit at t uses
only origins s with s + 1 + 63 <= t (asserted on every fit). An origin is scored in a period only if the origin and
its whole target lie inside it.

`old` writes forecast_old.txt / .json and the forecasts for timing.py; `new` scores the 16 unseen months and first
checks that its forecasts over the old span equal the old run's exactly; `fwd` scores forward data.
"""
import json
import sys
import time

sys.path.insert(0, '/home/bhoult/programming/correlation/research/variance_risk_premium')
import numpy as np  # noqa: E402

import vrp_common as V  # noqa: E402

t_start = time.time()
tag = sys.argv[1]
assert tag in V.TAGS
D = V.load(tag)
dates, T = D['dates'], D['T']
S = V.build(D)
y = V.target(S['r'], D['rf'], V.H)
train_mask = dates >= V.TRAIN_FROM

if tag == 'old':
    masks = {'dev': V.period_mask(D, *V.DEV, V.H), 'conf': V.period_mask(D, *V.CONF, V.H),
             'all_2008_2025': V.period_mask(D, V.DEV[0], V.CONF[1], V.H)}
elif tag == 'new':
    masks = {'unseen': V.period_mask(D, *V.UNSEEN, V.H)}
else:
    masks = {'forward': V.period_mask(D, *V.FWD, V.H)}
MAIN = 'dev' if tag == 'old' else list(masks)[0]


def models_for(B):
    return {'mean': [], 'vrp': [B['vrp']],
            'a': [B['lvix'], B['lrv']], 'b': [B['lvix']], 'c': [B['lrv']], 'd': [B['iv'], B['rv']],
            'a_vrp': [B['lvix'], B['lrv'], B['vrp']]}


GATE = {'a': 'log VIX + log RV (the brief\'s baseline)', 'b': 'log VIX alone', 'c': 'log RV alone',
        'd': 'IV and RV in levels (BTZ\'s unrestricted model)'}


def gate_eval(y_, fits, mask):
    m = mask & np.isfinite(y_)
    for k in ('vrp', 'a', 'b', 'c', 'd'):
        m &= np.isfinite(fits[k]['yhat'])
    out = {'n': int(m.sum())}
    if m.sum() < 30:
        return out
    for k in ('mean', 'vrp', 'a', 'b', 'c', 'd'):
        out[f'mse_{k}'] = float(np.mean((y_[m] - fits[k]['yhat'][m]) ** 2))
    out['pass'] = bool(all(out['mse_vrp'] < out[f'mse_{k}'] for k in 'abcd'))
    return out


def full_eval(y_, fits, mask, lags=V.NW_LAGS, block=V.BLOCK, sens=()):
    out = {'p1': V.evaluate(y_, fits['mean']['yhat'], fits['vrp']['yhat'], mask, lags, block, blocks_sens=sens),
           'gate': gate_eval(y_, fits, mask)}
    for k in 'abcd':
        out[f'{k}_vs_mean'] = V.evaluate(y_, fits['mean']['yhat'], fits[k]['yhat'], mask, lags, block)
    out['d_vs_vrp'] = V.evaluate(y_, fits['vrp']['yhat'], fits['d']['yhat'], mask, lags, block)
    out['a_vrp_vs_a'] = V.evaluate(y_, fits['a']['yhat'], fits['a_vrp']['yhat'], mask, lags, block)
    return out


# ---- Noise calibration (a code check, run before the primary is read) ----------------------------------------------
if '--noise-calibration' in sys.argv:
    n_draw = int(sys.argv[sys.argv.index('--noise-calibration') + 1])
    v = S['vrp']
    dm = (dates >= V.TRAIN_FROM) & (dates <= '2007-12-31') & np.isfinite(v)
    pair = dm[1:] & dm[:-1]
    phi = V.corr(v[1:][pair], v[:-1][pair])
    rows = []
    for i in range(n_draw):
        g = np.random.default_rng(V.SEED + 1000 + i)
        e = g.standard_normal(T)
        x = np.empty(T)
        x[0] = e[0]
        for t in range(1, T):
            x[t] = phi * x[t - 1] + np.sqrt(1 - phi ** 2) * e[t]
        x = np.where(np.isfinite(v), x, np.nan)
        fits, _ = V.compare(D, {'mean': [], 'vrp': [x]}, y, V.H, start_mask=train_mask)
        row = {}
        for pn, pm in masks.items():
            if pn == 'all_2008_2025':
                continue
            q = V.evaluate(y, fits['mean']['yhat'], fits['vrp']['yhat'], pm, seed=V.SEED + i)
            row[pn] = {k: q[k] for k in ('r2_oos', 'cw_t', 'cw_boot_p', 'cw_p_normal')}
        rows.append(row)
        print(i, {pn: (round(r_['cw_t'], 2), round(r_['cw_boot_p'], 4), round(100 * r_['r2_oos'], 2)) for pn, r_ in row.items()}, flush=True)
    summ = {}
    for pn in rows[0]:
        t_ = np.array([r_[pn]['cw_t'] for r_ in rows])
        bp = np.array([r_[pn]['cw_boot_p'] for r_ in rows])
        r2 = np.array([r_[pn]['r2_oos'] for r_ in rows])
        summ[pn] = {'cw_t_mean': float(t_.mean()), 'cw_t_sd': float(t_.std()), 'cw_t_p95': float(np.percentile(t_, 95)),
                    'share_t_ge_2': float(np.mean(t_ >= 2)), 'share_t_ge_3': float(np.mean(t_ >= 3)),
                    'share_t_ge_1645': float(np.mean(t_ >= 1.645)), 'share_boot_lt_0023': float(np.mean(bp < V.BAR_P)),
                    'share_boot_lt_005': float(np.mean(bp < 0.05)),
                    'share_full_pass': float(np.mean((t_ >= V.BAR_T) & (bp < V.BAR_P) & (r2 > 0))),
                    'share_r2_pos': float(np.mean(r2 > 0)), 'r2_median': float(np.median(r2)), 'r2_p95': float(np.percentile(r2, 95))}
    cal = {'phi': phi, 'draws': n_draw, 'summary': summ, 'rows': rows}
    with open(V.HERE / f'check_noise_{tag}.json', 'w') as f:
        json.dump(cal, f, indent=1)
    L = [f'Noise calibration of P1 ({n_draw} AR(1) noise predictors, phi = {phi:.4f}, the VRP\'s daily lag-1 autocorrelation 1993-03..2007-12)']
    for pn, q in summ.items():
        L.append(f"{pn}: CW t mean {q['cw_t_mean']:+.2f} sd {q['cw_t_sd']:.2f} (95th pct {q['cw_t_p95']:+.2f}); share t >= 1.645 {q['share_t_ge_1645']:.3f}, "
                 f">= 2 {q['share_t_ge_2']:.3f}, >= 3 {q['share_t_ge_3']:.3f}; bootstrap p < 0.05 {q['share_boot_lt_005']:.3f}, < 0.023 {q['share_boot_lt_0023']:.3f}; "
                 f"full P1 pass {q['share_full_pass']:.3f}; R2_oos > 0 in {q['share_r2_pos']:.3f} (median {100 * q['r2_median']:+.2f}%, 95th pct {100 * q['r2_p95']:+.2f}%)")
    (V.HERE / f'check_noise_{tag}.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    sys.exit(0)

res = {'tag': tag, 'axis': [dates[0], dates[-1], T]}

# ---- The main fits ---------------------------------------------------------------------------------------------------
fits, common = V.compare(D, models_for(S), y, V.H, start_mask=train_mask)
fm = fits['vrp']
res['first_fit'] = {'day': dates[fm['fit_days'][0]], 'eligible_origins': int(fm['nobs'][0]),
                    'first_origin': dates[np.nonzero(common)[0][0]]}
res['periods'] = {}
for pn, pm in masks.items():
    r_ = full_eval(y, fits, pm, sens=V.BLOCKS_SENS if pn == MAIN else ())
    ev = pm & np.isfinite(y) & np.isfinite(fits['vrp']['yhat'])
    r_['first'], r_['last'] = (dates[ev][0], dates[ev][-1]) if ev.any() else (None, None)
    res['periods'][pn] = r_
if tag == 'old':
    q = res['periods']['dev']['p1']
    res['P1_pass'] = bool(q['cw_t'] >= V.BAR_T and q['cw_boot_p'] < V.BAR_P and q['r2_oos'] > 0)
    res['P1_clears_t3'] = bool(q['cw_t'] >= 3)
    res['G_pass'] = res['periods']['dev']['gate']['pass']
    c = res['periods']['conf']['p1']
    res['conf_consistent'] = bool(c['r2_oos'] > 0 and c['cw_p_normal'] < 0.05)
    dev, conf = masks['dev'], masks['conf']
    res['dev_without_2008_2009'] = full_eval(y, fits, dev & (dates >= '2010-01-01'))
    res['conf_without_2020'] = full_eval(y, fits, conf & (dates >= '2021-01-01'))

# ---- Code checks: one pure-noise predictor and a leaky oracle through the same pipeline --------------------------
rng = np.random.default_rng(V.SEED + 99)
noise = np.where(np.isfinite(S['vrp']), rng.standard_normal(T), np.nan)
sd_y = float(np.nanstd(y[train_mask]))
oracle = y + 0.5 * sd_y * rng.standard_normal(T)
res['code_checks'] = {}
for nm, x in (('pure noise', noise), ('leaky oracle (future target + noise)', oracle)):
    f2, _ = V.compare(D, {'mean': [], 'vrp': [x]}, y, V.H, start_mask=train_mask)
    res['code_checks'][nm] = {pn: V.evaluate(y, f2['mean']['yhat'], f2['vrp']['yhat'], pm) for pn, pm in masks.items()}

# ---- Reported ---------------------------------------------------------------------------------------------------------
rep = {}
# horizons
for hh in V.HORIZONS:
    yh = V.target(S['r'], D['rf'], hh)
    f2, _ = V.compare(D, {'mean': [], 'vrp': [S['vrp']]}, yh, hh, start_mask=train_mask)
    mk = ({'dev': V.period_mask(D, *V.DEV, hh), 'conf': V.period_mask(D, *V.CONF, hh)} if tag == 'old'
          else {list(masks)[0]: V.period_mask(D, *(V.UNSEEN if tag == 'new' else V.FWD), hh)})
    rep[f'h = {hh}'] = {pn: V.evaluate(yh, f2['mean']['yhat'], f2['vrp']['yhat'], pm, 2 * hh, max(63, 2 * hh)) for pn, pm in mk.items()}
# variants
iv, rvv = S['iv'], S['rv']
volform = np.sqrt(iv) - np.sqrt(rvv)
logratio = np.log(iv / rvv)
rv22 = 1e4 * V.window_sum(S['r'] ** 2, np.arange(T) - 21, np.arange(T))
vrp22 = iv - rv22
# expected premium: IV - HAR forecast of RV over t + 1 .. t + 21, the HAR fitted under eligibility (s + 21 <= t_k)
t_idx = np.arange(T)
rv_next = 1e4 * V.window_sum(S['r'] ** 2, t_idx + 1, t_idx + 21)
rv1 = 1e4 * S['r'] ** 2
rv5 = 1e4 * V.window_sum(S['r'] ** 2, t_idx - 4, t_idx)
har_sample = train_mask & np.isfinite(rv_next) & np.isfinite(rv1) & np.isfinite(rv5) & np.isfinite(rvv)


def har_model(h_):
    """HAR on the calendar from the first calendar date with 252 eligible origins (eligibility s + h <= t_k)."""
    cal = D['cal']
    X = np.column_stack([rv1, rv5, rvv])
    out = np.full(T, np.nan)
    xok = np.isfinite(X).all(1)
    nfit = 0
    for k in range(len(cal)):
        tk = int(cal[k])
        t_next = int(cal[k + 1]) if k + 1 < len(cal) else T
        elig = np.nonzero(har_sample & (t_idx + h_ <= tk))[0]
        if len(elig) < 252:
            continue
        assert elig.max() + h_ <= tk
        beta, _ = V.ols(X[elig], rv_next[elig])
        rg = np.arange(tk, t_next)
        ok = xok[rg]
        out[rg[ok]] = V.predict(beta, X[rg[ok]])
        nfit += 1
    return out, nfit


erv, n_har = har_model(21)
evrp = iv - erv
res['har_fits'] = n_har
res['evrp_first'] = dates[np.nonzero(np.isfinite(evrp))[0][0]]
variants = {'volatility form sqrt(IV) - sqrt(RV)': volform, 'log ratio ln(IV / RV)': logratio,
            'expected premium IV - HAR forecast': evrp, 'RV over 22 days': vrp22}
for nm, x in variants.items():
    f2, _ = V.compare(D, {'mean': [], 'vrp': [x]}, y, V.H, start_mask=train_mask)
    rep[nm] = {pn: V.evaluate(y, f2['mean']['yhat'], f2['vrp']['yhat'], pm) for pn, pm in masks.items() if pn != 'all_2008_2025'}
ct = V.run_model(D, [S['vrp']], y, V.H, common, restrict='ct')
rep['Campbell-Thompson restrictions'] = {pn: V.evaluate(y, fits['mean']['yhat'], ct['yhat'], pm) for pn, pm in masks.items() if pn != 'all_2008_2025'}
if np.isfinite(D['spy_store']).any():
    Ss = V.build(D, 'spy_store', 'vix')
    ys = V.target(Ss['r'], D['rf'], V.H)
    f2, cm2 = V.compare(D, {'mean': [], 'vrp': [Ss['vrp']]}, ys, V.H, start_mask=train_mask)
    rep["the store's SPY"] = {pn: V.evaluate(ys, f2['mean']['yhat'], f2['vrp']['yhat'], pm) for pn, pm in masks.items() if pn != 'all_2008_2025'}
    rep["the store's SPY"]['first_fit_origins'] = int(f2['vrp']['nobs'][0]) if len(f2['vrp']['nobs']) else 0
res['reported'] = rep
# non-overlapping phases and the monthly version
cal = D['cal']
is_cal = np.zeros(T, bool)
is_cal[cal] = True
phase = np.full(T, -1)
phase[cal] = np.arange(len(cal)) % 3
res['phases'] = {}
res['monthly'] = {}
for pn, pm in masks.items():
    if pn == 'all_2008_2025':
        continue
    ph = {}
    for j in range(3):
        m = pm & (phase == j) & np.isfinite(y) & np.isfinite(fits['vrp']['yhat'])
        if m.sum() < 5:
            continue
        eb, ea = y[m] - fits['mean']['yhat'][m], y[m] - fits['vrp']['yhat'][m]
        f = eb ** 2 - ea ** 2 + (fits['mean']['yhat'][m] - fits['vrp']['yhat'][m]) ** 2
        ph[j] = {'n': int(m.sum()), 'r2_oos': float(1 - np.sum(ea ** 2) / np.sum(eb ** 2)), 'plain_t': V.plain_t(f),
                 'first': dates[m][0]}
    res['phases'][pn] = ph
    res['monthly'][pn] = V.evaluate(y, fits['mean']['yhat'], fits['vrp']['yhat'], pm & is_cal, lags=6, boot=False)
# by year (origin year), the slope at every fit, overlap with the volatility levels
res['by_year'] = {}
allm = np.zeros(T, bool)
for pm in masks.values():
    allm |= pm
for yr in sorted(set(d[:4] for d in dates[allm])):
    m = allm & np.char.startswith(dates.astype(str), yr)
    res['by_year'][yr] = {'p1': V.evaluate(y, fits['mean']['yhat'], fits['vrp']['yhat'], m, boot=False),
                          'gate': gate_eval(y, fits, m)}
res['slope_by_fit'] = {dates[d]: [float(b[0]), float(b[1])] for d, b in zip(fm['fit_days'], fm['betas'])}
res['overlap'] = {pn: {'corr_vrp_lvix': V.corr(S['vrp'][pm], S['lvix'][pm]), 'corr_vrp_lrv': V.corr(S['vrp'][pm], S['lrv'][pm]),
                       'corr_vrp_iv': V.corr(S['vrp'][pm], S['iv'][pm]), 'share_vrp_negative': float(np.mean(S['vrp'][pm] < 0)),
                       'vrp_mean': float(np.nanmean(S['vrp'][pm])), 'vrp_sd': float(np.nanstd(S['vrp'][pm]))}
                  for pn, pm in masks.items()}

# ---- Replications: VXN with QQQ, RVX with IWM -------------------------------------------------------------------------
res['replications'] = {}
repl_fc = {}
i0 = int(np.searchsorted(dates, V.REPL_FROM))
repl_start = dates[i0 + V.RV_WIN]
res['repl_start'] = repl_start
for fund, vol in (('qqq', 'vxn'), ('iwm', 'rvx')):
    if not np.isfinite(D[fund]).any():
        continue
    B = V.build(D, fund, vol, start=repl_start)
    yb = V.target(B['r'], D['rf'], V.H)
    fb, cb = V.compare(D, models_for(B), yb, V.H, start_mask=B['ok_from'])
    rr = {'first_fit_origins': int(fb['vrp']['nobs'][0]), 'first_origin': dates[np.nonzero(cb)[0][0]], 'periods': {}}
    for pn, pm in masks.items():
        if pn == 'all_2008_2025':
            continue
        rr['periods'][pn] = full_eval(yb, fb, pm)
    rr['slope_last_fit'] = float(fb['vrp']['betas'][-1][1])
    res['replications'][f'{vol.upper()} with {fund.upper()}'] = rr
    repl_fc[fund] = {'mean': fb['mean']['yhat'], 'vrp': fb['vrp']['yhat'], 'a': fb['a']['yhat']}
res['eligibility_assertions'] = V.FIT_COUNT['fits']

# ---- Save forecasts for timing.py; the new run must reproduce the old run's forecasts exactly -----------------------
save = {'days': D['days'], 'yhat_mean': fits['mean']['yhat'], 'yhat_vrp': fits['vrp']['yhat'], 'yhat_a': fits['a']['yhat'],
        'fit_days': fm['fit_days'], 'betas_vrp': fm['betas'], 'y63': y}
for fund, d_ in repl_fc.items():
    for k, v_ in d_.items():
        save[f'{fund}_yhat_{k}'] = v_
np.savez(V.WORK / f'fc_{tag}.npz', **save)
assert tag != 'new' or (V.WORK / 'fc_old.npz').exists(), 'run forecast.py old first'
if tag in ('new', 'fwd') and (V.WORK / f"fc_{'old' if tag == 'new' else 'new'}.npz").exists():
    prev = 'old' if tag == 'new' else 'new'
    zo = np.load(V.WORK / f'fc_{prev}.npz')
    n0 = len(zo['days'])
    assert np.array_equal(zo['days'], D['days'][:n0])
    chk = {}
    for k in ('yhat_mean', 'yhat_vrp', 'yhat_a'):
        a, b = zo[k], save[k][:n0]
        chk[k] = {'equal': bool(np.array_equal(a, b, equal_nan=True)), 'max_abs_diff': float(np.nanmax(np.abs(a - b)))}
    for fund in repl_fc:
        if f'{fund}_yhat_vrp' not in zo:
            continue
        a, b = zo[f'{fund}_yhat_vrp'], save[f'{fund}_yhat_vrp'][:n0]
        ok = np.isfinite(a) & np.isfinite(b)
        chk[f'{fund}_yhat_vrp'] = {'equal': bool(np.array_equal(a, b, equal_nan=True)), 'corr': V.corr(a[ok], b[ok]),
                                  'max_abs_diff': float(np.nanmax(np.abs(a - b)))}
    res['previous_run_match'] = chk
    if tag == 'new':
        assert all(chk[k]['equal'] for k in ('yhat_mean', 'yhat_vrp', 'yhat_a')), f'old-span forecasts differ: {chk}'
res['seconds'] = round(time.time() - t_start)
with open(V.HERE / f'forecast_{tag}.json', 'w') as f:
    json.dump(res, f, indent=1, default=float)

# ---- Report ---------------------------------------------------------------------------------------------------------
L = []
p = L.append
f3 = V.fmt
pct = lambda v: 'n/a' if v is None or not np.isfinite(v) else f'{100 * v:+.2f}%'  # noqa: E731
p(f"Forecast ({tag}). Day axis {dates[0]}..{dates[-1]}. First fit {res['first_fit']['day']} on {res['first_fit']['eligible_origins']} eligible origins "
  f"(from {res['first_fit']['first_origin']}). Eligibility assertion (s + 1 + h <= t) held on all {res['eligibility_assertions']} fits.")
p('\nP1: the VRP model against the historical mean (h = 63; CW t with Newey-West at 126 lags; bootstrap with 126-day blocks):')
p('| Period | Origins | First..last | MSE mean | MSE VRP | R2_oos (95%) | CW t | p normal | p bootstrap |')
p('|---|---|---|---|---|---|---|---|---|')
for pn, r_ in list(res['periods'].items()) + ([('dev without 2008-2009', res['dev_without_2008_2009']), ('conf without 2020', res['conf_without_2020'])] if tag == 'old' else []):
    q = r_['p1']
    if q.get('n', 0) < 30:
        p(f'| {pn} | {q.get("n", 0)} | too few |')
        continue
    ci = f" ({pct(q['r2_lo95'])}..{pct(q['r2_hi95'])})" if 'r2_lo95' in q else ''
    p(f"| {pn} | {q['n']} | {r_.get('first')}..{r_.get('last')} | {q['mse_base']:.5f} | {q['mse_alt']:.5f} | {pct(q['r2_oos'])}{ci} | "
      f"{q['cw_t']:+.2f} | {f3(q['cw_p_normal'], 4)} | {f3(q.get('cw_boot_p', np.nan), 4)} |")
q = res['periods'][MAIN]['p1']
p('Bootstrap p with other blocks (' + MAIN + '): ' + ', '.join(f"{b} days {f3(q.get(f'cw_boot_p_block{b}', np.nan), 4)}" for b in V.BLOCKS_SENS))
if tag == 'old':
    p(f"P1 pass (2008-2019: CW t >= 2, bootstrap p < 0.023, R2_oos > 0): {'PASS' if res['P1_pass'] else 'FAIL'}; clears t >= 3: "
      f"{'yes' if res['P1_clears_t3'] else 'no'}; 2020-2025-06 consistent (R2_oos > 0 and normal p < 0.05): {'yes' if res['conf_consistent'] else 'no'}")
p('\nG: mean squared error of each model (x 1e3), same origins:')
p('| Period | Origins | Historical mean | VRP | (a) log VIX + log RV | (b) log VIX | (c) log RV | (d) IV, RV levels | Gate |')
p('|---|---|---|---|---|---|---|---|---|')
for pn, r_ in res['periods'].items():
    g = r_['gate']
    if g.get('n', 0) < 30:
        continue
    p(f"| {pn} | {g['n']} | {1e3 * g['mse_mean']:.4f} | {1e3 * g['mse_vrp']:.4f} | {1e3 * g['mse_a']:.4f} | {1e3 * g['mse_b']:.4f} | "
      f"{1e3 * g['mse_c']:.4f} | {1e3 * g['mse_d']:.4f} | {'pass' if g['pass'] else 'fail'} |")
p('\nReported with G (R2_oos, CW t, p bootstrap):')
p('| Period | ' + ' | '.join(['(a) vs mean', '(b) vs mean', '(c) vs mean', '(d) vs mean', '(d) vs VRP (BTZ\'s restriction)', '(a) + VRP vs (a) (functional form)']) + ' |')
p('|---|---|---|---|---|---|---|')
for pn, r_ in res['periods'].items():
    cells = []
    for k in ('a_vs_mean', 'b_vs_mean', 'c_vs_mean', 'd_vs_mean', 'd_vs_vrp', 'a_vrp_vs_a'):
        q = r_[k]
        cells.append('too few' if q.get('n', 0) < 30 else f"{pct(q['r2_oos'])}, {q['cw_t']:+.2f}, {f3(q.get('cw_boot_p', np.nan), 3)}")
    p(f"| {pn} | " + ' | '.join(cells) + ' |')
p('\nCode checks (P1 pipeline):')
for nm, d2 in res['code_checks'].items():
    p(f"  {nm}: " + '; '.join(f"{pn} R2_oos {pct(q['r2_oos'])}, CW t {q['cw_t']:+.2f}, p boot {f3(q.get('cw_boot_p', np.nan), 4)}" for pn, q in d2.items() if q.get('n', 0) >= 30))
p('\nReported (each against the historical mean; R2_oos, CW t, p bootstrap):')
for nm, d2 in res['reported'].items():
    p(f"  {nm}: " + '; '.join(f"{pn} n {q['n']} {pct(q['r2_oos'])}, t {q['cw_t']:+.2f}, p {f3(q.get('cw_boot_p', np.nan), 3)}"
                               for pn, q in d2.items() if isinstance(q, dict) and q.get('n', 0) >= 30))
p(f"  (HAR fits for the expected premium: {res['har_fits']}; it starts {res['evrp_first']})")
p('\nNon-overlapping phases (origins 63 days apart on the calendar; R2_oos, plain t of the CW mean):')
for pn, ph in res['phases'].items():
    p(f"  {pn}: " + '; '.join(f"phase {j} n {q['n']} from {q['first']}: {pct(q['r2_oos'])}, t {q['plain_t']:+.2f}" for j, q in ph.items()))
for pn, q in res['monthly'].items():
    if q.get('n', 0) >= 30:
        p(f"Monthly origins (the calendar), {pn}: n {q['n']}, R2_oos {pct(q['r2_oos'])}, CW t (NW 6) {q['cw_t']:+.2f}")
p('\nBy origin year: R2_oos, CW t; gate')
p('| Year | Origins | R2_oos | CW t | MSE VRP below all four | ')
p('|---|---|---|---|---|')
for yr, d2 in res['by_year'].items():
    q, g = d2['p1'], d2['gate']
    if q.get('n', 0) < 30:
        continue
    p(f"| {yr} | {q['n']} | {pct(q['r2_oos'])} | {q['cw_t']:+.2f} | {'yes' if g.get('pass') else 'no'} |")
sl = res['slope_by_fit']
ks = list(sl)
p('\nVRP slope (decimal 63-day log excess return per monthly %^2), every 12th fit: ' + ', '.join(f"{k} {sl[k][1] * 1e3:+.3f}e-3" for k in ks[::12]) + f"; last {ks[-1]} {sl[ks[-1]][1] * 1e3:+.3f}e-3")
p('Overlap: ' + '; '.join(f"{pn}: corr(VRP, log VIX) {q['corr_vrp_lvix']:+.2f}, corr(VRP, log RV) {q['corr_vrp_lrv']:+.2f}, corr(VRP, IV) {q['corr_vrp_iv']:+.2f}, "
                          f"VRP mean {q['vrp_mean']:.2f} sd {q['vrp_sd']:.2f}, negative {q['share_vrp_negative']:.1%}" for pn, q in res['overlap'].items()))
p(f'\nReplications (origins from {repl_start}):')
for nm, rr in res['replications'].items():
    p(f"  {nm}: first fit on {rr['first_fit_origins']} origins (from {rr['first_origin']}); last-fit slope {rr['slope_last_fit'] * 1e3:+.3f}e-3")
    for pn, r_ in rr['periods'].items():
        q, g = r_['p1'], r_['gate']
        if q.get('n', 0) < 30:
            continue
        p(f"    {pn}: n {q['n']}, R2_oos {pct(q['r2_oos'])} ({pct(q.get('r2_lo95', np.nan))}..{pct(q.get('r2_hi95', np.nan))}), CW t {q['cw_t']:+.2f}, "
          f"p boot {f3(q.get('cw_boot_p', np.nan), 3)}; gate {'pass' if g.get('pass') else 'fail'} (MSE x1e3: mean {1e3 * g['mse_mean']:.4f}, VRP {1e3 * g['mse_vrp']:.4f}, "
          f"a {1e3 * g['mse_a']:.4f}, b {1e3 * g['mse_b']:.4f}, c {1e3 * g['mse_c']:.4f}, d {1e3 * g['mse_d']:.4f})")
if 'previous_run_match' in res:
    p(f"\nForecasts over the previous run's span: {res['previous_run_match']}")
p(f'Runtime {res["seconds"]} s')
txt = '\n'.join(L)
print(txt)
(V.HERE / f'forecast_{tag}.txt').write_text(txt + '\n')
