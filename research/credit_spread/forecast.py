"""P1 (primary), S1 (secondary) and the forecasting checks C2-C10.

    python3 -I -B forecast.py old --noise-calibration 200    (code checks only; the spread model is never fitted)
    python3 -I -B forecast.py old|new|fwd                    (after inputs.py <tag>)

Target: y(t) = log P(t + 22) - log P(t + 1), SPY's (Yahoo adjusted close) log total return over the 21 days from the
trade's close. Baseline B: least squares on log RV1, log RV5, log RV22, log VIX and R(t) = log P(t) - log P(t - 6).
Spread model: B + dS, dS = BAA10Y(t - 1) - BAA10Y(t - 6). Expanding window from the first origin with every input,
refit every 21 trading days on -3.2's calendar extended backward, at least 1,260 eligible origins per fit; a fit at
day t uses only origins s with s + 1 + h <= t (asserted on every fit). Both models are fitted on the same origins.
Clark-West with Newey-West at 2h lags and a 63-day moving-block bootstrap (21-day reported beside it).

`old` writes p1_old.txt / p1_old.json here and fc_old.npz to the work dir; `new` evaluates only the 16 unseen months.
"""
import json
import sys
import time

sys.path.insert(0, '/home/bhoult/programming/correlation/research/credit_spread')
import numpy as np  # noqa: E402

import cs_common as K  # noqa: E402

t_start = time.time()
tag = sys.argv[1]
assert tag in K.TAGS
z = K.load_inputs(tag)
days = z['days']
dates = K.iso_days(days)
T = len(days)
offset = int(z['offset'])
refits = K.refit_days(T, offset)
cal = refits[refits >= offset + 252] - offset
assert np.array_equal(cal, np.arange(252, T - offset, K.H)), "refit calendar differs from -3.2's"
if tag == 'old':
    sys.path.insert(1, '/home/bhoult/programming/correlation/research/market_timing')
    mt = np.load(K.Path.home() / '.local/share/correlation2/experiments/market_timing/states_old.npz')
    assert np.array_equal(mt['rebal'], cal), "calendar differs from -3.2's saved one"

lp = np.log(z['P'])
lx = np.log(z['X'])


def target(logp, h):
    """y[t] = logp(t + 1 + h) - logp(t + 1); NaN where t + 1 + h is past the axis."""
    y = np.full(T, np.nan)
    y[:T - 1 - h] = logp[1 + h:] - logp[1:T - h]
    for t in range(500, T - h - 2, 1013):               # direct check of the indices
        if np.isfinite(y[t]):
            assert abs(y[t] - (logp[t + 1 + h] - logp[t + 1])) < 1e-12
    return y


y21 = target(lp, K.H)
BASE = [z['lrv1'], z['lrv5'], z['lrv22'], z['lvix'], z['R']]
BASE_NAMES = ['log RV1', 'log RV5', 'log RV22', 'log VIX', 'R']
dS = z['dS1']
I_up = np.where(np.isfinite(z['R']), (z['R'] >= 0).astype(float), np.nan)
LAGS = {5: 10, 21: 42, 63: 126}
BLOCKS = {5: 63, 21: 63, 63: 189}

if tag == 'old':
    masks = {'early': K.sel(dates, *K.EARLY), 'dev': K.sel(dates, *K.DEV), 'check': K.sel(dates, *K.CHECK)}
elif tag == 'new':
    masks = {'unseen': K.sel(dates, *K.UNSEEN)}
else:
    masks = {'forward': K.sel(dates, *K.FWD)}


def med_coef(m, mask, col=-1):
    """Median, min, max and share negative of a coefficient over the fits whose forecasts score origins in mask."""
    fo = m['fit_of'][mask & np.isfinite(m['yhat'])]
    fo = np.unique(fo[fo >= 0])
    if len(fo) == 0:
        return {'fits': 0}
    b = m['betas'][fo, col]
    return {'fits': int(len(fo)), 'median': float(np.median(b)), 'min': float(b.min()), 'max': float(b.max()),
            'share_negative': float(np.mean(b < 0))}


def evaluate_all(mb, ma, mh, y, h, coef_col=-1, extra=None):
    out = {}
    for pn, pm in (extra or masks).items():
        q = K.evaluate(y, mb['yhat'], ma['yhat'], pm, LAGS[h], fh=mh['yhat'], block=BLOCKS[h])
        q['coef'] = med_coef(ma, pm, coef_col)
        ev = pm & np.isfinite(y) & np.isfinite(mb['yhat']) & np.isfinite(ma['yhat'])
        if ev.any():
            q['first'], q['last'] = dates[ev][0], dates[ev][-1]
        out[pn] = q
    return out


def by_year(y, mb, ma, mh, mask, h):
    out = {}
    ev = mask & np.isfinite(y) & np.isfinite(mb['yhat']) & np.isfinite(ma['yhat'])
    for yr in sorted(set(d[:4] for d in dates[ev])):
        m = ev & np.char.startswith(dates.astype(str), yr)
        out[yr] = K.evaluate(y, mb['yhat'], ma['yhat'], m, LAGS[h], fh=mh['yhat'], boot=False)
    return out


res = {'tag': tag}

# ---- Code checks: noise built like dS, and a leaky oracle (run before P1's result is read) ------------------------
if '--noise-calibration' in sys.argv:
    n_draw = int(sys.argv[sys.argv.index('--noise-calibration') + 1])
    S = z['S']
    dd = np.diff(S)
    okd = np.isfinite(dd[1:]) & np.isfinite(dd[:-1])
    phi = float(np.corrcoef(dd[1:][okd], dd[:-1][okd])[0, 1])        # daily spread changes' lag-1 autocorrelation
    common = np.isfinite(np.column_stack(BASE + [dS])).all(1) & np.isfinite(y21)
    mb = K.run_model(np.column_stack(BASE), y21, common, refits, K.H, K.MIN_FIT)
    mh = K.run_model(np.zeros((T, 0)), y21, common, refits, K.H, K.MIN_FIT)
    nanmask = ~np.isfinite(dS)
    out = {'phi_daily_spread_changes': phi, 'draws': n_draw}
    rng = np.random.default_rng(K.SEED + 99)
    sd_dev = float(np.nanstd(y21[masks['dev']]))
    oracle = y21 + rng.standard_normal(T) * sd_dev
    ma = K.run_model(np.column_stack(BASE + [oracle]), y21, common & np.isfinite(oracle), refits, K.H, K.MIN_FIT)
    out['leaky_oracle'] = {pn: K.evaluate(y21, mb['yhat'], ma['yhat'], pm, 42, fh=mh['yhat']) for pn, pm in masks.items()}
    for kind in ('iid', 'ar1'):
        rows = []
        for i in range(n_draw):
            g = np.random.default_rng(K.SEED + 1000 + i + (0 if kind == 'iid' else 50_000))
            e = g.standard_normal(T)
            if kind == 'ar1':
                x = np.empty(T)
                x[0] = e[0]
                for t in range(1, T):
                    x[t] = phi * x[t - 1] + np.sqrt(1 - phi ** 2) * e[t]
                e = x
            ns = np.full(T, np.nan)
            ns[5:] = np.convolve(e, np.ones(5), 'valid')[:T - 5]       # e(t-5) + .. + e(t-1): like dS1's five changes
            ns[nanmask] = np.nan
            ma = K.run_model(np.column_stack(BASE + [ns]), y21, common, refits, K.H, K.MIN_FIT)
            row = {}
            for pn, pm in masks.items():
                if pn == 'early':
                    continue
                q = K.evaluate(y21, mb['yhat'], ma['yhat'], pm, 42, fh=mh['yhat'], seed=K.SEED + i)
                row[pn] = {k: q[k] for k in ('rel_impr', 'cw_t', 'cw_boot_p', 'cw_p_normal')}
                row[pn]['coef_median'] = med_coef(ma, pm)['median']
            rows.append(row)
            if i % 20 == 0:
                print(kind, i, {pn: (round(r_['cw_t'], 2), round(r_['cw_boot_p'], 4), round(100 * r_['rel_impr'], 2)) for pn, r_ in row.items()}, flush=True)
        summ = {}
        for pn in rows[0]:
            t_ = np.array([r_[pn]['cw_t'] for r_ in rows])
            bp = np.array([r_[pn]['cw_boot_p'] for r_ in rows])
            ri = np.array([r_[pn]['rel_impr'] for r_ in rows])
            cm = np.array([r_[pn]['coef_median'] for r_ in rows])
            summ[pn] = {'cw_t_mean': float(t_.mean()), 'cw_t_sd': float(t_.std()), 'cw_t_p95': float(np.percentile(t_, 95)),
                        'cw_t_max': float(t_.max()), 'share_cw_t_ge_1645': float(np.mean(t_ >= 1.645)),
                        'share_cw_t_ge_3': float(np.mean(t_ >= 3)), 'share_boot_p_lt_005': float(np.mean(bp < 0.05)),
                        'share_boot_p_lt_00013': float(np.mean(bp < 0.0013)),
                        'share_full_pass': float(np.mean((t_ >= 3) & (bp < 0.0013) & (ri > 0) & (cm < 0))),
                        'share_rel_impr_pos': float(np.mean(ri > 0)), 'rel_impr_median': float(np.median(ri)),
                        'rel_impr_p95': float(np.percentile(ri, 95))}
        out[kind] = {'summary': summ, 'rows': rows}
    with open(K.HERE / f'check_noise_{tag}.json', 'w') as f:
        json.dump(out, f, indent=1, default=float)
    L = [f"Code checks for P1 ({tag}): {n_draw} noise series of each kind built like dS (five daily changes ending t - 1); "
         f"AR(1) uses the daily spread changes' lag-1 autocorrelation, {phi:.3f}. Fits checked: {K.FIT_LOG}"]
    for pn, q in out['leaky_oracle'].items():
        if q.get('n', 0) >= 30:
            L.append(f"leaky oracle (y + noise, sd {sd_dev:.4f}), {pn}: 1 - MSE ratio {100 * q['rel_impr']:+.1f}%, CW t {q['cw_t']:+.2f}")
    for kind in ('iid', 'ar1'):
        for pn, q in out[kind]['summary'].items():
            L.append(f"{kind} {pn}: CW t mean {q['cw_t_mean']:+.2f} sd {q['cw_t_sd']:.2f} (95th pct {q['cw_t_p95']:+.2f}, max {q['cw_t_max']:+.2f}); "
                     f"share t >= 1.645 {q['share_cw_t_ge_1645']:.3f}, >= 3 {q['share_cw_t_ge_3']:.3f}; bootstrap p < 0.05 {q['share_boot_p_lt_005']:.3f}, "
                     f"< 0.0013 {q['share_boot_p_lt_00013']:.3f}; full P1 pass {q['share_full_pass']:.3f}; lower error in {q['share_rel_impr_pos']:.3f} "
                     f"(median {100 * q['rel_impr_median']:+.2f}%, 95th pct {100 * q['rel_impr_p95']:+.2f}%)")
    txt = '\n'.join(L)
    print(txt)
    (K.HERE / f'check_noise_{tag}.txt').write_text(txt + '\n')
    sys.exit(0)

# ---- P1 (primary) -------------------------------------------------------------------------------------------------
mb, ms, mh = K.compare(BASE, BASE + [dS], y21, K.H, refits, K.MIN_FIT)
res['primary'] = evaluate_all(mb, ms, mh, y21, K.H)
res['first_fit_day'] = dates[ms['fit_days'][0]]
res['first_fit_origins'] = int(ms['nobs'][0])
if tag == 'old':
    dev, chk = masks['dev'], masks['check']
    for nm, m in (('dev_without_2008_2009', dev & ~K.sel(dates, '2008-01-01', '2009-12-31')),
                  ('check_without_2020', chk & ~K.sel(dates, '2020-01-01', '2020-12-31')),
                  ('all_2008_2025', dev | chk)):
        q = K.evaluate(y21, mb['yhat'], ms['yhat'], m, 42, fh=mh['yhat'])
        q['coef'] = med_coef(ms, m)
        res['primary'][nm] = q
    d = res['primary']['dev']
    res['P1_conditions'] = {'cw_t_ge_3': bool(d['cw_t'] >= 3), 'boot_p_lt_00013': bool(d['cw_boot_p'] < 0.0013),
                            'lower_mse': bool(d['mse_alt'] < d['mse_base']), 'negative_median_coef': bool(d['coef']['median'] < 0)}
    res['P1_pass'] = bool(all(res['P1_conditions'].values()))
    c = res['primary']['check']
    res['check_consistent'] = bool(c['rel_impr'] > 0 and c['cw_p_normal'] < 0.05 and c['coef']['median'] < 0)
    res['by_year'] = by_year(y21, mb, ms, mh, masks['early'] | dev | chk, K.H)
res['coef_by_fit'] = {dates[d_]: [float(x) for x in b] for d_, b in zip(ms['fit_days'][::12], ms['betas'][::12])}
res['coef_names'] = ['const'] + BASE_NAMES + ['dS']
res['last_fit'] = {'day': dates[ms['fit_days'][-1]], 'base': mb['betas'][-1].tolist(), 'spread': ms['betas'][-1].tolist()}
np.savez(K.WORK / f'fc_{tag}.npz', days=days, y21=y21, yhat_base=mb['yhat'], yhat_spread=ms['yhat'], yhat_hist=mh['yhat'],
         fit_of=ms['fit_of'], fit_days=ms['fit_days'], betas_base=mb['betas'], betas_spread=ms['betas'])

# ---- S1 (secondary): the conditional form ---------------------------------------------------------------------------
dS_up = dS * I_up
mb1, m1, mh1 = K.compare(BASE, BASE + [dS_up], y21, K.H, refits, K.MIN_FIT)
res['S1'] = evaluate_all(mb1, m1, mh1, y21, K.H)
if tag == 'old':
    d = res['S1']['dev']
    res['S1_conditions'] = {'cw_p_normal_lt_005': bool(d['cw_p_normal'] < 0.05), 'boot_p_lt_005': bool(d['cw_boot_p'] < 0.05),
                            'lower_mse': bool(d['mse_alt'] < d['mse_base']), 'negative_median_coef': bool(d['coef']['median'] < 0)}
    res['S1_pass_own_bar'] = bool(all(res['S1_conditions'].values()))
    res['S1_counts'] = bool(res['S1_pass_own_bar'] and res['P1_pass'])

# ---- Reported checks ------------------------------------------------------------------------------------------------
rep = {}


def add(name, base_cols, alt_cols, y, h=K.H, min_fit=K.MIN_FIT):
    b_, a_, h_ = K.compare(base_cols, alt_cols, y, h, refits, min_fit)
    rep[name] = evaluate_all(b_, a_, h_, y, h)
    return b_, a_, h_


add('S1 increment: B + dS -> B + dS + dS*1[R >= 0]', BASE + [dS], BASE + [dS, dS_up], y21)
add('C2 default spread: B + dD (BAA10Y - AAA10Y)', BASE, BASE + [z['dD1']], y21)
add('C2 default spread, conditional: B + dD*1[R >= 0]', BASE, BASE + [z['dD1'] * I_up], y21)
yX = target(lx, K.H)
BX = [z['lrv1X'], z['lrv5X'], z['lrv22X'], z['lvix'], z['RX']]
add("C3 XLF's next 21 days: XLF's own baseline + dS", BX, BX + [dS], yX, min_fit=K.MIN_FIT_XLF)
add('C3 XLF - SPY over the next 21 days: B + dS', BASE, BASE + [dS], yX - y21, min_fit=K.MIN_FIT_XLF)
add('C4 Treasury leg: B + dG -> B + dG + dS', BASE + [z['dG1']], BASE + [z['dG1'], dS], y21)
add('C5 optimistic timing: B + dS through t (S(t) - S(t-5))', BASE, BASE + [z['dS0']], y21)
add('C5 extra caution: B + dS through t - 2', BASE, BASE + [z['dS2']], y21)
add("C6 the spread's level: B + S(t-1) -> B + S(t-1) + dS", BASE + [z['L1']], BASE + [z['L1'], dS], y21)
add('C6 the level alone: B -> B + S(t-1)', BASE, BASE + [z['L1']], y21)
for h in (5, 63):
    add(f'C7 h = {h}: B + dS', BASE, BASE + [dS], target(lp, h), h=h)
add('C8 traded credit: B + -(LQD - IEF, 5 days through t)', BASE, BASE + [z['cETF']], y21)
res['reported'] = rep
res['fits'] = dict(K.FIT_LOG)
if tag == 'new':                                           # data check: the two runs' forecasts on the old span
    zo = np.load(K.WORK / 'fc_old.npz')
    n0 = len(zo['days'])
    for nm in ('yhat_base', 'yhat_spread'):
        a, b = zo[nm], np.load(K.WORK / 'fc_new.npz')[nm][:n0]
        ok = np.isfinite(a) & np.isfinite(b)
        res[f'old_vs_new_{nm}'] = {'corr': float(np.corrcoef(a[ok], b[ok])[0, 1]), 'max_abs_diff': float(np.abs(a[ok] - b[ok]).max()),
                                   'days': int(ok.sum())}
res['seconds'] = round(time.time() - t_start)
with open(K.HERE / f'p1_{tag}.json', 'w') as f:
    json.dump(res, f, indent=1, default=float)

# ---- Report -------------------------------------------------------------------------------------------------------
L = []
p = L.append
f3 = K.fmt


def row(nm, pn, q):
    if q.get('n', 0) < 30:
        return f'| {nm} | {pn} | {q.get("n", 0)} | too few | | | | | | |'
    ci = f" ({100 * q['rel_impr_lo95']:+.2f}..{100 * q['rel_impr_hi95']:+.2f})" if 'rel_impr_lo95' in q else ''
    c = q.get('coef', {})
    return (f"| {nm} | {pn} | {q['n']} | {100 * q['rel_impr']:+.2f}%{ci} | {q['cw_t']:+.2f} | {f3(q['cw_p_normal'], 4)} | "
            f"{f3(q.get('cw_boot_p', np.nan), 4)} | {f3(q.get('cw_boot_p_21', np.nan), 4)} | "
            f"{100 * q.get('r2os_base', np.nan):+.2f}% / {100 * q.get('r2os_alt', np.nan):+.2f}% | "
            f"{f3(c.get('median', np.nan), 4)} ({100 * c.get('share_negative', np.nan):.0f}% < 0) |")


HEAD = ('| Model | Period | Origins | 1 - MSE ratio (95%) | CW t | p normal | p boot 63 | p boot 21 | R2 OS base / alt | '
        'Added coef, median over fits |\n|---|---|---|---|---|---|---|---|---|---|')
p(f"P1 ({tag}). First fit {res['first_fit_day']} on {res['first_fit_origins']} eligible origins; eligibility assertion "
  f"(s + 1 + h <= t) held on {res['fits']['fits']} fits ({res['fits']['lstsq_checks']} checked against lstsq).")
p(HEAD)
for pn, q in res['primary'].items():
    p(row('P1: B + dS', pn, q))
for pn, q in res['S1'].items():
    p(row('S1: B + dS*1[R>=0]', pn, q))
if tag == 'old':
    p(f"\nP1 pass (2008-2019: CW t >= 3, bootstrap p < 0.0013, lower MSE, negative median coefficient): "
      f"{'PASS' if res['P1_pass'] else 'FAIL'} {res['P1_conditions']}")
    p(f"P1 check period consistent (lower MSE, CW p < 0.05, negative coefficient): {'yes' if res['check_consistent'] else 'no'}")
    p(f"S1 own bar (p < 0.05 both ways, lower MSE, negative coefficient): {'met' if res['S1_pass_own_bar'] else 'not met'} "
      f"{res['S1_conditions']}; counts: {res['S1_counts']}")
    p('\nBy year (P1):')
    p('| Year | Origins | MSE base | MSE spread | 1 - MSE ratio | CW t | R2 OS base / spread |')
    p('|---|---|---|---|---|---|---|')
    for yr, q in res['by_year'].items():
        if q.get('n', 0) >= 30:
            p(f"| {yr} | {q['n']} | {q['mse_base']:.5f} | {q['mse_alt']:.5f} | {100 * q['rel_impr']:+.2f}% | {q['cw_t']:+.2f} | "
              f"{100 * q['r2os_base']:+.1f}% / {100 * q['r2os_alt']:+.1f}% |")
p('\nReported checks (each against its own baseline, refit on the same origins):')
p(HEAD)
for nm, d2 in rep.items():
    for pn, q in d2.items():
        p(row(nm, pn, q))
p('\nCoefficients every 12th fit (const, ' + ', '.join(BASE_NAMES) + ', dS):')
for d_, b in res['coef_by_fit'].items():
    p(f"  {d_}: " + ' '.join(f'{x:+.4f}' for x in b))
for k2 in [k for k in res if k.startswith('old_vs_new')]:
    p(f'{k2}: {res[k2]}')
p(f"Runtime {res['seconds']} s")
txt = '\n'.join(L)
print(txt)
(K.HERE / f'p1_{tag}.txt').write_text(txt + '\n')
