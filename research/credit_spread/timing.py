"""S2 (secondary): a SPY-or-cash rule against variance timing, and the README's two numbers.

    python3 -I -B timing.py old|new|fwd        (after inputs.py <tag>)

-3.2's set-up with an earlier timing test's timing: decisions on -3.2's fixed calendar (export days 252 + 21k), made after the close of
t0, bought at the close of t0 + 1 and held to the close of t0 + 22; SPY or cash at 0; 5 bps on every SPY trade (the
first buy, each month's change, the trim back to target). Prices: Yahoo's adjusted-close SPY on export days, given
to research/market_timing/engine.py's functions unchanged (rel_paths, measures, random_timing,
block_bootstrap_sharpe). Rule B: w = min(1, c / RV(t0)), RV = variance (ddof 0) of the 21 daily log returns ending at
t0, c = median of RV over the previous 1,260 trading days (>= 252). The spread rule: 0 for the month when the warning
fires (dS > 0, dS >= the 80th percentile of the previous 1,260 days' dS (>= 252), and R >= 0), else B's weight.
"""
import json
import sys
import time

sys.path.insert(0, '/home/bhoult/programming/correlation/research/credit_spread')
import numpy as np  # noqa: E402

import cs_common as K  # noqa: E402

sys.path.insert(1, '/home/bhoult/programming/correlation/research/market_timing')
import engine as G  # noqa: E402  research/market_timing/engine.py (read-only)

t_start = time.time()
tag = sys.argv[1]
assert tag in K.TAGS
z = K.load_inputs(tag)
days = z['days']
dates = K.iso_days(days)
T = len(days)
off = int(z['offset'])
COST = K.COST
DPY = G.DPY


def realized_var(lp):
    """-3.2's quantity on every axis day: variance (ddof 0) of the 21 daily log returns ending at t (all 21 finite)."""
    r = np.diff(lp, prepend=np.nan)
    out = np.full(T, np.nan)
    for t in range(K.H, T):
        w = r[t - K.H + 1:t + 1]
        if np.isfinite(w).all():
            out[t] = float(np.var(w))
    return out


def trailing(x, t, fn):
    """fn over the previous (not current) finite values x[t-1260 .. t-1], if at least 252; else NaN."""
    prev = x[max(0, t - K.PCT_MAX):t]
    prev = prev[np.isfinite(prev)]
    return float(fn(prev)) if len(prev) >= K.PCT_MIN else float('nan')


lpY = np.log(z['P'])
lpS = np.log(z['Pst'])
lpX = np.log(z['X'])
RV = {'yahoo': realized_var(lpY), 'store': realized_var(lpS), 'xlf': realized_var(lpX)}
Rs = {'yahoo': z['R'], 'store': z['Rst'], 'xlf': z['RX']}
Px = {'yahoo': z['P'][off:], 'store': z['Pst'][off:], 'xlf': z['X'][off:]}      # on export days
Te = T - off
rebal = np.arange(252, Te, K.H)
if tag == 'old':
    lo, hi = K.DEV[0], K.CHECK[1]
elif tag == 'new':
    lo, hi = K.UNSEEN
else:
    lo, hi = K.FWD
ed = dates[off:]
t0s = np.array([int(t) for t in rebal if lo <= ed[t] <= hi and t + 1 + K.H <= Te - 1])
assert np.all(np.diff(t0s) == K.H), 'decision dates must be contiguous'
A0 = t0s + off                                     # axis index of each decision day
dd = ed[t0s]
M = len(t0s)


def weights_B(src):
    rv = RV[src][A0]
    c = np.array([trailing(RV[src], a, np.median) for a in A0])
    return np.minimum(1.0, c / rv), rv, c


def fire(dS, src, q, stock=True):
    """The warning at each decision: dS > 0, dS at or above the q-th percentile of the previous 1,260 days' dS
    (q None: no percentile condition), and (stock) R >= 0."""
    x = dS[A0]
    f = np.isfinite(x) & (x > 0)
    thr = np.full(M, np.nan)
    if q is not None:
        thr = np.array([trailing(dS, a, lambda v: np.percentile(v, 100 * q)) for a in A0])
        f &= np.isfinite(thr) & (x >= thr)
    if stock:
        rr = Rs[src][A0]
        f &= np.isfinite(rr) & (rr >= 0)
    return f, thr


def path(P, t0, w, cost):
    """Daily returns [M, 21]: buy w at the close of t0 + 1 (paying cost on |w - drifted weight|), hold to the close of
    t0 + 22, cash at 0. Copied from an earlier timing test's code."""
    out = np.empty((len(t0), K.H))
    wd, prev = 0.0, 1.0
    for m, t in enumerate(t0):
        s = t + 1
        v = prev * (1.0 - abs(w[m] - wd) * cost)
        rel = P[s + 1:s + K.H + 1] / P[s]
        val = v * ((1 - w[m]) + w[m] * rel)
        out[m] = val / np.concatenate([[prev], val[:-1]]) - 1.0
        prev = val[-1]
        wd = w[m] * rel[-1] / ((1 - w[m]) + w[m] * rel[-1])
    return out


def boot_idx(Mm, block, n=K.N_BOOT, seed=G.C.SEED + 1):
    """engine.block_bootstrap_sharpe's paired circular block draws of months (same generator and seed)."""
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(Mm / block))
    starts = rng.integers(0, Mm, size=(n, nb))
    return ((starts[:, :, None] + np.arange(block)[None, None, :]) % Mm).reshape(n, nb * block)[:, :Mm]


def sharpe_draws(dr, idx):
    s1, s2 = dr.sum(1), (dr ** 2).sum(1)
    nd = dr.shape[0] * dr.shape[1]
    S1, S2 = s1[idx].sum(1), s2[idx].sum(1)
    mu = S1 / nd
    var = (S2 - nd * mu ** 2) / (nd - 1)
    return mu / np.sqrt(var) * np.sqrt(DPY)


def boot_sharpe(dr_a, dr_b, block=G.C.BLOCK):
    """Sharpe(a) - Sharpe(b): point difference, p = share of draws <= 0, 2.5/97.5 percentiles. At block 12 it must
    give engine.block_bootstrap_sharpe's p exactly (asserted)."""
    idx = boot_idx(dr_a.shape[0], block)
    d = sharpe_draws(dr_a, idx) - sharpe_draws(dr_b, idx)
    point = G.measures(dr_a)['sharpe'] - G.measures(dr_b)['sharpe']
    if block == G.C.BLOCK:
        ref = G.block_bootstrap_sharpe(dr_a, dr_b)
        assert abs(ref['p'] - float(np.mean(d <= 0))) < 1e-12
    return {'diff': float(point), 'p': float(np.mean(d <= 0)), 'lo2.5': float(np.percentile(d, 2.5)),
            'hi97.5': float(np.percentile(d, 97.5))}


def romano_wolf(drs, dr_b, block=G.C.BLOCK):
    """Romano-Wolf step-down max-t adjusted one-sided p-values for Sharpe(rule k) - Sharpe(B) > 0, studentized by
    the bootstrap sd, from the same paired circular block draws for every rule."""
    idx = boot_idx(dr_b.shape[0], block, seed=G.C.SEED + 11)
    sb = sharpe_draws(dr_b, idx)
    point = np.array([G.measures(d)['sharpe'] - G.measures(dr_b)['sharpe'] for d in drs])
    draws = np.stack([sharpe_draws(d, idx) - sb for d in drs])          # [k, n]
    sd = draws.std(1, ddof=1)
    # a rule that never fires equals B exactly: its difference and its draws are all 0, so its t is set to 0 and it
    # adds nothing to the max (found after the unseen run, where a 0/0 t had made every adjusted p 0)
    same = sd == 0
    sd_ = np.where(same, 1.0, sd)
    tstat = np.where(same, 0.0, point / sd_)
    zc = np.where(same[:, None], 0.0, (draws - point[:, None]) / sd_[:, None])
    order = np.argsort(-tstat)
    adj = np.empty(len(drs))
    prev = 0.0
    for j, k in enumerate(order):
        rest = order[j:]
        mx = zc[rest].max(0)
        pj = max(prev, float(np.mean(mx >= tstat[k])))
        adj[k] = pj
        prev = pj
    raw = np.mean(draws <= 0, 1)
    return {'diff': point.tolist(), 't': tstat.tolist(), 'raw_p': raw.tolist(), 'rw_p': adj.tolist()}


def measures(src, w, cost):
    return G.measures(path(Px[src], t0s, w, cost))


def after_cost_vs_spy(w, src='yahoo'):
    """The file-name number, -3.2's way: the rule held in Yahoo SPY at 5 bps on every SPY trade, daily returns
    compounded per calendar year (by each day's date), minus SPY's total return over the same days, averaged."""
    P = Px[src]
    dr = path(P, t0s, w, COST).ravel()
    span = np.arange(t0s[0] + 1, t0s[-1] + 1 + K.H + 1)
    assert np.isfinite(P[span]).all()
    spy_d = Px['yahoo'][span[1:]] / Px['yahoo'][span[:-1]] - 1
    yr = np.array([d[:4] for d in ed[span[1:]]])
    out = {}
    for y in sorted(set(yr)):
        m = yr == y
        a, s = 100 * (np.prod(1 + dr[m]) - 1), 100 * (np.prod(1 + spy_d[m]) - 1)
        out[y] = {'rule5': float(a), 'spy': float(s), 'diff': float(a - s), 'days': int(m.sum())}
    n = len(dr)
    ann_rule = float(np.prod(1 + dr)) ** (DPY / n) - 1
    ann_spy = float(np.prod(1 + spy_d)) ** (DPY / n) - 1
    return {'years': out, 'number': float(np.mean([v['diff'] for v in out.values()])),
            'span_annualized_diff': 100 * (ann_rule - ann_spy), 'first_close': ed[span[0]], 'last_close': ed[span[-1]]}


# ---- The rules ----------------------------------------------------------------------------------------------------
wB, rvB, cB = weights_B('yahoo')
assert np.isfinite(wB).all(), 'rule B has no weight on some decision date'
f_main, thr_main = fire(z['dS1'], 'yahoo', K.Q_FIRE)
wS = np.where(f_main, 0.0, wB)
rules = {'spread rule': wS, 'rule B (variance timing)': wB, 'buy and hold': np.ones(M)}
grid = {}
for q in K.GRID_Q:
    for stock in (True, False):
        nm = f"{'dS > 0' if q is None else f'dS >= {int(100 * q)}th pct'}{' and R >= 0' if stock else ', no stock condition'}"
        f, _ = fire(z['dS1'], 'yahoo', q, stock)
        grid[nm] = np.where(f, 0.0, wB)
alts = {}
f, _ = fire(z['dD1'], 'yahoo', K.Q_FIRE)
alts['C2 default spread (BAA - AAA)'] = (np.where(f, 0.0, wB), f)
f, _ = fire(z['dS0'], 'yahoo', K.Q_FIRE)
alts['C5 optimistic timing (spread through t)'] = (np.where(f, 0.0, wB), f)
f, _ = fire(z['dS2'], 'yahoo', K.Q_FIRE)
alts['C5 extra caution (spread through t - 2)'] = (np.where(f, 0.0, wB), f)
rel = G.rel_paths(Px['yahoo'], t0s + 1)                 # [M, 22]: P[t0 + 1 + j] / P[t0 + 1]

res = {'tag': tag, 'first_decision': dd[0], 'last_decision': dd[-1], 'last_hold_end': ed[t0s[-1] + 1 + K.H], 'months': M,
       'fires': int(f_main.sum()), 'fire_dates': dd[f_main].tolist()}

if tag == 'old':
    periods = {'2008-01..2019-12 (pass window)': K.sel(dd, *K.DEV), '2020-01..2025-06': K.sel(dd, *K.CHECK),
               '2008-01..2025-06': np.ones(M, bool), '2008-2009': K.sel(dd, '2008-01-01', '2009-12-31'),
               '2010-2014': K.sel(dd, '2010-01-01', '2014-12-31'), '2015-2019': K.sel(dd, '2015-01-01', '2019-12-31'),
               '2010-2019 (without 2008-2009)': K.sel(dd, '2010-01-01', '2019-12-31'),
               '2021-2025-06 (without 2020)': K.sel(dd, '2021-01-01', K.CHECK[1])}
else:
    periods = {'unseen months' if tag == 'new' else 'forward': np.ones(M, bool)}


def sub_path(src, w, m, cost):
    return path(Px[src], t0s[m], w[m], cost)


def period_eval(m):
    out = {'months': int(m.sum()), 'fires': int(f_main[m].sum())}
    if m.sum() < 3:
        return out
    r = rel[m]
    for nm, w in rules.items():
        d5, d0 = sub_path('yahoo', w, m, COST), sub_path('yahoo', w, m, 0.0)
        out[nm] = G.measures(d5) | {'exposure': float(np.mean(w[m])), 'annual_0bps': G.measures(d0)['annual'],
                                    'sharpe_0bps': G.measures(d0)['sharpe'], 'switches': G.switches(w[m])}
        if nm != 'buy and hold':
            rd = G.random_timing(w[m], r, seed=G.C.SEED + (0 if nm == 'spread rule' else 3))
            out[nm]['random_median_annual'] = float(np.median(rd['annual']))
            out[nm]['random_p_annual'] = G.random_p(G.measures(d0)['annual'], rd['annual'])
            out[nm]['over_random'] = out[nm]['annual_0bps'] - out[nm]['random_median_annual']
    dS_, dB_ = sub_path('yahoo', wS, m, COST), sub_path('yahoo', wB, m, COST)
    out['boot_spread_vs_B'] = boot_sharpe(dS_, dB_)
    out['boot_spread_vs_B_1month_blocks'] = boot_sharpe(dS_, dB_, block=1)
    out['boot_spread_vs_bh'] = boot_sharpe(dS_, sub_path('yahoo', np.ones(M), m, COST))
    out['boot_B_vs_bh'] = boot_sharpe(dB_, sub_path('yahoo', np.ones(M), m, COST))
    # mean SPY return over the next 21 days (from the trade's close) in warning months and the others (descriptive)
    nxt = rel[m, -1] - 1
    out['spy_next21_fire_mean'] = float(nxt[f_main[m]].mean()) if f_main[m].any() else float('nan')
    out['spy_next21_nofire_mean'] = float(nxt[~f_main[m]].mean())
    return out


res['periods'] = {k: period_eval(m) for k, m in periods.items()}
main_key = list(periods)[0]
pw = res['periods'][main_key]
if tag == 'old':
    res['S2_pass_own_bar'] = bool(pw['spread rule']['sharpe'] > pw['rule B (variance timing)']['sharpe'] and pw['boot_spread_vs_B']['p'] < 0.10)
    res['fire_share_dev'] = float(f_main[periods[main_key]].mean())
whole_key = '2008-01..2025-06' if tag == 'old' else main_key
res['improvement_over_random'] = res['periods'][whole_key]['spread rule']['over_random']
res['improvement_over_random_p'] = res['periods'][whole_key]['spread rule']['random_p_annual']
res['improvement_over_random_B'] = res['periods'][whole_key]['rule B (variance timing)']['over_random']

# ---- The grid (reported), with Romano-Wolf, on the pass window (old) or the whole window -------------------------
gm = periods[main_key]
dB_g = sub_path('yahoo', wB, gm, COST)
g_names = list(grid)
rw = romano_wolf([sub_path('yahoo', grid[n], gm, COST) for n in g_names], dB_g)
res['grid'] = {n: {'fires': int((grid[n][gm] == 0).sum() - (wB[gm] == 0).sum()), 'diff': rw['diff'][i], 't': rw['t'][i],
                   'raw_p': rw['raw_p'][i], 'rw_p': rw['rw_p'][i],
                   'sharpe': G.measures(sub_path('yahoo', grid[n], gm, COST))['sharpe']} for i, n in enumerate(g_names)}
res['alts'] = {}
for n, (w, f) in alts.items():
    res['alts'][n] = {'fires': int(f[gm].sum()), 'sharpe': G.measures(sub_path('yahoo', w, gm, COST))['sharpe'],
                      'boot_vs_B': boot_sharpe(sub_path('yahoo', w, gm, COST), dB_g)}
    if tag == 'old':
        mc = periods['2020-01..2025-06']
        res['alts'][n]['check'] = boot_sharpe(sub_path('yahoo', w, mc, COST), sub_path('yahoo', wB, mc, COST))

# ---- The store's SPY (repeat) and XLF (C3) --------------------------------------------------------------------------
for src in ('store', 'xlf'):
    wb_, rv_, c_ = weights_B(src)
    ok = np.isfinite(wb_)
    f_, _ = fire(z['dS1'], src, K.Q_FIRE)
    ws_ = np.where(f_, 0.0, wb_)
    out = {'months_with_weight': int(ok.sum())}
    for pn, m in periods.items():
        mm = m & ok
        if mm.sum() < 12 or not np.all(np.diff(t0s[mm]) == K.H):
            continue
        dS_, dB_ = path(Px[src], t0s[mm], ws_[mm], COST), path(Px[src], t0s[mm], wb_[mm], COST)
        out[pn] = {'months': int(mm.sum()), 'fires': int(f_[mm].sum()), 'spread': G.measures(dS_) | {'exposure': float(ws_[mm].mean())},
                   'B': G.measures(dB_) | {'exposure': float(wb_[mm].mean())},
                   'hold': G.measures(path(Px[src], t0s[mm], np.ones(mm.sum()), COST)), 'boot': boot_sharpe(dS_, dB_)}
    res[f'repeat_{src}'] = out

res['after_cost'] = {'spread rule': after_cost_vs_spy(wS), 'rule B': after_cost_vs_spy(wB), 'buy and hold': after_cost_vs_spy(np.ones(M))}
res['file_name_number'] = res['after_cost']['spread rule']['number']
res['seconds'] = round(time.time() - t_start)
with open(K.HERE / f's2_{tag}.json', 'w') as f:
    json.dump(res, f, indent=1, default=float)
with open(K.HERE / f'months_{tag}.tsv', 'w') as f:
    f.write('date\trv21\tc\tw_B\tdS\tdS_p80\tR\twarning\tw_spread\tspy_next21\n')
    for m, t in enumerate(t0s):
        a = A0[m]
        f.write(f"{dd[m]}\t{rvB[m]:.4e}\t{cB[m]:.4e}\t{wB[m]:.4f}\t{z['dS1'][a]:+.3f}\t{thr_main[m]:+.4f}\t{z['R'][a]:+.4f}\t"
                f"{int(f_main[m])}\t{wS[m]:.4f}\t{rel[m, -1] - 1:+.4f}\n")

# ---- Report ---------------------------------------------------------------------------------------------------------
L = []
p = L.append
pc = lambda v: f'{100 * v:+.1f}%'  # noqa: E731
p(f"S2 ({tag}): {M} decisions {dd[0]} .. {dd[-1]} (last hold ends {res['last_hold_end']}); held from the close of t0 + 1 "
  f"to the close of t0 + 22; 5 bps on every SPY trade; Yahoo SPY. The warning fired on {res['fires']} decisions: {', '.join(res['fire_dates'])}")
for k, q in res['periods'].items():
    if q['months'] < 3:
        continue
    p(f"\n## {k}: {q['months']} months, warning on {q['fires']}")
    p('| Rule | Annual (5 bps) | Sharpe (5 bps) | Max drawdown | Avg exposure | Switches | Annual (0 bps) | Median random (0 bps) | Over random | Random p |')
    p('|---|---|---|---|---|---|---|---|---|---|')
    for nm in rules:
        a = q[nm]
        extra = (f" | {pc(a['random_median_annual'])} | {100 * a['over_random']:+.2f} | {a['random_p_annual']:.3f} |"
                 if 'random_median_annual' in a else ' | | | |')
        p(f"| {nm} | {pc(a['annual'])} | {a['sharpe']:.3f} | {pc(-a['mdd'])} | {a['exposure']:.0%} | {a['switches']} | {pc(a['annual_0bps'])}{extra}")
    b = q['boot_spread_vs_B']
    b1 = q['boot_spread_vs_B_1month_blocks']
    p(f"Sharpe spread rule - rule B: {b['diff']:+.3f}, bootstrap p {b['p']:.4f} (95%: {b['lo2.5']:+.3f}..{b['hi97.5']:+.3f}); "
      f"1-month blocks p {b1['p']:.4f}")
    p(f"Sharpe spread rule - buy and hold: {q['boot_spread_vs_bh']['diff']:+.3f} (p {q['boot_spread_vs_bh']['p']:.3f}); "
      f"rule B - buy and hold: {q['boot_B_vs_bh']['diff']:+.3f} (p {q['boot_B_vs_bh']['p']:.3f})")
    p(f"SPY's next 21 days (from the trade's close), mean: warning months {pc(q['spy_next21_fire_mean'])}, others {pc(q['spy_next21_nofire_mean'])}")
if tag == 'old':
    p(f"\nS2 own bar (2008-2019: higher Sharpe than rule B, bootstrap p < 0.10): {'met' if res['S2_pass_own_bar'] else 'not met'}; "
      f"warning share of 2008-2019 decisions {res['fire_share_dev']:.3f}")
p(f"Improvement over random ({whole_key}, 0 bps): spread rule {100 * res['improvement_over_random']:+.2f} pts/yr "
  f"(p {res['improvement_over_random_p']:.3f}); rule B {100 * res['improvement_over_random_B']:+.2f}")
p(f'\nThreshold grid against rule B ({main_key}; Romano-Wolf step-down max-t, one-sided):')
p('| Rule | Warning months | Sharpe | Sharpe - B | t | raw p | Romano-Wolf p |')
p('|---|---|---|---|---|---|---|')
for n, q in res['grid'].items():
    p(f"| {n} | {q['fires']} | {q['sharpe']:.3f} | {q['diff']:+.3f} | {q['t']:+.2f} | {q['raw_p']:.4f} | {q['rw_p']:.4f} |")
p(f'\nVariants of the main rule ({main_key}):')
for n, q in res['alts'].items():
    extra = f"; 2020-2025-06 {q['check']['diff']:+.3f} (p {q['check']['p']:.3f})" if 'check' in q else ''
    p(f"  {n}: warning on {q['fires']} months, Sharpe {q['sharpe']:.3f}, minus B {q['boot_vs_B']['diff']:+.3f} (p {q['boot_vs_B']['p']:.3f}){extra}")
for src in ('store', 'xlf'):
    p(f"\nRepeat on {'the store' + chr(39) + 's SPY' if src == 'store' else 'XLF (C3; the store, its own variance timing and R)'}:")
    for pn, q in res[f'repeat_{src}'].items():
        if not isinstance(q, dict):
            continue
        p(f"  {pn}: {q['months']} months, warning {q['fires']}; Sharpe spread {q['spread']['sharpe']:.3f} vs B {q['B']['sharpe']:.3f} "
          f"(diff {q['boot']['diff']:+.3f}, p {q['boot']['p']:.3f}); hold {q['hold']['sharpe']:.3f}; annual {pc(q['spread']['annual'])} vs "
          f"{pc(q['B']['annual'])}; exposure {q['spread']['exposure']:.0%} / {q['B']['exposure']:.0%}")
for nm, a in res['after_cost'].items():
    p(f"\nAfter costs, {nm}: yearly return at 5 bps a side minus SPY's, {a['first_close']} .. {a['last_close']}: "
      f"{a['number']:+.2f} points a year (whole span annualized {a['span_annualized_diff']:+.2f})")
    p(' | '.join(f"{y} {v['rule5']:+.1f} vs {v['spy']:+.1f} ({v['diff']:+.1f})" for y, v in a['years'].items()))
p(f"\nFile-name number (spread rule): {res['file_name_number']:+.1f}")
p(f"Runtime {res['seconds']} s")
txt = '\n'.join(L)
print(txt)
(K.HERE / f's2_{tag}.txt').write_text(txt + '\n')
