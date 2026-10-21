"""Bystanders: Test 5, the money test.

    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python3 -I money.py old            # walk-forward 2006-07 .. 2024-25
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python3 -I money.py new            # the 16 unseen months (once)

Two parts:
1. Judge scores: for each bystander event (B, t), score = -mismatch (the twist passed Test 2) on days t..t+4, added up
   when B has several events in that span, NaN elsewhere. Also a plain variant, -sympathy move on every bystander
   event, as a secondary check. Graded with c2eval.judge(rebalance=1).
2. The plan's own money test: each event day, every bystander event traded against its mismatch (or sympathy move),
   equal dollar, held 5 days, P&L = direction x 5-day market-relative return; 1,000 random-direction guessers drawing
   the same number of trades from the same day's bystanders.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import analyze  # noqa: E402

c2eval = common.c2eval
iso = c2eval.iso
OUT = analyze.OUT


def event_values(ev, start_day, end_day):
    ok, byst, part, insym = analyze.classify(ev)
    m = byst & ~ev['expiry'] & (ev['d'] >= start_day) & (ev['d'] <= end_day)
    rb = analyze.collapse(ev, m)
    mis = ev['xB'] - ev['resp'] * ev['xA']
    return rb, mis


def make_scores(ev, rows, vals, N, T, hold=5):
    sc = np.full((N, T), np.nan, np.float32)
    B, D = ev['B'][rows], ev['d'][rows]
    v = vals[rows]
    f = np.isfinite(v)
    B, D, v = B[f], D[f], v[f]
    for k in range(hold):
        dd = D + k
        g = dd < T
        cur = sc[B[g], dd[g]]
        # add, treating NaN as 0; np.add.at handles repeated (B, d)
        base = np.where(np.isnan(cur), 0, cur)
        sc[B[g], dd[g]] = base
        np.add.at(sc, (B[g], dd[g]), -v[g])
    return sc


def plan_test(ev, rows_pool, rows_trade, vals, days, years, n_rand=1000, seed=analyze.SEED, label=''):
    """rows_pool: all bystander events (one per stock-day); rows_trade: the traded subset; vals: signal (direction =
    -sign). years: list of (start_idx, end_idx) day ranges."""
    rng = np.random.default_rng(seed)
    y = ev['fB5']
    pool = rows_pool[np.isfinite(y[rows_pool])]
    trade = rows_trade[np.isfinite(y[rows_trade]) & np.isfinite(vals[rows_trade]) & (vals[rows_trade] != 0)]
    pnl = -np.sign(vals[trade]) * y[trade]
    tday = ev['d'][trade]
    pday = ev['d'][pool]
    # random guessers per day
    rand_by_day = {}
    ud = np.unique(tday)
    k_by_day = dict(zip(*np.unique(tday, return_counts=True)))
    pool_sorted = pool[np.argsort(pday, kind='stable')]
    ps = ev['d'][pool_sorted]
    tot_rand_day = np.zeros((n_rand, len(ud)))
    for i, d in enumerate(ud):
        a, b = np.searchsorted(ps, d), np.searchsorted(ps, d, side='right')
        p = pool_sorted[a:b]
        k = k_by_day[d]
        yp = y[p]
        if len(p) == k:
            pick = np.broadcast_to(np.arange(k), (n_rand, k))
        else:
            pick = rng.random((n_rand, len(p))).argsort(axis=1)[:, :k]
        sgn = rng.choice([-1.0, 1.0], size=(n_rand, k))
        tot_rand_day[:, i] = (sgn * yp[pick]).sum(1)
    strat_day = np.array([pnl[tday == d].sum() for d in ud])
    res = {'label': label, 'years': []}
    lines = []
    W = lines.append
    W(f'| Year | Trades | Trades per day | Gross bps per trade | Net at 5 bps/side | Net at 10 bps/side | Random p (same costs) |')
    W('|---|---|---|---|---|---|---|')
    pos5 = 0; ny = 0
    for (a, b) in years:
        sel = (tday >= a) & (tday < b)
        if sel.sum() == 0:
            continue
        ny += 1
        g = pnl[sel].mean() * 1e4
        dsel = (ud >= a) & (ud < b)
        s_tot = strat_day[dsel].sum(); r_tot = tot_rand_day[:, dsel].sum(1)
        p = (np.sum(r_tot >= s_tot) + 1) / (n_rand + 1)
        n5 = g - 10; n10 = g - 20
        pos5 += n5 > 0
        tdays = b - a
        W(f'| {iso(days[a])[:4]}-{iso(days[min(b, len(days)) - 1])[2:4]} | {sel.sum():,} | {sel.sum() / tdays:.1f} | {g:+.1f} | '
          f'{n5:+.1f} | {n10:+.1f} | {p:.3f} |')
        res['years'].append(dict(start=iso(days[a]), trades=int(sel.sum()), gross_bps=g, p=p))
    allsel = (tday >= years[0][0]) & (tday < years[-1][1])
    dsel = (ud >= years[0][0]) & (ud < years[-1][1])
    s_tot = strat_day[dsel].sum(); r_tot = tot_rand_day[:, dsel].sum(1)
    p_all = (np.sum(r_tot >= s_tot) + 1) / (n_rand + 1)
    g_all = pnl[allsel].mean() * 1e4
    se = pnl[allsel].std(ddof=1) / np.sqrt(allsel.sum()) * 1e4
    W(f'| **All** | {allsel.sum():,} | {allsel.sum() / (years[-1][1] - years[0][0]):.1f} | {g_all:+.1f} (se {se:.1f}) | '
      f'{g_all - 10:+.1f} | {g_all - 20:+.1f} | {p_all:.3f} |')
    W(f'\nYears net-positive at 5 bps per side: {pos5} of {ny}. Pass needs random p < 0.05 and at least two-thirds of years.')
    res.update(p_all=p_all, gross_all=g_all, se=se, pos5=int(pos5), n_years=ny)
    return '\n'.join(lines), res


def judge_years(days, first='2006-06-28', last=None):
    """Judge-style years: 06-28 to the next 06-27."""
    out = []
    y = int(first[:4])
    while True:
        a = c2eval.day_index(days, f'{y}-06-28')
        b = c2eval.day_index(days, f'{y + 1}-06-28')
        if a >= len(days) - 1:
            break
        out.append((a, min(b, len(days))))
        y += 1
        if b >= len(days):
            break
    return out


def main(tag):
    ev = analyze.load_events(tag)
    days = ev['days']
    P = common.load_panels(tag, names=('close',))
    N, T = P['close'].shape
    exp = c2eval.EXPORT_OLD if tag == 'old' else c2eval.EXPORT_NEW
    if tag == 'old':
        start, end = c2eval.day_index(days, '2005-01-01'), T - 1
    else:
        start, end = c2eval.day_index(days, '2025-06-30'), T - 1
    rb, mis = event_values(ev, start, end)
    rt = rb[np.isfinite(ev['resp'][rb])]
    sub = common.BIG.name  # 'bystanders'
    results = {}
    L = []
    W = L.append
    for name, rows, vals in (('mismatch', rt, mis), ('sympathy', rb, ev['xB'])):
        sc = make_scores(ev, rows, vals, N, T)
        nm = f'{sub}/bystanders_{name}_{tag}'
        path = c2eval.save_scores(sc, nm, export=exp)
        per_day = np.isfinite(sc).sum(0)
        W(f'### Scores: -{name} ({tag} export)\n')
        W(f'- Events scored: {np.isfinite(vals[rows]).sum():,}; scored stocks per day (days with any): median '
          f'{int(np.median(per_day[per_day > 0]))}, days with any score {int((per_day > 0).sum()):,}.')
        if tag == 'old':
            txt, js = c2eval.judge(path, rebalance=1)
        else:
            txt, js = c2eval.judge(path, export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30')
        cor = c2eval.change_over_random({'costs': js['report']['strategy']} if 'report' in js else js)
        (OUT / f'judge_{name}_{tag}.txt').write_text(txt)
        if tag == 'old':
            W(f'- Judge: {c2eval.summary_line(js)}')
        else:
            rp = js['report']
            st = rp['strategy']
            W(f"- Judge (single holdout): picks {st[0]['final_pct']:+.1f}% vs random median {st[0]['random_median_pct']:+.1f}% (p {st[0]['p']:.3f}); "
              f"5 bps {st[1]['final_pct']:+.1f}% vs {st[1]['random_median_pct']:+.1f}% (p {st[1]['p']:.3f}); 20 bps {st[2]['final_pct']:+.1f}% vs {st[2]['random_median_pct']:+.1f}%; "
              f"long-short {rp['ls_bps_per_day']:+.2f} bps/day (t {rp['ls_t']:+.2f}); rank IC in actual returns t {rp['ic_ret_t']:+.2f}; SPY {rp['benchmarks'][0]['return_pct']:+.1f}%")
        W(f'- Change over random: {cor["change"]:+.2f} pts/yr (se {cor["se"]:.2f}, ahead {cor["ahead"]}/{cor["years"]}); '
          f'per year {cor["per_year"]}')
        results[name] = dict(cor=cor, judge_json=str(path).replace('.npy', '__r1_d0_uniform.json'))
        W('')
    # the plan's own money test
    years = judge_years(days) if tag == 'old' else [(c2eval.day_index(days, '2025-06-30'), T)]
    for name, rows, vals in (('mismatch', rt, mis), ('sympathy', rb, ev['xB'])):
        txt, res = plan_test(ev, rb, rows, vals, days, years, label=name)
        W(f'### The plan\'s money test: against the {name} ({tag} export), 5-day hold, market-relative\n')
        W(txt)
        W('')
        results[f'plan_{name}'] = res
    txt = '\n'.join(L)
    (OUT / f'money_{tag}.md').write_text(txt + '\n')
    (OUT / f'money_{tag}.json').write_text(json.dumps(results, indent=1, default=float))
    print(txt)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'old')
