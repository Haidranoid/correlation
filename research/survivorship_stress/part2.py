"""Part 2: the haircut. Seed 2's monthly top-10 picks (the tie-break run behind the file's $230,343), each pick's
return replaced by (1 - k h) r + k h L, h from Part 1's hazard model at the decision day; benchmarks (the average
eligible stock and each pick's 20 comparable stocks) with k = 1 and their own h. Grid L in {-30%, -55%, -100%},
k in {1, 2, 3}; break-even k; crisis sensitivity; the grader's account at 5 and 20 bps against SPY.

Rules: experiments/+18.6-monthly-top10-picks.md (Survivorship stress test); decisions.md here.
Writes out/part2.json and out/part2_months.tsv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402

import common as C  # noqa: E402

LS = (-0.30, -0.55, -1.00)
KS = (1, 2, 3)


def adj(f, kh, L):
    kh = np.minimum(kh, 1.0)
    return (1.0 - kh) * f + kh * L


def build(E, specs, s2, model, spy_dd):
    """Per graded month: eligible returns, pick weights, hazards, comparable parts; plus each account year's blocks
    with their picks' hazards."""
    close = E['close']
    months, dropped_picks, miss = [], 0, {'picks': 0, 'eligible': 0}
    years = []
    for y, h0, t, uni, graded in specs:
        for d in graded:
            d = int(d)
            f = C.fwd_col(close, uni, d)
            s = np.asarray(s2[uni, d], dtype=np.float64)
            ok = np.isfinite(f) & np.isfinite(s)
            u, f = uni[ok], f[ok]
            picks = C.grader_picks(s2, uni, d)
            pos = {c: i for i, c in enumerate(u.tolist())}
            w = np.zeros(u.size)
            for c in picks.tolist():
                if c in pos:
                    w[pos[c]] = 1.0
                else:
                    dropped_picks += 1
            vv, rr = C.vol_col(close, u, d), C.ratio_col(close, u, d)
            parts = C.comparable_parts(f, w, vv, rr)
            feat = C.features_col(E, u, d)
            miss['eligible'] += int(sum((~np.isfinite(feat[k])).sum() for k in C.CONT))
            miss['picks'] += int(sum((~np.isfinite(feat[k][w > 0])).sum() for k in C.CONT))
            months.append({'year': y, 'd': d, 'date': C.iso(E['days'][d]), 'f': f, 'w': w, 'parts': parts,
                           'p': C.predict_p(feat, model), 'feat': feat, 'crisis': bool(spy_dd[d] <= -0.20)})
        gy = C.grader_year(E, s2, y, cand=uni)
        for blk in gy['blocks']:
            if blk['picks'].size:
                fb = C.features_col(E, blk['picks'], blk['d'])
                blk['feat'] = fb
                blk['p'] = C.predict_p(fb, model)
            else:
                blk['feat'], blk['p'] = None, np.zeros(0)
            blk['r'] = C.block_pick_returns(blk)
            blk['crisis'] = bool(spy_dd[blk['d']] <= -0.20)
        years.append(gy)
    return months, years, dropped_picks, miss


def hazards(p, m, mult):
    """Monthly hazard from the window probability, times the crisis multiplier (capped at 1)."""
    return np.minimum(mult * C.monthly_hazard(p, m), 1.0)


def month_stats(months, m, L, k, crisis_mode='none', bench_k=1.0):
    """Adjusted pick return, average-stock return and comparable excess per month (k = 0 and L = None: none)."""
    pick, allr, comp = [], [], []
    for mo in months:
        mult = 2.0 if (crisis_mode == 'all' or (crisis_mode == 'drawdown' and mo['crisis'])) else 1.0
        h = hazards(mo['p'], m, mult)
        f, w = mo['f'], mo['w']
        if L is None:
            fp, fb = f, f
        else:
            fp, fb = adj(f, k * h, L), adj(f, bench_k * h, L)
        pick.append(float(np.dot(w, fp) / C.TOP))
        allr.append(float(fb.mean()))
        if mo['parts'] is None:
            comp.append(float('nan'))
        else:
            comp.append(sum(wi * (fp[i] - fb[nb].mean()) for i, wi, nb in mo['parts']) / C.TOP)
    return np.array(pick), np.array(allr), np.array(comp)


def account(years, m, L, k, cost, crisis_mode='none'):
    """The grader's 14-year $10,000 account with each block's growth factor shifted by the haircut."""
    end = 10000.0
    for gy in years:
        delta = []
        for blk in gy['blocks']:
            if L is None or not blk['picks'].size:
                delta.append(0.0)
                continue
            mult = 2.0 if (crisis_mode == 'all' or (crisis_mode == 'drawdown' and blk['crisis'])) else 1.0
            hm = hazards(blk['p'], m, mult)
            hb = 1.0 - (1.0 - hm) ** (blk['len'] / C.H)
            kh = np.minimum(k * hb, 1.0)
            delta.append(float(np.sum(kh * (L - blk['r'])) / C.TOP))
        end *= C.grader_value(gy, cost, delta)
    return end


def cell(months, years, m, L, k, crisis_mode='none'):
    pick, allr, comp = month_stats(months, m, L, k, crisis_mode)
    ex = pick - allr
    return {'L': L, 'k': k, 'pick': pick.mean(), 'all': allr.mean(), 'excess': ex.mean(), 'excess_t': C.tstat(ex),
            'comparable': np.nanmean(comp), 'comparable_t': C.tstat(comp),
            'acct5': account(years, m, L, k, 5, crisis_mode), 'acct20': account(years, m, L, k, 20, crisis_mode)}


def breakeven(fn, target, lo=0.0, hi=100.0):
    """Smallest k in [lo, hi] with fn(k) <= target (fn decreasing in k), by bisection; None if fn(hi) > target,
    'below' if fn(lo) <= target already."""
    if fn(lo) <= target:
        return 'below'
    if fn(hi) > target:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if fn(mid) <= target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def main():
    E = C.load()
    cutoff = C.dev_cutoff(E['days'])
    specs = C.graded_days(E, cutoff)
    p1 = json.loads((C.OUT / 'part1.json').read_text())
    s2 = np.load(C.RUNS / 'jitter/lgbm_s2.npy', mmap_mode='r')
    rep = json.loads((C.OUT / 'reproduce.json').read_text())
    spy_tr, spy_st = rep['spy_tr_end'], rep['spy_store_end']
    # SPY drawdown from its trailing 252-day high, on the export's days (Stooq store, the grader's SPY)
    dates, sc = C.spy_store()
    pos = np.searchsorted(dates, E['days'])
    spy_on = np.where((pos < dates.size) & (dates[np.minimum(pos, dates.size - 1)] == E['days']),
                      sc[np.minimum(pos, dates.size - 1)], np.nan)
    spy_dd = np.full(spy_on.size, np.nan)
    for i in range(spy_on.size):
        win = spy_on[max(0, i - 251):i + 1]
        spy_dd[i] = spy_on[i] / np.nanmax(win) - 1.0
    out = {'spy_tr_end': spy_tr, 'spy_store_end': spy_st}
    for tag, model in (('main', p1['model']), ('no_logp', p1['model_no_logp'])):
        m = p1['model']['m']
        months, years, dropped, miss = build(E, specs, s2, model, spy_dd)
        res = {'months': len(months), 'dropped_picks': dropped, 'missing_features': miss,
               'crisis_months': sum(mo['crisis'] for mo in months),
               'crisis_dates': [mo['date'] for mo in months if mo['crisis']],
               'crisis_blocks': sum(b['crisis'] for gy in years for b in gy['blocks']),
               'months_in_2008_2009': sum(mo['date'][:4] in ('2008', '2009') for mo in months)}
        # hazards: picks vs their comparable stocks vs every eligible stock (monthly, base)
        hp, hc, ha, hp_coll = [], [], [], []
        for mo in months:
            h = hazards(mo['p'], m, 1.0)
            hp += h[mo['w'] > 0].tolist()
            ha.append(float(h.mean()))
            if mo['parts']:
                hc += [float(h[nb].mean()) for _, _, nb in mo['parts']]
            r12 = mo['feat']['ret12']
            hp_coll += h[(mo['w'] > 0) & (r12 < -0.5)].tolist()
        res['hazard_month'] = {'picks_mean': float(np.mean(hp)), 'picks_median': float(np.median(hp)),
                               'picks_p90': float(np.quantile(hp, 0.9)), 'comparables_mean': float(np.mean(hc)),
                               'eligible_mean': float(np.mean(ha)),
                               'collapsed_picks_mean': float(np.mean(hp_coll)) if hp_coll else float('nan'),
                               'collapsed_pick_share': len(hp_coll) / max(1, len(hp))}
        res['base'] = cell(months, years, m, None, 0)
        grids = {}
        for mode in ('none', 'drawdown', 'all'):
            if tag == 'no_logp' and mode != 'none':
                continue
            g = []
            for L in LS:
                for k in KS:
                    c = cell(months, years, m, L, k, mode)
                    c['pass'] = bool(c['comparable_t'] >= 2 and c['acct20'] > spy_tr)
                    c['pass_store_spy'] = bool(c['comparable_t'] >= 2 and c['acct20'] > spy_st)
                    g.append(c)
                    print(tag, mode, L, k, {kk: round(v, 4) if isinstance(v, float) else v for kk, v in c.items()},
                          flush=True)
            grids[mode] = g
        res['grids'] = grids
        # break-even k (base hazards): comparable t = 2, and the 20 bps account = SPY total return
        be = {}
        for L in LS:
            tk = breakeven(lambda k: C.tstat(month_stats(months, m, L, k)[2]), 2.0)
            ak = breakeven(lambda k: account(years, m, L, k, 20) - spy_tr, 0.0)
            be[str(L)] = {'comparable_t2': tk, 'account_spy': ak}
            if tag == 'main':
                tkd = breakeven(lambda k: C.tstat(month_stats(months, m, L, k, 'drawdown')[2]), 2.0)
                tka = breakeven(lambda k: C.tstat(month_stats(months, m, L, k, 'all')[2]), 2.0)
                be[str(L)].update(comparable_t2_drawdown=tkd, comparable_t2_all_doubled=tka)
            print(tag, 'break-even', L, be[str(L)], flush=True)
        res['breakeven'] = be
        out[tag] = res
        if tag == 'main':
            with open(C.OUT / 'part2_months.tsv', 'w') as fh:
                fh.write('date\tcrisis\tpicks\tpick_ret\tall_ret\tcomparable\tpick_h_mean\teligible_h_mean\n')
                pick, allr, comp = month_stats(months, m, None, 0)
                for mo, a, b, c in zip(months, pick, allr, comp):
                    h = hazards(mo['p'], m, 1.0)
                    fh.write(f"{mo['date']}\t{int(mo['crisis'])}\t{int(mo['w'].sum())}\t{a:.6f}\t{b:.6f}\t{c:.6f}\t"
                             f"{h[mo['w'] > 0].mean():.6f}\t{h.mean():.6f}\n")
    # verdict (main model, base hazards)
    g = {(c['L'], c['k']): c for c in out['main']['grids']['none']}
    worth = g[(-0.55, 2)]['pass']
    explain = not g[(-0.30, 1)]['pass']
    out['verdict'] = ('Worth the Sharadar rerun' if worth else
                      'Survivorship can explain it' if explain else 'Unresolved')
    gs = {(c['L'], c['k']): c for c in out['main']['grids']['none']}
    out['verdict_store_spy'] = ('Worth the Sharadar rerun' if gs[(-0.55, 2)]['pass_store_spy'] else
                                'Survivorship can explain it' if not gs[(-0.30, 1)]['pass_store_spy'] else
                                'Unresolved')
    for mode in ('drawdown', 'all'):
        gm = {(c['L'], c['k']): c for c in out['main']['grids'][mode]}
        out[f'verdict_crisis_{mode}'] = ('Worth the Sharadar rerun' if gm[(-0.55, 2)]['pass'] else
                                         'Survivorship can explain it' if not gm[(-0.30, 1)]['pass'] else
                                         'Unresolved')
    gn = {(c['L'], c['k']): c for c in out['no_logp']['grids']['none']}
    out['verdict_no_logp'] = ('Worth the Sharadar rerun' if gn[(-0.55, 2)]['pass'] else
                              'Survivorship can explain it' if not gn[(-0.30, 1)]['pass'] else 'Unresolved')
    print('verdict', out['verdict'], '| store SPY', out['verdict_store_spy'], '| crisis', out['verdict_crisis_drawdown'],
          out['verdict_crisis_all'], '| no logp', out['verdict_no_logp'])
    C.save_json(out, 'part2.json')


if __name__ == '__main__':
    main()
