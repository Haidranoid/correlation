"""Overnight auctions: the plan's money test (its own, not the judge's), and the judge's plumbing check.

    python3 -I money.py sanity   # 5-day reversal through the judge, rebalance 5: should give +11.8 pts/yr, 8 of 18
    python3 -I money.py old      # 2008 .. 2025-06-27, by calendar year
    python3 -I money.py new      # the unseen months 2025-06-30 .. 2026-10-19 as one period (once, at the end)

The account: each day t the 20 highest S of the all-stocks ranked set (stable sort), bought at close_t (closing
auction) and sold at open_{t+1} (opening auction), long only, unhedged; the day's return is the mean simple overnight
return of the picks with a valid exit (decisions D6). 1,000 random accounts each draw 20 distinct stocks uniformly
from the same day's ranked set (numpy default_rng, seed common.SEED). Yearly returns compound the daily returns by
the exit day's calendar year. Number = mean over years of (account - median random) in points, at 0 bps; also at
2 bps per side (4 bps a day). Sensitivity: exits with open_{t+1} = close_t kept at 0 instead of dropped.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np = C.np
c2eval = C.c2eval
iso = c2eval.iso

tag = sys.argv[1] if len(sys.argv) > 1 else 'old'


def sanity():
    e = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
    c = np.asarray(e['close'], dtype=np.float64)
    s = np.full(c.shape, np.nan, np.float32)
    with np.errstate(all='ignore'):
        s[:, 5:] = -np.log(c[:, 5:] / c[:, :-5])
    p = c2eval.save_scores(s, 'overnight_auctions/sanity_reversal5')
    txt, js = c2eval.judge(p, rebalance=5)
    line = c2eval.summary_line(js)
    cor = c2eval.change_over_random(js)
    out = line + '\n' + json.dumps(cor) + '\n'
    (C.OUT / 'sanity_reversal5.txt').write_text(out)
    (C.OUT / 'sanity_reversal5_judge.txt').write_text(txt)
    print(out)


def daily(P, d_lo, d_hi, seed=C.SEED):
    """Per exit day: the account's return, the random accounts' [1000] (both with stale exits dropped and kept at 0),
    the number of ranked stocks and the account's mean volatility percentile (trailing 63-day close-to-close vol
    through t-1, within the ranked set; descriptive)."""
    import build as B  # noqa: F401
    rng = np.random.default_rng(seed)
    E = c2eval.load_export(C.EXPORTS[P['tag']], arrays=('close',))
    close = E['close']
    c0, rebal = P['c0'], P['rebal']
    ST = P['S'].T
    onT, okT, stT = P['on_s'].T, P['ok_s'].T, P['st_s'].T
    out = {k: [] for k in ('d', 'acc', 'acc0', 'n', 'nv', 'volp', 'in500')}
    rnd, rnd0 = [], []
    for d in range(d_lo, d_hi + 1):
        t = d - 1
        j = d - c0
        if j < 1:
            continue
        k = C.kidx(rebal, t - 1)
        if k < 0:
            continue
        U = P['uni'][k] & (P['fund_row'][k] >= 0)
        s = np.asarray(ST[t], np.float64)
        rows = np.nonzero(U & np.isfinite(s))[0]
        n = len(rows)
        if n < C.MIN_NAMES:
            continue
        top = rows[np.argsort(-s[rows], kind='stable')][:C.PICKS]
        ok, on, st = np.asarray(okT[j]), np.asarray(onT[j], np.float64), np.asarray(stT[j])
        r = np.where(ok, np.expm1(on), np.nan)               # stale exits dropped (headline)
        r0 = np.where(ok, np.expm1(on), np.where(st, 0.0, np.nan))  # stale exits kept at 0 (sensitivity)
        acc = np.nanmean(r[top]) if np.isfinite(r[top]).any() else 0.0
        acc0 = np.nanmean(r0[top]) if np.isfinite(r0[top]).any() else 0.0
        draws = np.argpartition(rng.random((C.N_RANDOM, n)), C.PICKS, axis=1)[:, :C.PICKS]
        rr = r[rows][draws]
        rr0 = r0[rows][draws]
        with np.errstate(all='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                a1 = np.nan_to_num(np.nanmean(rr, 1))
                a0 = np.nan_to_num(np.nanmean(rr0, 1))
        rnd.append(a1.astype(np.float32)); rnd0.append(a0.astype(np.float32))
        # descriptive: volatility percentile of the picks within the ranked set
        cw = np.asarray(close[rows, t - 63:t], np.float64)
        with np.errstate(all='ignore'):
            lr = np.log(cw[:, 1:] / cw[:, :-1])
        lr[~np.isfinite(lr) | (np.abs(lr) > C.MAX_LOG)] = np.nan
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            vol = np.nanstd(lr, 1)
        pr = np.argsort(np.argsort(np.nan_to_num(vol, nan=np.nanmedian(vol)))) / max(n - 1, 1)
        pos = np.searchsorted(rows, top)
        out['d'].append(d); out['acc'].append(acc); out['acc0'].append(acc0); out['n'].append(n)
        out['nv'].append(int(np.isfinite(r[top]).sum())); out['volp'].append(float(pr[pos].mean()))
        out['in500'].append(float(P['big'][k][top].mean()))
    o = {k: np.array(v) for k, v in out.items()}
    o['rnd'] = np.array(rnd)
    o['rnd0'] = np.array(rnd0)
    return o


def compound(r, cost):
    return float(np.prod(1.0 + r - cost) - 1.0)


def yearly(D, days, key='acc', rkey='rnd', cost=0.0, groups=None):
    """Per group (calendar year, or one period): account %, median random %, difference, p, random 5-95%."""
    rows = []
    for name, sel in groups:
        a = 100 * compound(D[key][sel], cost)
        R = 100 * (np.prod(1.0 + D[rkey][sel].astype(np.float64) - cost, 0) - 1.0)
        med = float(np.median(R))
        rows.append({'period': name, 'days': int(sel.sum()), 'account_pct': a, 'random_median_pct': med,
                     'diff': a - med, 'p': float((R >= a).mean()), 'random_p5': float(np.percentile(R, 5)),
                     'random_p95': float(np.percentile(R, 95))})
    return rows


def agg(rows, years=None):
    rr = [r for r in rows if years is None or int(r['period']) in years]
    d = np.array([r['diff'] for r in rr])
    return {'change': float(d.mean()), 'se': float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float('nan'),
            'ahead': int((d > 0).sum()), 'years': len(d), 'p_lt_0_1': int(sum(r['p'] < 0.1 for r in rr)),
            'per_year': [round(x, 2) for x in d]}


def main():
    if tag == 'sanity':
        return sanity()
    import build as B
    P = B.load(tag)
    days = P['days']
    if tag == 'old':
        d_lo = c2eval.day_index(days, C.FIRST)
        d_hi = int(np.searchsorted(days, C.iso_to_day('2025-06-27'), side='right')) - 1
    else:
        d_lo = c2eval.day_index(days, '2025-06-30')
        d_hi = int(np.searchsorted(days, C.iso_to_day('2026-10-19'), side='right')) - 1
    D = daily(P, d_lo, d_hi)
    np.savez(C.WORK / f'money_daily_{tag}.npz', **D)
    C.log('daily done', len(D['d']), 'days')
    yrs = np.array([int(iso(x)[:4]) for x in days[D['d']]])
    if tag == 'old':
        groups = [(str(y), yrs == y) for y in np.unique(yrs)]
    else:
        groups = [('unseen 2025-06-30..2026-10-19', np.ones(len(yrs), bool))] + [(str(y), yrs == y) for y in np.unique(yrs)]
    res = {'tag': tag, 'seed': C.SEED, 'n_random': C.N_RANDOM, 'picks': C.PICKS,
           'first_exit_day': iso(days[D['d'][0]]), 'last_exit_day': iso(days[D['d'][-1]]), 'days': int(len(D['d'])),
           'ranked_median': float(np.median(D['n'])), 'valid_picks_mean': float(D['nv'].mean()),
           'picks_vol_percentile_mean': float(D['volp'].mean()), 'picks_in_500_largest_mean': float(D['in500'].mean())}
    for label, key, rkey, cost in (('0bps', 'acc', 'rnd', 0.0), ('2bps', 'acc', 'rnd', 4e-4),
                                   ('0bps_stale_kept', 'acc0', 'rnd0', 0.0), ('2bps_stale_kept', 'acc0', 'rnd0', 4e-4)):
        rows = yearly(D, days, key, rkey, cost, groups)
        res[label] = {'rows': rows}
        if tag == 'old':
            res[label]['all'] = agg(rows)
            res[label]['2008-2022'] = agg(rows, set(range(2008, 2023)))
            res[label]['2023-2025'] = agg(rows, {2023, 2024, 2025})
    # mean daily excess over the median... (descriptive): account minus the mean random account, bps a day
    ex = D['acc'] - D['rnd'].astype(np.float64).mean(1)
    m, t_, _ = C.tstat(ex)
    res['daily_excess_vs_random_mean_bps'] = m * 1e4
    res['daily_excess_t'] = t_
    res['daily_excess_nw_t'] = C.nw_t(ex)
    (C.OUT / f'money_{tag}.json').write_text(json.dumps(res, indent=1))
    L = []
    W = L.append
    W(f'# Overnight auctions: the money test on the {tag} export (seed {C.SEED}, {C.N_RANDOM} random accounts)\n')
    W(f"Exit days {res['first_exit_day']}..{res['last_exit_day']} ({res['days']}); median ranked stocks {res['ranked_median']:.0f}; "
      f"valid picks a day {res['valid_picks_mean']:.2f} of 20; picks' mean volatility percentile {res['picks_vol_percentile_mean']:.2f}; "
      f"share in the 500 largest {res['picks_in_500_largest_mean']:.2f}")
    W(f"Daily excess over the mean random account: {m * 1e4:+.2f} bps a day (t {t_:+.2f}, NW {res['daily_excess_nw_t']:+.2f})\n")
    for label in ('0bps', '2bps', '0bps_stale_kept', '2bps_stale_kept'):
        W(f'## {label}\n')
        if tag == 'old':
            for nm in ('all', '2008-2022', '2023-2025'):
                a = res[label][nm]
                W(f"- {nm}: change over random {a['change']:+.2f} pts/yr (se {a['se']:.2f}), ahead {a['ahead']}/{a['years']}, p<0.1 in {a['p_lt_0_1']}")
        W('\n| Period | Days | Account % | Median random % | Difference | p | Random 5-95% |')
        W('|---|---|---|---|---|---|---|')
        for r in res[label]['rows']:
            W(f"| {r['period']} | {r['days']} | {r['account_pct']:+.1f} | {r['random_median_pct']:+.1f} | {r['diff']:+.1f} | {r['p']:.3f} | "
              f"{r['random_p5']:+.1f}..{r['random_p95']:+.1f} |")
        W('')
    txt = '\n'.join(L) + '\n'
    (C.OUT / f'money_{tag}.md').write_text(txt)
    print(txt)


if __name__ == '__main__':
    main()
