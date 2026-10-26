"""S2 (secondary): hold ALC for the next 21 days when the combination forecasts ALC - SPY > 0, otherwise SPY; and
the README numbers (before costs vs random reorderings; after costs vs SPY by calendar year).

    python3 -I -B timing.py old     # -> s2_old.txt, s2_old.json, months_old.tsv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ad_common as K  # noqa: E402
import numpy as np  # noqa: E402


def holds(z, D):
    """Per decision: the ALC path and SPY path (relative to the buy close t0 + 1) over t0 + 1 .. t0 + 22, and the
    ALC names' end-of-hold weights (for the rebalancing cost at the next buy)."""
    P = {t: z[f'P_{t}'] for t in K.ALC + ('SPY',)}
    out = []
    for t0 in D:
        b, e = t0 + 1, t0 + 1 + K.H
        names = [n for n in K.ALC if np.isfinite(P[n][t0]) and np.isfinite(P[n][b]) and np.isfinite(P[n][e])]
        rel = np.array([P[n][b:e + 1] / P[n][b] for n in names])
        alc = rel.mean(0)
        spy = P['SPY'][b:e + 1] / P['SPY'][b]
        w_end = rel[:, -1] / rel[:, -1].sum()
        out.append({'names': names, 'alc': alc, 'spy': spy, 'w_end': w_end, 'days': np.arange(b, e + 1)})
    return out


def account(H, choice, cost):
    """Daily returns (days t0 + 2 .. t0 + 22 of each hold), per-hold log returns after costs, and the cost paid."""
    dr, days, mlog, paid = [], [], [], []
    prev = None
    for i, (h, c) in enumerate(zip(H, choice)):
        if prev is None:
            traded = 1.0
        elif prev != c:
            traded = 2.0
        elif c == 1:
            hp = H[i - 1]
            assert hp['names'] == h['names']
            traded = float(np.abs(hp['w_end'] - 1.0 / len(h['names'])).sum())
        else:
            traded = 0.0
        k = cost * traded
        path = h['alc'] if c == 1 else h['spy']
        r = path[1:] / path[:-1] - 1
        r[0] = (1 + r[0]) * (1 - k) - 1
        dr.append(r)
        days.append(h['days'][1:])
        mlog.append(float(np.log(path[-1]) + np.log(1 - k)))
        paid.append(k)
        prev = c
    return np.concatenate(dr), np.concatenate(days), np.array(mlog), np.array(paid)


def measures(dr):
    lg = np.log1p(dr)
    car = float(np.exp(lg.sum() / (len(dr) / 252)) - 1)
    sh = float(dr.mean() / dr.std(ddof=1) * np.sqrt(252))
    cum = np.exp(np.cumsum(lg))
    mdd = float((cum / np.maximum.accumulate(cum) - 1).min())
    return {'car': car, 'sharpe': sh, 'maxdd': mdd}


def paired_p(a, b, seed):
    """One-sided paired moving-block bootstrap p that mean(a - b) > 0 (12-decision blocks, centred)."""
    d = a - b
    idx = K.block_index(len(d), K.BLOCK, K.N_BOOT, seed)
    dr = (d - d.mean())[idx].mean(1)
    return float(np.mean(dr >= d.mean())), float(d.mean())


def random_vs(a, s, c, seed):
    """Rule's compound annual return (gross) minus the median of N_RANDOM random reorderings of its own choices."""
    years = len(a) * K.H / 252
    rule = float(np.exp((c * a + (1 - c) * s).sum() / years) - 1)
    rng = np.random.default_rng(seed)
    perm = np.array([rng.permutation(c) for _ in range(K.N_RANDOM)])
    rand = np.exp((perm * a + (1 - perm) * s).sum(1) / years) - 1
    return {'rule': rule, 'random_median': float(np.median(rand)), 'diff': rule - float(np.median(rand)),
            'p': float(np.mean(rand >= rule)), 'n': int(len(a)), 'share_alc': float(c.mean())}


def by_year(dr, days, axis, spy_dr):
    yrs = np.array([K.iso(axis[d])[:4] for d in days])
    out = {}
    for y in sorted(set(yrs)):
        m = yrs == y
        out[y] = float(np.prod(1 + dr[m]) - np.prod(1 + spy_dr[m]))
    return out


def main(tag):
    z = np.load(K.WORK / f'inputs_{tag}.npz')
    f = np.load(K.WORK / f'forecasts_{tag}.npz')
    D, dates, axis = z['D'], z['dates'], z['axis']
    fc, f0 = f['fc'], f['f0']
    use = np.isfinite(fc) & (dates <= K.CHECK[1])
    D, dates, fc, f0 = D[use], dates[use], fc[use], f0[use]
    H = holds(z, D)
    rule = (fc > 0).astype(int)
    excess = (fc > f0).astype(int)
    lines, res = [], {}

    def out(s=''):
        print(s, flush=True)
        lines.append(s)

    out(f'S2: decisions {dates[0]}..{dates[-1]} ({len(D)}); the rule holds ALC in {rule.mean():.2f} of them, '
        f'the excess variant in {excess.mean():.2f}; switches {int(np.abs(np.diff(rule)).sum())} '
        f'(variant {int(np.abs(np.diff(excess)).sum())})')
    acc = {}
    for nm, c in (('rule', rule), ('excess variant', excess), ('always ALC', np.ones(len(D), int)),
                  ('always SPY', np.zeros(len(D), int))):
        for cost in (0.0, K.COST):
            acc[(nm, cost)] = account(H, c, cost)
    spy_dr = acc[('always SPY', 0.0)][0]
    dev = K.sel(dates, *K.DEV)
    pers = {'2008-2009': K.sel(dates, '2008-01-01', '2009-12-31'), '2010-2014': K.sel(dates, '2010-01-01', '2014-12-31'),
            '2015-2019': K.sel(dates, '2015-01-01', '2019-12-31'), '2020-2025-06': K.sel(dates, *K.CHECK),
            'all': np.ones(len(D), bool)}
    res['accounts'] = {}
    for (nm, cost), (dr, days, ml, paid) in acc.items():
        if cost == 0.0:
            continue
        row = {}
        for pn, m in pers.items():
            dmask = np.repeat(m, K.H)
            row[pn] = measures(dr[dmask])
        row['share_alc'] = None
        row['cost_paid_per_year'] = float(paid.sum() / (len(D) * K.H / 252))
        res['accounts'][nm] = row
        out(f'  {nm:15s} at 5 bps: ' + '; '.join(f'{pn} CAR {v["car"]:+.3f} Sharpe {v["sharpe"]:.2f} maxDD '
                                                 f'{v["maxdd"]:+.2f}' for pn, v in row.items()
                                                 if isinstance(v, dict)) + f'; costs {row["cost_paid_per_year"]:.4f} a year')
    # the bar: development decisions, after costs, the rule's monthly log returns vs always-ALC and always-SPY
    r_ml = acc[('rule', K.COST)][2]
    res['bar'] = {}
    for nm, seed in (('always ALC', 1), ('always SPY', 2)):
        p, d = paired_p(r_ml[dev], acc[(nm, K.COST)][2][dev], K.SEED + 30 + seed)
        res['bar'][nm] = {'mean_monthly_log_diff': d, 'p': p}
        out(f'  development, rule minus {nm}: mean monthly log return {d:+.5f}, paired block-bootstrap p {p:.3f}')
    res['bar']['pass'] = all(v['mean_monthly_log_diff'] > 0 and v['p'] < 0.10 for v in res['bar'].values()
                             if isinstance(v, dict))
    out(f'  S2 bar met (counts only if P1 passes; P1 failed): {res["bar"]["pass"]}')

    # README: before costs vs random reorderings of the rule's own choices
    a = acc[('always ALC', 0.0)][2]
    s = acc[('always SPY', 0.0)][2]
    res['random'] = {}
    for pn, m in (('2008-2025-06', np.ones(len(D), bool)), ('2008-2019', dev), ('2020-2025-06', pers['2020-2025-06'])):
        for nm, c in (('rule', rule), ('excess variant', excess)):
            r = random_vs(a[m], s[m], c[m], K.SEED + 40)
            res['random'][f'{nm} {pn}'] = r
            out(f'  vs random, {nm}, {pn}: {r["rule"]:+.4f} against median {r["random_median"]:+.4f} -> '
                f'{100 * r["diff"]:+.2f} points a year, random p {r["p"]:.3f} ({r["n"]} decisions)')
    # README: after costs vs SPY, calendar years by trading day
    res['after_costs'] = {}
    for nm in ('rule', 'excess variant', 'always ALC'):
        dr, days, _, _ = acc[(nm, K.COST)]
        yy = by_year(dr, days, axis, spy_dr)
        res['after_costs'][nm] = {'by_year': yy, 'mean': float(np.mean(list(yy.values()))), 'years': len(yy)}
        out(f'  after costs vs SPY, {nm}: {100 * np.mean(list(yy.values())):+.2f} points a year over {len(yy)} '
            f'calendar years ({min(yy)}..{max(yy)})')
    out('  by year (rule): ' + ', '.join(f'{k} {100 * v:+.1f}' for k, v in res['after_costs']['rule']['by_year'].items()))
    first_day = K.iso(axis[acc[('rule', K.COST)][1][0]])
    last_day = K.iso(axis[acc[('rule', K.COST)][1][-1]])
    out(f'  daily returns from {first_day} to {last_day}')
    res['span'] = [first_day, last_day]

    with open(K.HERE / f'months_{tag}.tsv', 'w') as fh:
        fh.write('decision\tforecast\tprevailing_mean\trule_alc\talc_log\tspy_log\n')
        for i in range(len(D)):
            fh.write(f'{dates[i]}\t{fc[i]:+.5f}\t{f0[i]:+.5f}\t{rule[i]}\t{a[i]:+.5f}\t{s[i]:+.5f}\n')
    (K.HERE / f's2_{tag}.txt').write_text('\n'.join(lines) + '\n')
    K.dump(res, K.HERE / f's2_{tag}.json')


if __name__ == '__main__':
    main(sys.argv[1])
