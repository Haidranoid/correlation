"""Tables, bootstrap and the plan's tests from build.py's series (the .md, "Tests" and "Decisions").

    python3 -I report.py old     # 2006-2019, 2020-2025-06, 2006-2025-06 (+ 2008-2019 beside test 1); writes report_old.*
    python3 -I report.py new     # the unseen months 2025-06-30..2026-10-19 (run once, at the end); writes report_new.*
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, PERIODS, PORTS, UNSEEN, WORK, c2eval, ffill_rows, np, rf_last_date  # noqa: E402

B, BLOCK, SEED = 10000, 21, 0
PAIRS = [('MV-LW', 'IV'), ('MV-LW', 'EW'), ('MV-LW', 'MV-TW'), ('MV-LW', 'HRP'), ('MV-TW', 'IV'), ('MV-TW', 'EW'),
         ('MV-TW', 'HRP'), ('HRP', 'IV'), ('HRP', 'EW'), ('IV', 'EW')]
PI = {p: i for i, p in enumerate(PORTS)}
lines = []


def P(s=''):
    print(s, flush=True)
    lines.append(s)


def to_day(iso):
    return int((np.datetime64(iso) - np.datetime64('1970-01-01')).astype(int))


def cagr(r):
    return float(np.prod(1 + r) ** (252 / len(r)) - 1)


def maxdd(r):
    w = np.concatenate([[1.0], np.cumprod(1 + r)])
    return float((w / np.maximum.accumulate(w) - 1).min())


def sharpe(x):
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252))


def beta(y, x):
    xc = x - x.mean()
    return float(xc @ (y - y.mean()) / (xc @ xc))


def boot_idx(rng, n, b):
    nb = -(-n // BLOCK)
    st = rng.integers(0, n, (b, nb))
    return ((st[:, :, None] + np.arange(BLOCK)[None, None, :]) % n).reshape(b, -1)[:, :n]


def bootstrap(ex_net, r_net, ex_gross, seed=SEED):
    """Paired circular block bootstrap of every portfolio's Sharpe (net and gross excess) and volatility (net)."""
    rng = np.random.default_rng(seed)
    n = ex_net.shape[1]
    sh, shg, vol = [], [], []
    for a in range(0, B, 500):
        idx = boot_idx(rng, n, min(500, B - a))
        for arr, out, kind in ((ex_net, sh, 's'), (ex_gross, shg, 's'), (r_net, vol, 'v')):
            X = arr[:, idx]                                   # [P, b, n]
            if kind == 's':
                out.append(X.mean(2) / X.std(2, ddof=1) * np.sqrt(252))
            else:
                out.append(X.std(2, ddof=1) * np.sqrt(252))
    return np.concatenate(sh, 1), np.concatenate(shg, 1), np.concatenate(vol, 1)


def p_two(stat_b, stat):
    return float(np.mean(np.abs(stat_b - stat) >= abs(stat)))


def p_greater(stat_b, stat):
    """One-sided p for H1: stat > 0 (centred bootstrap)."""
    return float(np.mean((stat_b - stat) >= stat))


def p_less(stat_b, stat):
    """One-sided p for H1: stat < 0."""
    return float(np.mean((stat_b - stat) <= stat))


def period_stats(z, a, b, label, spy_r, res):
    days = z['days']
    gross, net, rf = z['gross'], z['net'], z['rf']
    M = (days >= to_day(a)) & (days <= to_day(b)) & np.isfinite(gross).all(0) & np.isfinite(rf) & np.isfinite(spy_r)
    n = int(M.sum())
    g, r, f, s = gross[:, M], net[:, M], rf[M], spy_r[M]
    exn, exg, exs = r - f, g - f, s - f
    t0 = z['t0']
    first = t0 + 1
    K = (days[np.minimum(first, len(days) - 1)] >= to_day(a)) & (days[np.minimum(first, len(days) - 1)] <= to_day(b))
    K &= np.arange(len(t0)) > 0                         # the first formation (from cash) is not a turnover
    out = {'label': label, 'start': c2eval.iso(days[M][0]), 'end': c2eval.iso(days[M][-1]), 'days': n,
           'rebalances': int(K.sum()), 'ports': {}}
    P(f'### {label}: {out["start"]} .. {out["end"]}, {n} days, {int(K.sum())} rebalances')
    P()
    P('| | Return (CAGR) | Volatility | Sharpe net | Sharpe gross | Max drawdown | One-way turnover per rebalance | '
      'Names > 0.1% | Beta to SPY (realized) | Beta (formation) |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    for p, name in enumerate(PORTS):
        d = {'cagr': cagr(r[p]), 'vol': float(r[p].std(ddof=1) * np.sqrt(252)), 'sharpe': sharpe(exn[p]),
             'sharpe_gross': sharpe(exg[p]), 'maxdd': maxdd(r[p]), 'turnover': float(0.5 * z['turn'][K, p].mean()),
             'names': float(z['names'][K, p].mean()), 'beta': beta(exn[p], exs),
             'beta_formation': float(z['beta_ea'][K, p].mean()), 'cagr_gross': cagr(g[p]),
             'cost_per_year': float((g[p] - r[p]).mean() * 252)}
        out['ports'][name] = d
        P(f"| {name} | {d['cagr']:+.1%} | {d['vol']:.1%} | {d['sharpe']:.2f} | {d['sharpe_gross']:.2f} | "
          f"{d['maxdd']:.1%} | {d['turnover']:.1%} | {d['names']:.0f} | {d['beta']:.2f} | {d['beta_formation']:.2f} |")
    out['spy'] = {'cagr': cagr(s), 'vol': float(s.std(ddof=1) * np.sqrt(252)), 'sharpe': sharpe(exs), 'maxdd': maxdd(s)}
    d = out['spy']
    P(f"| SPY | {d['cagr']:+.1%} | {d['vol']:.1%} | {d['sharpe']:.2f} | | {d['maxdd']:.1%} | | | 1.00 | |")
    P()
    P(f"Costs (5 bps a side) per year: " + ', '.join(f"{nm} {out['ports'][nm]['cost_per_year']:.2%}" for nm in PORTS)
      + f". Mean Ledoit-Wolf intensity {z['delta'][K].mean():.2f}, mean sample correlation {z['rbar'][K].mean():.2f}; "
      f"stocks per rebalance {z['n'][K].min()}..{z['n'][K].max()}; twins per rebalance {z['n_twin'][K].mean():.0f} "
      f"(funds {z['n_funds'][K].mean():.0f}).")
    P()
    # calendar-year drawdowns
    for yr in ('2008', '2020'):
        Y = (days[M] >= to_day(f'{yr}-01-01')) & (days[M] <= to_day(f'{yr}-12-31'))
        if Y.sum() > 200:
            out[f'dd_{yr}'] = {nm: maxdd(r[p][Y]) for p, nm in enumerate(PORTS)} | {'SPY': maxdd(s[Y])}
    # bootstrap
    shb, shgb, vb = bootstrap(exn, r, exg)
    sh = np.array([out['ports'][nm]['sharpe'] for nm in PORTS])
    shg = np.array([out['ports'][nm]['sharpe_gross'] for nm in PORTS])
    vo = np.array([out['ports'][nm]['vol'] for nm in PORTS])
    P('| A vs B | Volatility ratio A/B (95% interval) | p (two-sided) | Sharpe A − B, net (95% interval) | p (two-sided) | '
      'Sharpe A − B, gross | p (two-sided) |')
    P('|---|---|---|---|---|---|---|')
    out['pairs'] = {}
    for A, Bn in PAIRS:
        i, j = PI[A], PI[Bn]
        lr, lrb = np.log(vo[i] / vo[j]), np.log(vb[i] / vb[j])
        dsh, dshb = sh[i] - sh[j], shb[i] - shb[j]
        dg, dgb = shg[i] - shg[j], shgb[i] - shgb[j]
        rec = {'vol_ratio': float(np.exp(lr)), 'vol_ratio_ci': [float(np.exp(np.percentile(lrb, 2.5))), float(np.exp(np.percentile(lrb, 97.5)))],
               'vol_p2': p_two(lrb, lr), 'vol_p_less': p_less(lrb, lr),
               'dsharpe': float(dsh), 'dsharpe_ci': [float(np.percentile(dshb, 2.5)), float(np.percentile(dshb, 97.5))],
               'dsharpe_p2': p_two(dshb, dsh), 'dsharpe_p_greater': p_greater(dshb, dsh),
               'dsharpe_gross': float(dg), 'dsharpe_gross_p2': p_two(dgb, dg), 'dsharpe_gross_p_greater': p_greater(dgb, dg)}
        out['pairs'][f'{A} vs {Bn}'] = rec
        P(f"| {A} vs {Bn} | {rec['vol_ratio']:.3f} ({rec['vol_ratio_ci'][0]:.3f}–{rec['vol_ratio_ci'][1]:.3f}) | "
          f"{rec['vol_p2']:.3f} | {dsh:+.3f} ({rec['dsharpe_ci'][0]:+.3f} to {rec['dsharpe_ci'][1]:+.3f}) | "
          f"{rec['dsharpe_p2']:.3f} | {dg:+.3f} | {rec['dsharpe_gross_p2']:.3f} |")
    P()
    res[label] = out
    return out


def main(tag):
    z = dict(np.load(WORK / f'series_{tag}.npz'))
    days = z['days']
    spy = ffill_rows(z['spy'][None, :])[0]
    spy_r = np.full(len(days), np.nan)
    spy_r[1:] = spy[1:] / spy[:-1] - 1
    res = {}
    P(f'# Correlation-built portfolios: {tag} export')
    P()
    P(f'Bootstrap: paired circular blocks of {BLOCK} days, {B} draws, seed {SEED}. Net = after 5 bps a side on turnover. '
      f'Excess over DGS3MO (CSV through {rf_last_date()}, carried forward).')
    P()
    if tag == 'old':
        periods = dict(PERIODS)
        periods['2008-2019 (500 names)'] = ('2008-01-01', '2019-12-31')
    else:
        periods = {'unseen 2025-06-30..2026-10-19': UNSEEN}
    for label, (a, b) in periods.items():
        period_stats(z, a, b, label, spy_r, res)
    if tag == 'old':
        full = res['2006-2025-06']
        P('### Drawdowns within 2008 and 2020 (peak to trough inside the calendar year, net)')
        P()
        P('| | ' + ' | '.join(list(PORTS) + ['SPY']) + ' |')
        P('|---' * (len(PORTS) + 2) + '|')
        for yr in ('2008', '2020'):
            d = full[f'dd_{yr}']
            P(f'| {yr} | ' + ' | '.join(f'{d[k]:.1%}' for k in list(PORTS) + ['SPY']) + ' |')
        P()
        t1 = res['2006-2019']['pairs']['MV-LW vs IV']
        t2 = res['2020-2025-06']['pairs']['MV-LW vs IV']
        number = (full['ports']['MV-LW']['sharpe_gross'] - full['ports']['IV']['sharpe_gross']) * 15
        pass1 = (t1['vol_ratio'] <= 0.90) and (t1['dsharpe'] > 0) and (t1['dsharpe_p_greater'] < 0.10)
        pass2 = (t2['vol_ratio'] < 1.0) and (t2['dsharpe'] > 0)
        P('### The tests')
        P()
        P(f"Test 1 (2006-2019, net): volatility ratio MV-LW/IV {t1['vol_ratio']:.3f} (needs <= 0.90); Sharpe MV-LW − IV "
          f"{t1['dsharpe']:+.3f}, one-sided p {t1['dsharpe_p_greater']:.3f} (needs < 0.10) -> {'PASS' if pass1 else 'FAIL'}")
        P(f"Test 2 (2020-2025-06, net): volatility ratio {t2['vol_ratio']:.3f} (needs < 1); Sharpe difference "
          f"{t2['dsharpe']:+.3f} (needs > 0) -> {'PASS' if pass2 else 'FAIL'}")
        fp = full['pairs']['MV-LW vs IV']
        P(f"The number: (Sharpe MV-LW − Sharpe IV, gross, 2006-2025-06) x 15 = ({full['ports']['MV-LW']['sharpe_gross']:.3f} − "
          f"{full['ports']['IV']['sharpe_gross']:.3f}) x 15 = {number:+.1f} (gross difference two-sided p "
          f"{fp['dsharpe_gross_p2']:.3f}, one-sided {fp['dsharpe_gross_p_greater']:.3f})")
        res['tests'] = {'test1': pass1, 'test2': pass2, 'number': number}
    else:
        u = res['unseen 2025-06-30..2026-10-19']
        t3 = u['pairs']['MV-LW vs IV']
        pass3 = t3['vol_ratio'] < 1.0
        P(f"Test 3 (unseen months, net): volatility ratio MV-LW/IV {t3['vol_ratio']:.3f} (needs < 1) -> "
          f"{'PASS' if pass3 else 'FAIL'}; Sharpe MV-LW − IV {t3['dsharpe']:+.3f} (reported; two-sided p {t3['dsharpe_p2']:.3f})")
        res['tests'] = {'test3': pass3}
    (HERE / f'report_{tag}.txt').write_text('\n'.join(lines) + '\n')
    (HERE / f'report_{tag}.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main(sys.argv[1])
