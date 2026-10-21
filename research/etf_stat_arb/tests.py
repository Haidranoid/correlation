"""Tables from the daily books written by run.py: the plan's tests, both price sources, both executions, the 500
largest, periods, costs, break-even, filter shares, holding times, and FF5 + UMD alphas (when the shared data is
ready).

    python3 -I -B tests.py      # writes out/tests.txt and out/tests.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402

PUB = Path.home() / '.local/share/correlation2/experiments/published_data'
LINES = []
J = {}


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def factors(days):
    if not (PUB / 'READY').exists():
        return None
    sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data')
    import pubdata
    d5, f5 = pubdata.ff5_daily()
    du, fu = pubdata.umd_daily()
    X = np.column_stack([pubdata.on_days(d5, f5[k], days) for k in ('Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA')]
                        + [pubdata.on_days(du, fu['UMD'], days)])
    return X


def alpha(y, X, lags=5):
    """OLS of y on [1, X]; alpha (per day) and its Newey-West t."""
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    A = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ coef
    n, k = A.shape
    XtX_inv = np.linalg.inv(A.T @ A)
    Z = A * e[:, None]
    S = Z.T @ Z
    for L in range(1, lags + 1):
        G = Z[L:].T @ Z[:-L]
        S += (1 - L / (lags + 1)) * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return float(coef[0]), float(coef[0] / np.sqrt(V[0, 0])), int(n), coef[1:]


def stats(B, pre, pm):
    g = B[f'{pre}_gross'][pm]
    cs = B[f'{pre}_cost_stock'][pm]
    cf = B[f'{pre}_cost_fund'][pm]
    net = B[f'{pre}_net'][pm]
    n = B[f'{pre}_n'][pm]
    mean, t, nd = C.tstat(net)
    out = dict(days=nd, net_bps=1e4 * mean, gross_bps=1e4 * float(g.mean()), cost_stock_bps=1e4 * float(cs.mean()),
               cost_fund_bps=1e4 * float(cf.mean()), t_plain=t, t_nw=C.nw_t(net),
               t_gross_nw=C.nw_t(g), sharpe_net=float(net.mean() / net.std(ddof=1) * np.sqrt(252)),
               sharpe_gross=float(g.mean() / g.std(ddof=1) * np.sqrt(252)), positions=float(n.mean()),
               longs=float(B[f'{pre}_n_long'][pm].mean()), shorts=float(B[f'{pre}_n_short'][pm].mean()),
               zero_days=int((n == 0).sum()))
    den = np.maximum(n, 1)
    out['stock_turnover'] = float((B[f'{pre}_stock_traded'][pm] / den).mean())   # $ traded per position per day
    out['fund_turnover'] = float((B[f'{pre}_fund_traded'][pm] / den).mean())
    out['breakeven_stock_bps'] = float((g.mean() - cf.mean()) / (cs.mean() / C.STOCK_BPS)) if cs.mean() > 0 else float('nan')
    return out


def fmt(st):
    return (f"net {st['net_bps']:+6.2f} bps/day (NW t {st['t_nw']:+5.2f}, plain t {st['t_plain']:+5.2f}), "
            f"gross {st['gross_bps']:+6.2f} (NW t {st['t_gross_nw']:+5.2f}), costs stock {st['cost_stock_bps']:.2f} "
            f"fund {st['cost_fund_bps']:.2f}; Sharpe net {st['sharpe_net']:+.2f} gross {st['sharpe_gross']:+.2f}; "
            f"positions {st['positions']:.0f} ({st['longs']:.0f} L / {st['shorts']:.0f} S); turnover/position/day "
            f"stock {st['stock_turnover']:.3f} fund {st['fund_turnover']:.3f}; break-even {st['breakeven_stock_bps']:.1f} bps/side")


def main():
    runs = {}
    for tag in ('old', 'new'):
        for src in ('export', 'store'):
            for fx in ('raw', 'fix'):
                p = C.WORK / f'books_{tag}_{src}_{fx}.npz'
                if p.exists():
                    runs[f'{tag}_{src}_{fx}'] = dict(np.load(p))
    say('runs available:', list(runs))
    for name, B in runs.items():
        days = B['days']
        tag = name.split('_')[0]
        X = factors(days)
        diag = json.loads((C.OUT / f"diag_{name}.json").read_text())
        say(f'\n===== {name} =====')
        pers = ('2006-2019', '2020-2025-06', 'post-2022-07') if tag == 'old' else ('unseen', 'unseen-ff')
        for setname in ('all', 'big'):
            dg = diag[setname]
            say(f'-- set {setname}: filter pass share {dg["pass_stock_days"] / dg["fit_stock_days"]:.3f} of fitted '
                f'stock-days; flips {dg["flips"]}; exits [count, mean days, median days]: {dg["hold"]}')
            for per in pers:
                pm = C.period_mask(days, per)
                fitn, okn = B[f'{setname}_fitn'][pm], B[f'{setname}_okn'][pm]
                share = float(okn.sum() / max(fitn.sum(), 1))
                opens = float(B[f'{setname}_opens'][pm].mean())
                say(f'   {per}: filter pass share {share:.3f}; fitted stocks/day {fitn.mean():.0f}; opens/day {opens:.1f}')
                for ex in ('same', 'next'):
                    pre = f'{setname}_{ex}'
                    st = stats(B, pre, pm)
                    if X is not None:
                        a, ta, na, betas = alpha(B[f'{pre}_net'][pm], X[pm])
                        st['alpha_bps'], st['alpha_t_nw'], st['alpha_days'] = 1e4 * a, ta, na
                        st['factor_betas'] = [round(float(b), 3) for b in betas]
                    J[f'{name}|{setname}|{per}|{ex}'] = st
                    extra = (f"; FF5+UMD alpha {st['alpha_bps']:+.2f} bps/day (NW t {st['alpha_t_nw']:+.2f}, "
                             f"{st['alpha_days']} days)" if 'alpha_bps' in st else '')
                    say(f'   {per} {ex:4s}: ' + fmt(st) + extra)
                # by year, same close, all set
            if setname == 'all':
                yrs = np.array([int(c2eval.iso(d)[:4]) for d in days])
                row = []
                for y in np.unique(yrs):
                    m = (yrs == y) & (B[f'{setname}_same_n'] > 0)
                    if m.sum() < 50:
                        continue
                    net = B[f'{setname}_same_net'][m]
                    nx = B[f'{setname}_next_net'][m]
                    row.append(f'{y}: same {1e4 * net.mean():+.1f} (Sh {net.mean() / net.std() * np.sqrt(252):+.1f}), '
                               f'next {1e4 * nx.mean():+.1f} (Sh {nx.mean() / nx.std() * np.sqrt(252):+.1f})')
                    J[f'{name}|year|{y}'] = dict(same_bps=1e4 * float(net.mean()),
                                                  same_sharpe=float(net.mean() / net.std() * np.sqrt(252)),
                                                  next_bps=1e4 * float(nx.mean()),
                                                  next_sharpe=float(nx.mean() / nx.std() * np.sqrt(252)))
                say('   by year, net bps/day:\n     ' + '\n     '.join(row))
    (C.OUT / 'tests.txt').write_text('\n'.join(LINES) + '\n')
    (C.OUT / 'tests.json').write_text(json.dumps(J, indent=1))


if __name__ == '__main__':
    main()
