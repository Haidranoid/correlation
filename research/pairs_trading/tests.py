"""Tests and tables for the pairs-trading replication (experiments/+0.8-pairs-trading.md).

    python3 -I -B tests.py old        # 2006-2019, 2020-2025-06: tests 1-3, data checks, alphas, the file-name number
    python3 -I -B tests.py new        # the 16 unseen months (blocks starting 2025-06-30 or later), once, at the end

Months are 21-trading-day blocks of the twins calendar (block b = export days rebal[b]+1 .. rebal[b]+21); a block's
return is the mean of the six live portfolios' P&L in it, divided by 20 pairs (committed capital) or by the pairs that
opened at least once in that portfolio's period (employed capital). Periods go by the block's last day.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data')
import pubdata  # noqa: E402

TAG = sys.argv[1]
PERIODS_OLD = {'2006-2019': ('2006-01-01', '2019-12-31'), '2020-2025-06': ('2020-01-01', '2025-06-27'),
               '2006-2025-06': ('2006-01-01', '2025-06-27'),
               '2006-2009': ('2006-01-01', '2009-12-31'), '2010-2014': ('2010-01-01', '2014-12-31'),
               '2015-2019': ('2015-01-01', '2019-12-31'), '2020-2022-06': ('2020-01-01', '2022-06-30'),
               '2022-07-2025-06': ('2022-07-01', '2025-06-27')}
UNSEEN_START = '2025-06-30'


def dnum(iso_):
    return int((np.datetime64(iso_) - np.datetime64('1970-01-01')).astype(int))


def load(tag, source, div, variant):
    p = C.WORK / f'run_{tag}_{source}_{div}_{variant}.npz'
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def series(R):
    """Block series: dict of committed/employed gross and cost [K], random gross/cost [NRAND, K], block first/last
    day (export index), number of live portfolios."""
    rebal, T = R['rebal'], int(R['T'])
    K = len(rebal)
    out = {k: np.full(K, np.nan) for k in ('cg', 'cc', 'eg', 'ec')}
    out['rg'] = np.full((C.NRAND, K), np.nan)
    out['rc'] = np.full((C.NRAND, K), np.nan)
    out['first'] = rebal + 1
    out['last'] = np.minimum(rebal + C.BLOCK, T - 1)
    out['nport'] = np.zeros(K, int)
    opened = (R['n_open'] > 0).sum(1)
    for b in range(K):
        if rebal[b] + 1 > T - 1:
            continue
        ks = np.array([k for k in range(b - C.NPORT + 1, b + 1) if k >= 0 and R['valid'][k]], int)
        if len(ks) == 0:
            continue
        m = b - ks
        g = R['pnl'][ks, :, m].sum(1)
        c = R['cost'][ks, :, m].sum(1)
        out['cg'][b] = (g / C.NPAIRS).mean()
        out['cc'][b] = (c / C.NPAIRS).mean()
        den = np.maximum(opened[ks], 1)
        out['eg'][b] = (g / den).mean()
        out['ec'][b] = (c / den).mean()
        out['rg'][:, b] = (R['rpnl'][ks, :, m].astype(np.float64) / C.NPAIRS).mean(0)
        out['rc'][:, b] = (R['rcost'][ks, :, m].astype(np.float64) / C.NPAIRS).mean(0)
        out['nport'][b] = len(ks)
    return out


def pmask(S, days, a, b):
    return (days[S['last']] >= dnum(a)) & (days[S['last']] <= dnum(b)) & np.isfinite(S['cg'])


def stats(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return {'n': len(x), 'mean_pct': float('nan'), 'nw_t': float('nan'), 't': float('nan'), 'yearly_pct': float('nan'),
                'sharpe': float('nan')}
    return {'n': int(len(x)), 'mean_pct': 100 * float(x.mean()), 'nw_t': C.nw_t(x, 6), 't': C.plain_t(x),
            'yearly_pct': 1200 * float(x.mean()), 'sharpe': float(x.mean() / x.std(ddof=1) * np.sqrt(12))}


def ols_nw(y, X, lags=6):
    """OLS with Newey-West (Bartlett) standard errors. Returns coefficients and t-stats."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xe[L:].T @ Xe[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return b, b / np.sqrt(np.diag(V))


_FAC = {}


def factors(days, S):
    """[K, 6] block sums of daily FF5 + UMD factors (decimal); NaN where a block day has no factor value."""
    key = (len(days), int(days[-1]))
    if key not in _FAC:
        d5, f5 = pubdata.ff5_daily()
        du, fu = pubdata.umd_daily()
        cols = [pubdata.on_days(d5, f5[c], days) for c in ('Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA')]
        cols.append(pubdata.on_days(du, fu['UMD'], days))
        _FAC[key] = np.stack(cols, 1)
    F = _FAC[key]
    out = np.full((len(S['first']), 6), np.nan)
    for b, (a, z) in enumerate(zip(S['first'], S['last'])):
        if a <= z:
            blk = F[a:z + 1]
            out[b] = blk.sum(0) if np.isfinite(blk).all() else np.nan
    return out


def alpha(S, days, mask, y):
    F = factors(days, S)
    ok = mask & np.isfinite(y) & np.isfinite(F).all(1)
    if ok.sum() < 24:
        return {'n': int(ok.sum()), 'alpha_yearly_pct': float('nan'), 'alpha_t': float('nan')}
    X = np.column_stack([np.ones(ok.sum()), F[ok]])
    b, t = ols_nw(y[ok], X)
    names = ['alpha', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD']
    return {'n': int(ok.sum()), 'alpha_yearly_pct': 1200 * float(b[0]), 'alpha_t': float(t[0]),
            'betas': {nm: round(float(v), 3) for nm, v in zip(names[1:], b[1:])},
            'beta_t': {nm: round(float(v), 2) for nm, v in zip(names[1:], t[1:])}}


def random_cmp(S, mask, cost_mult):
    """Book mean vs the random books' means over the masked blocks (committed capital)."""
    y = S['cg'] - cost_mult * S['cc']
    rb = (S['rg'] - cost_mult * S['rc'])[:, mask].mean(1)
    bk = float(y[mask].mean())
    return {'book_yearly_pct': 1200 * bk, 'rand_median_yearly_pct': 1200 * float(np.median(rb)),
            'rand_p5_yearly_pct': 1200 * float(np.percentile(rb, 5)), 'rand_p95_yearly_pct': 1200 * float(np.percentile(rb, 95)),
            'share_random_ge_book': float((rb >= bk).mean()), 'above_p95': bool(bk > np.percentile(rb, 95)),
            'book_minus_median_pts': 1200 * (bk - float(np.median(rb)))}


def trade_stats(R, mask_k):
    """GGR Table 2-style statistics over the formations in mask_k."""
    no, nc, nf, do = R['n_open'][mask_k], R['n_conv'][mask_k], R['n_forced'][mask_k], R['days_open'][mask_k]
    opened = no > 0
    return {'formations': int(mask_k.sum()), 'pairs_opened_of_20': float(opened.sum(1).mean()),
            'round_trips_per_pair': float(no.mean()), 'months_open_per_pair': float((do / C.BLOCK).mean()),
            'share_pairs_never_open': float((~opened).mean()),
            'share_trades_not_converging': float(nf.sum() / max(1, no.sum())),
            'share_opened_pairs_never_converging': float(((nc == 0) & opened).sum() / max(1, opened.sum())),
            'trigger_2sigma': float(2 * np.nanmean(R['sigma'][mask_k])),
            'universe_median': int(np.median(R['nuni'][mask_k])),
            'random_pairs_open_rate': float(np.nanmean(R['r_open_rate'][mask_k])),
            'random_round_trips_per_pair': float(np.nanmean(R['r_trips'][mask_k])),
            'random_trigger_2sigma_median': float(2 * np.nanmedian(R['r_sigma'][mask_k])),
            'random_share_trades_not_converging': float(np.nanmean(R['r_forced_share'][mask_k]))}


def fmt(x, nd=2):
    return '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:+.{nd}f}'


def report_old():
    cfgs = [(s, d, v) for s in ('export', 'store') for d in ('div', 'nodiv') for v in ('all', 'big')]
    res, lines = {}, []
    for (src, div, var) in cfgs:
        R = load('old', src, div, var)
        if R is None:
            continue
        S = series(R)
        days = R['days']
        key = f'{src}_{div}_{var}'
        out = {'periods': {}}
        for pname, (a, b) in PERIODS_OLD.items():
            m = pmask(S, days, a, b)
            o = {}
            for cm, lab in ((0.0, 'gross'), (1.0, 'net5'), (4.0, 'net20')):
                o[f'committed_{lab}'] = stats((S['cg'] - cm * S['cc'])[m])
                o[f'employed_{lab}'] = stats((S['eg'] - cm * S['ec'])[m])
            o['random_gross'] = random_cmp(S, m, 0.0)
            o['random_net5'] = random_cmp(S, m, 1.0)
            o['alpha_net5'] = alpha(S, days, m, S['cg'] - S['cc'])
            o['alpha_gross'] = alpha(S, days, m, S['cg'])
            o['months_with_6_portfolios'] = int((S['nport'][m] == 6).sum())
            out['periods'][pname] = o
        # by calendar year
        yrs = {}
        for y in range(2006, 2026):
            m = pmask(S, days, f'{y}-01-01', f'{y}-12-31')
            if m.sum() == 0:
                continue
            rb = S['rg'][:, m].mean(1)
            yrs[y] = {'committed_gross_yearly_pct': 1200 * float(S['cg'][m].mean()),
                      'committed_net5_yearly_pct': 1200 * float((S['cg'] - S['cc'])[m].mean()),
                      'random_median_gross_yearly_pct': 1200 * float(np.median(rb)), 'months': int(m.sum())}
        out['years'] = yrs
        # trading statistics by formation period (formation's last day)
        fdays = days[R['rebal']]
        for pname in ('2006-2019', '2020-2025-06'):
            a, b = PERIODS_OLD[pname]
            mk = R['valid'] & (fdays >= dnum(a) - 183) & (fdays <= dnum(b) - 183)
            out[f'trades_{pname}'] = trade_stats(R, mk)
        res[key] = out
    return res


def daily_unseen(R, start=UNSEEN_START):
    """Daily book and random-book returns from the unseen start to the export's end (each day = mean of the six live
    portfolios, P&L / 20), re-blocked into 21-trading-day months starting on the unseen start's first trading day."""
    rebal, T, days = R['rebal'], int(R['T']), R['days']
    u0 = c2eval.day_index(days, start)
    dd = np.arange(u0, T)
    opened = (R['n_open'] > 0).sum(1)
    cg, cc, eg, ec = (np.zeros(len(dd)) for _ in range(4))
    rg = np.zeros((C.NRAND, len(dd)))
    rc = np.zeros((C.NRAND, len(dd)))
    for n, d in enumerate(dd):
        ks = np.array([k for k in range(len(rebal)) if R['valid'][k] and rebal[k] < d <= rebal[k] + C.H], int)
        assert len(ks) == C.NPORT, (d, ks)
        h = d - rebal[ks]
        g = R['pnl_d'][ks, :, h].astype(np.float64).sum(1)
        c = R['cost_d'][ks, :, h].astype(np.float64).sum(1)
        cg[n], cc[n] = (g / C.NPAIRS).mean(), (c / C.NPAIRS).mean()
        den = np.maximum(opened[ks], 1)
        eg[n], ec[n] = (g / den).mean(), (c / den).mean()
        rg[:, n] = (R['rpnl_d'][ks, :, h].astype(np.float64) / C.NPAIRS).mean(0)
        rc[:, n] = (R['rcost_d'][ks, :, h].astype(np.float64) / C.NPAIRS).mean(0)
    blk = (dd - u0) // C.BLOCK
    nb = blk.max() + 1

    def bsum(x):
        return np.stack([x[..., blk == b].sum(-1) for b in range(nb)], -1)

    S = {'cg': bsum(cg), 'cc': bsum(cc), 'eg': bsum(eg), 'ec': bsum(ec), 'rg': bsum(rg), 'rc': bsum(rc),
         'first': np.array([dd[blk == b][0] for b in range(nb)]), 'last': np.array([dd[blk == b][-1] for b in range(nb)])}
    return S


def report_new():
    res = {}
    for src in ('export', 'store'):
        for div in ('div', 'nodiv'):
            for var in ('all', 'big'):
                R = load('new', src, div, var)
                if R is None:
                    continue
                days = R['days']
                # consistency: daily arrays re-summed over the fixed calendar's blocks equal the stored month sums
                Sf = series(R)
                k = int(np.nonzero(R['valid'])[0][-40])
                assert np.allclose(C.month_sums(R['pnl_d'][k].astype(np.float64)), R['pnl'][k], atol=1e-5)
                S = daily_unseen(R)
                m = np.ones(len(S['cg']), bool)
                o = {'blocks': int(m.sum()), 'first': c2eval.iso(days[S['first'][0]]), 'last': c2eval.iso(days[S['last'][-1]]),
                     'days': int((S['last'] - S['first'] + 1).sum()), 'last_block_days': int(S['last'][-1] - S['first'][-1] + 1)}
                for cm, lab in ((0.0, 'gross'), (1.0, 'net5')):
                    o[f'committed_{lab}'] = stats((S['cg'] - cm * S['cc'])[m])
                    o[f'employed_{lab}'] = stats((S['eg'] - cm * S['ec'])[m])
                o['random_gross'] = random_cmp(S, m, 0.0)
                o['random_net5'] = random_cmp(S, m, 1.0)
                o['alpha_net5'] = alpha(S, days, m, S['cg'] - S['cc'])
                o['monthly_committed_net5_pct'] = [round(100 * float(v), 3) for v in (S['cg'] - S['cc'])[m]]
                # first pass (fixed-calendar blocks starting on or after 2025-06-30: misses 2025-06-30..07-16)
                mf = (days[Sf['first']] >= dnum(UNSEEN_START)) & np.isfinite(Sf['cg'])
                o['first_pass_fixed_blocks_committed_net5'] = stats((Sf['cg'] - Sf['cc'])[mf])
                o['first_pass_first_day'] = c2eval.iso(days[Sf['first'][mf][0]])
                m2 = pmask(Sf, days, '2006-01-01', '2019-12-31')
                o['new_export_2006_2019_committed_net5'] = stats((Sf['cg'] - Sf['cc'])[m2])
                res[f'{src}_{div}_{var}'] = o
    return res


if __name__ == '__main__':
    res = report_old() if TAG == 'old' else report_new()
    (C.OUT / f'tests_{TAG}.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float)[:3000])
