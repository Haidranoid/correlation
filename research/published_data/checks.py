"""Checks of the published data through pubdata's loaders (system python3 + numpy).

    python3 -I -B checks.py      # -> out/checks.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402
import pubdata as P  # noqa: E402

HERE = Path(__file__).resolve().parent
LINES = []
CHECK_NAMES = ('XOM', 'T', 'VZ', 'KO', 'CVX', 'PFE', 'MO', 'JNJ', 'IBM', 'PG')


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def cover(name, dates, cols):
    say(f'{name}: {dates[0]} .. {dates[-1]}  n {len(dates)}  cols {list(cols)}  '
        + '  '.join(f'{k} mean {np.nanmean(v) * (252 if "daily" in name else 12):+.4f}/yr' for k, v in cols.items()))


def main():
    for fn in ('ff3_daily', 'ff5_daily', 'umd_daily', 'ff3_monthly', 'ff5_monthly', 'umd_monthly',
               'aqr_bab_us_monthly', 'aqr_tsmom_monthly'):
        d, c = getattr(P, fn)()
        assert d.dtype == np.dtype('datetime64[D]') and all(v.dtype == np.float64 for v in c.values())
        assert np.all(np.diff(d.astype(np.int64)) > 0)
        cover(fn, d, c)
    # monthly dates are calendar month-ends
    for fn in ('ff3_monthly', 'aqr_bab_us_monthly', 'aqr_tsmom_monthly'):
        d, _ = getattr(P, fn)()
        assert np.all((d + 1).astype('datetime64[M]') != d.astype('datetime64[M]')), fn
    # French daily days vs export days
    E = c2eval.load_export(c2eval.EXPORT_NEW, arrays=())
    days = E['days']
    d5, _ = P.ff5_daily()
    dd = P.to_days(d5)
    lo, hi = max(dd[0], days[0]), min(dd[-1], days[-1])
    a, b = set(dd[(dd >= lo) & (dd <= hi)].tolist()), set(days[(days >= lo) & (days <= hi)].tolist())
    say(f'FF5 daily vs new export days over {c2eval.iso(lo)}..{c2eval.iso(hi)}: French {len(a)}, export {len(b)}, '
        f'French only {len(a - b)}, export only {len(b - a)} {[c2eval.iso(x) for x in sorted(b - a)][:10]}')
    # FF3 vs FF5 market factor
    d3, c3 = P.ff3_daily()
    m = np.isin(d3, d5)
    say(f'FF3 vs FF5 daily Mkt-RF max |diff| on common days: {np.nanmax(np.abs(c3["Mkt-RF"][m] - P.ff5_daily()[1]["Mkt-RF"])):.6f}')
    # rf
    rf = P.rf_daily(days)
    say(f'rf_daily on new export days: first finite {c2eval.iso(days[np.isfinite(rf)][0])}, last day {c2eval.iso(days[-1])} '
        f'= {rf[-1] * 252:.4f}/yr; 2024 mean {np.nanmean(rf[(days >= P.to_days(np.datetime64("2024-01-01"))) & (days <= P.to_days(np.datetime64("2024-12-31")))]) * 252:.4f}/yr')
    # ETFs
    d, adj = P.etf_adjclose(P.ETF_TICKERS)
    _, raw = P.etf_close(P.ETF_TICKERS)
    _, dv = P.etf_dividends(P.ETF_TICKERS)
    say(f'etf: union dates {d[0]} .. {d[-1]}  n {len(d)}')
    for t in P.ETF_TICKERS:
        f = np.isfinite(adj[t])
        say(f'  {t}: {d[f][0]} .. {d[f][-1]}  bars {int(f.sum())}  dividends {int((dv[t] > 0).sum())}')
    i0 = np.searchsorted(d, np.datetime64('2009-12-31'))
    i1 = np.searchsorted(d, np.datetime64('2019-12-31'))
    assert d[i0] == np.datetime64('2009-12-31') and d[i1] == np.datetime64('2019-12-31')
    spy_a = (adj['SPY'][i1] / adj['SPY'][i0]) ** 0.1 - 1
    spy_r = (raw['SPY'][i1] / raw['SPY'][i0]) ** 0.1 - 1
    say(f'SPY 2009-12-31..2019-12-31 a year: adjusted {spy_a:.4f}, raw close {spy_r:.4f}')
    assert 0.12 < spy_a < 0.145
    # dividend panel
    for tag, ex, end in (('old', c2eval.EXPORT_OLD, '2025-06-27'), ('new', c2eval.EXPORT_NEW, '2025-12-31')):
        E = c2eval.load_export(ex, arrays=('close',))
        F = P.div_fix(tag)
        assert F.shape == E['close'].shape and F.dtype == np.float32
        days = E['days']
        t0 = c2eval.day_index(days, '2023-01-03') - 1          # 2022-12-30
        t1 = int(np.searchsorted(days, P.to_days(np.datetime64(end)), side='right')) - 1
        yrs = (days[t1] - days[t0]) / 365.25
        say(f'\ndiv_fix {tag}: shape {F.shape}, nonzero {int((np.asarray(F) != 0).sum())}, first nonzero day '
            f'{c2eval.iso(days[np.nonzero(np.asarray(F).any(0))[0][0]])}; check {c2eval.iso(days[t0])}..{c2eval.iso(days[t1])} '
            f'({yrs:.2f} yr), bps a year (export + fix minus Yahoo adjusted; export alone minus Yahoo adjusted):')
        import csv
        rowsY = {}
        for f in sorted((P.DATA / 'raw' / 'yahoo_stocks').glob('batch_*.csv')):
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    if r['sym'] in CHECK_NAMES:
                        rowsY.setdefault(r['sym'], {})[r['date']] = float(r['adj_close'])
        for t in CHECK_NAMES:
            j = int(np.nonzero(E['tickers'] == t)[0][0])
            c = np.asarray(E['close'][j], np.float64)
            ya = rowsY[t]
            yl = np.log(ya[c2eval.iso(days[t1])] / ya[c2eval.iso(days[t0])])
            el = np.log(c[t1] / c[t0])
            fx = float(np.asarray(F[j, t0 + 1:t1 + 1], np.float64).sum())
            n_ev = int((np.asarray(F[j, t0 + 1:t1 + 1]) != 0).sum())
            say(f'  {t:4s} events {n_ev:2d}  with fix {1e4 * (el + fx - yl) / yrs:+7.1f}   without {1e4 * (el - yl) / yrs:+7.1f}')
    (HERE / 'out' / 'checks.txt').write_text('\n'.join(LINES) + '\n')


if __name__ == '__main__':
    main()
