"""The plan's tests and the reported tables, from books.py's series.

    python3 -I -B tests.py seen      -> out/tests.txt     (old export: 2006-2019, 2020-2025-06, post-2022 with/without fix)
    python3 -I -B tests.py unseen    -> out/unseen.txt    (new export: periods ending after 2025-06-27; once, at the end)

Units: % per 21-day period ("month") unless marked. t: plain for the 21-day series (non-overlapping periods);
Newey-West with 6 lags for the 6-month overlapping calendar-time series and the event-time 6-month cumulative.
Alphas: OLS of the period return on French's FF5 (Mkt-RF, SMB, HML, RMW, CMA) and UMD, each compounded over the
same 21 trading days; French's daily files end 2026-08-31, so later periods have no alpha.
Primary: 'divfix' (the dividend correction on signals and holding returns from 2022-07; identical to 'plain' before).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import c2eval, np, pubdata, WORK, OUT, WINDOWS, UNSEEN, N_ID, log, ols, window_mask, tstat  # noqa: E402

NW = 6
BOOKS = ('FIP', 'DIP', 'FIP-DIP', 'MOM')
FACT = ('Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD')


def load(tag, px):
    z = np.load(WORK / f'books_{tag}_{px}.npz')
    return {k: z[k] for k in z.files}


def factors(days, rebal):
    d5, f5 = pubdata.ff5_daily()
    du, fu = pubdata.umd_daily()
    X = np.column_stack([pubdata.on_days(d5, f5[c], days) for c in FACT[:5]] + [pubdata.on_days(du, fu['UMD'], days)])
    K = len(rebal)
    out = np.full((K, len(FACT)), np.nan)
    for k in range(K - 1):
        x = X[rebal[k] + 1:rebal[k + 1] + 1]
        if np.isfinite(x).all():
            out[k] = np.prod(1 + x, 0) - 1
    return out


def series(z, v, book, kind):
    """kind: net21, gross21, net6, gross6, cum6."""
    if book == 'FIP-DIP':
        return z[f'{v}__FIP__{kind}'] - z[f'{v}__DIP__{kind}']
    return z[f'{v}__{book}__{kind}']


def mean_t(y, m, nw=None):
    y = y[m]
    y = y[np.isfinite(y)]
    if len(y) < 3:
        return float('nan'), float('nan'), len(y)
    if nw is None:
        mu, t, n = tstat(y)
        return mu * 100, t, n
    b, t, n, _ = ols(y, None, nw_lags=nw)
    return b[0] * 100, t[0], n


def alpha(y, X, m, nw=None):
    ok = m & np.isfinite(y) & np.isfinite(X).all(1)
    if ok.sum() < 12:
        return float('nan'), float('nan'), int(ok.sum())
    b, t, n, _ = ols(y[ok], X[ok], nw_lags=nw)
    return b[0] * 100, t[0], n


def fmt(mu, t, n=None):
    if not np.isfinite(mu):
        return 'n/a'
    return f'{mu:+.2f} (t {t:+.2f})' if n is None else f'{mu:+.2f} (t {t:+.2f}, n {n})'


def book_table(z, X, v, windows, say, cum6_from=None):
    """cum6_from: if set, the event-time column uses only cohorts formed on or after that day (the unseen months:
    a cohort formed before them would carry seen returns; decided after the unseen run, a relabeling)."""
    say(f'| Window | Book | 21-day gross | 21-day net | FF5+UMD alpha, net | 6-month hold, net (NW t) | '
        f'6-month cumulative, event time, gross (NW t) | periods |')
    say('|---|---|---|---|---|---|---|---|')
    end = z['end_iso']
    for wn, (a, b) in windows.items():
        m = window_mask(end, a, b)
        # the event-time cohort belongs to the window by the end of its 126 days
        end6 = np.array([end[k + 5] if k + 5 < len(end) else '' for k in range(len(end))])
        m6 = window_mask(end6, a, b)
        if cum6_from is not None:
            m6 = z['start_iso'] >= cum6_from
        for book in BOOKS:
            g = mean_t(series(z, v, book, 'gross21'), m)
            nt = mean_t(series(z, v, book, 'net21'), m)
            al = alpha(series(z, v, book, 'net21'), X, m)
            n6 = mean_t(series(z, v, book, 'net6'), m, NW)
            c6 = mean_t(series(z, v, book, 'cum6'), m6, NW)
            say(f'| {wn} | {book} | {g[0]:+.2f} | {fmt(nt[0], nt[1])} | {fmt(al[0], al[1])} (n {al[2]}) | '
                f'{fmt(n6[0], n6[1])} | {fmt(c6[0], c6[1])} | {nt[2]} |')


def grid(z, v, a, b, say, kind='21'):
    m = window_mask(z['end_iso'], a, b)
    R = z[f'{v}__R{kind}'] - z[f'{v}__U{kind}'][:, None]
    nw = None if kind == '21' else NW
    say(f'Market-relative cell returns, {"21-day hold" if kind == "21" else "6-month overlapping hold"}, '
        f'% per period, periods ending {a}..{b} ({v}):')
    say('| PRET fifth | ID low (continuous) | ID mid | ID high (discrete) | low - high |')
    say('|---|---|---|---|---|')
    for f in range(5):
        cells = [mean_t(R[:, f * N_ID + c], m, nw) for c in range(3)]
        d = mean_t(R[:, f * N_ID] - R[:, f * N_ID + 2], m, nw)
        say(f'| {f + 1}{" (losers)" if f == 0 else " (winners)" if f == 4 else ""} | '
            + ' | '.join(fmt(c[0], c[1]) for c in cells) + f' | {fmt(d[0], d[1])} |')


def seen():
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    E = c2eval.load_export(c2eval.EXPORT_OLD, arrays=())
    Z = {px: load('old', px) for px in ('divfix', 'plain')}
    X = factors(E['days'], Z['divfix']['rebal'])
    z = Z['divfix']
    end = z['end_iso']
    say('# Frog in the pan: tests on the old export (periods by end day)')
    say()
    for f in ('checks_signals.txt', 'checks_umd.txt'):
        say(f'## Construction check: {f}')
        say((OUT / f).read_text())
    say('## The plan\'s tests (all = the liquid set; divfix)')
    a, b = WINDOWS['test1 2006-2019']
    m1 = window_mask(end, a, b)
    t1 = mean_t(series(z, 'all', 'FIP', 'net21'), m1)
    say(f'Test 1, FIP net, 2006-2019: {fmt(*t1)} -> {"PASS" if t1[1] >= 2 else "FAIL"} (t >= 2)')
    t2 = mean_t(series(z, 'all', 'FIP-DIP', 'net21'), m1)
    say(f'Test 2, FIP - DIP net, 2006-2019: {fmt(*t2)} -> {"PASS" if t2[0] > 0 and t2[1] >= 2 else "FAIL"} (positive, t >= 2)')
    a, b = WINDOWS['test3 2020-2025-06']
    m3 = window_mask(end, a, b)
    for px in ('divfix', 'plain'):
        t3 = mean_t(series(Z[px], 'all', 'FIP', 'net21'), m3)
        say(f'Test 3 (first half), FIP net, 2020-2025-06, {px}: {fmt(*t3)} -> {"same sign (positive)" if t3[0] > 0 else "WRONG SIGN"}')
    say()
    for v, title in (('all', 'the liquid set'), ('big', 'the 500 largest')):
        say(f'## Books, {title} (divfix)')
        book_table(z, X, v, WINDOWS, say)
        say()
    say('## Post-2022-07 periods with and without the dividend correction')
    w = {'post 2022-07 .. 2025-06': WINDOWS['post 2022-07 .. 2025-06'], 'test3 2020-2025-06': WINDOWS['test3 2020-2025-06']}
    for px in ('divfix', 'plain'):
        for v in ('all', 'big'):
            say(f'### {px}, {v}')
            book_table(Z[px], X, v, w, say)
            say()
    say('## The 5 x 3 grid')
    for v in ('all', 'big'):
        for wn in ('test1 2006-2019', 'test3 2020-2025-06'):
            for kind in ('21', '6'):
                grid(z, v, *WINDOWS[wn], say, kind)
                say()
    # names and turnover
    say('## Book sizes and turnover (divfix, all; 2006-2025-06)')
    m = window_mask(end, *WINDOWS['all 2006-2025-06'])
    for v in ('all', 'big'):
        for book in ('FIP', 'DIP', 'MOM'):
            nL = z[f'{v}__{book}__nL'][m]
            say(f'{v} {book}: long names mean {nL.mean():.0f} (min {nL.min()}), turnover per period 21-day '
                f'{np.nanmean(z[f"{v}__{book}__turn21"][m]):.2f}, 6-month {np.nanmean(z[f"{v}__{book}__turn6"][m]):.2f} '
                f'(of 4 = both legs fully replaced)')
    (OUT / 'tests.txt').write_text('\n'.join(lines) + '\n')


def unseen():
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    E = c2eval.load_export(c2eval.EXPORT_NEW, arrays=())
    Z = {px: load('new', px) for px in ('divfix', 'plain')}
    X = factors(E['days'], Z['divfix']['rebal'])
    w = {'unseen (periods ending 2025-06-28..2026-10-19)': UNSEEN}
    say('# Frog in the pan: the unseen months (new export; first period 2025-06-13 -> 2025-07-16)')
    say('(event-time 6-month column: cohorts formed on or after 2025-06-13 only)')
    for px in ('divfix', 'plain'):
        z = Z[px]
        m = window_mask(z['end_iso'], *UNSEEN)
        say(f'periods: {int(m.sum())}, {z["start_iso"][m][0]} .. {z["end_iso"][m][-1]}')
        t = mean_t(series(z, 'all', 'FIP', 'net21'), m)
        say(f'Test 3 (second half), FIP net, unseen, {px}: {fmt(*t)} -> {"same sign (positive)" if t[0] > 0 else "WRONG SIGN"}')
        for v in ('all', 'big'):
            say(f'### {px}, {v}')
            book_table(z, X, v, w, say, cum6_from='2025-06-13')
            say()
        grid(z, 'all', *UNSEEN, say, '21')
        say()
    (OUT / 'unseen.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    {'seen': seen, 'unseen': unseen}[sys.argv[1]]()
