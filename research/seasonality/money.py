"""The money test: daily scores = the stock's percentile of `same` for the calendar month of the next trading day, among
the universe stocks (at that month's formation) holding a signal; NaN elsewhere. Constant from the close of the last
trading day of month m-1 through the close of the second-to-last day of month m, so with rebalance=1 the picks change
only when the month does.

    python3 -I -B money.py scores old|new [raw|fix]   -> WORK/scores/season_<tag>_<variant>.npy, plus leak checks
    python3 -I -B money.py judge old [raw|fix]        -> walk-forward 2006-07..2024-25, money_old_<variant>.txt/.json
    python3 -I -B money.py judge new [raw|fix]        -> the unseen months (once, at the end), money_new_<variant>.txt
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import c2eval, np, WORK, HERE, EXPORTS, log, month_of, month_label, day_num  # noqa: E402
from build import monthly, signals, SAME_LAGS  # noqa: E402

(WORK / 'scores').mkdir(parents=True, exist_ok=True)


def next_month_index(days, months):
    """[T] index into months of each day's next trading day (the next weekday for the export's last day)."""
    T = len(days)
    nxt = np.empty(T, np.int64)
    nxt[:-1] = days[1:]
    d = int(days[-1]) + 1
    while (d + 3) % 7 in (5, 6):      # 1970-01-01 was a Thursday: (d + 3) % 7 == 5 Saturday, 6 Sunday
        d += 1
    nxt[-1] = d
    mi = np.searchsorted(months, month_of(nxt))
    return mi


def scores(tag, variant, start_iso=None):
    z = np.load(WORK / f'panel_{tag}.npz')
    days, months, e, kf = z['days'], z['months'], z['e'], z['kf']
    if variant == 'fix':
        cf = np.load(WORK / f'divfix_{tag}.npy').astype(np.float64)
        same = signals(z['r'].astype(np.float64) + cf)['same']
    else:
        same = z['same'].astype(np.float64)
    uni_f = z['uni_f']
    N, M = same.shape
    T = len(days)
    E = c2eval.load_export(EXPORTS[tag], arrays=('close',))
    tw_rebal = c2eval.twins(E)['rebal']
    mi = next_month_index(days, months)
    # percentiles per holding month
    pct = np.full((M, N), np.nan, np.float32)
    for i in range(1, M):
        s = same[:, i]
        m = np.nonzero(uni_f[i] & np.isfinite(s))[0]
        if len(m) < 2:
            continue
        rk = np.argsort(np.argsort(s[m], kind='mergesort'), kind='mergesort')
        pct[i, m] = rk / (len(m) - 1)
    S = np.full((N, T), np.nan, np.float32)
    first = c2eval.day_index(days, start_iso) if start_iso else 0
    n_checked = 0
    for d in range(first, T):
        i = int(mi[d])
        if i >= M or i < 1 or not np.isfinite(pct[i]).any():
            continue
        # point in time: the signal's latest month (lag 12) ends before day d; the formation day e[i-1] <= d; the
        # universe comes from a twins calendar day <= e[i-1]
        assert e[i - min(SAME_LAGS)] < d and e[i - 1] <= d and tw_rebal[kf[i]] <= e[i - 1], (d, i)
        assert month_of(days[e[i - 1]:e[i - 1] + 1])[0] == months[i] - 1
        S[:, d] = pct[i]
        n_checked += 1
    name = f'season_{tag}_{variant}'
    p = c2eval.save_scores(S, f'seasonality/scores/{name}', export=EXPORTS[tag])
    have = np.nonzero(np.isfinite(S).any(0))[0]
    log(f'saved {p}; scored days {len(have)} ({c2eval.iso(days[have[0]])}..{c2eval.iso(days[have[-1]])}), '
        f'point-in-time asserts passed on {n_checked} days')
    return E, S, z


def leak_check(tag, E, S, z, n=8, seed=1):
    """Recompute the scores of sample days from the export's closes truncated at that day (data through the close of
    d only; the trading calendar is kept) and compare. Also check one signal by hand from dates."""
    days, months = z['days'], z['months']
    uni_f = z['uni_f']
    rng = np.random.default_rng(seed)
    scored = np.nonzero(np.isfinite(S).any(0))[0]
    # month-ends (the switch days) and random mid-month days
    mi = next_month_index(days, months)
    switch = scored[mi[scored] != np.searchsorted(months, month_of(days[scored]))]
    pick = np.concatenate([rng.choice(switch, n // 2, replace=False), rng.choice(scored, n - n // 2, replace=False)])
    lines = []
    for d in sorted(pick):
        P = monthly(E['close'][:, :d + 1], days[:d + 1])
        same_t = signals(P['r'])['same']
        i_full = int(mi[d])
        i_t = int(np.searchsorted(P['months'], months[i_full]))
        if i_t < same_t.shape[1]:
            s_t = same_t[:, i_t]
        else:   # the predicted month has no day in the truncated panel: shift the lags by hand
            r_t = P['r']
            M_t = r_t.shape[1]
            vals = []
            for l in SAME_LAGS:
                j = i_t - l
                vals.append(r_t[:, j] if 0 <= j < M_t else np.full(r_t.shape[0], np.nan))
            V = np.array(vals)
            cnt = np.isfinite(V).sum(0)
            with np.errstate(all='ignore'):
                s_t = np.nanmean(V, 0)
            s_t[cnt < 5] = np.nan
        m = np.nonzero(uni_f[i_full] & np.isfinite(s_t))[0]
        rk = np.argsort(np.argsort(s_t[m], kind='mergesort'), kind='mergesort')
        pc = np.full(len(S), np.nan)
        pc[m] = rk / (len(m) - 1)
        a, b = S[:, d].astype(np.float64), pc
        same_nan = np.array_equal(np.isfinite(a), np.isfinite(b))
        diff = float(np.nanmax(np.abs(a - b))) if np.isfinite(a).any() else 0.0
        lines.append(f'{c2eval.iso(days[d])} (holds {month_label(months[i_full])}): stocks {int(np.isfinite(a).sum())}, '
                     f'same NaN pattern {same_nan}, max |score diff| {diff:.2e}')
        assert same_nan and diff < 1e-5, lines[-1]
    # one signal by hand: a stock with a full signal at a mid-sample month
    i = int(np.searchsorted(months, 2016 * 12 + 2))   # holding month 2016-03
    j = int(np.nonzero(np.isfinite(z['same'][:, i]) & (z['n_same'][:, i] == 10))[0][0])
    c = np.asarray(E['close'][j], np.float64)
    vals = []
    for y in range(2006, 2016):
        def last_close(yy, mm):
            a0, b0 = day_num(f'{yy}-{mm:02d}-01'), day_num(f'{yy + (mm == 12)}-{mm % 12 + 1:02d}-01')
            ix = np.nonzero((days >= a0) & (days < b0) & np.isfinite(c))[0]
            return c[ix[-1]]
        vals.append(np.log(last_close(y, 3) / last_close(y, 2)))
    hand = float(np.mean(vals))
    lines.append(f'by hand: {E["tickers"][j]} March 2006..2015 mean log return {hand:+.6f}, panel {z["same"][j, i]:+.6f}')
    assert abs(hand - z['same'][j, i]) < 1e-5
    lines.append('LEAK CHECK PASSED')
    return lines


def grade(tag, variant):
    p = WORK / 'scores' / f'season_{tag}_{variant}.npy'
    res, out = {}, []
    for rm in ('uniform', 'volatility'):
        if tag == 'old':
            txt, js = c2eval.judge(p, rebalance=1, random_match=rm)
        else:
            txt, js = c2eval.judge(p, export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30', rebalance=1,
                                   random_match=rm)
        ch = [c2eval.change_over_random(js, ci) for ci in range(3)]
        res[rm] = dict(change=ch, skipped=js.get('skipped'),
                       summary=c2eval.summary_line(js) if tag == 'old' else None,
                       years=[dict(start=y['start'], end=y['end'], costs=y['costs'], spy=y.get('spy_pct'))
                              for y in js.get('years', [])] if tag == 'old' else None,
                       report=js.get('report') if tag == 'new' else None)
        out.append(f'===== random_match={rm} =====\n{txt}')
        if tag == 'old':
            out.append('SUMMARY ' + c2eval.summary_line(js))
        for ci, cb in enumerate((0, 5, 20)):
            out.append(f"CHANGE OVER RANDOM ({rm}, {cb} bps): {ch[ci]['change']:+.2f} pts/yr; per year {ch[ci]['per_year']}; "
                       f"ahead {ch[ci]['ahead']}/{ch[ci]['years']}; se {ch[ci]['se']:.2f}")
        log('\n'.join(l for l in out[-4:] if l.startswith(('SUMMARY', 'CHANGE'))))
    (HERE / f'money_{tag}_{variant}.txt').write_text('\n'.join(out) + '\n')
    (HERE / f'money_{tag}_{variant}.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    what, tag = sys.argv[1], sys.argv[2]
    variant = sys.argv[3] if len(sys.argv) > 3 else 'raw'
    if what == 'scores':
        start = '2007-04-30' if tag == 'old' else None
        E, S, z = scores(tag, variant, start)
        if tag == 'old' and variant == 'raw':
            lines = leak_check(tag, E, S, z)
            print('\n'.join(lines))
            (HERE / 'leak_check.txt').write_text('\n'.join(lines) + '\n')
    elif what == 'judge':
        grade(tag, variant)
