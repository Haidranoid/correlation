"""The money test. Account A (the file-name number): daily scores = ILLIQ's percentile within the latest grid day's
U_J sample, after the 1% rule on the day's ILLIQ, so the judge buys the 20 most illiquid. Account B: the controlled
residual's percentile from the latest grid day. Graded with rebalance=21, delay=1 (A also at delay=0), uniform and
volatility-matched random accounts, one judge run at a time.

    python3 -I -B money.py scores old|new
    python3 -I -B money.py judge old|new
    python3 -I -B money.py costs old|new     (rebuild A's trades, check them, the square-root cost of trading, regrade)
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import c2eval, np, WORK, OUT, EXPORTS, TRIM, COST_BPS, M77, log, pct_rank, dumps  # noqa: E402
import tests as TT  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/after_costs/group_b')
import ac  # noqa: E402

SC = {'A': 'illiq_money_A', 'B': 'illiq_money_B'}


def scores(tag):
    E = c2eval.load_export(EXPORTS[tag], arrays=())
    Z = dict(np.load(WORK / f'form_{tag}.npz', allow_pickle=False))
    rebal, okj, sj = Z['rebal'], Z['ok_j'], Z['samp_j']
    ILL = np.load(WORK / f'illiq_{tag}.npy')
    N, T = ILL.shape
    SA = np.full((N, T), np.nan, np.float32)
    SB = np.full((N, T), np.nan, np.float32)
    resid_k = {}
    for d in range(T):
        k = int(np.searchsorted(rebal, d, side='right')) - 1
        if k < 0 or not okj[k].any():
            continue
        col = np.asarray(ILL[:, d], np.float64)
        m = np.nonzero(okj[k] & np.isfinite(col))[0]
        if len(m) >= 100:
            v = col[m]
            lo, hi = np.quantile(v, [TRIM, 1 - TRIM])
            m = m[(v >= lo) & (v <= hi)]
            SA[m, d] = pct_rank(col[m])
        if k not in resid_k:
            j = np.nonzero(sj[k])[0]
            resid_k[k] = (j, pct_rank(TT.residual(Z, k, j, 'both')) if len(j) >= 100 else None)
        j, rr = resid_k[k]
        if rr is not None:
            SB[j, d] = rr
    for nm, S in (('A', SA), ('B', SB)):
        p = c2eval.save_scores(S, f'amihud_illiquidity/{SC[nm]}_{tag}', export=EXPORTS[tag])
        fs = int(np.argmax(np.isfinite(S).any(0)))
        log('saved', p, 'first scored day', c2eval.iso(E['days'][fs]), 'mean scored per day',
            float(np.isfinite(S[:, fs:]).sum(0).mean()))
    look(tag, E, SA, 'A')
    look(tag, E, SB, 'B')


def sec_names():
    try:
        return {r['ticker'].upper(): r['name'] for r in json.loads(Path(c2eval.SEC_MAP).read_text())}
    except (OSError, ValueError, KeyError):
        return {}


def look(tag, E, S, nm):
    """Top-20 scores on the judge's first decision day of each test year, with SEC names, and the most frequent picks.
    (Copied from research/max_lottery/money.py.)"""
    names = sec_names()
    tk = [str(t) for t in E['tickers']]
    days = E['days']
    lines, cnt = [], Counter()
    years = range(2006, 2025) if tag == 'old' else [2025]
    for y in years:
        h0 = c2eval.day_index(days, f'{y}-06-28' if tag == 'old' else '2025-06-30')
        for d in range(h0 - 1, min(h0 + 251, len(days) - 1), 21):
            s = S[:, d]
            ok = np.isfinite(s)
            if not ok.any():
                continue
            top = np.nonzero(ok)[0][np.argsort(-s[ok], kind='mergesort')[:20]]
            cnt.update(tk[j] for j in top)
            if d == h0 - 1:
                lines.append(f'{c2eval.iso(days[d])}: ' + ', '.join(
                    f"{tk[j]} ({names.get(tk[j].replace('.', '-'), '?')[:24]})" for j in top))
    lines.append('most frequent top-20 names on the judge decision days: ' +
                 ', '.join(f"{t} {c} ({names.get(t.replace('.', '-'), '?')[:20]})" for t, c in cnt.most_common(60)))
    (OUT / f'picks_{nm}_{tag}.txt').write_text('\n'.join(lines) + '\n')


def after_cost_single(js, cost_bps=5.0):
    """A single holdout's account at cost minus SPY's Yahoo total return over the same window, in points."""
    r = js['report']
    c = [x for x in r['strategy'] if abs(x['cost_bps'] - cost_bps) < 1e-9][0]
    spy = 100 * ac.spy_ret(r['first_day'], r['last_day'])
    return c['final_pct'] - spy, c['final_pct'], spy, r['first_day'], r['last_day']


def grade(tag):
    runs = [('A', 1, 'uniform'), ('A', 1, 'volatility'), ('A', 0, 'uniform'), ('A', 0, 'volatility'),
            ('B', 1, 'uniform'), ('B', 1, 'volatility')]
    res, out = {}, []
    for nm, dl, rm in runs:
        p = WORK / f'{SC[nm]}_{tag}.npy'
        log('judge', nm, 'delay', dl, rm)
        if tag == 'old':
            txt, js = c2eval.judge(p, rebalance=21, delay=dl, random_match=rm)
        else:
            txt, js = c2eval.judge(p, export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30', rebalance=21,
                                   delay=dl, random_match=rm)
        cor = c2eval.change_over_random(js)
        key = f'{nm}_d{dl}_{rm}'
        row = {'change0': cor}
        row['change5'] = c2eval.change_over_random(js, cost_index=1)
        row['change20'] = c2eval.change_over_random(js, cost_index=2)
        jp = Path(str(p).replace('.npy', f'__r21_d{dl}_{rm}.json'))
        if tag == 'old':
            a = ac.judge_after_cost(jp)
            row['after_cost5'] = a['after_cost']
            row['strat5'] = a['strat5']
            row['spy'] = a['spy']
            row['years'] = a['n']
            row['first'], row['last'] = a['first'], a['last']
            row['per_year'] = [(r[0], r[1], r[2], r[4], r[2] - r[4]) for r in a['rows']]
            a20 = ac.judge_after_cost(jp, 20.0)
            row['after_cost20'] = a20['after_cost']
            out.append(f'===== {key} =====\nSUMMARY ' + c2eval.summary_line(js))
        else:
            ac5, s5, spy, f0, f1 = after_cost_single(js)
            row.update(after_cost5_total=ac5, strat5=s5, spy=spy, first=f0, last=f1)
            ac20, s20, _, _, _ = after_cost_single(js, 20.0)
            row['after_cost20_total'] = ac20
            r = js['report']
            row['strategy'] = r['strategy']
            row['candidates'] = r['candidates']
            row['ic_ret_t'] = r.get('ic_ret_t')
            out.append(f'===== {key} =====\n' + json.dumps({k: r[k] for k in ('candidates', 'days', 'ic_ret_mean', 'ic_ret_t',
                                                                          'ls_bps_per_day', 'ls_t', 'ls_turnover')}))
        out.append(f"CHANGE OVER RANDOM ({key}, 0 bps): {cor['change']:+.2f}; per year {cor['per_year']}; ahead "
                   f"{cor['ahead']}/{cor['years']}; se {cor['se']:.2f}")
        out.append(f"  at 5 / 20 bps: {row['change5']['change']:+.2f} / {row['change20']['change']:+.2f}")
        if tag == 'old':
            out.append(f"  AFTER COSTS (5 bps minus SPY, mean over {row['years']} judge years {row['first']}..{row['last']}): "
                       f"{row['after_cost5']:+.2f}; at 20 bps {row['after_cost20']:+.2f}; strategy {row['strat5']:+.2f}, SPY "
                       f"{row['spy']:+.2f}")
        else:
            out.append(f"  AFTER COSTS over the holdout ({row['first']}..{row['last']}): {row['after_cost5_total']:+.2f} "
                       f"points (strategy {row['strat5']:+.2f}%, SPY {row['spy']:+.2f}%); at 20 bps "
                       f"{row['after_cost20_total']:+.2f}")
        res[key] = row
        print('\n'.join(out[-4:]), flush=True)
    (OUT / f'money_{tag}.txt').write_text('\n'.join(out) + '\n')
    (OUT / f'money_{tag}.json').write_text(dumps(res))
    if tag == 'old':
        year_table(tag)


def year_table(tag):
    u = json.load(open(WORK / f'{SC["A"]}_{tag}__r21_d1_uniform.json'))
    v = json.load(open(WORK / f'{SC["A"]}_{tag}__r21_d1_volatility.json'))
    L = ['| Year | Candidates | Picks, 0 bps | Picks, 5 bps | Median random, uniform | Median random, volatility-matched | '
         'SPY (Yahoo) | 5 bps minus SPY | p (uniform) | p (volatility) |', '|---|---|---|---|---|---|---|---|---|---|']
    for a, b in zip(u['years'], v['years']):
        c0, c5 = a['costs'][0], a['costs'][1]
        spy = 100 * ac.spy_ret(a['first_day'], a['last_day'])
        L.append(f"| {a['start'][:4]}-{a['end'][2:4]} | {a['candidates']} | {c0['final_pct']:+.1f}% | {c5['final_pct']:+.1f}% | "
                 f"{c0['random_median_pct']:+.1f}% | {b['costs'][0]['random_median_pct']:+.1f}% | {spy:+.1f}% | "
                 f"{c5['final_pct'] - spy:+.1f} | {c0['p']:.2f} | {b['costs'][0]['p']:.2f} |")
    (OUT / f'money_years_{tag}.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


# ---- A's trades, rebuilt with a copy of an earlier test's copy of the judge's Judge, and the square-root cost of trading ----
WINDOW, LOOKBACK, MIN_PRICE, MIN_DV = 756, 252, 5.0, 5e6


def judge_universe(E, h0):
    """an earlier test's code judge_universe (holdout.rs liquid_universe), copied."""
    v, close, dv = E['v'], E['close'], E['dollar_volume']
    w0 = max(h0 - WINDOW, 0)
    n_days = h0 - w0
    tail = min(n_days, LOOKBACK)
    vv = np.asarray(v[:, w0:h0], dtype=np.float32)
    enough = np.isfinite(vv).sum(axis=1) * 10 >= n_days * 9
    cc = np.asarray(close[:, h0 - tail:h0], dtype=np.float32)
    dd = np.asarray(dv[:, h0 - tail:h0], dtype=np.float32)

    def upper_median(x):
        k = np.isfinite(x).sum(axis=1)
        s = np.sort(np.where(np.isfinite(x), x, np.inf), axis=1)
        idx = np.minimum(k // 2, x.shape[1] - 1)
        m = np.take_along_axis(s, idx[:, None], axis=1)[:, 0]
        return np.where(k > 0, m, np.float32(0.0))

    ok = enough & (upper_median(cc) >= np.float32(MIN_PRICE)) & (upper_median(dd) >= np.float32(MIN_DV))
    return np.flatnonzero(ok)


def judge_year(E, close32, S, start, end, picks, rebalance, delay):
    """an earlier test's code Judge.year, copied: account values at 0 and 5 bps and the trades."""
    days = E['days']
    h0 = c2eval.day_index(days, start)
    t = c2eval.day_index(days, end) if end else len(days)
    cand = judge_universe(E, h0)
    dlist = list(range(h0 - 1, t - 1))
    pk = []
    for k, d in enumerate(dlist):
        if k % rebalance == 0 and d < t:
            s = S[cand, d].astype(np.float64)
            pos = np.flatnonzero(s > 0)
            order = pos[np.argsort(-s[pos], kind='stable')][:picks]
            pk.append(order)
        else:
            pk.append(pk[-1] if pk else np.zeros(0, int))
    w = 1.0 / picks
    v0 = v5 = 1e4
    prev = np.zeros(0, int)
    trades = []
    for k, d in enumerate(dlist):
        now = pk[k - delay] if k >= delay else np.zeros(0, int)
        rows = cand[now]
        a, b = close32[rows, d], close32[rows, d + 1]
        with np.errstate(invalid='ignore', divide='ignore'):
            r = (b / a - np.float32(1.0)).astype(np.float64)
        r = np.where(np.isfinite(a) & np.isfinite(b) & (a > 0), r, 0.0)
        gross = float((w * r).sum())
        new = np.setdiff1d(now, prev)
        gone = np.setdiff1d(prev, now)
        turn = w * (len(new) + len(gone))
        v0 *= 1.0 + gross
        v5 *= 1.0 + gross - turn * 5.0 / 1e4
        for c in np.concatenate([new, gone]):
            trades.append((int(cand[c]), d))
        prev = now
    return 100 * (v0 / 1e4 - 1), 100 * (v5 / 1e4 - 1), trades, len(cand)


def ref77(tag, A77, E):
    """the earlier saved reference: the median Amihud of its 500 most-traded stocks (saved universe500_<tag>), per day."""
    if tag == 'old' and (M77 / 'amihud_ref_old.npy').exists():
        return np.load(M77 / 'amihud_ref_old.npy')
    z = np.load(M77 / f'universe500_{tag}.npz')
    tw_rebal = np.arange(252, len(E['days']), 21)
    T = A77.shape[1]
    ref = np.full(T, np.nan)
    for k, t0 in enumerate(tw_rebal):
        a, b = int(t0), (int(tw_rebal[k + 1]) if k + 1 < len(tw_rebal) else T)
        ref[a:b] = np.nanmedian(np.asarray(A77[z[f'k{k}'], a:b], np.float64), axis=0)
    return ref


def costs(tag):
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume', 'v'))
    close32 = np.asarray(E['close'], np.float32)
    p = WORK / f'{SC["A"]}_{tag}.npy'
    S = np.load(p, mmap_mode='r')
    js = json.loads(Path(str(p).replace('.npy', '__r21_d1_uniform.json')).read_text())
    if tag == 'old' and (M77 / 'amihud_old.npy').exists():
        A77 = np.load(M77 / 'amihud_old.npy', mmap_mode='r')
    else:
        A77 = np.load(WORK / f'illiq77_{tag}.npy', mmap_mode='r')
    ref = ref77(tag, A77, E)
    if js.get('years'):
        yrs = [(y['start'], y['end'], {float(c['cost_bps']): c['final_pct'] for c in y['costs']}, y['candidates'])
               for y in js['years']]
    else:
        r = js['report']
        yrs = [(js['options']['start'], None, {float(c['cost_bps']): c['final_pct'] for c in r['strategy']},
                r['candidates'])]
    trades, checks = [], []
    for st, en, cst, nc in yrs:
        a0, a5, tr, n = judge_year(E, close32, S, st, en, 20, 21, 1)
        ok = abs(a0 - cst[0.0]) < 0.01 and abs(a5 - cst[5.0]) < 0.01 and n == nc
        checks.append(dict(start=st, acct0=a0, judge0=cst[0.0], acct5=a5, judge5=cst[5.0], cand=n, judge_cand=nc, match=bool(ok)))
        trades += tr
    ratios = np.array([A77[j, d] / ref[d] for j, d in trades], float)
    ratios = ratios[np.isfinite(ratios)]
    med = float(np.median(ratios))
    cost = max(COST_BPS, COST_BPS * np.sqrt(med))
    res = dict(checks=checks, all_match=all(c['match'] for c in checks), n_trades=len(trades), median_ratio=med,
               p25=float(np.percentile(ratios, 25)), p75=float(np.percentile(ratios, 75)), cost_sqrt=cost,
               cost_linear=max(COST_BPS, COST_BPS * med))
    c = round(cost, 1)
    log('cost of trading', c, 'bps; regrading at it')
    jo = WORK / f'{SC["A"]}_{tag}__r21_d1_uniform_cost77.json'
    if tag == 'old':
        txt, js2 = c2eval.judge(p, rebalance=21, delay=1, random_match='uniform', costs=f'0,5,{c}', json_out=jo)
        a = ac.judge_after_cost(jo, c)
        res.update(cost_used=c, after_cost_at_cost=a['after_cost'], strat_at_cost=a['strat5'],
                   change_at_cost=c2eval.change_over_random(js2, cost_index=2)['change'])
    else:
        txt, js2 = c2eval.judge(p, export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30', rebalance=21, delay=1,
                                random_match='uniform', costs=f'0,5,{c}', json_out=jo)
        acx, sx, spy, f0, f1 = after_cost_single(js2, c)
        res.update(cost_used=c, after_cost_at_cost_total=acx, strat_at_cost=sx)
    (OUT / f'money_costs_{tag}.json').write_text(dumps(res))
    print(dumps({k: v for k, v in res.items() if k != 'checks'}))
    print('rebuilt years matching the judge:', sum(c['match'] for c in checks), 'of', len(checks))


if __name__ == '__main__':
    what, tag = sys.argv[1], sys.argv[2]
    {'scores': scores, 'judge': grade, 'costs': costs}[what](tag)
