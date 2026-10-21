"""Close location: Tests 1-3 on the old export (experiments/+2.5-close-location.md).

    python3 -I tests.py          # -> out/tests_old.txt, out/tests_old.json, out/weekly_old.tsv

Each formation week (every 5 export days from 2008-01-08), on the ranked set (universe members with P, the same-day
and 5-day returns and the next 5-day return):
  - Fama-MacBeth: OLS of y - m5 (next 5-day simple return minus the export market's) on ranks in [0, 1] of P, the
    same-day return and the 5-day return. P's coefficient = next-week return of the top of P minus the bottom,
    the two controls held fixed. Plain t over the non-overlapping weeks.
  - raw effect: the same with P alone; the decile spread; diagnostics (ranked y, stock-only CLV, rank correlations).
  - the book: long the top 10% of P and short the bottom 10% (or the reverse, per Test 1), equal weight, held 5 days,
    5 bps per side on traded notional against the previous week's drifted book.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np = C.np
c2eval = C.c2eval

LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def weekly(R, sel=None):
    """Per-week statistics on the ranked set (optionally restricted by the boolean row mask `sel`). Returns a dict of
    arrays over the graded weeks. The book here is long top P / short bottom P; the reverse book is its negative with
    the same turnover."""
    t = R['t']
    if sel is not None:
        R = {k: v[sel] for k, v in R.items() if k != 'days'} | {'days': R['days']}
        t = R['t']
    bounds = np.r_[0, np.nonzero(np.diff(t))[0] + 1, len(t)]
    out = {k: [] for k in ('d', 'n', 'q', 'b_P', 'b_r1', 'b_r5', 'raw_P', 'rky_P', 'st_P', 'st_raw', 'c_P_r1', 'c_P_r5',
                           'gross', 'long_x', 'short_x', 'traded', 'dec')}
    prev_l, prev_s, prev_d = {}, {}, None
    weeks_all = np.unique(t)
    wk_pos = {d: n for n, d in enumerate(weeks_all)}
    for a, b in zip(bounds[:-1], bounds[1:]):
        d = int(t[a])
        n = b - a
        if n < C.MIN_NAMES:
            prev_l, prev_s, prev_d = {}, {}, None
            continue
        P, r1, r5, y, m5, ii = (R['P'][a:b].astype(np.float64), R['r1'][a:b].astype(np.float64), R['r5'][a:b].astype(np.float64),
                                R['y'][a:b], R['m5'][a:b], R['i'][a:b])
        Pst = R['Pst'][a:b].astype(np.float64)
        yx = y - m5
        rp, rr1, rr5 = C.rank01(P), C.rank01(r1), C.rank01(r5)
        one = np.ones(n)
        bb = ols(np.c_[one, rp, rr1, rr5], yx)
        braw = ols(np.c_[one, rp], yx)
        bky = ols(np.c_[one, rp, rr1, rr5], C.rank01(y))
        okst = np.isfinite(Pst)
        rst = C.rank01(Pst[okst])
        bst = ols(np.c_[one[okst], rst, rr1[okst], rr5[okst]], yx[okst])
        bstr = ols(np.c_[one[okst], rst], yx[okst])
        # the book: top 10% of P long, bottom 10% short (ties broken by stock row, deterministic)
        q = int(round(0.1 * n))
        order = np.lexsort((ii, P))
        lo, hi = order[:q], order[n - q:]
        gross = y[hi].mean() - y[lo].mean()
        tgt_l = {int(ii[k]): 1.0 / q for k in hi}
        tgt_s = {int(ii[k]): 1.0 / q for k in lo}
        consecutive = prev_d is not None and wk_pos[d] == wk_pos[prev_d] + 1
        if not consecutive:
            prev_l, prev_s = {}, {}
        traded = 0.0
        for tgt, prev in ((tgt_l, prev_l), (tgt_s, prev_s)):
            for k in set(tgt) | set(prev):
                traded += abs(tgt.get(k, 0.0) - prev.get(k, 0.0))
        yi = dict(zip(ii.tolist(), y.tolist()))
        prev_l = {k: w * (1 + yi[k]) for k, w in tgt_l.items()}
        prev_s = {k: w * (1 + yi[k]) for k, w in tgt_s.items()}
        prev_d = d
        # deciles of P (raw): mean of y minus the ranked set's mean, per decile
        dec = np.array([yx[order[int(np.floor(n * g / 10)):int(np.floor(n * (g + 1) / 10))]].mean() - yx.mean() for g in range(10)])
        out['d'].append(d)
        out['n'].append(n)
        out['q'].append(q)
        out['b_P'].append(bb[1])
        out['b_r1'].append(bb[2])
        out['b_r5'].append(bb[3])
        out['raw_P'].append(braw[1])
        out['rky_P'].append(bky[1])
        out['st_P'].append(bst[1])
        out['st_raw'].append(bstr[1])
        out['c_P_r1'].append(np.corrcoef(rp, rr1)[0, 1])
        out['c_P_r5'].append(np.corrcoef(rp, rr5)[0, 1])
        out['gross'].append(gross)
        out['long_x'].append(y[hi].mean() - y.mean())
        out['short_x'].append(y[lo].mean() - y.mean())
        out['traded'].append(traded)
        out['dec'].append(dec)
    return {k: np.array(v) for k, v in out.items()}


def window(W, days, a, b):
    """Mask of weeks formed in [a, b) (ISO dates)."""
    ia, ib = c2eval.day_index(days, a), c2eval.day_index(days, b)
    return (W['d'] >= ia) & (W['d'] < ib)


def stats(W, m, sign):
    """Summary of the weeks in mask m, with the book in direction `sign` (+1 = long top P)."""
    cost = C.COST_BPS / 1e4 * W['traded'][m]
    g = sign * W['gross'][m]
    net = g - cost
    return dict(weeks=int(m.sum()), n_median=float(np.median(W['n'][m])) if m.any() else 0,
                b_P=float(W['b_P'][m].mean()), t_P=C.tstat(W['b_P'][m]),
                b_r1=float(W['b_r1'][m].mean()), t_r1=C.tstat(W['b_r1'][m]),
                b_r5=float(W['b_r5'][m].mean()), t_r5=C.tstat(W['b_r5'][m]),
                raw_P=float(W['raw_P'][m].mean()), t_raw=C.tstat(W['raw_P'][m]),
                rky_P=float(W['rky_P'][m].mean()), t_rky=C.tstat(W['rky_P'][m]),
                st_P=float(W['st_P'][m].mean()), t_st=C.tstat(W['st_P'][m]),
                st_raw=float(W['st_raw'][m].mean()), t_st_raw=C.tstat(W['st_raw'][m]),
                c_P_r1=float(W['c_P_r1'][m].mean()), c_P_r5=float(W['c_P_r5'][m].mean()),
                gross=float(g.mean()), t_gross=C.tstat(g), net=float(net.mean()), t_net=C.tstat(net),
                sd_gross=float(g.std(ddof=1)),
                long_x=float((W['long_x'][m] if sign > 0 else W['short_x'][m]).mean()),
                short_x=float((W['short_x'][m] if sign > 0 else W['long_x'][m]).mean()),
                traded=float(W['traded'][m].mean()), cost=float(cost.mean()),
                dec=[float(x) for x in W['dec'][m].mean(0)] if m.any() else [])


def pct(x, nd=3):
    return f'{100 * x:+.{nd}f}'


def fmt_row(s):
    return (f"{s['weeks']:>4}  {pct(s['b_P'])} ({s['t_P']:+.2f})  {pct(s['raw_P'])} ({s['t_raw']:+.2f})  "
            f"{pct(s['gross'])} ({s['t_gross']:+.2f})  {pct(s['net'])} ({s['t_net']:+.2f})")


def years_of(W, days):
    return np.array([int(c2eval.iso(days[d])[:4]) for d in W['d']])


def main():
    R = C.load_rows('old')
    days = R['days']
    say(f'rows {len(R["t"])}')
    W = weekly(R)
    W5 = weekly(R, R['in500'])
    # ---- Test 1: direction from P's mean coefficient over 2008-2019
    m1 = window(W, days, *C.TEST1)
    s1 = stats(W, m1, +1)
    sign = 1 if s1['b_P'] > 0 else -1
    s1 = stats(W, m1, sign)
    say('\n== Test 1: weekly Fama-MacBeth, 2008-2019 (y = next 5-day simple return minus the market; regressors ranked 0..1)')
    say(f"weeks {s1['weeks']}, median names {s1['n_median']:.0f}")
    say(f"P (controls)      {pct(s1['b_P'])}% a week, t {s1['t_P']:+.2f}")
    say(f"same-day return   {pct(s1['b_r1'])}% a week, t {s1['t_r1']:+.2f}")
    say(f"5-day return      {pct(s1['b_r5'])}% a week, t {s1['t_r5']:+.2f}")
    say(f"P alone (raw)     {pct(s1['raw_P'])}% a week, t {s1['t_raw']:+.2f}")
    say(f"diag: P with ranked y (rank units) {s1['rky_P']:+.4f}, t {s1['t_rky']:+.2f}; stock-only CLV (controls) "
        f"{pct(s1['st_P'])} t {s1['t_st']:+.2f}, alone {pct(s1['st_raw'])} t {s1['t_st_raw']:+.2f}")
    say(f"mean rank correlation of P with the same-day return {s1['c_P_r1']:+.3f}, with the 5-day return {s1['c_P_r5']:+.3f}")
    p1 = abs(s1['t_P']) >= 3
    say(f"Direction fixed: {'long high P (continuation)' if sign > 0 else 'long low P (reversal)'} (sign {sign:+d}); "
        f"Test 1 {'PASSES' if p1 else 'FAILS'} (|t| {abs(s1['t_P']):.2f} vs 3)")
    # ---- Test 2: the book in that direction
    say('\n== Test 2: the weekly book in that direction, 2008-2019 (% a week on $1 long + $1 short)')
    say(f"gross {pct(s1['gross'])} (t {s1['t_gross']:+.2f}), net {pct(s1['net'])} (t {s1['t_net']:+.2f}); long leg vs ranked "
        f"mean {pct(s1['long_x'])}, short leg {pct(s1['short_x'])} (the book earns long - short); weekly sd {pct(s1['sd_gross'], 2)}; "
        f"traded {s1['traded']:.2f} per week (of 4 max), cost {pct(s1['cost'])}")
    p2 = s1['t_net'] >= 2
    say(f"Test 2 {'PASSES' if p2 else 'FAILS'} (net t {s1['t_net']:+.2f} vs 2)")
    # ---- Test 3
    m3 = window(W, days, *C.TEST3)
    s3 = stats(W, m3, sign)
    say('\n== Test 3: 2020..2025-06 (holds ending by 2025-06-27)')
    say(f"weeks {s3['weeks']}; P (controls) {pct(s3['b_P'])} t {s3['t_P']:+.2f}; raw {pct(s3['raw_P'])} t {s3['t_raw']:+.2f}; "
        f"book gross {pct(s3['gross'])} (t {s3['t_gross']:+.2f}), net {pct(s3['net'])} (t {s3['t_net']:+.2f})")
    p3 = np.sign(s3['b_P']) == np.sign(s1['b_P'])
    say(f"Test 3 (old export part) {'PASSES' if p3 else 'FAILS'}: P's coefficient sign {np.sign(s3['b_P']):+.0f} vs Test 1 {np.sign(s1['b_P']):+.0f}")
    # ---- 500 largest
    say('\n== The 500 largest (ranked within the 500)')
    res500 = {}
    for name, (a, b) in (('2008-2019', C.TEST1), ('2020-2025-06', C.TEST3), ('2008-2025-06', ('2008-01-01', '2025-06-28'))):
        s = stats(W5, window(W5, days, a, b), sign)
        res500[name] = s
        say(f"{name}: weeks {s['weeks']}; P (controls) {pct(s['b_P'])} t {s['t_P']:+.2f}; raw {pct(s['raw_P'])} t {s['t_raw']:+.2f}; "
            f"book gross {pct(s['gross'])} t {s['t_gross']:+.2f}, net {pct(s['net'])} t {s['t_net']:+.2f}; traded {s['traded']:.2f}")
    # ---- periods
    say('\n== By period: weeks, P coef % (t), raw P % (t), book gross % (t), book net % (t) | 500 largest: P coef (t), book net (t)')
    per = {}
    for name, a, b in C.PERIODS:
        s = stats(W, window(W, days, a, b), sign)
        s5 = stats(W5, window(W5, days, a, b), sign)
        per[name] = {'all': s, '500': s5}
        say(f"{name:<13} {fmt_row(s)} | {pct(s5['b_P'])} ({s5['t_P']:+.2f})  {pct(s5['net'])} ({s5['t_net']:+.2f})")
    sall = stats(W, window(W, days, '2008-01-01', '2025-06-28'), sign)
    s5all = res500['2008-2025-06']
    say(f"{'2008-2025-06':<13} {fmt_row(sall)} | {pct(s5all['b_P'])} ({s5all['t_P']:+.2f})  {pct(s5all['net'])} ({s5all['t_net']:+.2f})")
    # ---- calendar years
    yrs = years_of(W, days)
    say('\n== By calendar year: P coef % (t), book gross %, book net % (sum over the year\'s weeks)')
    by_year = {}
    for y in sorted(set(yrs)):
        m = yrs == y
        s = stats(W, m, sign)
        g = sign * W['gross'][m]
        net = g - C.COST_BPS / 1e4 * W['traded'][m]
        by_year[int(y)] = dict(b_P=s['b_P'], t_P=s['t_P'], gross_sum=float(g.sum()), net_sum=float(net.sum()))
        say(f"{y}  {pct(s['b_P'])} ({s['t_P']:+.2f})  {100 * g.sum():+.1f}  {100 * net.sum():+.1f}")
    same = sum(1 for v in by_year.values() if np.sign(v['b_P']) == sign)
    say(f"years with P's coefficient in the Test-1 direction: {same} of {len(by_year)}")
    # ---- deciles
    say('\n== Raw deciles of P, 2008-2019: mean next-week return minus the ranked set, % (decile 1 = lowest P)')
    say('  '.join(f'{100 * x:+.3f}' for x in s1['dec']))
    say('2020-2025-06: ' + '  '.join(f'{100 * x:+.3f}' for x in s3['dec']))
    res = dict(sign=sign, test1=s1, test1_pass=bool(p1), test2_pass=bool(p2), test3=s3, test3_pass=bool(p3), all=sall,
               top500=res500, periods=per, by_year=by_year)
    C.OUT.mkdir(exist_ok=True)
    (C.OUT / 'tests_old.json').write_text(json.dumps(res, indent=1, default=float))
    (C.OUT / 'tests_old.txt').write_text('\n'.join(LOG) + '\n')
    with open(C.OUT / 'weekly_old.tsv', 'w') as f:
        f.write('date\tnames\tleg\tb_P\tb_r1\tb_r5\traw_P\tbook_gross_longtopP\ttraded\n')
        for k in range(len(W['d'])):
            f.write(f"{c2eval.iso(days[W['d'][k]])}\t{W['n'][k]}\t{W['q'][k]}\t{W['b_P'][k]:.6f}\t{W['b_r1'][k]:.6f}\t"
                    f"{W['b_r5'][k]:.6f}\t{W['raw_P'][k]:.6f}\t{W['gross'][k]:.6f}\t{W['traded'][k]:.4f}\n")


if __name__ == '__main__':
    main()
