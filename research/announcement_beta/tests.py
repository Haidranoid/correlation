"""Slopes, the A-day strategy and the plan's tests 1-3 for one export.

    python3 -I -B tests.py old|new     -> results_{tag}.txt, daily_{tag}.tsv

Per trading day d (universe = the latest rebalance at or before d-1; beta[:, d] uses returns through d-1):
  - slope_d: cross-sectional OLS of the stocks' close-to-close log returns on d (|r| > 0.5 dropped) on beta[:, d];
  - ls_d: equal-weight top beta tenth minus bottom tenth, simple returns from export closes (close d-1 -> close d);
    a ranked stock with no return on d counts as 0. On A days the strategy holds it; net = ls_d - 20 bps
    (5 bps per side, both legs bought at the close before and sold at the A day's close). On N days it is flat;
    ls_d on N days is shown only for comparison.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (HERE, WORK, EXPORTS, PERIODS, COST_BPS, CRASH_MOVE, CRASH_YEARS, DECILE, c2eval, calendar, dnum,  # noqa: E402
                    kidx, log, log_returns, np, tstat, welch)

KINDS = ('fomc', 'cpi', 'jobs')


def daily(tag):
    E = c2eval.load_export(EXPORTS[tag], arrays=('close',))
    z = np.load(WORK / f'meta_{tag}.npz')
    days, m, rebal, uni = z['days'], z['m'], z['rebal'], z['uni']
    T = len(days)
    log('loading beta and closes')
    B = np.load(WORK / f'beta_{tag}.npy')
    C = np.asarray(E['close'], np.float64)
    R = np.full(C.shape, np.nan)
    with np.errstate(all='ignore'):
        R[:, 1:] = C[:, 1:] / C[:, :-1] - 1.0
    R[~np.isfinite(R)] = np.nan
    LR = log_returns(C)
    del C
    cal = calendar(days)
    out = {k: np.full(T, np.nan) for k in ('slope', 'n', 'ls', 'n10', 'b_top', 'b_bot', 'bspread', 'n_miss')}
    for d in range(1, T):
        k = kidx(rebal, d - 1)
        if k < 0:
            continue
        b = B[:, d]
        sel = uni[k] & np.isfinite(b)
        if sel.sum() < 50:
            continue
        idx = np.nonzero(sel)[0]
        bb = b[idx].astype(np.float64)
        r = LR[idx, d]
        ok = np.isfinite(r)
        if ok.sum() >= 50:
            x, y = bb[ok], r[ok]
            xc = x - x.mean()
            out['slope'][d] = float((xc * (y - y.mean())).sum() / (xc * xc).sum())
            out['n'][d] = ok.sum()
        n10 = int(np.floor(DECILE * len(idx)))
        o = np.argsort(bb, kind='stable')
        bot, top = o[:n10], o[-n10:]
        rs = R[idx, d]
        out['n_miss'][d] = int((~np.isfinite(rs[top])).sum() + (~np.isfinite(rs[bot])).sum())
        rs = np.nan_to_num(rs, nan=0.0)
        out['ls'][d] = rs[top].mean() - rs[bot].mean()
        out['n10'][d] = n10
        out['b_top'][d] = bb[top].mean()
        out['b_bot'][d] = bb[bot].mean()
        out['bspread'][d] = bb[top].mean() - bb[bot].mean()
    out['days'] = days
    out['m'] = m
    out.update({k: cal[k] for k in ('fomc', 'cpi', 'jobs', 'A')})
    out['off'] = cal['off']
    return out


def fmt_bps(x):
    return f'{1e4 * x:+.1f}'


def report(tag, D):
    days = D['days']
    T = len(days)
    year = np.array([int(c2eval.iso(d)[:4]) for d in days])
    crash = np.isin(year, CRASH_YEARS) & np.isfinite(D['m']) & (np.abs(D['m']) >= CRASH_MOVE)
    A = D['A']
    have = np.isfinite(D['slope'])
    lines = []
    say = lambda *a: lines.append(' '.join(str(x) for x in a))  # noqa: E731
    say(f'Announcement-day beta, export {tag} ({EXPORTS[tag]})')
    say('Units: slopes in bps of daily log return per unit of beta; strategy in bps per A day (simple returns).')
    say('Calendar releases in the export span that are not trading days (dropped):', ', '.join(D['off']) or 'none')
    first = int(np.nonzero(have)[0][0])
    say('First day with a slope:', c2eval.iso(days[first]))
    say(f'Crash days (|SPY| >= {CRASH_MOVE:.0%} in {CRASH_YEARS}): {int(crash.sum())}, of which A days {int((crash & A).sum())}:',
        ', '.join(f"{c2eval.iso(days[i])}{'*' if A[i] else ''}" for i in np.nonzero(crash)[0]))
    res = {}
    for name, a, b in PERIODS[tag]:
        P = have & (days >= dnum(a)) & (days <= dnum(b))
        for crash_mode in ('with crash days', 'without crash days'):
            Q = P & (~crash if crash_mode.startswith('without') else True)
            key = (name, crash_mode)
            s = D['slope']
            say()
            say(f'=== {name} ({a}..{b}), {crash_mode}')
            nA, nN = int((Q & A).sum()), int((Q & ~A).sum())
            mA, tA, _ = tstat(s[Q & A])
            mN, tN, _ = tstat(s[Q & ~A])
            dif, tw = welch(s[Q & A], s[Q & ~A])
            say(f'  slope A days: {fmt_bps(mA)} (t {tA:+.2f}, n {nA});  N days: {fmt_bps(mN)} (t {tN:+.2f}, n {nN})')
            say(f'  A - N: {fmt_bps(dif)}  Welch t {tw:+.2f}')
            r = {'A': mA, 'tA': tA, 'nA': nA, 'N': mN, 'tN': tN, 'nN': nN, 'diff': dif, 'welch': tw, 'kinds': {}}
            for kd in KINDS:
                K = Q & D[kd]
                mk, tk, nk = tstat(s[K])
                dk, twk = welch(s[K], s[Q & ~A])
                say(f'  {kd:5s} days: slope {fmt_bps(mk)} (t {tk:+.2f}, n {nk});  minus N {fmt_bps(dk)} (Welch t {twk:+.2f})')
                r['kinds'][kd] = {'mean': mk, 't': tk, 'n': nk, 'diff': dk, 'welch': twk}
            # market
            mmA, tmA, _ = tstat(D['m'][Q & A])
            mmN, tmN, _ = tstat(D['m'][Q & ~A])
            say(f'  SPY log return: A {fmt_bps(mmA)} (t {tmA:+.2f}), N {fmt_bps(mmN)} (t {tmN:+.2f})')
            # strategy
            ls = D['ls']
            g, tg, ng = tstat(ls[Q & A])
            net = ls[Q & A] - 4 * COST_BPS / 1e4
            nm, tn, _ = tstat(net)
            gN, tgN, _ = tstat(ls[Q & ~A])
            hit = float((net > 0).mean()) if len(net) else float('nan')
            per_year = len(net) / max(1e-9, (Q.sum()) / 252)
            say(f'  strategy (top - bottom beta tenth, A days): gross {fmt_bps(g)} (t {tg:+.2f}, n {ng}); '
                f'net of 20 bps {fmt_bps(nm)} (t {tn:+.2f}); net > 0 on {hit:.0%} of A days; '
                f'about {per_year:.0f} A days a year -> net {100 * nm * per_year:+.1f}% a year (simple sum)')
            say(f'  the same long-short on N days (not traded): gross {fmt_bps(gN)} (t {tgN:+.2f}); '
                f'beta spread top - bottom {np.nanmean(D["bspread"][Q]):.2f} (top {np.nanmean(D["b_top"][Q]):.2f}, '
                f'bottom {np.nanmean(D["b_bot"][Q]):.2f}); stocks per tenth {np.nanmean(D["n10"][Q]):.0f}; '
                f'slope regression stocks {np.nanmean(D["n"][Q]):.0f}')
            for kd in KINDS:
                K = Q & D[kd]
                gk, tgk, nk = tstat(ls[K])
                say(f'    strategy on {kd:5s} days: gross {fmt_bps(gk)} (t {tgk:+.2f}, n {nk}); net {fmt_bps(gk - 4 * COST_BPS / 1e4)}')
            r['strat'] = {'gross': g, 't_gross': tg, 'n': ng, 'net': nm, 't_net': tn, 'hit': hit, 'gross_N': gN, 't_gross_N': tgN}
            res[key] = r
    say()
    say('Per year (with crash days): A days, slope A, slope N, A - N (Welch t), strategy net per A day, SPY A, SPY N')
    for y in sorted(set(year[have])):
        Y = have & (year == y)
        dif, tw = welch(D['slope'][Y & A], D['slope'][Y & ~A])
        say(f'  {y}: {int((Y & A).sum()):3d}  {fmt_bps(np.nanmean(D["slope"][Y & A])):>7s} {fmt_bps(np.nanmean(D["slope"][Y & ~A])):>7s}  '
            f'{fmt_bps(dif):>7s} ({tw:+.2f})  {fmt_bps(np.nanmean(D["ls"][Y & A]) - 4 * COST_BPS / 1e4):>7s}  '
            f'{fmt_bps(np.nanmean(D["m"][Y & A])):>7s} {fmt_bps(np.nanmean(D["m"][Y & ~A])):>7s}')
    # missing returns inside the tenths on A days
    say()
    say('Ranked stocks with no return on the day, inside the two tenths (counted as 0): A days total',
        int(np.nansum(D['n_miss'][A & have])), 'over', int((A & have).sum()), 'A days')
    return lines, res


def verdict(res_old, res_new=None):
    lines = []
    say = lambda *a: lines.append(' '.join(str(x) for x in a))  # noqa: E731
    r1 = res_old[('2006-2019', 'with crash days')]
    t1a = r1['diff'] > 0 and r1['welch'] >= 3
    t1b = r1['strat']['net'] > 0 and r1['strat']['t_net'] >= 2
    say(f"Test 1 (2006-2019): A - N slope {fmt_bps(r1['diff'])} bps, Welch t {r1['welch']:+.2f} -> {'pass' if t1a else 'FAIL'}; "
        f"strategy net {fmt_bps(r1['strat']['net'])} bps per A day, t {r1['strat']['t_net']:+.2f} -> {'pass' if t1b else 'FAIL'}")
    t2 = all(r1['kinds'][k]['mean'] > 0 for k in KINDS)
    say('Test 2 (2006-2019, each kind positive): ' + ', '.join(f"{k} {fmt_bps(r1['kinds'][k]['mean'])} (t {r1['kinds'][k]['t']:+.2f})" for k in KINDS)
        + f" -> {'pass' if t2 else 'FAIL'}")
    r3 = res_old[('2020-2025-06', 'with crash days')]
    sign = np.sign(r1['diff'])
    t3a = np.sign(r3['diff']) == sign
    msg = f"Test 3 (2020-2025-06): A - N {fmt_bps(r3['diff'])} (Welch t {r3['welch']:+.2f}), same sign as 2006-2019: {'yes' if t3a else 'NO'}"
    t3b = None
    if res_new is not None:
        rn = res_new[('unseen 2025-06-30..2026-10-19', 'with crash days')]
        t3b = np.sign(rn['diff']) == sign
        msg += f"; unseen months A - N {fmt_bps(rn['diff'])} (Welch t {rn['welch']:+.2f}), same sign: {'yes' if t3b else 'NO'}"
    say(msg)
    first_fail = next((n for n, ok in (('1', t1a and t1b), ('2', t2), ('3', t3a and (t3b is not False))) if not ok), None)
    say('Verdict stops at:', f'test {first_fail}' if first_fail else 'all pass')
    return lines


def main(tag):
    D = daily(tag)
    lines, res = report(tag, D)
    days = D['days']
    with open(HERE / f'daily_{tag}.tsv', 'w') as f:
        f.write('date\tA\tfomc\tcpi\tjobs\tslope\tn\tls\tspy\n')
        for i in np.nonzero(np.isfinite(D['slope']))[0]:
            f.write(f"{c2eval.iso(days[i])}\t{int(D['A'][i])}\t{int(D['fomc'][i])}\t{int(D['cpi'][i])}\t{int(D['jobs'][i])}\t"
                    f"{D['slope'][i]:.6g}\t{int(D['n'][i])}\t{D['ls'][i]:.6g}\t{D['m'][i]:.6g}\n")
    import pickle
    (WORK / f'res_{tag}.pkl').write_bytes(pickle.dumps(res))
    if tag == 'old':
        lines += [''] + verdict(res)
    else:
        res_old = pickle.loads((WORK / 'res_old.pkl').read_bytes())
        lines += [''] + verdict(res_old, res)
    (HERE / f'results_{tag}.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1])
