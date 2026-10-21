"""Decoupling: Tests 1-3 and the reported splits, with the matched non-breaks.

Outcome of an event = D x fwd, fwd = sum of e over t+1..t+63 with the twin and beta of the event's rebalance.
Matched non-breaks: universe stock-days in the event's calendar month with a valid break measure, Drop < 0.10,
the same D, and the same decile of |S3| (deciles over all universe stock-days of that month with a valid measure,
events included), other stocks, finite fwd. Up to 20, seeded, without replacement. Standard errors clustered by the
calendar month of t (CR1).

    python3 -I -B tests.py old|new      # writes out/tests_{tag}.txt and .json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402

LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def match(ev, P, U, days, require_rb=False, seed=C.SEED):
    """Per event: mean control outcome (D x fwd of the controls), number of controls, pool size of its bin."""
    rebal = U['rebal']
    blk = np.searchsorted(rebal, np.arange(len(days)), side='right') - 1
    mon = C.month_of(days)
    ev_mon = mon[ev['t']]
    n = len(ev['t'])
    ctrl_mean = np.full(n, np.nan)
    n_ctrl = np.zeros(n, int)
    bin_size = np.zeros(n, int)
    ev_dec = np.full(n, -1)
    rng = np.random.default_rng(seed)
    order = np.lexsort((ev['j'], ev['t']))
    for m in np.unique(ev_mon):
        dd = np.nonzero((mon == m) & (blk >= 0))[0]
        Um = U['uni'][blk[dd]].T
        rb = np.asarray(P['rho_b'][:, dd], np.float64)
        rn = np.asarray(P['rho_n'][:, dd], np.float64)
        s3 = np.asarray(P['S3'][:, dd], np.float64)
        fw = np.asarray(P['fwd'][:, dd], np.float64)
        drop = rb - rn
        with np.errstate(invalid='ignore'):
            pool = Um & np.isfinite(drop) & np.isfinite(s3)
            edges = np.quantile(np.abs(s3[pool]), np.arange(1, 10) / 10)
            dec = np.searchsorted(edges, np.abs(s3), side='right')
            ctrl = pool & (drop < C.NONBREAK_MAX) & np.isfinite(fw)
            if require_rb:
                ctrl &= rb >= C.RHO_BEFORE_MIN
        sg = np.sign(s3)
        for i in order[ev_mon[order] == m]:
            de = int(np.searchsorted(edges, abs(ev['S3'][i]), side='right'))
            ev_dec[i] = de
            c = ctrl & (sg == ev['D'][i]) & (dec == de)
            c[ev['j'][i]] = False
            rr, cc = np.nonzero(c)
            bin_size[i] = len(rr)
            if len(rr) == 0:
                continue
            pick = rng.choice(len(rr), size=min(C.N_MATCH, len(rr)), replace=False)
            ctrl_mean[i] = float(np.mean(ev['D'][i] * fw[rr[pick], cc[pick]]))
            n_ctrl[i] = len(pick)
    return ctrl_mean, n_ctrl, bin_size, ev_dec


def fmt(mu, t, n, G=None, scale=1e4, unit=' bps'):
    if not np.isfinite(mu):
        return f'n/a (n {n})'
    g = f', {G} months' if G is not None else ''
    return f'{scale * mu:+.0f}{unit} (t {t:+.2f}, n {n}{g})'


def main(tag):
    ev = dict(np.load(C.WORK / f'events_{tag}.npz'))
    U = dict(np.load(C.WORK / f'univ_{tag}.npz'))
    days = U['days']
    P = {p: np.load(C.WORK / f'{p}_{tag}.npy', mmap_mode='r') for p in C.PANELS}
    ev['out'] = ev['D'] * ev['fwd']
    ev['out_all'] = ev['Dall'] * ev['fwd']
    ev['mon'] = C.month_of(ev['day'])
    ev['ctrl'], ev['n_ctrl'], ev['bin'], ev['dec'] = match(ev, P, U, days)
    ev['ctrl_rb'], ev['n_ctrl_rb'], _, _ = match(ev, P, U, days, require_rb=True)
    ev['diff'] = ev['out'] - ev['ctrl']
    ev['diff_rb'] = ev['out'] - ev['ctrl_rb']
    d64 = lambda s: (np.datetime64(s) - np.datetime64('1970-01-01')).astype(int)  # noqa: E731
    if tag == 'old':
        periods = [('2005 (before the test period)', '2004-01-01', '2005-12-31'),
                   ('2006-2012', '2006-01-01', '2012-12-31'), ('2013-2019', '2013-01-01', '2019-12-31'),
                   ('2006-2019 (Tests 1 and 2)', '2006-01-01', '2019-12-31'),
                   ('2020-2025-06-27 (Test 3)', '2020-01-01', '2025-06-27'),
                   ('All, old export', '2004-01-01', '2025-06-27')]
    else:
        periods = [('Unseen months 2025-06-30..2026-10-19 (Test 3)', '2025-06-30', '2026-10-19')]
    res = {'tag': tag, 'periods': {}}
    say(f'# Decoupling tests, {tag} export\n')
    say('Outcome = D x (sum of e over t+1..t+63), in log-return bps. t = month-clustered (CR1) [iid t].\n')
    for name, a, b in periods:
        m = (ev['day'] >= d64(a)) & (ev['day'] <= d64(b))
        if not m.any():
            continue
        r = {}
        mo = ev['mon'][m]
        o = ev['out'][m]
        has = np.isfinite(o)
        r['events'] = int(m.sum())
        r['with_outcome'] = int(has.sum())
        r['d_pos_share'] = float(np.mean(ev['D'][m] > 0))
        mu, t, n, G, ti = C.cr1_mean(o, mo)
        r['outcome'] = dict(mean=mu, t=t, n=n, months=G, t_iid=ti, hit=float(np.mean(o[has] > 0)) if has.any() else None,
                            median=float(np.median(o[has])) if has.any() else None)
        say(f'## {name}: {r["events"]} events, {r["with_outcome"]} with a 63-day outcome, D > 0: {r["d_pos_share"]:.0%}')
        say(f'  outcome D x fwd:             {fmt(mu, t, n, G)} [iid t {ti:+.2f}]; median '
            f'{1e4 * (r["outcome"]["median"] or np.nan):+.0f} bps; share > 0 {r["outcome"]["hit"] or np.nan:.0%}')
        for lab, sel in (('D > 0', ev['D'][m] > 0), ('D < 0', ev['D'][m] < 0)):
            mu2, t2, n2, G2, ti2 = C.cr1_mean(o[sel], mo[sel])
            mu3, t3, _, _, _ = C.cr1_mean(ev['fwd'][m][sel], mo[sel])
            r[f'outcome_{lab}'] = dict(mean=mu2, t=t2, n=n2, months=G2, raw_fwd=mu3, raw_fwd_t=t3)
            say(f'    {lab}:                     {fmt(mu2, t2, n2, G2)}; raw hedged drift {1e4 * mu3:+.0f} bps (t {t3:+.2f})')
        mu4, t4, n4, G4, _ = C.cr1_mean(ev['out_all'][m], mo)
        r['outcome_3_days_back'] = dict(mean=mu4, t=t4, n=n4)
        say(f'  3 big days back in D (Dall): {fmt(mu4, t4, n4, G4)}')
        mu5, t5, n5, G5, _ = C.cr1_mean(ev['fwd'][m], mo)
        r['raw_fwd'] = dict(mean=mu5, t=t5, n=n5)
        say(f'  unsigned hedged drift fwd:   {fmt(mu5, t5, n5, G5)}')
        bigm = ev['big'][m].astype(bool)
        mu6, t6, n6, G6, _ = C.cr1_mean(o[bigm], mo[bigm])
        r['outcome_top500'] = dict(mean=mu6, t=t6, n=n6)
        say(f'  the 500 largest only:        {fmt(mu6, t6, n6, G6)}')
        for key, cm, nc, lab in (('matched', 'ctrl', 'n_ctrl', 'matched non-breaks (the plan)'),
                                 ('matched_rb', 'ctrl_rb', 'n_ctrl_rb', 'matches also with rho_before >= 0.5')):
            c = ev[cm][m]
            ok = np.isfinite(c) & has
            muc, tc, nc_, Gc, _ = C.cr1_mean(c[ok], mo[ok])
            mud, td, nd, Gd, tid = C.cr1_mean(o[ok] - c[ok], mo[ok])
            r[key] = dict(events=int(ok.sum()), no_match=int((has & ~np.isfinite(c)).sum()),
                          mean_matches=float(np.mean(ev[nc][m][ok])) if ok.any() else None,
                          ctrl=muc, ctrl_t=tc, event=float(np.mean(o[ok])) if ok.any() else None,
                          diff=mud, diff_t=td, diff_t_iid=tid, months=Gd)
            say(f'  {lab}: events {int(ok.sum())} (no match {r[key]["no_match"]}), matches/event '
                f'{r[key]["mean_matches"] or np.nan:.1f}; event {1e4 * (r[key]["event"] or np.nan):+.0f} bps, '
                f'controls {1e4 * muc:+.0f} bps (t {tc:+.2f}); difference {fmt(mud, td, nd, Gd)} [iid t {tid:+.2f}]')
        res['periods'][name] = r
        say('')

    # per year (events, outcome, difference)
    yr = C.year_of(ev['day'])
    say('## By year: events, D>0, mean outcome bps (n), matched difference bps')
    res['by_year'] = []
    for y in sorted(set(yr.tolist())):
        m = yr == y
        o = ev['out'][m]
        d = ev['diff'][m]
        say(f'  {y}: {int(m.sum()):>4} events, D>0 {np.mean(ev["D"][m] > 0):4.0%}, outcome '
            f'{1e4 * np.nanmean(o) if np.isfinite(o).any() else np.nan:+6.0f} ({int(np.isfinite(o).sum())}), '
            f'difference {1e4 * np.nanmean(d) if np.isfinite(d).any() else np.nan:+6.0f} ({int(np.isfinite(d).sum())})')
        res['by_year'].append(dict(year=int(y), events=int(m.sum()), d_pos=float(np.mean(ev['D'][m] > 0)),
                                   outcome=float(np.nanmean(o)) if np.isfinite(o).any() else None,
                                   n_out=int(np.isfinite(o).sum()),
                                   diff=float(np.nanmean(d)) if np.isfinite(d).any() else None))
    # months with the most events (clustering)
    um, uc = np.unique(ev['mon'], return_counts=True)
    o = np.argsort(-uc)[:8]
    say('\nMonths with the most events: ' + ', '.join(
        f'{str(np.datetime64(int(um[i]), "M"))} {uc[i]}' for i in o))
    say(f'matching bins: median pool size of the event\'s bin {int(np.median(ev["bin"]))}; '
        f'event deciles (0-9) counts {np.bincount(ev["dec"][ev["dec"] >= 0], minlength=10).tolist()}')

    # verdict
    if tag == 'old':
        t1 = res['periods']['2006-2019 (Tests 1 and 2)']
        p1 = t1['outcome']['mean'] >= 0.01 and t1['outcome']['t'] >= 3
        p2 = t1['matched']['diff'] > 0 and t1['matched']['diff_t'] >= 2
        t3 = res['periods']['2020-2025-06-27 (Test 3)']
        p3 = t3['outcome']['mean'] > 0
        res['tests'] = dict(test1=p1, test2=p2, test3_old=p3)
        say(f'\nTest 1 (2006-2019 mean >= +100 bps, month t >= 3): {"PASS" if p1 else "FAIL"} '
            f'({1e4 * t1["outcome"]["mean"]:+.0f} bps, t {t1["outcome"]["t"]:+.2f})')
        say(f'Test 2 (beats matched non-breaks, t >= 2): {"PASS" if p2 else "FAIL"} '
            f'({1e4 * t1["matched"]["diff"]:+.0f} bps, t {t1["matched"]["diff_t"]:+.2f})')
        say(f'Test 3, 2020-2025-06 (mean outcome > 0): {"PASS" if p3 else "FAIL"} ({1e4 * t3["outcome"]["mean"]:+.0f} bps; '
            f'matched difference {1e4 * t3["matched"]["diff"]:+.0f} bps)')
    else:
        u = list(res['periods'].values())[0]
        res['tests'] = dict(test3_unseen=bool(u['outcome']['mean'] > 0))
        say(f'\nTest 3, unseen months (mean outcome > 0): {"PASS" if res["tests"]["test3_unseen"] else "FAIL"}')
    # the event list with outcomes, for the record
    with open(C.OUT / f'events_{tag}.tsv', 'w') as fh:
        fh.write('ticker\tdate\ttwin\tbeta\trho_before\trho_now\tdrop\tdrop_t21\tS3\tSall\tD\tfwd\toutcome\tctrl\tn_ctrl\ttop500\n')
        for i in range(len(ev['t'])):
            fh.write(f'{ev["ticker"][i]}\t{c2eval.iso(ev["day"][i])}\t{ev["twin"][i]}\t{ev["beta"][i]:.3f}\t'
                     f'{ev["rho_b"][i]:.3f}\t{ev["rho_n"][i]:.3f}\t{ev["drop"][i]:.3f}\t{ev["drop21"][i]:.3f}\t'
                     f'{ev["S3"][i]:.4f}\t{ev["Sall"][i]:.4f}\t{int(ev["D"][i])}\t{ev["fwd"][i]:.4f}\t'
                     f'{ev["out"][i]:.4f}\t{ev["ctrl"][i]:.4f}\t{ev["n_ctrl"][i]}\t{int(ev["big"][i])}\n')
    (C.OUT / f'tests_{tag}.txt').write_text('\n'.join(LINES) + '\n')
    (C.OUT / f'tests_{tag}.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main(sys.argv[1])
