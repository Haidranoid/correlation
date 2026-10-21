"""Step 2: thrusts, the tests, the scores file and the judge (run after bt_refits.py).

    python3 -I bt_tests.py old      # tests 1-3 on 2006-2025-06, the money test (walk-forward); results_old.txt
    python3 -I bt_tests.py new      # the unseen months, once, at the end; results_new.txt

Thrust (fixed rules): on day d, with the members of the latest refit, the group's breadth is >= 0.615 and was
<= 0.40 on some day of the 10 trading days before d. Only the first day of such a run counts (the condition was false
on d-1, with the same members), and none within 21 trading days of the same group's previous thrust. A group is "the
same" across a refit as the previous refit's group holding most of its members (its predecessor), whose last thrust
day it inherits.

Outcome: the members' mean log return from the close of d to the close of d+21 minus the export's market over
d+1..d+21. t statistics are clustered by thrust date (CR1).
"""
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_common import (COOLDOWN, EXPORTS, HERE, HIGH, HOLD, K_GROUPS, LOOKBACK, LOW, MIN_MEMBERS, big, c2eval,  # noqa: E402
                       cluster_ols, clustered_mean, iso)

os.environ.setdefault('RAYON_NUM_THREADS', '4')

P1 = ('2006-01-01', '2019-12-31')
P2 = ('2020-01-01', '2025-06-27')
UNSEEN = ('2025-06-30', '2026-10-19')
PASS_BPS, PASS_T, PASS_T2 = 50.0, 3.0, 2.0

OUT = []


def say(s=''):
    print(s, flush=True)
    OUT.append(s)


def find_thrusts(R):
    """Walk the refits in order. Returns (events list of dicts, group-day arrays dict)."""
    refits = R['refits']
    N = R['N']
    ev = []
    gd = {k: [] for k in ('r', 'g', 'd', 'size', 'breadth', 'cond', 'thrust', 'cool', 'mom', 'y21', 'ny')}
    prev_lab = None
    prev_last = None
    for ri, rf in enumerate(refits):
        t0, t1, U, lab, br = rf['t0'], rf['t1'], rf['U'], rf['lab'], rf['breadth']
        sizes = rf['sizes']
        # predecessor of each group: the previous refit's group holding most of its members
        last = np.full(K_GROUPS, -10 ** 9, np.int64)
        if prev_lab is not None and refits[ri - 1]['t1'] == t0:
            for g in range(K_GROUPS):
                pl = prev_lab[U[lab == g]]
                pl = pl[pl >= 0]
                if len(pl):
                    last[g] = prev_last[np.bincount(pl, minlength=K_GROUPS).argmax()]
        nb = br.shape[1]
        off = LOOKBACK + 1                     # breadth column c <-> day t0 - off + c

        def cond(row, c):
            b = row[c]
            win = row[c - LOOKBACK:c]
            win = win[np.isfinite(win)]
            return bool(np.isfinite(b) and b >= HIGH and len(win) and win.min() <= LOW)

        for g in range(K_GROUPS):
            if sizes[g] < MIN_MEMBERS:
                continue
            row = br[g]
            for c in range(off, nb):
                d = t0 - off + c
                cnow = cond(row, c)
                since = d - last[g]
                th = cnow and not cond(row, c - 1) and since >= COOLDOWN
                cool = 1 <= since <= COOLDOWN - 1
                if th:
                    last[g] = d
                    idx = np.flatnonzero(lab == g)
                    ev.append(dict(r=ri, g=g, d=d, size=int(sizes[g]), label=rf['glabel'][g], breadth=float(row[c]),
                                   low=float(np.nanmin(row[c - LOOKBACK:c])), mom=float(rf['mom'][g, d - t0]),
                                   y21=float(rf['y21'][g, d - t0]), ny=int(rf['ny'][g, d - t0]), members=U[idx],
                                   rho=rf['rho'][idx]))
                for key, val in (('r', ri), ('g', g), ('d', d), ('size', sizes[g]), ('breadth', row[c]),
                                 ('cond', cnow), ('thrust', th), ('cool', cool and not th), ('mom', rf['mom'][g, d - t0]),
                                 ('y21', rf['y21'][g, d - t0]), ('ny', rf['ny'][g, d - t0])):
                    gd[key].append(val)
        pl = np.full(N, -1, np.int64)
        pl[U] = lab
        prev_lab, prev_last = pl, last.copy()
    gd = {k: np.array(v) for k, v in gd.items()}
    return ev, gd


def in_period(days, d, period):
    a, b = c2eval.day_index(days, period[0]), c2eval.day_index(days, period[1])
    if c2eval.iso(days[min(b, len(days) - 1)]) <= period[1] and b < len(days):
        b += 1
    return (d >= a) & (d < b)


def block_t(y, d):
    """reported-only: t clustered by 21-trading-day blocks of the export calendar."""
    return clustered_mean(y, np.asarray(d) // 21)[2]


def analyse(name, days, ev, gd, mask_ev, mask_gd, show=True):
    """Event mean, baselines (a) and (b), for the thrusts in mask_ev against the group-days in mask_gd."""
    d_ev = np.array([e['d'] for e in ev])[mask_ev]
    y_ev = np.array([e['y21'] for e in ev])[mask_ev]
    m_ev = np.array([e['mom'] for e in ev])[mask_ev]
    ok = np.isfinite(y_ev)
    d_ev, y_ev, m_ev = d_ev[ok], y_ev[ok], m_ev[ok]
    res = dict(name=name, n=len(y_ev), dates=len(np.unique(d_ev)))
    if len(y_ev) == 0:
        if show:
            say(f'{name}: no thrusts with a full 21-day outcome')
        return res
    mean, se, t, n, nc = clustered_mean(y_ev, d_ev)
    res.update(mean=mean * 1e4, se=se * 1e4, t=t, t_block=block_t(y_ev, d_ev) if len(y_ev) > 2 else np.nan,
               hit=float((y_ev > 0).mean()))
    # (a) the group's ordinary returns: every other group-day in the period
    gy, gm, gdd = gd['y21'], gd['mom'], gd['d']
    base = mask_gd & np.isfinite(gy)
    ctrl_a = base & ~gd['thrust']
    mu_a = float(gy[ctrl_a].mean())
    da = clustered_mean(y_ev - mu_a, d_ev)
    res.update(mu_a=mu_a * 1e4, diff_a=da[0] * 1e4, t_a=da[2])
    # (b) momentum-matched non-thrusts: deciles of the group's 10-day market-relative return over the period's
    # group-days; controls are group-days where the thrust condition is false and no thrust of the group in 20 days
    pool = base & np.isfinite(gm)
    edges = np.quantile(gm[pool], np.linspace(0, 1, 11)[1:-1])
    dec_gd = np.searchsorted(edges, gm, side='right')
    ok_m = np.isfinite(m_ev)
    dec_ev = np.searchsorted(edges, m_ev, side='right')
    ctrl_b = pool & ~gd['cond'] & ~gd['cool'] & ~gd['thrust']
    mu_b = np.array([gy[ctrl_b & (dec_gd == q)].mean() if (ctrl_b & (dec_gd == q)).any() else np.nan for q in range(10)])
    yb = y_ev[ok_m] - mu_b[dec_ev[ok_m]]
    db = clustered_mean(yb, d_ev[ok_m])
    res.update(diff_b=db[0] * 1e4, t_b=db[2], n_b=int(ok_m.sum()), n_ctrl_b=int(ctrl_b.sum()),
               dec_counts=np.bincount(dec_ev[ok_m], minlength=10), mu_b=mu_b * 1e4)
    # reported-only: (b) as a regression with decile fixed effects on thrusts + controls, clustered by date
    sel = ctrl_b | (pool & gd['thrust'])
    X = np.zeros((sel.sum(), 11))
    X[np.arange(sel.sum()), dec_gd[sel]] = 1
    X[:, 10] = gd['thrust'][sel]
    b, s, _ = cluster_ols(gy[sel], X, gdd[sel])
    res.update(reg_b=b[10] * 1e4, reg_t_b=b[10] / s[10])
    if show:
        say(f'{name}: {res["n"]} thrusts on {res["dates"]} dates; mean {res["mean"]:+.1f} bps (se {res["se"]:.1f}, '
            f't {t:+.2f}; t by 21-day block {res["t_block"]:+.2f}), {res["hit"]:.0%} positive')
        say(f'   (a) ordinary group-days {mu_a * 1e4:+.1f} bps over {int(ctrl_a.sum())} group-days; '
            f'thrust minus (a) {res["diff_a"]:+.1f} bps, t {res["t_a"]:+.2f}')
        say(f'   (b) momentum-matched non-thrusts ({res["n_ctrl_b"]} group-days): thrust minus (b) {res["diff_b"]:+.1f} bps, '
            f't {res["t_b"]:+.2f}  [regression with decile effects, both sides clustered by date: '
            f'{res["reg_b"]:+.1f} bps, t {res["reg_t_b"]:+.2f}]')
        say(f'   thrusts by momentum decile (1 = lowest): {" ".join(str(x) for x in res["dec_counts"])}; '
            f'control means by decile (bps): {" ".join(f"{x:+.0f}" for x in res["mu_b"])}')
    return res


def build_scores(R, ev, shape):
    """Members of a thrusting group score 1 + their leave-one-out correlation with the group on days d..d+20 (the
    judge buys only positive scores; 1 + r keeps the ranking). Overlaps keep the larger score."""
    N, T = shape
    s = np.full((N, T), np.nan, np.float32)
    for e in ev:
        a, b = e['d'], min(e['d'] + HOLD, T)
        v = (1.0 + e['rho']).astype(np.float32)
        ok = np.isfinite(v)
        rows = e['members'][ok]
        cur = s[rows, a:b]
        s[rows, a:b] = np.fmax(cur, v[ok][:, None])
    return s


def scored_counts(s, days, start):
    a = c2eval.day_index(days, start)
    n = np.isfinite(s[:, a:len(days) - 1]).sum(0)
    return n


def judge_report(path, tag, **kw):
    out = {}
    for rm in ('uniform', 'volatility'):
        txt, js = c2eval.judge(path, rebalance=1, random_match=rm, **kw)
        (HERE / f'judge_{tag}_{rm}.txt').write_text(txt)
        out[rm] = js
    return out


def main(tag):
    with open(big(tag) / 'refits.pkl', 'rb') as fh:
        R = pickle.load(fh)
    days, T, N = R['days'], R['T'], R['N']
    ev, gd = find_thrusts(R)
    d_ev = np.array([e['d'] for e in ev])
    yr_ev = np.array([int(iso(days[d])[:4]) for d in d_ev])
    say(f'# Group breadth thrust, {tag} export ({N} stocks x {T} days, {iso(days[0])}..{iso(days[-1])})')
    say(f'refits: {len(R["refits"])} ({iso(days[R["refits"][0]["t0"]])} .. {iso(days[R["refits"][-1]["t0"]])}); '
        f'group sizes {min(r["sizes"].min() for r in R["refits"])}-{max(r["sizes"].max() for r in R["refits"])}, '
        f'groups under {MIN_MEMBERS} members: {sum(int((r["sizes"] < MIN_MEMBERS).sum()) for r in R["refits"])}')
    say(f'group-days: {len(gd["d"])}; thrust condition true on {int(gd["cond"].sum())} ({gd["cond"].mean():.2%}); '
        f'thrusts {len(ev)} on {len(np.unique(d_ev))} dates')
    # events file
    with open(HERE / f'events_{tag}.tsv', 'w') as fh:
        fh.write('date\trefit\tgroup\tlabel\tmembers\tlow_breadth\tbreadth\tmom10_bps\ty21_bps\tmembers_with_outcome\n')
        for e in ev:
            fh.write(f"{iso(days[e['d']])}\t{e['r']}\t{e['g']}\t{e['label']}\t{e['size']}\t{e['low']:.3f}\t"
                     f"{e['breadth']:.3f}\t{e['mom'] * 1e4:.1f}\t{e['y21'] * 1e4:.1f}\t{e['ny']}\n")

    if tag == 'old':
        # ---------------------------------------------------------------- counts
        say()
        say('## Thrusts per year')
        say('year  thrusts  dates  groups/date(max)  mean y21 bps')
        for y in range(2006, 2026):
            m = yr_ev == y
            if not m.any():
                say(f'{y}  0')
                continue
            ys = np.array([ev[i]['y21'] for i in np.flatnonzero(m)])
            per = Counter(d_ev[m])
            say(f'{y}  {m.sum():5d}  {len(per):5d}  {max(per.values()):3d}  '
                f'{np.nanmean(ys) * 1e4 if np.isfinite(ys).any() else np.nan:+8.1f}')
        n_gp = sum(int((r['sizes'] >= MIN_MEMBERS).sum()) for r in R['refits'])
        with_thr = len({(e['r'], e['g']) for e in ev})
        say(f'group-periods (30 groups x 21 days): {n_gp}; with a thrust {with_thr} ({with_thr / n_gp:.1%}); '
            f'thrusts per group per year {len(ev) / (n_gp * 21 / 252):.2f}')
        per_date = Counter(d_ev)
        dist = Counter(per_date.values())
        say('groups thrusting on the same date: ' + ', '.join(f'{k}: {v} dates' for k, v in sorted(dist.items())))
        say()
        say('## Thrusts per group (label = the most common twin fund among members; groups change each refit)')
        lab_ct = Counter(e['label'] for e in ev)
        gp_ct = Counter(r['glabel'][g] for r in R['refits'] for g in range(K_GROUPS) if r['sizes'][g] >= MIN_MEMBERS)
        say(f'{len(lab_ct)} labels; label  thrusts  group-periods  thrusts per group-period')
        for lb, c in lab_ct.most_common(20):
            say(f'  {lb:6s} {c:5d} {gp_ct[lb]:6d}  {c / gp_ct[lb]:.2f}')

        # ---------------------------------------------------------------- tests
        say()
        say('## Test 1: 2006-2019, after-thrust 21-day market-relative return (pass: >= +50 bps, t >= 3)')
        m1_ev, m1_gd = in_period(days, d_ev, P1), in_period(days, gd['d'], P1)
        r1 = analyse('2006-2019', days, ev, gd, m1_ev, m1_gd)
        pass1 = r1['mean'] >= PASS_BPS and r1['t'] >= PASS_T
        say(f'TEST 1: {"PASS" if pass1 else "FAIL"}')
        say()
        say('## Test 2: beats the baselines, difference t >= 2 (both (b), the plan\'s test, and (a), the TL;DR\'s)')
        pass2 = r1['t_b'] >= PASS_T2 and r1['t_a'] >= PASS_T2
        say(f'thrust - (b) {r1["diff_b"]:+.1f} bps t {r1["t_b"]:+.2f}; thrust - (a) {r1["diff_a"]:+.1f} bps t {r1["t_a"]:+.2f}')
        say(f'TEST 2: {"PASS" if pass2 else "FAIL"}')
        say()
        say('## Test 3: 2020-2025-06, same sign as 2006-2019')
        m2_ev, m2_gd = in_period(days, d_ev, P2), in_period(days, gd['d'], P2)
        r2 = analyse('2020-2025-06', days, ev, gd, m2_ev, m2_gd)
        pass3 = np.sign(r2['mean']) == np.sign(r1['mean'])
        say(f'TEST 3 (2020-2025-06 part): {"PASS" if pass3 else "FAIL"} (sign {np.sign(r1["mean"]):+.0f} then {np.sign(r2["mean"]):+.0f})')
        say()
        say('## Reported only: with and without 2009 and 2020')
        y_gd = np.array([int(iso(days[d])[:4]) for d in np.unique(gd['d'])])
        yr_of = dict(zip(np.unique(gd['d']), y_gd))
        y_gd_all = np.array([yr_of[d] for d in gd['d']])
        both_ev, both_gd = in_period(days, d_ev, (P1[0], P2[1])), in_period(days, gd['d'], (P1[0], P2[1]))
        for nm, mev, mgd in (('2006-2019 without 2009', m1_ev & (yr_ev != 2009), m1_gd & (y_gd_all != 2009)),
                             ('2009 alone', m1_ev & (yr_ev == 2009), m1_gd & (y_gd_all == 2009)),
                             ('2020-2025-06 without 2020', m2_ev & (yr_ev != 2020), m2_gd & (y_gd_all != 2020)),
                             ('2020 alone', m2_ev & (yr_ev == 2020), m2_gd & (y_gd_all == 2020)),
                             ('2006-2025-06 all', both_ev, both_gd),
                             ('2006-2025-06 without 2009 and 2020', both_ev & (yr_ev != 2009) & (yr_ev != 2020),
                              both_gd & (y_gd_all != 2009) & (y_gd_all != 2020))):
            analyse(nm, days, ev, gd, mev, mgd)
        say()
        say('## Reported only: by year (mean after-thrust return, bps; t clustered by date)')
        for y in range(2006, 2026):
            m = yr_ev == y
            ys = np.array([ev[i]['y21'] for i in np.flatnonzero(m)])
            dd = d_ev[m]
            ok = np.isfinite(ys)
            if ok.sum() >= 2:
                mm, se, t, n, nc = clustered_mean(ys[ok], dd[ok])
                say(f'  {y}: {n:4d} thrusts, {nc:3d} dates, {mm * 1e4:+7.1f} bps, t {t:+.2f}')

        # ---------------------------------------------------------------- money test
        say()
        say('## Money test: the judge, rebalance=1, walk-forward 2006-2025 (scores: 1 + member correlation, days d..d+20)')
        s = build_scores(R, ev, (N, T))
        n = scored_counts(s, days, '2006-06-28')
        say(f'decision days from 2006-06-28: {len(n)}; with no scored stock {int((n == 0).sum())}, '
            f'with 1-19 {int(((n > 0) & (n < 20)).sum())} (the account holds cash for the missing picks), '
            f'>= 20 {int((n >= 20).sum())}; median scored {int(np.median(n))}')
        path = c2eval.save_scores(s, 'breadth_thrust/scores_old')
        del s
        js = judge_report(path, 'old')
        for rm, j in js.items():
            say(f'{rm}: {c2eval.summary_line(j)}')
            for ci, c in enumerate((0, 5, 20)):
                cor = c2eval.change_over_random(j, ci)
                say(f'   {c:2d} bps: change over random {cor["change"]:+.2f} pts/yr (se {cor["se"]:.2f}, ahead {cor["ahead"]}/{cor["years"]})')
            cor = c2eval.change_over_random(j, 0)
            say('   per year (0 bps): ' + ' '.join(f'{x:+.1f}' for x in cor['per_year']))
            say('   picks / random median per year (0 bps): ' + '; '.join(
                f"{y['start'][:4]} {y['costs'][0]['final_pct']:+.1f}/{y['costs'][0]['random_median_pct']:+.1f}"
                for y in j['years']))
        num = c2eval.change_over_random(js['uniform'], 0)['change']
        say()
        say(f'FILE NAME NUMBER (uniform, 0 bps): {num:+.1f}')
        verdict = 'test 1' if not pass1 else 'test 2' if not pass2 else 'test 3 (2020-25)' if not pass3 else 'none so far'
        say(f'first failure: {verdict}')
        (HERE / 'results_old.txt').write_text('\n'.join(OUT) + '\n')
    else:
        # ---------------------------------------------------------------- the unseen months, once
        say()
        say('## The unseen months: thrusts dated 2025-06-30 on, with a full 21-day outcome')
        mu_ev, mu_gd = in_period(days, d_ev, UNSEEN), in_period(days, gd['d'], UNSEEN)
        say(f'thrusts dated in the unseen months: {int(mu_ev.sum())} on {len(np.unique(d_ev[mu_ev]))} dates')
        for i in np.flatnonzero(mu_ev):
            e = ev[i]
            say(f"  {iso(days[e['d']])} {e['label']:6s} {e['size']:3d} members, 10-day {e['mom'] * 1e4:+.0f} bps, "
                f"21-day after {e['y21'] * 1e4:+.0f} bps" if np.isfinite(e['y21']) else
                f"  {iso(days[e['d']])} {e['label']:6s} {e['size']:3d} members, 10-day {e['mom'] * 1e4:+.0f} bps, no full outcome")
        analyse('unseen months', days, ev, gd, mu_ev, mu_gd)
        say()
        say('## The judge on the unseen months (holdout from 2025-06-30, rebalance=1)')
        s = build_scores(R, ev, (N, T))
        n = scored_counts(s, days, '2025-06-30')
        say(f'decision days: {len(n)}; with no scored stock {int((n == 0).sum())}, with 1-19 {int(((n > 0) & (n < 20)).sum())}, '
            f'>= 20 {int((n >= 20).sum())}')
        path = c2eval.save_scores(s, 'breadth_thrust/scores_new', export=c2eval.EXPORT_NEW)
        del s
        js = judge_report(path, 'new', export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30')
        for rm, j in js.items():
            st = j['report']['strategy'] if 'report' in j else None
            for ci, c in enumerate((0, 5, 20)):
                cor = c2eval.change_over_random(j, ci)
                e = st[ci] if st else {}
                say(f'{rm} {c:2d} bps: picks {e.get("final_pct", np.nan):+.1f}% vs random median '
                    f'{e.get("random_median_pct", np.nan):+.1f}% (p {e.get("p", np.nan)}), change {cor["change"]:+.2f}')
        (HERE / 'results_new.txt').write_text('\n'.join(OUT) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
