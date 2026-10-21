"""Step 3: the plan's tests, in order, on the refits from pp_refits.py.

    python3 -I pp_tests.py old            -> results_old.txt, events_old.tsv, scores file + judge (walk-forward)
    python3 -I pp_tests.py new            -> results_new.txt: the 16 unseen months (2025-06-30..2026-10-19), once

Test 1 structure (null, stability, overlap with groups, style labels); test 2 known episodes; test 3 13F (blocked,
see the .md); test 4 bounce beyond own residual reversal (2005-2019); test 5 the later years (2020-2025-06 here,
the 16 months in `new`); test 6 money test (the plan's own: 5-day hedged trades vs random guessers from clusters
that moved >= 2 sigma without unwinding) and the judge (scores on t..t+4, long-only top 20 + long-short).
"""
import os
import sys

for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[v] = '8'

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pp_common import (BREADTH, HERE, SPIKE_SHARE, TREND_WIN, VOL_MED, Z_EVENT, adjusted_rand, big, c2eval,  # noqa: E402
                       cluster_ols)

COOL = 5          # a qualifying day within 5 trading days after another of the same type, direction and cluster is the same event
HOLD = 5
EPISODES = [('Aug 2007 quant meltdown', '2007-08-01', '2007-08-14'),
            ('2009 momentum crash', '2009-03-09', '2009-05-29'),
            ('late Jan 2021 de-grossing', '2021-01-20', '2021-02-05'),
            ('summer 2025 quant wobble (MSCI: 2025-06-01..07-25)', '2025-06-02', '2025-07-31')]
CHAR_NAMES = ['mom', 'rev', 'vol', 'beta', 'size', 'price']

OUT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def load(tag):
    files = sorted((big(tag) / 'refits').glob('R*.npz'))
    f0 = np.load(files[0])
    rc = list(f0['row_cols'])
    mc = list(f0['member_cols'])
    info, labels = [], []
    crow, cmem, grow, gmem = [], [], [], []
    oc = og = 0
    for f in files:
        z = np.load(f)
        R = int(z['R'])
        for c in range(len(z['sizes'])):
            info.append([R, c, z['sizes'][c], z['mcorr'][c], z['pval'][c], z['kept'][c], z['purity'][c], z['modal'][c]]
                        + list(z['style'][c]) + list(z['lab_corr'][c]))
        labels.append((R, z['U'], z['clab'], z['glab'], z['kept'], z['null_means']))
        cr, cm, gr, gm = z['c_rows'], z['c_mem'].copy(), z['g_rows'], z['g_mem'].copy()
        cm[:, 0] += oc
        gm[:, 0] += og
        oc += len(cr)
        og += len(gr)
        crow.append(cr)
        cmem.append(cm)
        grow.append(gr)
        gmem.append(gm)
    lab_names = list(f0['lab_names'])
    fz = np.load(big(tag) / 'factors.npz')
    return dict(rc={k: i for i, k in enumerate(rc)}, mc={k: i for i, k in enumerate(mc)},
                info=np.array(info, np.float64), lab_names=lab_names, labels=labels,
                crow=np.concatenate(crow), cmem=np.concatenate(cmem), grow=np.concatenate(grow), gmem=np.concatenate(gmem),
                days=fz['days'], tickers=fz['tickers'], M=fz['M'])


def classify(rows, rc, kept_only=True, zthr=Z_EVENT):
    z, tr, vm = rows[:, rc['z']], rows[:, rc['trend']], rows[:, rc['vmed']]
    dn, up, sp = rows[:, rc['down']], rows[:, rc['up']], rows[:, rc['spike']]
    kept = rows[:, rc['kept']] > 0 if kept_only else np.ones(len(rows), bool)
    with np.errstate(invalid='ignore'):
        unwind = kept & (z <= -zthr) & (tr > 0) & (vm >= VOL_MED) & (dn >= BREADTH) & (sp <= SPIKE_SHARE)
        squeeze = kept & (z >= zthr) & (tr < 0) & (vm >= VOL_MED) & (up >= BREADTH) & (sp <= SPIKE_SHARE)
        big_ = kept & (np.abs(z) >= zthr)
        quiet = big_ & (vm < VOL_MED)
        quiet_strict = quiet & (((z < 0) & (tr > 0) & (dn >= BREADTH)) | ((z > 0) & (tr < 0) & (up >= BREADTH))) & (sp <= SPIKE_SHARE)
        mover2 = kept & (np.abs(z) >= 2.0)
    return dict(unwind=unwind, squeeze=squeeze, event=unwind | squeeze, big=big_, quiet=quiet, quiet_strict=quiet_strict,
                mover2=mover2)


def cooldown(rows, rc, mask):
    """First day of each run: drop a qualifying row if the same refit's cluster had a qualifying row in the same
    direction within the previous COOL trading days."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return mask.copy()
    R, c, d, s = rows[idx, rc['R']], rows[idx, rc['c']], rows[idx, rc['d']], np.sign(rows[idx, rc['z']])
    o = np.lexsort((d, s, c, R))
    keep = np.ones(len(idx), bool)
    for a, b in zip(o[:-1], o[1:]):
        if R[a] == R[b] and c[a] == c[b] and s[a] == s[b] and d[b] - d[a] <= COOL:
            keep[b] = False
    out = np.zeros_like(mask)
    out[idx[keep]] = True
    return out


def month_of(days_idx, days):
    dd = days[days_idx.astype(int)].astype('datetime64[D]')
    return dd.astype('datetime64[M]').astype(int)


def year_of(days_idx, days):
    return days[days_idx.astype(int)].astype('datetime64[D]').astype('datetime64[Y]').astype(int) + 1970


def date_mask(days_idx, days, a, b):
    lo, hi = c2eval.day_index(days, a), c2eval.day_index(days, b)
    di = days_idx.astype(int)
    return (di >= lo) & (di < hi)


def describe_cluster(D, R, c, k=6):
    inf = D['info']
    r = np.flatnonzero((inf[:, 0] == R) & (inf[:, 1] == c))[0]
    row = inf[r]
    style = row[8:14]
    lab = row[14:]
    parts = []
    if np.isfinite(lab).any():
        j = int(np.nanargmax(np.abs(lab)))
        if abs(lab[j]) >= 0.3:
            parts.append(f"{'+' if lab[j] > 0 else '-'}{D['lab_names'][j]} r={lab[j]:+.2f}")
    st = [f"{CHAR_NAMES[i]}{style[i]:+.1f}" for i in np.argsort(-np.abs(style)) if abs(style[i]) >= 0.5][:3]
    if st:
        parts.append('style ' + ' '.join(st))
    parts.append(f'group purity {row[6]:.2f}')
    lab_ = [x for x in D['labels'] if x[0] == R][0]
    U, clab = lab_[1], lab_[2]
    mem = U[clab == c]
    return mem, '; '.join(parts), ' '.join(D['tickers'][mem[:k]])


def test1(D):
    say('\n=== Test 1: is there structure? ===')
    inf = D['info']
    ok8 = inf[:, 2] >= 8
    kept = inf[:, 5] > 0
    nref = len(D['labels'])
    by_r = {}
    for row in inf:
        by_r.setdefault(int(row[0]), []).append(row)
    kept_per = np.array([sum(1 for r in v if r[5] > 0) for v in by_r.values()])
    n8_per = np.array([sum(1 for r in v if r[2] >= 8) for v in by_r.values()])
    q99 = np.array([np.quantile(x[5], 0.99) for x in D['labels']])
    qmed = np.array([np.median(x[5]) for x in D['labels']])
    say(f'refits: {nref} ({c2eval.iso(D["days"][D["labels"][0][0]])} .. {c2eval.iso(D["days"][D["labels"][-1][0]])}), 50 clusters each')
    say(f'clusters with >= 8 members per refit: mean {n8_per.mean():.1f}; kept (beat the shifted-dates null at p < 0.01): '
        f'mean {kept_per.mean():.1f} per refit ({kept.sum() / ok8.sum():.0%} of clusters), min {kept_per.min()}, max {kept_per.max()}')
    say(f'mean pairwise residual correlation: kept clusters {np.nanmean(inf[kept, 3]):.3f} (median {np.nanmedian(inf[kept, 3]):.3f}); '
        f'other clusters {np.nanmean(inf[ok8 & ~kept, 3]):.3f}; null clusters median {qmed.mean():.3f}, null 99th percentile {q99.mean():.3f}')
    sz = inf[kept, 2]
    say(f'kept cluster size: median {np.median(sz):.0f}, 10th-90th pct {np.quantile(sz, 0.1):.0f}-{np.quantile(sz, 0.9):.0f}; '
        f'stocks in kept clusters per refit: {np.mean([sz_.sum() for sz_ in [inf[(inf[:, 0] == R) & kept, 2] for R in by_r]]):.0f} of ~1,500')
    # by year
    yrs = year_of(np.array([x[0] for x in D['labels']]), D['days'])
    say('kept clusters per refit by year: ' + ' '.join(f'{y}:{kept_per[yrs == y].mean():.0f}' for y in np.unique(yrs)))
    # stability
    ari_c, ari_g, ari_k, ari_cg = [], [], [], []
    for a, b in zip(D['labels'][:-1], D['labels'][1:]):
        Ua, Ub = a[1], b[1]
        com, ia, ib = np.intersect1d(Ua, Ub, return_indices=True)
        ari_c.append(adjusted_rand(a[2][ia], b[2][ib]))
        ari_g.append(adjusted_rand(a[3][ia], b[3][ib]))
        ka = a[4][a[2][ia]]
        ari_k.append(adjusted_rand(a[2][ia][ka], b[2][ib][ka]) if ka.sum() > 50 else np.nan)
    for x in D['labels']:
        ari_cg.append(adjusted_rand(x[2], x[3]))
    say(f'stability, one refit to the next (21 days apart), adjusted Rand index on common stocks: phantom clusters {np.mean(ari_c):.3f} '
        f'(members of kept clusters only: {np.nanmean(ari_k):.3f}); the groups map refit on the same schedule {np.mean(ari_g):.3f}')
    say(f'overlap with the groups map: adjusted Rand index phantom vs groups (same refit) {np.mean(ari_cg):.3f}; '
        f'modal-group share of a kept cluster mean {np.mean(inf[kept, 6]):.2f}')
    # chance purity: members drawn at random from the universe's group sizes
    pur_chance = []
    rng = np.random.default_rng(1)
    for x in D['labels'][::10]:
        glab = x[3]
        for c in np.flatnonzero(x[4]):
            n = int((x[2] == c).sum())
            pur_chance.append(np.mean([np.bincount(rng.choice(glab, n, replace=False), minlength=20).max() / n for _ in range(20)]))
    say(f'  (chance modal-group share for random baskets of the same sizes: {np.mean(pur_chance):.2f})')
    # labels
    lab = inf[:, 14:]
    best = np.where(np.isfinite(lab).any(1), np.nanargmax(np.abs(np.where(np.isfinite(lab), lab, 0)), 1), -1)
    bestv = np.array([lab[i, j] if j >= 0 else np.nan for i, j in enumerate(best)])
    labeled = kept & (np.abs(bestv) >= 0.3)
    say(f'kept clusters whose residual moves track a label fund (|r| >= 0.3 over the 126 days, fund taken relative to the market): '
        f'{labeled.sum() / kept.sum():.0%}')
    names = D['lab_names']
    cnt = {}
    for i in np.flatnonzero(labeled):
        key = f"{'+' if bestv[i] > 0 else '-'}{names[best[i]]}"
        cnt[key] = cnt.get(key, 0) + 1
    top = sorted(cnt.items(), key=lambda x: -x[1])[:15]
    say('  most common labels (share of kept clusters): ' + ', '.join(f'{k} {v / kept.sum():.1%}' for k, v in top))
    st = inf[kept, 8:14]
    strong = np.abs(st) >= 0.75
    say('kept clusters with a strong price-style tilt (|mean z| >= 0.75 of the universe): ' +
        ', '.join(f'{CHAR_NAMES[i]} {strong[:, i].mean():.0%}' for i in range(6)) +
        f'; any {strong.any(1).mean():.0%}')
    style_labels = [('value (IWD-IWF)', 'IWD-IWF'), ('small value (IWN-IWO)', 'IWN-IWO'), ('momentum (MTUM-SPY)', 'MTUM-SPY'),
                    ('quality (QUAL-SPY)', 'QUAL-SPY'), ('min vol (USMV-SPY)', 'USMV-SPY'), ('value (VLUE-SPY)', 'VLUE-SPY')]
    say('  share of kept clusters tracking a style fund at |r| >= 0.3: ' + ', '.join(
        f"{lbl} {(kept & (np.abs(lab[:, names.index(k)]) >= 0.3)).sum() / max((kept & np.isfinite(lab[:, names.index(k)])).sum(), 1):.0%}"
        for lbl, k in style_labels))
    fail = kept_per.mean() < 5 or np.mean(ari_c) < 0.05
    say(f'Test 1 kill rule (fixed before results: fail if < 5 kept clusters per refit on average, or mean ARI < 0.05): '
        f'{"FAIL" if fail else "pass"}')
    return not fail


def events_table(D, cls, ev, fn):
    rows, rc = D['crow'], D['rc']
    idx = np.flatnonzero(ev)
    with open(fn, 'w') as f:
        f.write('date\ttype\tz\tmove_bps\ttrend21_bps\tvol_median_x\tfell_or_rose\tmembers\tmarket_bps\tdescription\ttop_members\n')
        for i in idx:
            r = rows[i]
            R, c = int(r[rc['R']]), int(r[rc['c']])
            _, desc, tk = describe_cluster(D, R, c)
            typ = 'unwind' if r[rc['z']] < 0 else 'squeeze'
            br = r[rc['down']] if r[rc['z']] < 0 else r[rc['up']]
            f.write(f"{c2eval.iso(D['days'][int(r[rc['d']])])}\t{typ}\t{r[rc['z']]:+.2f}\t{1e4 * r[rc['X']]:+.0f}\t{1e4 * r[rc['trend']]:+.0f}\t"
                    f"{r[rc['vmed']]:.2f}\t{br:.0%}\t{int(r[rc['n']])}\t{1e4 * r[rc['M']]:+.0f}\t{desc}\t{tk}\n")


def test2(D, cls, ev, episodes):
    say('\n=== Test 2: do they light up when they should? ===')
    rows, rc, days = D['crow'], D['rc'], D['days']
    d = rows[:, rc['d']]
    kept = rows[:, rc['kept']] > 0
    first_d, last_d = int(d.min()), int(d.max())
    ndays = len(np.unique(d))
    rate = ev.sum() / ndays
    rate_big = (cls['big'] & cooldown(rows, rc, cls['big'])).sum() / ndays
    say(f'unwind or squeeze events (after the 5-day cool-down): {ev.sum()} on {ndays} days with clusters '
        f'({c2eval.iso(days[first_d])}..{c2eval.iso(days[last_d])}): {rate * 252:.1f} a year '
        f'({(ev & (rows[:, rc["z"]] < 0)).sum()} unwinds, {(ev & (rows[:, rc["z"]] > 0)).sum()} squeezes); '
        f'any |z| >= 3 move of a kept cluster: {rate_big * 252:.1f} a year')
    yr = year_of(rows[ev, rc['d']], days)
    say('  events by year: ' + ' '.join(f'{y}:{n}' for y, n in zip(*np.unique(yr, return_counts=True))))
    mo = days[rows[ev, rc['d']].astype(int)].astype('datetime64[D]').astype('datetime64[M]')
    um, cm = np.unique(mo, return_counts=True)
    say('  months with 3 or more events: ' + ', '.join(f'{m}: {n}' for m, n in zip(um, cm) if n >= 3))
    for name, a, b in episodes:
        lo, hi = c2eval.day_index(days, a), c2eval.day_index(days, b)
        if lo >= len(days) or hi <= first_d or lo > last_d:
            say(f'- {name}: not in this export')
            continue
        w = (d >= lo) & (d < hi)
        n_days = hi - lo
        e_in = ev & w
        b_in = cls['big'] & w
        say(f'- {name} ({a}..{b}, {n_days} trading days): {e_in.sum()} unwind/squeeze events (expected at the average rate '
            f'{rate * n_days:.1f}); {b_in.sum()} cluster-days with |z| >= 3 (expected {rate_big * n_days:.1f}); '
            f'kept-cluster days with |z| >= 2: {(cls["mover2"] & w).sum()}')
        show = np.flatnonzero(b_in)
        for i in show[np.argsort(rows[show, rc['d']])][:12]:
            r = rows[i]
            _, desc, tk = describe_cluster(D, int(r[rc['R']]), int(r[rc['c']]), 5)
            tag = 'UNWIND' if e_in[i] and r[rc['z']] < 0 else 'SQUEEZE' if e_in[i] else 'big move'
            why = []
            if not e_in[i]:
                if r[rc['vmed']] < VOL_MED:
                    why.append(f"volume {r[rc['vmed']]:.2f}x")
                if np.sign(r[rc['trend']]) == np.sign(r[rc['z']]):
                    why.append('with its trend')
                br = r[rc['down']] if r[rc['z']] < 0 else r[rc['up']]
                if br < BREADTH:
                    why.append(f'breadth {br:.0%}')
                if r[rc['spike']] > SPIKE_SHARE:
                    why.append('one member with news')
            say(f"    {c2eval.iso(days[int(r[rc['d']])])} {tag:8s} z {r[rc['z']]:+.1f} move {1e4 * r[rc['X']]:+.0f} bps, vol {r[rc['vmed']]:.2f}x, "
                f"{int(r[rc['n']])} members [{tk}] {desc}" + (f" (not an event: {', '.join(why)})" if why else ''))
        # the same window with the looser 2-sigma threshold
        cl2 = classify(rows, rc, zthr=2.0)
        e2 = cooldown(rows, rc, cl2['event']) & w
        say(f'    at the 2-sigma sensitivity threshold: {e2.sum()} events in the window '
            f'(expected {cooldown(rows, rc, cl2["event"]).sum() / ndays * n_days:.1f})')


def member_frame(D, keys_mask, which='c'):
    rows = D['crow'] if which == 'c' else D['grow']
    mem = D['cmem'] if which == 'c' else D['gmem']
    mc, rc = D['mc'], D['rc']
    k = mem[:, mc['key']].astype(int)
    sel = keys_mask[k]
    m = mem[sel]
    r = rows[k[sel]]
    sg = np.sign(r[:, rc['X']])
    out = dict(key=k[sel], d=r[:, rc['d']], R=r[:, rc['R']], z=r[:, rc['z']], X=r[:, rc['X']], sg=sg, j=m[:, mc['j']].astype(int),
               C=sg * m[:, mc['loo']], O=sg * m[:, mc['own']], Op=sg * m[:, mc['p0']])
    for h in (1, 5, 10, 20):
        out[f'B{h}'] = -sg * m[:, mc[f'e{h}']]
        out[f'Bo{h}'] = -sg * m[:, mc[f'eo{h}']]
        out[f'Bp{h}'] = -sg * m[:, mc[f'p{h}']]
    out['h5'] = m[:, mc['h5']]
    out['ho5'] = m[:, mc['ho5']]
    out['raw5'] = m[:, mc['raw5']]
    out['hm'] = m[:, mc['hedge_m']]
    out['hg'] = m[:, mc['hedge_g']]
    out['dir'] = -sg  # trade direction: +1 buy (after a fall), -1 sell short (after a jump)
    return out


def bounce_regression(frames, h, days, label, use_open=False, group_units=False):
    """B_h = b_k * C + g_k * O per type, no intercept, SE clustered by calendar month of the event day.
    frames: list of (name, frame). Returns dict name -> (b, se, n_members, n_events), and the difference rows."""
    Xs, ys, gs, names = [], [], [], []
    K = len(frames)
    for k, (name, f) in enumerate(frames):
        if name.startswith('G') or group_units:
            B = f[f'Bp{h}']
            O = f['Op']
        else:
            B = f[f'Bo{h}'] if use_open else f[f'B{h}']
            O = f['O']
        C = f['C']
        ok = np.isfinite(B) & np.isfinite(C) & np.isfinite(O)
        X = np.zeros((ok.sum(), 2 * K))
        X[:, k] = C[ok]
        X[:, K + k] = O[ok]
        Xs.append(X)
        ys.append(B[ok])
        gs.append(month_of(f['d'][ok], days))
        names.append((name, ok.sum(), len(np.unique(f['key'][ok]))))
    X, y, g = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)
    if len(y) < 10:
        return None
    b, se, cov = cluster_ols(y, X, g)
    res = {}
    for k, (name, n, ne) in enumerate(names):
        res[name] = (b[k], se[k], n, ne, b[K + k], se[K + k], float(np.mean(ys[k])) if len(ys[k]) else np.nan)
    diffs = {}
    for k in range(1, K):
        dvec = np.zeros(2 * K)
        dvec[0], dvec[k] = 1, -1
        dv = dvec @ b
        dse = np.sqrt(dvec @ cov @ dvec)
        diffs[f'{names[0][0]}-{names[k][0]}'] = (dv, dse)
    return res, diffs


def print_regression(title, out):
    if out is None:
        say(f'{title}: too few rows')
        return
    res, diffs = out
    say(title)
    say('    type      bounce fraction b (t)    own-move coef g (t)   mean bounce bps   members  events')
    for name, (b, se, n, ne, g, gse, mb) in res.items():
        say(f'    {name:8s}  {b:+.3f} ({b / se:+.2f})          {g:+.3f} ({g / gse:+.2f})        {1e4 * mb:+7.1f}        {n:6d}  {ne:5d}')
    for k, (dv, dse) in diffs.items():
        say(f'    {k}: {dv:+.3f} (t {dv / dse:+.2f})')


def event_level(f, h, days, key='B'):
    """Basket-level: each event's member-mean bounce regressed on its |move| (no intercept), SE by month."""
    keys = np.unique(f['key'])
    B, C, d = [], [], []
    for k in keys:
        s = (f['key'] == k) & np.isfinite(f[f'{key}{h}'])
        if s.sum() == 0:
            continue
        B.append(f[f'{key}{h}'][s].mean())
        C.append(abs(f['X'][s][0]))
        d.append(f['d'][s][0])
    if len(B) < 5:
        return np.nan, np.nan, len(B), np.nan
    B, C, d = np.array(B), np.array(C), np.array(d)
    b, se, _ = cluster_ols(B, C[:, None], month_of(d, days))
    return b[0], se[0], len(B), B.mean()


def bounce_tests(D, cls, period, title, kill_t, fomc=None):
    say(f'\n=== {title} ===')
    rows, rc, days = D['crow'], D['rc'], D['days']
    grows = D['grow']
    a, b = period
    pm = date_mask(rows[:, rc['d']], days, a, b)
    gpm = date_mask(grows[:, rc['d']], days, a, b)
    U = cooldown(rows, rc, cls['event']) & pm
    N = cooldown(rows, rc, cls['quiet']) & pm
    Ns = cooldown(rows, rc, cls['quiet_strict']) & pm
    gcls = classify(grows, rc, kept_only=False)
    Gm = cooldown(grows, rc, gcls['big']) & gpm
    Gs = cooldown(grows, rc, gcls['event']) & gpm
    fU, fN, fNs = member_frame(D, U), member_frame(D, N), member_frame(D, Ns)
    fG, fGs = member_frame(D, Gm, 'g'), member_frame(D, Gs, 'g')
    say(f'period {a}..{b}: events U (unwinds+squeezes) {U.sum()} ({(U & (rows[:, rc["z"]] < 0)).sum()} unwinds), '
        f'N (same-size cluster moves, volume < {VOL_MED}x) {N.sum()}, N-strict (all other unwind rules hold) {Ns.sum()}, '
        f'G (same-size group moves) {Gm.sum()}, G-strict (group days meeting every unwind rule) {Gs.sum()}')
    say('Bounce = the member\'s residual over the next h days, signed so that a reversal is positive; C = the basket\'s '
        'event-day move without the member (Wardlaw: the member\'s own event-day return is not in it), in the event\'s '
        'direction; O = the member\'s own event-day residual (its own residual reversal, Da-Liu-Schaumburg). '
        'b = fraction of the basket\'s move that comes back beyond the member\'s own reversal. t from SEs clustered by month.')
    results = {}
    for h in (1, 5, 10, 20):
        out = bounce_regression([('U', fU), ('N', fN), ('G', fG)], h, days, 'primary')
        results[h] = out
        print_regression(f'  h = {h} days (primary: U vs N vs G; G in returns before the group is removed)', out)
    out5 = results[5]
    say('  strict controls, h = 5:')
    print_regression('    U vs N-strict vs G-strict', bounce_regression([('U', fU), ('N-strict', fNs), ('G-strict', fGs)], 5, days, 's'))
    say('  U and N in the same units as G (returns before the group is removed), h = 5:')
    print_regression('    ', bounce_regression([('U', fU), ('N', fN), ('G', fG)], 5, days, 'p', group_units=True))
    if a < '2008-01-01':
        a2 = '2008-01-01'
    else:
        a2 = a
    U8 = U & date_mask(rows[:, rc['d']], days, a2, b)
    N8 = N & date_mask(rows[:, rc['d']], days, a2, b)
    fU8, fN8 = member_frame(D, U8), member_frame(D, N8)
    for h in (1, 5):
        r_close = bounce_regression([('U', fU8), ('N', fN8)], h, days, 'c8')
        r_open = bounce_regression([('U', fU8), ('N', fN8)], h, days, 'o8', use_open=True)
        print_regression(f'  from {a2}, h = {h}, from the event close', r_close)
        print_regression(f'  from {a2}, h = {h}, from the next open', r_open)
    for h in (5,):
        for nm, f in (('U', fU), ('N', fN), ('G', fG)):
            bb, se, n, mb = event_level(f, h, days, 'Bp' if nm == 'G' else 'B')
            say(f'  basket level, h = {h}, {nm}: bounce per unit of the basket move {bb:+.3f} (t {bb / se:+.2f}), '
                f'{n} events, mean basket bounce {1e4 * mb:+.1f} bps')
    # same-day market
    ev_rows = rows[U]
    say(f'  same-day market on U events: mean {1e4 * np.nanmean(ev_rows[:, rc["M"]]):+.0f} bps, '
        f'|market| > 2%: {np.mean(np.abs(ev_rows[:, rc["M"]]) > 0.02):.0%} of events')
    # post-hoc (after seeing that many events fall on huge market days): split U by the same-day market move
    Mabs = np.abs(rows[:, rc['M']])
    for lbl, mm in (('|market| <= 2%', Mabs <= 0.02), ('|market| > 2%', Mabs > 0.02)):
        Ux = U & mm
        if Ux.sum() >= 3:
            print_regression(f'  post-hoc split, U events with same-day {lbl} ({Ux.sum()} events) vs all N and G, h = 5',
                             bounce_regression([('U', member_frame(D, Ux)), ('N', fN), ('G', fG)], 5, days, 'm'))
    if fomc is not None and U.sum():
        dd = rows[:, rc['d']].astype(int)
        on = np.isin(dd, fomc)
        Ux = U & ~on
        Nx = N & ~on
        Gx = Gm & ~np.isin(grows[:, rc['d']].astype(int), fomc)
        say(f'  robustness, FOMC and CPI days dropped ({(U & on).sum()} of {U.sum()} U events were on one):')
        print_regression('    h = 5', bounce_regression([('U', member_frame(D, Ux)), ('N', member_frame(D, Nx)), ('G', member_frame(D, Gx, 'g'))], 5, days, 'x'))
    # sensitivity: 2-sigma threshold (decided after seeing that out-of-sample cluster z has sd ~0.6, before outcomes)
    cl2 = classify(rows, rc, zthr=2.0)
    U2 = cooldown(rows, rc, cl2['event']) & pm
    say(f'  sensitivity, events at 2 sigma instead of 3 ({U2.sum()} events), h = 5:')
    N2 = cooldown(rows, rc, cl2['big'] & (rows[:, rc['vmed']] < VOL_MED)) & pm
    print_regression('    U2 vs N2 vs G', bounce_regression([('U2', member_frame(D, U2)), ('N2', member_frame(D, N2)), ('G', fG)], 5, days, 's2'))
    # kill rule
    if out5 is None:
        say(f'kill rule: too few events -> FAIL')
        return False, results
    res, diffs = out5
    tU = res['U'][0] / res['U'][1]
    tN = diffs['U-N'][0] / diffs['U-N'][1]
    tG = diffs['U-G'][0] / diffs['U-G'][1]
    ok = (diffs['U-N'][0] > 0) and (diffs['U-G'][0] > 0) and tN >= kill_t and tG >= kill_t
    say(f'kill rule (h = 5; fixed before results: U\'s bounce fraction exceeds N\'s and G\'s, each difference with t >= {kill_t}): '
        f'b_U {res["U"][0]:+.3f} (t {tU:+.2f}); U-N t {tN:+.2f}; U-G t {tG:+.2f} -> {"pass" if ok else "FAIL"}')
    return ok, results


def wf_years(days, start='2006-06-28', end=None):
    """The judge's walk-forward years: [Y-06-28, Y+1-06-28)."""
    out = []
    y = int(start[:4])
    last = c2eval.iso(days[-1])
    while True:
        a, b = f'{y}-{start[5:]}', f'{y + 1}-{start[5:]}'
        if a > last or (end and a >= end):
            break
        out.append((a, b))
        y += 1
    return out


def money_test(D, cls, periods, title, runs=1000, costs=(0, 5, 10)):
    say(f'\n=== {title} ===')
    rows, rc, days = D['crow'], D['rc'], D['days']
    ev = cooldown(rows, rc, cls['event'])
    lo, hi = periods[0][0], periods[-1][1]
    ev &= date_mask(rows[:, rc['d']], days, lo, hi)
    f = member_frame(D, ev)
    pnl = f['dir'] * f['h5']
    ok = np.isfinite(pnl)
    pool_mask = cls['mover2'] & ~cls['event'] & date_mask(rows[:, rc['d']], days, lo, hi)
    fp = member_frame(D, pool_mask)
    ppnl = fp['dir'] * fp['h5']
    pok = np.isfinite(ppnl)
    pd = fp['d'][pok].astype(int)
    ppnl = ppnl[pok]
    tdays = f['d'][ok].astype(int)
    tp = pnl[ok]
    hedge_notional = (np.abs(f['hm']) + np.abs(f['hg']))[ok]
    say(f'trades: {ok.sum()} stock trades from {len(np.unique(f["key"][ok]))} events on {len(np.unique(tdays))} days; '
        f'random pool: {len(ppnl)} member-days of kept clusters that moved >= 2 sigma that day without unwinding/squeezing')
    # random guessers: per event day, the same number of stocks from that day's pool (nearest day within 5 if empty)
    pool_days = np.unique(pd)
    by_day = {d: ppnl[pd == d] for d in pool_days}
    udays, cnts = np.unique(tdays, return_counts=True)
    rng = np.random.default_rng(2026)
    rand_sum = np.zeros((runs, len(periods)))
    rand_n = np.zeros(len(periods))
    fallback = 0
    yidx = np.zeros(len(udays), int)
    for k, (a, b) in enumerate(periods):
        yidx[date_mask(udays, days, a, b)] = k
    for d, n, yk in zip(udays, cnts, yidx):
        p = by_day.get(d)
        if p is None or len(p) == 0:
            near = pool_days[np.argsort(np.abs(pool_days - d))[:1]]
            if len(near) and abs(near[0] - d) <= 5:
                p = by_day[near[0]]
                fallback += 1
            else:
                continue
        draw = rng.integers(0, len(p), size=(runs, n)) if len(p) < n else np.argsort(rng.random((runs, len(p))), 1)[:, :n]
        rand_sum[:, yk] += p[draw].sum(1)
        rand_n[yk] += n
    say(f'  event days whose pool was empty and took the nearest day within 5: {fallback}')
    ty = np.zeros(len(tdays), int)
    for k, (a, b) in enumerate(periods):
        ty[date_mask(tdays, days, a, b)] = k
    say('  per year, mean 5-day hedged return per stock trade (bps; hedged with the group and the market):')
    say('    year         trades   gross   after 5 bps/side  after 10   random median   p(random >= picks)')
    pos5 = 0
    yr_rows = []
    for k, (a, b) in enumerate(periods):
        s = ty == k
        n = s.sum()
        if n == 0:
            say(f'    {a[:4]}-{b[2:4]}        0       -          -             -          -')
            yr_rows.append((a, 0, np.nan, np.nan))
            continue
        m = tp[s].mean()
        rm = rand_sum[:, k] / max(rand_n[k], 1)
        pval = (rm >= m).mean() if rand_n[k] else np.nan
        pos5 += (m - 10e-4) > 0
        say(f'    {a[:4]}-{b[2:4]}   {n:6d}  {1e4 * m:+7.1f}   {1e4 * (m - 10e-4):+7.1f}        {1e4 * (m - 20e-4):+7.1f}      '
            f'{1e4 * np.median(rm) if rand_n[k] else np.nan:+7.1f}        {pval:.3f}')
        yr_rows.append((a, n, m, pval))
    allm = tp.mean() if len(tp) else np.nan
    rall = rand_sum.sum(1) / max(rand_n.sum(), 1)
    p_all = (rall >= allm).mean()
    se = tp.std(ddof=1) / np.sqrt(len(tp)) if len(tp) > 1 else np.nan
    # event-clustered SE (members of one event move together)
    ek = f['key'][ok]
    em = np.array([tp[ek == k].mean() for k in np.unique(ek)])
    se_ev = em.std(ddof=1) / np.sqrt(len(em)) if len(em) > 1 else np.nan
    say(f'  all years: {len(tp)} trades, mean {1e4 * allm:+.1f} bps gross (se {1e4 * se:.1f} by trade, {1e4 * se_ev:.1f} by event; '
        f'event-weighted mean {1e4 * em.mean():+.1f}), after 5 bps per side {1e4 * (allm - 10e-4):+.1f}, after 10 bps per side '
        f'{1e4 * (allm - 20e-4):+.1f}; random guessers median {1e4 * np.median(rall):+.1f}, p(random >= picks) = {p_all:.3f}')
    hc = 10e-4 * (1 + hedge_notional)
    say(f'  charging 5 bps per side on the hedge legs too (mean hedge notional {hedge_notional.mean():.2f}): '
        f'after costs {1e4 * (tp - hc).mean():+.1f} bps per trade')
    unw = f['dir'][ok] > 0
    say(f'  unwinds (bought) {unw.sum()} trades mean {1e4 * tp[unw].mean() if unw.any() else np.nan:+.1f} bps; '
        f'squeezes (shorted) {(~unw).sum()} trades mean {1e4 * tp[~unw].mean() if (~unw).any() else np.nan:+.1f} bps')
    o8 = np.isfinite(f['ho5'][ok]) & (days[tdays] >= (np.datetime64('2008-01-01') - np.datetime64('1970-01-01')).astype(int))
    if o8.any():
        po = (f['dir'] * f['ho5'])[ok][o8]
        say(f'  from 2008, entered at the next open instead of the event close: {o8.sum()} trades, mean {1e4 * po.mean():+.1f} bps '
            f'(same trades from the close: {1e4 * tp[o8].mean():+.1f})')
    # post-hoc: random guessers that draw whole clusters (the picks hold whole clusters, so their noise is
    # event-level; stock-level draws across many clusters understate it)
    pk = fp['key'][pok]
    ev_keys = np.unique(f['key'][ok])
    ev_day = np.array([int(f['d'][ok][f['key'][ok] == k][0]) for k in ev_keys])
    pool_keys_by_day = {}
    for k in np.unique(pk):
        dk = int(pd[pk == k][0])
        pool_keys_by_day.setdefault(dk, []).append(k)
    sums = {k: ppnl[pk == k].sum() for k in np.unique(pk)}
    cnts_k = {k: (pk == k).sum() for k in np.unique(pk)}
    rs = np.zeros(runs)
    rn = np.zeros(runs)
    rng2 = np.random.default_rng(7)
    for dk in np.unique(ev_day):
        need = int((ev_day == dk).sum())
        cand = pool_keys_by_day.get(dk)
        if not cand:
            pdays = np.array(sorted(pool_keys_by_day))
            near = pdays[np.argmin(np.abs(pdays - dk))]
            if abs(near - dk) > 5:
                continue
            cand = pool_keys_by_day[near]
        cand = np.array(cand)
        draw = cand[rng2.integers(0, len(cand), size=(runs, need))]
        rs += np.vectorize(sums.get)(draw).sum(1)
        rn += np.vectorize(cnts_k.get)(draw).sum(1)
    rcl = rs / np.maximum(rn, 1)
    say(f'  post-hoc, random guessers drawing whole clusters (same number of baskets on the same days): median '
        f'{1e4 * np.median(rcl):+.1f} bps, 5th-95th pct {1e4 * np.quantile(rcl, 0.05):+.1f}..{1e4 * np.quantile(rcl, 0.95):+.1f}, '
        f'p(random >= picks) = {(rcl >= allm).mean():.3f}')
    years_with = sum(1 for r in yr_rows if r[1] > 0)
    passed = (p_all < 0.05) and (allm - 10e-4 > 0) and pos5 >= np.ceil(2 / 3 * len(periods))
    say(f'pass rule (fixed: beats random guessers at p < 0.05 after 5 bps per side, positive after costs in >= 2/3 of years): '
        f'p = {p_all:.3f}; positive after costs in {pos5} of {len(periods)} years ({years_with} years had trades) -> '
        f'{"PASS" if passed else "FAIL"}')
    return dict(p=p_all, mean=allm, pos=pos5, years=len(periods), trades=len(tp), passed=passed)


def judge_scores(D, cls, export, name, start=None):
    rows, rc, days = D['crow'], D['rc'], D['days']
    ev = cooldown(rows, rc, cls['event'])
    f = member_frame(D, ev)
    E = c2eval.load_export(export, arrays=())
    n, t = len(E['tickers']), len(E['days'])
    assert t == len(days)
    sc = np.full((n, t), np.nan, np.float32)
    val = -np.sign(f['X']) * np.abs(f['z'])   # + |z| after an unwind, - |z| after a squeeze
    for j, d, v in zip(f['j'], f['d'].astype(int), val):
        for dd in range(d, min(d + HOLD, t)):
            sc[j, dd] = v if not np.isfinite(sc[j, dd]) else sc[j, dd] + v
    path = c2eval.save_scores(sc, f'phantom_portfolios/{name}', export=export)
    scored_days = np.isfinite(sc).any(0)
    both = (np.nan_to_num(sc, nan=0) > 0).any(0) & (np.nan_to_num(sc, nan=0) < 0).any(0)
    for a, b in (wf_years(days) if start is None else [(start, '2100-01-01')]):
        m = date_mask(np.arange(t), days, a, b)
        say(f'    {a}..{b}: days with >= 2 stocks scored (long-short days) {int(((np.isfinite(sc).sum(0) >= 2) & m).sum())}, '
            f'with both sides {int((both & m).sum())}')
    say(f'scores file {path}: {np.isfinite(sc).sum()} stock-days scored on {scored_days.sum()} days; '
        f'days with both an unwind and a squeeze side: {both.sum()}; max stocks scored on a day {np.isfinite(sc).sum(0).max()}')
    return path


def main(tag):
    D = load(tag)
    days = D['days']
    rows, rc = D['crow'], D['rc']
    cls = classify(rows, rc)
    fomc = None
    cal = HERE / 'fomc_cpi_days.tsv'
    if cal.exists():
        ds = [l.split('\t')[0] for l in open(cal)][1:]
        fomc = np.array([c2eval.day_index(days, x) for x in ds if c2eval.day_index(days, x) < len(days)
                         and c2eval.iso(days[c2eval.day_index(days, x)]) == x])
    if tag == 'old':
        export = c2eval.EXPORT_OLD
        say(f'# Phantom portfolios on the old export (through {c2eval.iso(days[-1])})')
        t1 = test1(D)
        ev = cooldown(rows, rc, cls['event'])
        events_table(D, cls, ev, HERE / 'events_old.tsv')
        test2(D, cls, ev, EPISODES[:3])
        say('\n=== Test 3: ownership check (13F) ===\nskipped: sec.gov answered 403 to a request without a contact email '
            'in the User-Agent (see the .md)')
        t4, _ = bounce_tests(D, cls, ('2005-01-01', '2020-01-01'), 'Test 4: bounce beyond own residual reversal, 2005-2019', 3.0, fomc)
        t5, _ = bounce_tests(D, cls, ('2020-01-01', '2025-06-28'), 'Test 5 (part 1): the later years 2020-01..2025-06, same rules', 2.0, fomc)
        mt = money_test(D, cls, wf_years(days), 'Test 6: money test (the plan\'s rule), walk-forward years 2006-07..2024-25')
        say('\n=== Judge (scores on t..t+4, rebalance 1, walk-forward) ===')
        path = judge_scores(D, cls, export, 'scores_old')
        if '--no-judge' not in sys.argv:
            txt, js = c2eval.judge(path, rebalance=1)
            (HERE / 'judge_old.txt').write_text(txt)
            say(c2eval.summary_line(js))
            say(f"judge skipped years: {js.get('skipped')}")
            degenerate = [y['start'] for y in js['years'] if abs(y['ls_t']) > 1e3]
            say(f"years with a degenerate long-short t (|t| > 1000, too few long-short days): {degenerate}")
            cor = c2eval.change_over_random(js)
            say(f"change over random {cor['change']:+.2f} pts/yr, se {cor['se']:.2f}, ahead {cor['ahead']}/{cor['years']}; "
                f"per year {cor['per_year']}")
            for y in js['years']:
                say(f"  {y['start']}..{y['end']}: picks {y['costs'][0]['final_pct']:+.1f}% vs random median "
                    f"{y['costs'][0]['random_median_pct']:+.1f}% (p {y['costs'][0].get('p', y['costs'][0].get('share', float('nan')))}), "
                    f"long-short {y['ls_bps_per_day']:+.2f} bps/day (t {y['ls_t']:+.2f})")
        (HERE / 'results_old.txt').write_text('\n'.join(OUT) + '\n')
    else:
        export = c2eval.EXPORT_NEW
        say(f'# Phantom portfolios: the 16 unseen months on the new export (through {c2eval.iso(days[-1])}), run once')
        t1 = test1(D)
        ev = cooldown(rows, rc, cls['event'])
        events_table(D, cls, ev, HERE / 'events_new.tsv')
        test2(D, cls, ev, EPISODES)
        bounce_tests(D, cls, ('2025-06-30', '2026-10-20'), 'Test 5 (part 2): the 16 unseen months 2025-06-30..2026-10-19', 2.0, fomc)
        bounce_tests(D, cls, ('2020-01-01', '2026-10-20'), 'Test 5 (combined): 2020-01..2026-10 on the new export', 2.0, fomc)
        money_test(D, cls, [('2025-06-30', '2026-10-20')], 'Money test (the plan\'s rule), the 16 unseen months')
        say('\n=== Judge on the 16 unseen months (start 2025-06-30, no walk-forward) ===')
        path = judge_scores(D, cls, export, 'scores_new', start='2025-06-30')
        if '--no-judge' not in sys.argv:
            try:
                txt, js = c2eval.judge(path, export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30', rebalance=1)
                (HERE / 'judge_new.txt').write_text(txt)
                cor = c2eval.change_over_random({'costs': js['report']['strategy']})  # single holdout: report.strategy
                say(txt)
                say(f"change over random {cor['change']:+.2f} pts")
            except RuntimeError as err:
                say(f'judge did not grade the 16 months: {err}')
        (HERE / 'results_new.txt').write_text('\n'.join(OUT) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
