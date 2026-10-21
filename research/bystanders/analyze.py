"""Bystanders: tests 1-4 of experiments/-7.5-bystanders.md.

    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python3 -I analyze.py base old
    ... analyze.py replicate old
    ... analyze.py giveback old 2005-01-01 2019-12-31
    ... analyze.py giveback new 2025-06-30 2026-10-19      (the unseen months, run once)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

iso = common.c2eval.iso
OUT = Path(__file__).resolve().parent / 'out'
OUT.mkdir(exist_ok=True)
SEED = 20261021


def load_events(tag):
    z = np.load(common.BIG / f'events_{tag}.npz')
    return {k: z[k] for k in z.files}


def years_of(days, d):
    return np.array([int(iso(days[k])[:4]) for k in d]) if len(d) else np.array([], int)


def collapse(ev, mask):
    """One row per (B, day): the co-mover row of the most correlated A. Returns index array into ev."""
    idx = np.where(mask)[0]
    o = np.lexsort((-ev['corr'][idx], ev['d'][idx], ev['B'][idx]))
    idx = idx[o]
    key = ev['B'][idx].astype(np.int64) * 100000 + ev['d'][idx]
    first = np.r_[True, key[1:] != key[:-1]]
    return idx[first]


def classify(ev):
    ok = np.isfinite(ev['xB']) & np.isfinite(ev['vrB']) & np.isfinite(ev['sdB'])
    byst = ok & (ev['vrB'] <= 1.5)
    part = ok & (ev['vrB'] >= 2.0)
    insym = ok & (np.sign(ev['xB']) == np.sign(ev['xA'])) & (np.abs(ev['xB']) >= 1.5 * ev['sdB'])
    return ok, byst, part, insym


# ---------------------------------------------------------------- test 1
def base(tag):
    ev = load_events(tag)
    days = ev['days']
    ok, byst, part, insym = classify(ev)
    keep = ~ev['expiry']
    lines = []
    P = lines.append
    nd_solo = ev['news_solo']
    yrs_news = years_of(days, ev['news_d'])
    P('## Test 1: base rate (' + tag + ' export)\n')
    P('News days (vol >= 4x 60-day median, |move| >= 3 sd, first in 5 days) and solo news days, by year; '
      'co-movers per solo news day; bystander events per month.\n')
    P('| Year | News days | Solo | Solo with >= 1 co-mover | Co-movers per solo day | Bystanders / month | '
      'of which moved with A >= 1.5 sd | Participants / month |')
    P('|---|---|---|---|---|---|---|---|')
    bi = collapse(ev, byst & keep)
    bi_s = collapse(ev, byst & keep & insym)
    pi = collapse(ev, part & keep)
    yb = years_of(days, ev['d'][bi]); ybs = years_of(days, ev['d'][bi_s]); yp = years_of(days, ev['d'][pi])
    first_year = int(iso(days[ev['d'].min()])[:4])
    last = iso(days[-1])
    for y in range(first_year, int(last[:4]) + 1):
        months = 12 if y < int(last[:4]) else int(last[5:7]) - (0 if int(last[8:10]) > 20 else 1) or 1
        sel = yrs_news == y
        nco = ev['news_nco'][sel & nd_solo]
        P(f'| {y} | {sel.sum():,} | {(sel & nd_solo).sum():,} | {(nco > 0).sum():,} | {nco.mean():.1f} | '
          f'{(yb == y).sum() / months:.0f} | {(ybs == y).sum() / months:.0f} | {(yp == y).sum() / months:.0f} |')
    post = yb >= 2010
    P('')
    P(f'- Bystander events (unique stock-days, expiry days dropped): {len(bi):,}; after 2010: '
      f'{post.sum() / max(1, len(set(zip(yb[post], (ev["d"][bi][post]))))) * 0 + post.sum():,}.')
    # monthly minimum after 2010
    mon = np.array([iso(days[k])[:7] for k in ev['d'][bi]])
    um, cnt = np.unique(mon[yb >= 2010], return_counts=True)
    P(f'- Bystanders per calendar month since 2010: min {cnt.min()}, median {int(np.median(cnt))}, max {cnt.max()} '
      f'({len(um)} months). Kill line: about 20.')
    # sympathy vs ordinary-response prediction
    P('\n**Sympathy move against the ordinary-response prediction (beta x A move).** Slope of B\'s move on the '
      'prediction, through the origin, and the mean ratio of the two in A\'s direction:\n')
    P('| Group | Rows | Slope of move on prediction | Mean move in A\'s direction / mean prediction in A\'s direction | '
      'Mean abs move (sd units) |')
    P('|---|---|---|---|---|')
    pred = ev['beta'] * ev['xA']
    for name, msk in (('All co-movers on solo news days', ok & keep), ('Bystanders (vol <= 1.5x)', byst & keep),
                      ('Participants (vol >= 2x)', part & keep), ('Bystanders that moved with A >= 1.5 sd', byst & keep & insym)):
        xb = ev['xB'][msk]; pr = pred[msk]; s = np.sign(ev['xA'][msk])
        sl = (xb * pr).sum() / (pr * pr).sum()
        rat = (xb * s).mean() / (pr * s).mean()
        P(f'| {name} | {msk.sum():,} | {sl:.2f} | {rat:.2f} | {np.mean(np.abs(xb) / ev["sdB"][msk]):.2f} |')
    s = byst & keep & insym
    P(f'\n- Size of the "moved with A >= 1.5 sd" bystander moves: median |move| {np.median(np.abs(ev["xB"][s])) * 100:.2f}%, '
      f'interquartile {np.percentile(np.abs(ev["xB"][s]), 25) * 100:.2f}-{np.percentile(np.abs(ev["xB"][s]), 75) * 100:.2f}%.')
    s = byst & keep
    P(f'- All bystanders: median |move| {np.median(np.abs(ev["xB"][s])) * 100:.2f}%; median A move '
      f'{np.median(np.abs(ev["xA"][s])) * 100:.1f}%; median correlation {np.median(ev["corr"][s]):.2f}; median ordinary slope '
      f'{np.median(ev["beta"][s]):.2f}.')
    has = np.isfinite(ev['resp'][bi])
    P(f'- Bystander events with a news-day response (A has >= 8 earlier solo news days): {has.sum():,} of {len(bi):,} '
      f'({has.mean():.0%}).')
    txt = '\n'.join(lines)
    (OUT / f'test1_base_{tag}.md').write_text(txt + '\n')
    print(txt)


# ---------------------------------------------------------------- test 2
PERIODS = [('2005-01-01', '2009-12-31'), ('2010-01-01', '2014-12-31'), ('2015-01-01', '2019-12-31'),
           ('2020-01-01', '2026-12-31')]


def replicate(tag):
    """Is the news-day response a property of the pair? For each co-mover pair (A, B) seen on A's solo news days in a
    5-year period: ordinary slope (all days of the period) and news-day slope (A's solo news days in the period,
    >= 8). Residual gap = news slope minus its cross-pair fit on the ordinary slope (removes the common
    attenuation). Same sign in the next period more often than 50%?"""
    ev = load_events(tag)
    P = common.load_panels(tag, names=('x', 'r', 'U'))
    days = ev['days']
    x = np.asarray(P['x']); r = np.asarray(P['r']); U = np.asarray(P['U'])
    msum, mcnt = P['msum'], P['mcnt']
    nd_solo = ev['news_solo']
    nA, nD = ev['news_A'][nd_solo], ev['news_d'][nd_solo]
    per = []
    for a, b in PERIODS:
        lo, hi = common.c2eval.day_index(days, a), common.c2eval.day_index(days, b) if b < iso(days[-1]) else len(days)
        hi = min(hi, len(days))
        sel = (ev['d'] >= lo) & (ev['d'] < hi)
        pairs = np.unique(ev['A'][sel].astype(np.int64) * 100000 + ev['B'][sel])
        per.append((lo, hi, pairs))
    stats = []
    for pi, (lo, hi, _) in enumerate(per):
        stats.append({})
    def pair_stats(pairs, lo, hi):
        out = {}
        As = pairs // 100000; Bs = pairs % 100000
        for a in np.unique(As):
            Dn = nD[(nA == a) & (nD >= lo) & (nD < hi)]
            if len(Dn) < 8:
                continue
            mexA = (msum[Dn] - r[a, Dn]) / (mcnt[Dn] - 1)
            xa_n = r[a, Dn] - mexA
            xa = x[a, lo:hi]
            for b in Bs[As == a]:
                okn = U[b, Dn]
                if okn.sum() < 8:
                    continue
                xa1 = xa_n[okn]; xb1 = r[b, Dn[okn]] - mexA[okn]
                xc = xa1 - xa1.mean()
                sn = (xc * (xb1 - xb1.mean())).sum() / (xc * xc).sum()
                xb = x[b, lo:hi]
                f = np.isfinite(xa) & np.isfinite(xb)
                if f.sum() < 250:
                    continue
                xa2 = xa[f] - xa[f].mean()
                so = (xa2 * (xb[f] - xb[f].mean())).sum() / (xa2 * xa2).sum()
                out[a * 100000 + b] = (sn, so, okn.sum())
        return out
    lines = []
    Pn = lines.append
    Pn('## Test 2: is the news-day response a property of the pair? (' + tag + ' export)\n')
    Pn('| Periods | Pairs (>= 8 solo news days in both) | Raw gap same sign | Across-pair chance (raw) | '
       'Residual gap same sign | z (clustered by A) | Within-A shuffle chance (residual) | Mean sign(then) x residual next |')
    Pn('|---|---|---|---|---|---|---|---|')
    rng = np.random.default_rng(SEED)
    pooled = []
    for i in range(len(per) - 1):
        lo0, hi0, pairs0 = per[i]; lo1, hi1, _ = per[i + 1]
        s0 = pair_stats(pairs0, lo0, hi0)
        s1 = pair_stats(np.array(sorted(s0.keys()), dtype=np.int64), lo1, hi1)
        keys = np.array(sorted(set(s0) & set(s1)), dtype=np.int64)
        if len(keys) < 20:
            continue
        g = {}
        for j, s in ((0, s0), (1, s1)):
            sn = np.array([s[k][0] for k in keys]); so = np.array([s[k][1] for k in keys])
            X = np.c_[np.ones_like(so), so]
            coef = np.linalg.lstsq(X, sn, rcond=None)[0]
            g[j] = (sn - so, sn - X @ coef)
        As = keys // 100000
        raw_same = np.sign(g[0][0]) == np.sign(g[1][0])
        q0 = (g[0][0] < 0).mean(); q1 = (g[1][0] < 0).mean()
        chance_raw = q0 * q1 + (1 - q0) * (1 - q1)
        agree = (np.sign(g[0][1]) == np.sign(g[1][1])).astype(float)
        z = clustered_mean_z(agree - 0.5, As)
        # within-A shuffle of the next-period residual
        sh = []
        for _ in range(200):
            nxt = g[1][1].copy()
            for a in np.unique(As):
                w = np.where(As == a)[0]
                if len(w) > 1:
                    nxt[w] = nxt[rng.permutation(w)]
            sh.append((np.sign(g[0][1]) == np.sign(nxt)).mean())
        msx = np.mean(np.sign(g[0][1]) * g[1][1])
        label = f'{iso(days[lo0])[:4]}-{iso(days[hi0 - 1])[:4]} -> {iso(days[lo1])[:4]}-{iso(days[hi1 - 1])[:4]}'
        Pn(f'| {label} | {len(keys):,} | {raw_same.mean():.1%} | {chance_raw:.1%} | {agree.mean():.1%} | {z:+.2f} | '
           f'{np.mean(sh):.1%} | {msx:+.3f} |')
        pooled.append((label, agree, As + i * 10_000_000, np.mean(sh), len(keys), i))
    pre = [p for p in pooled if p[5] < 2]
    if pre:
        ag = np.concatenate([p[1] for p in pre]); cl = np.concatenate([p[2] for p in pre])
        Pn(f'\n- **Pooled 2005-2009 -> 2010-2014 and 2010-2014 -> 2015-2019 (the decision):** residual gap same sign '
           f'{ag.mean():.1%} of {len(ag):,} pairs, z = {clustered_mean_z(ag - 0.5, cl):+.2f} (clustered by A; pass needs z >= 2), '
           f'within-A shuffle chance {np.average([p[3] for p in pre], weights=[p[4] for p in pre]):.1%}.')
    txt = '\n'.join(lines)
    (OUT / f'test2_replicate_{tag}.md').write_text(txt + '\n')
    print(txt)


def clustered_mean_z(v, cl):
    """z of mean(v) with standard error clustered by cl."""
    v = np.asarray(v, float)
    mu = v.mean()
    e = v - mu
    u, inv = np.unique(cl, return_inverse=True)
    s = np.bincount(inv, weights=e)
    G = len(u)
    se = np.sqrt((s * s).sum() * G / (G - 1)) / len(v)
    return mu / se


# ---------------------------------------------------------------- test 3/4
def ols_cluster(X, y, cl):
    """OLS with Liang-Zeger standard errors clustered by cl (CR1: G/(G-1) x (n-1)/(n-k))."""
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    u, inv = np.unique(cl, return_inverse=True)
    G = len(u)
    k = X.shape[1]
    S = np.zeros((G, k))
    np.add.at(S, inv, X * e[:, None])
    meat = S.T @ S
    n = len(y)
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    return b, np.sqrt(np.diag(V)), G


def matched_days(ev, rows, P, lo_window=252, gap=5, per_event=5, seed=SEED):
    """For each bystander row: up to `per_event` of B's own days in [t-252, t-6] with B in the universe, volume
    0.5-1.5x its 60-day median, no news day (spike) of B or of any of the event's As within +-5 days, not an expiry
    day, and a market-relative move of the same sign within +-25% of the sympathy move. Returns
    (event_row_index, matched_day) pairs, deduplicated on (B, day)."""
    rng = np.random.default_rng(seed)
    U, vr, x, spike = P['U'], P['vr'], P['x'], P['spike']
    T = x.shape[1]
    ex = np.zeros(T, bool); ex[common.expiry_days(P['days'])] = True
    # all As of each (B, d)
    key_all = ev['B'].astype(np.int64) * 100000 + ev['d']
    order = np.argsort(key_all)
    ks = key_all[order]
    out_e, out_d = [], []
    seen = set()
    for i in rows:
        b, t, s = ev['B'][i], ev['d'][i], ev['xB'][i]
        if not np.isfinite(s) or s == 0:
            continue
        lo = max(66, t - lo_window); hi = t - gap
        if hi <= lo:
            continue
        k = b * 100000 + t
        a0, a1 = np.searchsorted(ks, k), np.searchsorted(ks, k, side='right')
        As = np.unique(ev['A'][order[a0:a1]])
        ds = np.arange(lo, hi)
        xb = x[b, ds]
        ok = U[b, ds] & (vr[b, ds] >= 0.5) & (vr[b, ds] <= 1.5) & ~ex[ds]
        with np.errstate(invalid='ignore'):
            ok &= (np.sign(xb) == np.sign(s)) & (np.abs(xb - s) <= 0.25 * abs(s))
        if not ok.any():
            continue
        # no spike of B or the As within +-5 days
        sp = spike[b, lo - gap:hi + gap].copy()
        for a in As:
            sp |= spike[a, lo - gap:hi + gap]
        csum = np.r_[0, np.cumsum(sp)]
        j = np.arange(len(ds))  # day ds[j] ~ index j+gap in sp
        near = csum[j + 2 * gap + 1] - csum[j]
        ok &= near == 0
        cand = ds[ok]
        if len(cand) == 0:
            continue
        pick = rng.choice(cand, size=min(per_event, len(cand)), replace=False)
        for d in pick:
            if (b, d) in seen:
                continue
            seen.add((b, d))
            out_e.append(i); out_d.append(d)
    return np.array(out_e, dtype=np.int64), np.array(out_d, dtype=np.int64)


def fwd_matched(P, Bs, Ds, h, open_entry=False):
    c, op, m, moc = P['close'], P['open'], P['m'], P['moc']
    T = c.shape[1]
    mm = np.where(np.isfinite(m), m, 0.0)
    M = np.concatenate([[1.0], np.cumprod(1 + mm)])
    dh = Ds + h
    ok = dh < T
    dh = np.minimum(dh, T - 1)
    with np.errstate(all='ignore'):
        if not open_entry:
            f = c[Bs, dh] / c[Bs, Ds] - 1 - (M[dh + 1] / M[Ds + 1] - 1)
        else:
            d1 = np.minimum(Ds + 1, T - 1)
            o1 = op[Bs, d1]
            f = c[Bs, dh] / o1 - 1 - ((1 + moc[d1]) * M[dh + 1] / M[d1 + 1] - 1)
            f[~(o1 > 0)] = np.nan
    f[~ok] = np.nan
    return f


def giveback(tag, start, end, label=None):
    ev = load_events(tag)
    P = common.load_panels(tag, names=('U', 'vr', 'x', 'spike', 'close', 'open'))
    P = {k: (np.asarray(v) if hasattr(v, 'shape') and v.ndim == 2 else v) for k, v in P.items()}
    days = ev['days']
    lo = common.c2eval.day_index(days, start)
    hi = common.c2eval.day_index(days, end) if end < iso(days[-1]) else len(days)
    if end >= iso(days[-1]):
        hi = len(days)
    else:
        hi = int(np.searchsorted(days, (np.datetime64(end) - np.datetime64('1970-01-01')).astype(int), side='right'))
    ok, byst, part, insym = classify(ev)
    inper = (ev['d'] >= lo) & (ev['d'] < hi) & ~ev['expiry']
    label = label or f'{start}..{end} ({tag} export)'
    L = []
    W = L.append
    W(f'## Give-back regressions, {label}\n')
    res = {}

    def run(rows_b, xb_vals, name, h_list=(1, 5, 10), open_entry=False, extra_mask=None):
        """bystander-type rows vs their matched sets; pooled regression with interaction, clustered by date."""
        me, md = matched_days(ev, rows_b, P)
        has = np.zeros(len(ev['d']), bool); has[me] = True
        rb = rows_b[has[rows_b]]
        n_drop = len(rows_b) - len(rb)
        outs = []
        for h in h_list:
            if open_entry:
                yb = ev[f'foB{h}'][rb]
                ym = fwd_matched(P, ev['B'][me], md, h, open_entry=True)
            else:
                yb = ev[f'fB{h}'][rb]
                ym = fwd_matched(P, ev['B'][me], md, h)
            xb = xb_vals[rb] if xb_vals is not None else ev['xB'][rb]
            xm = P['x'][ev['B'][me], md]
            fb = np.isfinite(yb) & np.isfinite(xb); fm = np.isfinite(ym) & np.isfinite(xm)
            y = np.r_[yb[fb], ym[fm]]; xx = np.r_[xb[fb], xm[fm]]
            I = np.r_[np.ones(fb.sum()), np.zeros(fm.sum())]
            cl = np.r_[ev['d'][rb][fb], md[fm]]
            X = np.c_[np.ones_like(y), xx, I, xx * I]
            b, se, G = ols_cluster(X, y, cl)
            # separate slopes and their own clustered SEs
            bb, sb, _ = ols_cluster(np.c_[np.ones(fb.sum()), xb[fb]], yb[fb], ev['d'][rb][fb])
            bm, sm, _ = ols_cluster(np.c_[np.ones(fm.sum()), xm[fm]], ym[fm], md[fm])
            # month clusters as a check on overlapping windows
            mon = (days[cl].astype('datetime64[D]').astype('datetime64[M]')).astype(int)
            _, se_m, _ = ols_cluster(X, y, mon)
            outs.append(dict(h=h, nb=int(fb.sum()), nm=int(fm.sum()), slope_b=bb[1], se_b=sb[1], slope_m=bm[1], se_m=sm[1],
                             diff=b[3], se=se[3], t=b[3] / se[3], t_month=b[3] / se_m[3], G=G, drop=n_drop))
        res[name] = outs
        return outs

    def table(name, outs):
        W(f'**{name}**\n')
        W('| h (days) | Events | Matched days | Event slope (t) | Matched slope (t) | Difference | t (date clusters) | t (month clusters) |')
        W('|---|---|---|---|---|---|---|---|')
        for o in outs:
            W(f"| {o['h']} | {o['nb']:,} | {o['nm']:,} | {o['slope_b']:+.4f} ({o['slope_b'] / o['se_b']:+.1f}) | "
              f"{o['slope_m']:+.4f} ({o['slope_m'] / o['se_m']:+.1f}) | {o['diff']:+.4f} | {o['t']:+.2f} | {o['t_month']:+.2f} |")
        W(f"\nEvents with no matched day, dropped from both sets: {outs[0]['drop']:,}.\n")

    rb = collapse(ev, byst & inper)
    table('Bystanders: future return on the sympathy move (the pre-chosen test is h = 5)', run(rb, None, 'byst'))
    rbs = collapse(ev, byst & inper & insym)
    table('Bystanders that moved with A by >= 1.5 sd (the cost paragraph\'s selection)', run(rbs, None, 'byst_insym'))
    rp = collapse(ev, part & inper)
    table('Participants (volume >= 2x): same recipe', run(rp, None, 'part'))
    # twist: mismatch in place of the sympathy move, events with a news-day response
    mis = ev['xB'] - ev['resp'] * ev['xA']
    rt = collapse(ev, byst & inper & np.isfinite(ev['resp']))
    table('Twist: bystanders with a news-day response, future return on the mismatch (matched set as above, on the move)',
          run(rt, mis, 'twist'))
    table('Same events as the twist row, on the sympathy move (for comparison)', run(rt, None, 'twist_base'))
    # decomposition: y on mismatch and predicted part together
    W('**Twist decomposition** (bystanders with a news-day response; y on mismatch and on the predicted part '
      'news-day response x A move, date clusters):\n')
    W('| h | Events | Mismatch coef (t) | Predicted-part coef (t) |')
    W('|---|---|---|---|')
    for h in (1, 5, 10):
        y = ev[f'fB{h}'][rt]; a1 = mis[rt]; a2 = (ev['resp'] * ev['xA'])[rt]
        f = np.isfinite(y) & np.isfinite(a1) & np.isfinite(a2)
        b, se, _ = ols_cluster(np.c_[np.ones(f.sum()), a1[f], a2[f]], y[f], ev['d'][rt][f])
        W(f'| {h} | {f.sum():,} | {b[1]:+.4f} ({b[1] / se[1]:+.1f}) | {b[2]:+.4f} ({b[2] / se[2]:+.1f}) |')
    W('')
    # diagnostic split by A's own next 5 days
    kept = np.sign(ev['fA5']) == np.sign(ev['xA'])
    rk = collapse(ev, byst & inper & kept & np.isfinite(ev['fA5']))
    rr = collapse(ev, byst & inper & ~kept & np.isfinite(ev['fA5']))
    table('Diagnostic: A kept going over the next 5 days (uses A\'s future, not tradable)', run(rk, None, 'A_kept', h_list=(5,)))
    table('Diagnostic: A reversed over the next 5 days', run(rr, None, 'A_rev', h_list=(5,)))
    # next-open entry, 2008+
    o_lo = max(lo, common.c2eval.day_index(days, '2008-01-01'))
    ro = collapse(ev, byst & inper & (ev['d'] >= o_lo))
    table(f'Next-open entry (from {iso(days[o_lo])}): bystanders, return from the next open', run(ro, None, 'byst_open', open_entry=True))
    roc = ro
    table(f'Same events (from {iso(days[o_lo])}), close-to-close for comparison', run(roc, None, 'byst_open_cc'))
    txt = '\n'.join(L)
    tagf = label.replace(' ', '_').replace('(', '').replace(')', '').replace('..', '_')
    (OUT / f'giveback_{tagf}.md').write_text(txt + '\n')
    np.save(OUT / f'giveback_{tagf}.npy', np.array([res], dtype=object), allow_pickle=True)
    print(txt)
    return res


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'base':
        base(sys.argv[2])
    elif cmd == 'replicate':
        replicate(sys.argv[2])
    elif cmd == 'giveback':
        giveback(sys.argv[2], sys.argv[3], sys.argv[4])
