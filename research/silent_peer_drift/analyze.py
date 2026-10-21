"""Silent-peer drift: classes, counts and Tests 1-4 of experiments/-8.2-silent-peer-drift.md.

    PY="systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 -- env OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python3 -I"
    $PY analyze.py counts old     # classes and counts by class and year (no forward returns)
    $PY analyze.py tests old      # Tests 1-3 and the context rows (forward returns)
    $PY analyze.py tests new      # Test 4, the unseen months (once, at the end)

Writes research/silent_peer_drift/out/{counts,tests}_<tag>.md and a .json with the key numbers.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build  # noqa: E402

c2eval = build.c2eval
iso = c2eval.iso
OUT = Path(__file__).resolve().parent / 'out'
OUT.mkdir(exist_ok=True)

MIN_CO = 3        # co-movers with a move that day needed for E (decided before outcomes; see the .md)
MIN_PRIOR = 100   # earlier classified news days (d-252..d-1) needed for the tercile breakpoints
WIN = 252
HS = (1, 5, 20, 62)   # 1 = the skipped day t+1 (close t -> t+1); 5/20/62 = close t+1 -> close t+6/t+21/t+63
CLS = {0: 'silent', 1: 'middle', 2: 'loud'}
EXPORT = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}


def load_events(tag):
    z = np.load(build.BIG / f'events_{tag}.npz')
    return {k: z[k] for k in z.files}


def year_of(days, d):
    return np.array([int(iso(days[k])[:4]) for k in d]) if len(d) else np.array([], int)


def trailing_split(d, val, ok, qs, min_prior=MIN_PRIOR, win=WIN):
    """For each row i: quantiles qs of val over rows with ok and day in [d_i - win, d_i - 1]. NaN if < min_prior."""
    o = np.argsort(d[ok], kind='stable')
    dd = d[ok][o]; vv = val[ok][o]
    out = np.full((len(d), len(qs)), np.nan)
    cache = {}
    for i, di in enumerate(d):
        if di not in cache:
            a, b = np.searchsorted(dd, di - win), np.searchsorted(dd, di)
            cache[di] = np.quantile(vv[a:b], qs) if b - a >= min_prior else np.full(len(qs), np.nan)
        out[i] = cache[di]
    return out


def classify(ev, min_co=MIN_CO, min_prior=MIN_PRIOR):
    """cls: 0 silent / 1 middle / 2 loud / -1 unclassified (fewer than min_co co-movers, or too few earlier events).
    size: 1 = A's 60-day median dollar volume at or above the median of the earlier year's news days, 0 below, -1 n/a."""
    d = ev['d']; E = ev['E']
    defined = (ev['n_co'] >= min_co) & np.isfinite(E)
    q = trailing_split(d, E, defined, [1 / 3, 2 / 3], min_prior)
    cls = np.full(len(d), -1, np.int8)
    okq = defined & np.isfinite(q[:, 0])
    cls[okq & (E < q[:, 0])] = 0
    cls[okq & (E > q[:, 1])] = 2
    cls[okq & (E >= q[:, 0]) & (E <= q[:, 1])] = 1
    dv = ev['dvA']
    sq = trailing_split(d, dv, np.isfinite(dv), [0.5], min_prior)[:, 0]
    size = np.full(len(d), -1, np.int8)
    oks = np.isfinite(sq) & np.isfinite(dv)
    size[oks] = (dv[oks] >= sq[oks]).astype(np.int8)
    return cls, size, q


# ---------------------------------------------------------------- counts (no forward returns)
def counts(tag):
    ev = load_events(tag)
    days = ev['days']
    cls, size, q = classify(ev)
    yr = year_of(days, ev['d'])
    L = []
    W = L.append
    W(f'## Counts by class and year ({tag} export; no forward returns used)\n')
    W(f'News day: volume >= 4x its 60-day median, |market-relative log move| >= 3 sd60, first in 5 days, A in the '
      f'judge-like liquid set (not fund-like). E needs >= {MIN_CO} co-movers; terciles from >= {MIN_PRIOR} classified '
      f'news days in the previous {WIN} trading days.\n')
    W('| Year | News days | With >= 1 co-mover | With >= 3 | Classified | Silent | Middle | Loud | Median co-movers | '
      'Good news % | Silent E cut | Loud E cut |')
    W('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for y in range(yr.min(), yr.max() + 1):
        s = yr == y
        if not s.any():
            continue
        qs = q[s & np.isfinite(q[:, 0])]
        W(f'| {y} | {s.sum():,} | {(ev["n_co"][s] >= 1).sum():,} | {(ev["n_co"][s] >= 3).sum():,} | {(cls[s] >= 0).sum():,} | '
          f'{(cls[s] == 0).sum():,} | {(cls[s] == 1).sum():,} | {(cls[s] == 2).sum():,} | {np.median(ev["n_co"][s]):.0f} | '
          f'{100 * (ev["xA"][s] > 0).mean():.0f} | {np.median(qs[:, 0]) if len(qs) else np.nan:.2f} | '
          f'{np.median(qs[:, 1]) if len(qs) else np.nan:.2f} |')
    s = cls >= 0
    W(f'\nAll years: {len(cls):,} news days, {s.sum():,} classified.\n')
    W('**Event-day profile by class** (all classified events; no forward returns):\n')
    W('| Class | Events | Median E | Median co-movers | Median mean corr | Mean share of co-movers with volume >= 2x | '
      'Median |A move| % | Median |A move| sd | Median A 60-day $ volume ($M) | Large half % | Good news % |')
    W('|---|---|---|---|---|---|---|---|---|---|---|')
    for c, nm in CLS.items():
        m = cls == c
        W(f'| {nm} | {m.sum():,} | {np.median(ev["E"][m]):+.2f} | {np.median(ev["n_co"][m]):.0f} | '
          f'{np.median(ev["mean_corr"][m]):.2f} | {100 * np.mean(ev["n_part"][m] / ev["n_co"][m]):.0f}% | '
          f'{100 * np.median(np.abs(ev["xA"][m])):.1f} | {np.median(np.abs(ev["xA"][m]) / ev["sdA"][m]):.1f} | '
          f'{np.median(ev["dvA"][m]) / 1e6:.0f} | {100 * np.mean(size[m] == 1):.0f} | {100 * np.mean(ev["xA"][m] > 0):.0f} |')
    nco = ev['n_co']
    W('\n**Co-movers per news day** (all years): ' + ', '.join(
        f'{k}: {100 * np.mean(nco == k):.0f}%' for k in range(0, 5)) + f', 5-19: {100 * np.mean((nco >= 5) & (nco < 20)):.0f}%, '
      f'20: {100 * np.mean(nco == 20):.0f}%')
    W(f'\nQuiet big moves (the no-news contrast): {len(ev["qd"]):,}.')
    txt = '\n'.join(L)
    (OUT / f'counts_{tag}.md').write_text(txt + '\n')
    print(txt)


# ---------------------------------------------------------------- outcomes and statistics
def outcomes(tag, A, d, sign):
    """Signed market-relative log returns: h=1 from close d to close d+1; h=5/20/62 from close d+1 to close d+1+h.
    NaN where either close is missing or the window runs past the export."""
    Ex = c2eval.load_export(EXPORT[tag], arrays=('close', 'market'))
    c = Ex['close']
    T = c.shape[1]
    mk = np.asarray(Ex['market'], np.float64)
    Mc = np.concatenate([[0.0], np.cumsum(np.nan_to_num(mk))])
    out = {}
    for h in HS:
        s0 = d if h == 1 else d + 1
        s1 = d + 1 if h == 1 else d + 1 + h
        ok = s1 <= T - 1
        s0c, s1c = np.minimum(s0, T - 1), np.minimum(s1, T - 1)
        c0 = np.asarray(c[A, s0c], np.float64); c1 = np.asarray(c[A, s1c], np.float64)
        with np.errstate(all='ignore'):
            r = np.log(c1 / c0) - (Mc[s1c + 1] - Mc[s0c + 1])
        r[~ok | ~np.isfinite(r)] = np.nan
        out[h] = sign * r
    return out


def ols_cluster(X, y, cl):
    """OLS with Liang-Zeger standard errors clustered by cl (CR1), as in research/bystanders/analyze.py."""
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    u, inv = np.unique(cl, return_inverse=True)
    G = len(u)
    k = X.shape[1]
    S = np.zeros((G, k))
    np.add.at(S, inv, X * e[:, None])
    n = len(y)
    V = XtX_inv @ (S.T @ S) @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    return b, np.sqrt(np.diag(V)), G


def month_of(days, d):
    dt = days[d].astype('datetime64[D]').astype('datetime64[M]').astype(int)
    return dt


def mean_t(y, dcl, mcl=None):
    f = np.isfinite(y)
    if f.sum() < 3:
        return dict(n=int(f.sum()), mean=np.nan, t=np.nan, t_month=np.nan)
    X = np.ones((f.sum(), 1))
    b, se, _ = ols_cluster(X, y[f], dcl[f])
    r = dict(n=int(f.sum()), mean=float(b[0]), t=float(b[0] / se[0]))
    if mcl is not None:
        b2, se2, _ = ols_cluster(X, y[f], mcl[f])
        r['t_month'] = float(b2[0] / se2[0])
    return r


def diff_t(y, grp1, grp2, dcl, mcl=None):
    """Mean(y | grp1) - mean(y | grp2) with clustered t, from one regression on [1, I(grp1)]."""
    s = (grp1 | grp2) & np.isfinite(y)
    if (grp1 & s).sum() < 3 or (grp2 & s).sum() < 3:
        return dict(n1=int((grp1 & s).sum()), n2=int((grp2 & s).sum()), diff=np.nan, t=np.nan, t_month=np.nan)
    X = np.column_stack([np.ones(s.sum()), grp1[s].astype(float)])
    b, se, G = ols_cluster(X, y[s], dcl[s])
    r = dict(n1=int((grp1 & s).sum()), n2=int((grp2 & s).sum()), diff=float(b[1]), t=float(b[1] / se[1]), clusters=G)
    if mcl is not None:
        b2, se2, _ = ols_cluster(X, y[s], mcl[s])
        r['t_month'] = float(b2[1] / se2[1])
    return r


def pct(v):
    return f'{100 * v:+.2f}%'


def tests(tag):
    ev = load_events(tag)
    days = ev['days']
    T = len(days)
    cls, size, _ = classify(ev)
    d = ev['d']; A = ev['A']
    sign = np.sign(ev['xA'])
    Y = outcomes(tag, A, d, sign)
    mcl = month_of(days, d)
    dcl = d
    q_y = outcomes(tag, ev['qA'], ev['qd'], np.sign(ev['qx']))
    qd = ev['qd']
    q_mcl = month_of(days, qd)
    di = lambda s: c2eval.day_index(days, s)  # noqa: E731
    if tag == 'old':
        periods = [('2006-2019 (test period)', di('2006-01-01'), di('2020-01-01')),
                   ('2006-2012', di('2006-01-01'), di('2013-01-01')),
                   ('2013-2019', di('2013-01-01'), di('2020-01-01')),
                   ('2020-01 .. 2025-06-27', di('2020-01-01'), T)]
    else:
        periods = [('The unseen months 2025-06-30 .. 2026-10-19', di('2025-06-30'), T)]
    res = {'tag': tag, 'periods': {}}
    L = []
    W = L.append
    W(f'## Tests ({tag} export)\n')
    W('Outcomes are signed by the direction of A\'s news move, market-relative log returns (stock minus the export\'s '
      '`market`). "t+1" is the skipped day (close t -> close t+1); 5, 20 and 62 days run from the close of t+1 to the '
      'close of t+6, t+21 and t+63. t statistics are clustered by event date (month clusters in brackets).\n')
    for pname, a, b in periods:
        inp = (d >= a) & (d < b)
        pr = res['periods'][pname] = {}
        W(f'### {pname}\n')
        W('News days dated in the period: ' + ', '.join(f'{nm} {int((inp & (cls == c)).sum()):,}' for c, nm in CLS.items())
          + f', unclassified {int((inp & (cls == -1)).sum()):,} (counts include events without a 20-day outcome).\n')
        pr['counts'] = {nm: int((inp & (cls == c)).sum()) for c, nm in list(CLS.items()) + [(-1, 'unclassified')]}
        W('**Signed drift by class**\n')
        W('| Class | Events | t+1 (t) | 5 days (t) | 20 days (t) [month t] | 62 days (t) |')
        W('|---|---|---|---|---|---|')
        groups = [('silent', cls == 0), ('middle', cls == 1), ('loud', cls == 2), ('unclassified', cls == -1),
                  ('all news days', np.ones(len(d), bool))]
        for nm, g in groups:
            s = inp & g
            cells = []
            pr[nm] = {}
            for h in HS:
                r = mean_t(np.where(s, Y[h], np.nan), dcl, mcl)
                pr[nm][h] = r
                cells.append(f'{pct(r["mean"])} ({r["t"]:+.2f})' + (f' [{r["t_month"]:+.2f}]' if h == 20 else ''))
            W(f'| {nm} | {pr[nm][20]["n"]:,} | ' + ' | '.join(cells) + ' |')
        W('\n**Silent minus loud**\n')
        W('| Horizon | Silent events | Loud events | Difference | t, date clusters | t, month clusters |')
        W('|---|---|---|---|---|---|')
        pr['silent_minus_loud'] = {}
        for h in HS:
            r = diff_t(Y[h], inp & (cls == 0), inp & (cls == 2), dcl, mcl)
            pr['silent_minus_loud'][h] = r
            lab = 't+1' if h == 1 else f'{h} days'
            W(f'| {lab} | {r["n1"]:,} | {r["n2"]:,} | {pct(r["diff"])} | {r["t"]:+.2f} | {r["t_month"]:+.2f} |')
        W('\n**By size half (A\'s 60-day median dollar volume against the median of the previous year\'s news days), '
          '20 days**\n')
        W('| Half | Silent (t) | Middle (t) | Loud (t) | Silent minus loud (t) [month t] |')
        W('|---|---|---|---|---|')
        pr['size'] = {}
        for sz, nm in ((1, 'large'), (0, 'small')):
            g = inp & (size == sz)
            rs = [mean_t(np.where(g & (cls == c), Y[20], np.nan), dcl) for c in (0, 1, 2)]
            r = diff_t(Y[20], g & (cls == 0), g & (cls == 2), dcl, mcl)
            pr['size'][nm] = dict(silent=rs[0], middle=rs[1], loud=rs[2], diff=r)
            W(f'| {nm} | ' + ' | '.join(f'{pct(x["mean"])} ({x["t"]:+.2f}, n {x["n"]:,})' for x in rs) +
              f' | {pct(r["diff"])} ({r["t"]:+.2f}) [{r["t_month"]:+.2f}] |')
        W('\n**Silent events by news direction, 20 days** (the long side is good news; the short side is bad news)\n')
        W('| Side | Events | Signed drift (t) |')
        W('|---|---|---|')
        pr['silent_side'] = {}
        for nm, g in (('good news (long)', sign > 0), ('bad news (short)', sign < 0)):
            r = mean_t(np.where(inp & (cls == 0) & g, Y[20], np.nan), dcl)
            pr['silent_side'][nm] = r
            W(f'| {nm} | {r["n"]:,} | {pct(r["mean"])} ({r["t"]:+.2f}) |')
        W('\n**The no-news contrast:** same-size moves (>= 3 sd60) on volume <= 1.5x the 60-day median, no news day in '
          'the previous 5 days, first in 5 days\n')
        W('| Group | Events | t+1 (t) | 5 days (t) | 20 days (t) [month t] | 62 days (t) |')
        W('|---|---|---|---|---|---|')
        qin = (qd >= a) & (qd < b)
        pr['quiet'] = {}
        for h in HS:
            pr['quiet'][h] = mean_t(np.where(qin, q_y[h], np.nan), qd, q_mcl)
        cells = [f'{pct(pr["quiet"][h]["mean"])} ({pr["quiet"][h]["t"]:+.2f})' +
                 (f' [{pr["quiet"][h]["t_month"]:+.2f}]' if h == 20 else '') for h in HS]
        W(f'| quiet big moves | {pr["quiet"][20]["n"]:,} | ' + ' | '.join(cells) + ' |')
        cells = [f'{pct(pr["all news days"][h]["mean"])} ({pr["all news days"][h]["t"]:+.2f})' for h in HS]
        W(f'| all news days (above) | {pr["all news days"][20]["n"]:,} | ' + ' | '.join(cells) + ' |')
        W('')
    # sensitivity to the minimum number of co-movers (context; the rule is MIN_CO)
    W(f'### Sensitivity: minimum co-movers for E (the rule is {MIN_CO}), silent minus loud at 20 days\n')
    W('| Minimum co-movers | ' + ' | '.join(p[0] for p in periods) + ' |')
    W('|---|' + '---|' * len(periods))
    res['min_co'] = {}
    for mc in (1, MIN_CO, 5, 10):
        c_mc, _, _ = classify(ev, min_co=mc)
        cells = []
        for pname, a, b in periods:
            inp = (d >= a) & (d < b)
            r = diff_t(Y[20], inp & (c_mc == 0), inp & (c_mc == 2), dcl, mcl)
            res['min_co'][f'{mc}|{pname}'] = r
            cells.append(f'{pct(r["diff"])} ({r["t"]:+.2f}; n {r["n1"]:,} / {r["n2"]:,})')
        W(f'| {mc} | ' + ' | '.join(cells) + ' |')
    W('')
    # year by year
    W('### Year by year, 20 days\n')
    W('| Year | Silent | Middle | Loud | Silent minus loud (t) |')
    W('|---|---|---|---|---|')
    yr = year_of(days, d)
    res['years'] = {}
    for y in range(max(yr.min(), 2004), yr.max() + 1):
        g = yr == y
        rs = [mean_t(np.where(g & (cls == c), Y[20], np.nan), dcl) for c in (0, 1, 2)]
        r = diff_t(Y[20], g & (cls == 0), g & (cls == 2), dcl)
        if not np.isfinite(r['diff']):
            continue
        res['years'][y] = dict(silent=rs[0]['mean'], middle=rs[1]['mean'], loud=rs[2]['mean'], diff=r['diff'], t=r['t'],
                               n_s=rs[0]['n'], n_l=rs[2]['n'])
        W(f'| {y} | {pct(rs[0]["mean"])} (n {rs[0]["n"]:,}) | {pct(rs[1]["mean"])} | {pct(rs[2]["mean"])} (n {rs[2]["n"]:,}) | '
          f'{pct(r["diff"])} ({r["t"]:+.2f}) |')
    txt = '\n'.join(L)
    (OUT / f'tests_{tag}.md').write_text(txt + '\n')
    (OUT / f'tests_{tag}.json').write_text(json.dumps(res, indent=1, default=float))
    print(txt)


if __name__ == '__main__':
    cmd, tag = sys.argv[1], sys.argv[2]
    {'counts': counts, 'tests': tests}[cmd](tag)
