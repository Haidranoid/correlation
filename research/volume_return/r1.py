"""R1 (primary) and its reported items 1-7 (plan: "R1 (primary)", "Reported (not pass/fail)").

    r1.py old   -> out/r1_old.txt, out/r1_old.json   (development 2008-2019, the check period, the reported items,
                   and the check period's dividend robustness)
    r1.py new   -> out/r1_new.txt, out/r1_new.json   (the 16 unseen months as one period; run by unseen.py)
    r1.py fwd   -> out/r1_fwd.txt, out/r1_fwd.json   (forward.py)

Per stock and period: OLS of R(t+1) on [1, R(t), V(t) R(t)] over the period's pairs (>= 200; >= 100 for the 2025
half-year). Delta = mean C2 of the lowest third by 252-day median dollar volume minus the highest, C2 winsorized at the
period's 1st/99th percentiles across its stocks. R1 = the mean of the yearly Deltas, 2008-2019. Standard errors: blocks
of 21 trading days within each period, resampled with replacement, the same draw for every stock.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/bhoult/programming/correlation/research/volume_return')
import numpy as np  # noqa: E402

import common as C  # noqa: E402

c2eval = C.c2eval


# ---- designs: columns of X from (r, v); the coefficient(s) the statistic reads ------------------------------------
def d_paper(r, v):
    return [np.ones_like(r), r, v * r]


def d_updown(r, v):
    rp, rm = np.maximum(r, 0.0), np.minimum(r, 0.0)
    return [np.ones_like(r), rp, rm, v * rp, v * rm]


DESIGNS = {'paper': (d_paper, {'C1': 1, 'C2': 2}), 'updown': (d_updown, {'C2u': 3, 'C2d': 4, 'C1u': 1, 'C1d': 2})}


def period_blocks(R, V, cal, uname, t_first, t_last, design, min_pairs, mask=None, mkt=None, fix=None, prox=None,
                  Y=None, shuffle=None):
    """Per-stock sums of X'X, X'y and n for each 21-day block of the period's pairs (t in [t_first, t_last])."""
    rebal = cal['rebal']
    T = R.shape[1]
    k = C.cal_index(rebal, t_first)
    members = cal[uname][k].copy()
    if prox is not None:
        members &= np.isfinite(prox[k])
    rows = np.nonzero(members)[0]
    ts = np.arange(t_first, min(t_last, T - 2) + 1)
    r = np.asarray(R[:, ts][rows], np.float64)
    y = np.asarray(R[:, ts + 1][rows], np.float64) if Y is None else np.asarray(Y[:, ts][rows], np.float64)
    v = np.asarray(V[:, ts][rows], np.float64)
    if shuffle is not None:      # code check: V shuffled across the period's days within each stock
        v = np.take_along_axis(v, np.argsort(shuffle.random(v.shape), axis=1), axis=1)
    if fix is not None:          # dividend repair (log fix on the day's return)
        r = (1 + r) * np.exp(np.asarray(fix[:, ts][rows], np.float64)) - 1
        y = (1 + y) * np.exp(np.asarray(fix[:, ts + 1][rows], np.float64)) - 1
    if mkt is not None:          # market-adjusted returns
        r = r - mkt[ts][None, :]
        y = y - mkt[ts + 1][None, :]
    ok = np.isfinite(r) & np.isfinite(y) & np.isfinite(v)
    if mask is not None:
        ok &= np.asarray(mask[:, ts][rows])
    keep = ok.sum(1) >= min_pairs
    rows, r, y, v, ok = rows[keep], r[keep], y[keep], v[keep], ok[keep]
    r, y, v = (np.where(ok, a, 0.0) for a in (r, y, v))
    X = design(r, v)
    X = [np.where(ok, x, 0.0) for x in X]
    p = len(X)
    L = len(ts)
    nbk = max(1, L // C.BLOCK)
    starts = np.arange(nbk) * C.BLOCK
    S = len(rows)
    XX = np.zeros((S, nbk, p, p))
    Xy = np.zeros((S, nbk, p))
    for i in range(p):
        for j in range(i, p):
            XX[:, :, i, j] = XX[:, :, j, i] = np.add.reduceat(X[i] * X[j], starts, axis=1)
        Xy[:, :, i] = np.add.reduceat(X[i] * y, starts, axis=1)
    n = np.add.reduceat(ok.astype(np.float64), starts, axis=1)
    return {'rows': rows, 'k': k, 'XX': XX, 'Xy': Xy, 'n': n, 'nbk': nbk, 'L': L}


def solve(A, b):
    """Batched solve of A x = b for [..., p, p] and [..., p]; NaN where singular."""
    p = A.shape[-1]
    if p == 3:
        a, bb, c = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
        d, e, f = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
        g, h, i = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
        co = np.stack([np.stack([e * i - f * h, -(bb * i - c * h), bb * f - c * e], -1),
                       np.stack([-(d * i - f * g), a * i - c * g, -(a * f - c * d)], -1),
                       np.stack([d * h - e * g, -(a * h - bb * g), a * e - bb * d], -1)], -2)
        det = a * co[..., 0, 0] + bb * co[..., 1, 0] + c * co[..., 2, 0]
        scale = np.abs(a * e * i) + 1e-300
        with np.errstate(all='ignore'):
            x = np.einsum('...ij,...j->...i', co, b) / det[..., None]
        x[np.abs(det) <= 1e-12 * scale] = np.nan
        return x
    flat_A = A.reshape(-1, p, p)
    flat_b = b.reshape(-1, p)
    out = np.full(flat_b.shape, np.nan)
    with np.errstate(all='ignore'):
        det = np.linalg.det(flat_A)
        diag = np.prod(np.abs(np.diagonal(flat_A, axis1=1, axis2=2)), axis=1) + 1e-300
    good = np.isfinite(det) & (np.abs(det) > 1e-12 * diag)
    if good.any():
        out[good] = np.linalg.solve(flat_A[good], flat_b[good][..., None])[..., 0]
    return out.reshape(b.shape)


def row_percentile(srt, n, q):
    """Linear-interpolation percentile q of each row of an ascending-sorted array whose first n[row] entries are valid
    (numpy's default method); NaN for rows with n == 0."""
    pos = (np.maximum(n, 1) - 1) * (q / 100.0)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, np.maximum(n, 1) - 1)
    a = np.take_along_axis(srt, lo[:, None], 1)[:, 0]
    b = np.take_along_axis(srt, hi[:, None], 1)[:, 0]
    out = a + (b - a) * (pos - lo)
    out[n == 0] = np.nan
    return out


def wins_rows(c, valid):
    """Winsorize each row of c [D, S] at its own 1st/99th percentiles over valid entries; NaN where invalid."""
    valid = valid & np.isfinite(c)
    c = np.where(valid, c, np.nan)
    srt = np.sort(c, axis=1)                     # NaNs sort last
    n = valid.sum(1)
    lo = row_percentile(srt, n, C.WIN_LO)
    hi = row_percentile(srt, n, C.WIN_HI)
    return np.clip(c, lo[:, None], hi[:, None])


def stats_from(coef, n, groups, gpct, cidx, min_draw):
    """Delta (low third - high third of winsorized coefficient), slope on gpct, per-third means; rows = draws."""
    out = {}
    valid = (n >= min_draw) & np.all(np.isfinite(coef), axis=-1)
    for name, j in cidx.items():
        w = wins_rows(coef[..., j], valid)
        lo = np.where(groups[None, :] == 0, w, np.nan)
        hi = np.where(groups[None, :] == 2, w, np.nan)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out[name + '_delta'] = np.nanmean(lo, 1) - np.nanmean(hi, 1)
            for g in (0, 1, 2):
                out[f'{name}_mean{g}'] = np.nanmean(np.where(groups[None, :] == g, w, np.nan), 1)
            gv = np.where(np.isfinite(w), gpct[None, :], np.nan)
            gm = np.nanmean(gv, 1, keepdims=True)
            wm = np.nanmean(w, 1, keepdims=True)
            out[name + '_slope'] = np.nansum((gv - gm) * (w - wm), 1) / np.nansum((gv - gm) ** 2, 1)
            if name.startswith('C2'):
                for g in (0, 1, 2):
                    out[f'{name}_neg{g}'] = np.nanmean(np.where(groups[None, :] == g, (w < 0).astype(float), np.nan), 1)
    return out


def run_period(R, V, cal, uname, gname, t_first, t_last, design_name, min_pairs, rng, draws, **kw):
    """Point estimates and bootstrap draws of the statistics for one period."""
    design, cidx = DESIGNS[design_name]
    blk = period_blocks(R, V, cal, uname, t_first, t_last, design, min_pairs, **kw)
    S = len(blk['rows'])
    if S < 30:
        return None
    if kw.get('prox') is not None:
        val = kw['prox'][blk['k']][blk['rows']]
        gpct = 1.0 - C.pct_rank(val.astype(np.float64))   # high proxy value = most asymmetry = "low third" side
    else:
        gpct = C.pct_rank(cal[gname][blk['k']][blk['rows']].astype(np.float64))
    groups = C.thirds_of(gpct)
    p = blk['XX'].shape[-1]
    coef0 = solve(blk['XX'].sum(1)[None], blk['Xy'].sum(1)[None])     # [1, S, p]
    n0 = blk['n'].sum(1)[None]
    point = {k: float(v[0]) for k, v in stats_from(coef0, n0, groups, gpct, cidx, min_pairs).items()}
    boots = {}
    if draws:
        nbk = blk['nbk']
        XXf = blk['XX'].transpose(1, 0, 2, 3).reshape(nbk, -1)
        Xyf = blk['Xy'].transpose(1, 0, 2).reshape(nbk, -1)
        nf = blk['n'].T
        chunk = max(50, min(1000, int(4e7 // max(1, S * p * p))))
        parts = []
        for a in range(0, draws, chunk):
            dd = min(chunk, draws - a)
            pick = rng.integers(0, nbk, size=(dd, nbk))
            cnt = np.zeros((dd, nbk))
            np.add.at(cnt, (np.repeat(np.arange(dd), nbk), pick.ravel()), 1.0)
            A = (cnt @ XXf).reshape(dd, S, p, p)
            b = (cnt @ Xyf).reshape(dd, S, p)
            nn = cnt @ nf
            parts.append(stats_from(solve(A, b), nn, groups, gpct, cidx, C.MIN_PAIRS_DRAW))
        boots = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    sizes = [int((groups == g).sum()) for g in (0, 1, 2)]
    return {'point': point, 'boots': boots, 'S': S, 'sizes': sizes, 'nbk': blk['nbk'], 'L': blk['L'],
            'first': int(t_first), 'last': int(min(t_last, R.shape[1] - 2))}


def periods_for(days, kind):
    yrs = np.array([int(c2eval.iso(x)[:4]) for x in days])
    out = []
    if kind == 'dev':
        for y in range(2008, 2020):
            ix = np.nonzero(yrs == y)[0]
            out.append((str(y), int(ix[0]), int(ix[-1]), C.MIN_PAIRS))
    elif kind == 'check':
        for y in range(2020, 2025):
            ix = np.nonzero(yrs == y)[0]
            out.append((str(y), int(ix[0]), int(ix[-1]), C.MIN_PAIRS))
        ix = np.nonzero(yrs == 2025)[0]
        last = c2eval.day_index(days, '2025-06-27')
        out.append(('2025H1', int(ix[0]), min(int(ix[-1]), last) - 1, C.MIN_PAIRS_HALF))
    elif kind == 'unseen':
        a = c2eval.day_index(days, C.UNSEEN_START)
        out.append(('unseen', a, len(days) - 2, C.MIN_PAIRS))
    elif kind.startswith('fwd:'):
        a = c2eval.day_index(days, kind[4:])
        out.append(('forward', a, len(days) - 2, C.MIN_PAIRS))
    return out


def combine(results, key):
    """Mean over periods of a statistic: point and bootstrap draws (draw i of each period paired by index)."""
    res = [r for r in results if r is not None]
    pt = float(np.mean([r['point'][key] for r in res]))
    if res[0]['boots']:
        bs = np.mean(np.stack([r['boots'][key] for r in res]), 0)
    else:
        bs = None
    return pt, bs


def family(R, V, cal, periods, uname, gname, design='paper', draws=C.DRAWS, seed=C.SEED, keys=('C2_delta',), **kw):
    rng = np.random.default_rng(seed)
    res = [run_period(R, V, cal, uname, gname, a, b, design, mp, rng, draws, **kw) for _, a, b, mp in periods]
    out = {'periods': [], 'stats': {}}
    for (name, a, b, mp), r in zip(periods, res):
        if r is None:
            out['periods'].append({'period': name, 'S': 0})
            continue
        out['periods'].append({'period': name, 'S': r['S'], 'sizes': r['sizes'], 'blocks': r['nbk'], **r['point']})
    for key in keys:
        pt, bs = combine(res, key)
        vals = np.array([r['point'][key] for r in res if r is not None])
        plain_t = float(vals.mean() / (vals.std(ddof=1) / np.sqrt(len(vals)))) if len(vals) > 1 else float('nan')
        s = C.boot_summary(pt, bs) if bs is not None else {'est': pt}
        s['plain_t'] = plain_t
        s['ahead'] = int((vals > 0).sum())
        s['n_periods'] = len(vals)
        out['stats'][key] = s
    return out


def fmt_stat(s, scale=1.0, unit=''):
    if 'sd' not in s:
        return f"{s['est'] * scale:+.4f}{unit}"
    return (f"{s['est'] * scale:+.4f}{unit} (90% {s['lo90'] * scale:+.4f}..{s['hi90'] * scale:+.4f}), bootstrap t "
            f"{s['t']:+.2f}, p {s['p']:.4f} (percentile p {s['p_boot']:.4f}); plain t over periods {s['plain_t']:+.2f}; "
            f"positive in {s['ahead']} of {s['n_periods']}")


def main(tag):
    t0 = time.time()
    rep = C.OUT / f'r1_{tag}.txt'
    rep.write_text('')
    E = C.export(tag, arrays=())
    days = E['days']
    R, V, cal = C.load_panels(tag)
    mkt = C.market_simple(tag)
    J = {'tag': tag}

    def section(title, res, keys, scale=1.0):
        C.log(rep, '', f'== {title} ==')
        for key in keys:
            C.log(rep, f'{key}: ' + fmt_stat(res['stats'][key], scale))
        J[title] = res

    if tag == 'old':
        dev = periods_for(days, 'dev')
        chk = periods_for(days, 'check')
        keys1 = ('C2_delta', 'C2_slope', 'C1_delta', 'C2_mean0', 'C2_mean1', 'C2_mean2', 'C1_mean0', 'C1_mean1',
                 'C1_mean2', 'C2_neg0', 'C2_neg1', 'C2_neg2')
        # ---- the primary
        prim = family(R, V, cal, dev, 'UW', 'GW', keys=keys1)
        section('R1 (primary): U_W, 2008-2019', prim, ('C2_delta', 'C2_slope'))
        C.log(rep, 'by year: period | stocks (low/mid/high third) | C2 low | C2 mid | C2 high | Delta | slope on G | '
                   'C1 low | C1 high | share C2<0 low/high')
        for p in prim['periods']:
            C.log(rep, f"{p['period']} | {p['S']} ({'/'.join(str(x) for x in p['sizes'])}) | {p['C2_mean0']:+.4f} | "
                       f"{p['C2_mean1']:+.4f} | {p['C2_mean2']:+.4f} | {p['C2_delta']:+.4f} | {p['C2_slope']:+.4f} | "
                       f"{p['C1_mean0']:+.4f} | {p['C1_mean2']:+.4f} | {p['C2_neg0']:.2f}/{p['C2_neg2']:.2f}")
        st = prim['stats']
        C.log(rep, '', 'Item 1, the paper\'s Table 3 form (means over the 12 years): third | C1 | C2 | share C2 < 0',
              *[f"{nm} | {st[f'C1_mean{g}']['est']:+.4f} | {st[f'C2_mean{g}']['est']:+.4f} | "
                f"{st[f'C2_neg{g}']['est']:.2f}" for g, nm in ((0, 'low DV'), (1, 'middle'), (2, 'high DV'))])
        C.log(rep, '', 'Item 2, C1 + C2*V by third (means over the 12 years): third | V=0 | V=1 | V=2 | V=3')
        for g, nm in ((0, 'low DV'), (1, 'middle'), (2, 'high DV')):
            c1, c2 = st[f'C1_mean{g}']['est'], st[f'C2_mean{g}']['est']
            C.log(rep, f'{nm} | ' + ' | '.join(f'{c1 + c2 * v:+.4f}' for v in (0, 1, 2, 3)))
        # ---- the check period
        chk_res = family(R, V, cal, chk, 'UW', 'GW', keys=keys1)
        section('R1 on the check period 2020-2025-06 (consistent if > 0 at one-sided p < 0.05)', chk_res,
                ('C2_delta', 'C2_slope'))
        for p in chk_res['periods']:
            C.log(rep, f"{p['period']} | {p['S']} | {p['C2_mean0']:+.4f} | {p['C2_mean1']:+.4f} | {p['C2_mean2']:+.4f} "
                       f"| {p['C2_delta']:+.4f} | {p['C2_slope']:+.4f}")
        # ---- the reported items
        for lbl, pers in (('2008-2019', dev), ('2020-2025-06', chk)):
            res = family(R, V, cal, pers, 'UW', 'GW', design='updown', draws=2000, seed=C.SEED + 1,
                         keys=('C2u_delta', 'C2d_delta'))
            section(f'Item 3, up and down moves apart, {lbl} (2,000 draws)', res, ('C2u_delta', 'C2d_delta'))
            spike = np.asarray(V) < C.SPIKE_V
            res = family(R, V, cal, pers, 'UW', 'GW', seed=C.SEED + 2, mask=spike)
            section(f'Item 4, without spike days (V >= {C.SPIKE_V}), {lbl}', res, ('C2_delta',))
            del spike
            res = family(R, V, cal, pers, 'UW', 'GW', seed=C.SEED + 3, mkt=mkt)
            section(f'Item 5, market-adjusted returns, {lbl}', res, ('C2_delta',))
            for un, gn, nm in (('UJ', 'GJ', 'U_J'), ('TOP', 'GT', 'the 500 largest'), ('UW1', 'GW1', 'U_W at $1')):
                res = family(R, V, cal, pers, un, gn, seed=C.SEED + 4, keys=('C2_delta', 'C2_slope'))
                section(f'Item 6, {nm}, {lbl}', res, ('C2_delta', 'C2_slope'))
            for pr in ('amihud', 'idio', 'cs'):
                pers2 = [p for p in pers if not (pr == 'cs' and p[0] == '2008')]
                res = family(R, V, cal, pers2, 'UW', 'GW', seed=C.SEED + 5, prox=cal[pr])
                section(f'Item 7, proxy {pr} (C2 of the highest third minus the lowest), {lbl}', res, ('C2_delta',))
        # ---- dividend robustness on the check period (decided before any result; coordinator's note)
        import pubdata
        fix = pubdata.div_fix('old')
        res = family(R, V, cal, chk, 'UW', 'GW', seed=C.SEED + 6, fix=fix)
        section('Dividends: check period with div_fix applied to every return', res, ('C2_delta',))
        exm = exdate_mask('old', R.shape)
        res = family(R, V, cal, chk, 'UW', 'GW', seed=C.SEED + 7, mask=exm)
        section('Dividends: check period with ex-dividend days dropped from the signal', res, ('C2_delta',))
        # the store has no dividend adjustment before about 2011-02 either (coordinator's note): the same check on
        # the development years, and on 2008-2010 alone beside the unchanged statistic for 2008-2010
        res = family(R, V, cal, dev, 'UW', 'GW', seed=C.SEED + 7, mask=exm)
        section('Dividends: 2008-2019 with ex-dividend days dropped from the signal', res, ('C2_delta',))
        res = family(R, V, cal, dev[:3], 'UW', 'GW', seed=C.SEED + 7, mask=exm)
        section('Dividends: 2008-2010 with ex-dividend days dropped from the signal', res, ('C2_delta',))
        res = family(R, V, cal, dev[:3], 'UW', 'GW', seed=C.SEED + 7)
        section('Dividends: 2008-2010 as run (for comparison)', res, ('C2_delta',))
        yrs = np.array([int(c2eval.iso(x)[:4]) for x in days])
        C.log(rep, 'ex-dates among U_W member stock-days, by year: ' + ', '.join(
            f"{y} {int(sum((~exm[cal['UW'][C.cal_index(cal['rebal'], t)], t]).sum() for t in np.nonzero(yrs == y)[0]))}"
            for y in range(2008, 2026)))
    else:
        kind = 'unseen' if tag == 'new' else 'fwd:' + C.os.environ.get('C2_FWD_START', '2026-10-20')
        pers = periods_for(days, kind)
        res = family(R, V, cal, pers, 'UW', 'GW', keys=('C2_delta', 'C2_slope', 'C2_mean0', 'C2_mean1', 'C2_mean2',
                                                          'C1_mean0', 'C1_mean2'))
        section(f'R1 on {pers[0][0]} ({c2eval.iso(days[pers[0][1]])}..{c2eval.iso(days[pers[0][2] + 1])})', res,
                ('C2_delta', 'C2_slope'))
        p = res['periods'][0]
        C.log(rep, f"stocks {p['S']}; C2 low {p['C2_mean0']:+.4f}, middle {p['C2_mean1']:+.4f}, high {p['C2_mean2']:+.4f}")
        for un, gn, nm in (('UJ', 'GJ', 'U_J'), ('TOP', 'GT', 'the 500 largest')):
            res = family(R, V, cal, pers, un, gn, seed=C.SEED + 4)
            section(f'Item 6, {nm}', res, ('C2_delta',))
        res = family(R, V, cal, pers, 'UW', 'GW', design='updown', draws=2000, seed=C.SEED + 1,
                     keys=('C2u_delta', 'C2d_delta'))
        section('Item 3, up and down moves apart (2,000 draws)', res, ('C2u_delta', 'C2d_delta'))
        if tag == 'new':
            import pubdata
            fix = pubdata.div_fix('new')
            res = family(R, V, cal, pers, 'UW', 'GW', seed=C.SEED + 6, fix=fix)
            section('Dividends: div_fix applied to every return', res, ('C2_delta',))
            exm = exdate_mask('new', R.shape)
            res = family(R, V, cal, pers, 'UW', 'GW', seed=C.SEED + 7, mask=exm)
            section('Dividends: ex-dividend days dropped from the signal', res, ('C2_delta',))
    json.dump(J, open(C.OUT / f'r1_{tag}.json', 'w'), indent=1, default=lambda o: None)
    C.log(rep, '', f'done in {time.time() - t0:.0f} s')


def exdate_mask(tag, shape):
    """[N, T] bool: True except on ex-dividend dates of the export's stocks. Union of two read-only sources:
    pubdata.div_events(tag) (Yahoo events from 2022-07 for the judge-like tickers) and the full Yahoo histories in
    published_data/raw/yahoo_full_stocks (fetched 2026-10-25 by another run; every year, about 2,958 tickers). Cached."""
    cache = C.tagdir(tag) / 'exdates.npy'
    if cache.exists():
        m = np.load(cache)
        if m.shape == tuple(shape):
            return m
    import pubdata
    m = np.ones(shape, bool)
    ev = pubdata.div_events(tag)
    ok = ev['col'] >= 0
    m[ev['row'][ok], ev['col'][ok]] = False
    E = C.export(tag, arrays=())
    row = {t: j for j, t in enumerate(E['tickers'])}
    col = {c2eval.iso(d): i for i, d in enumerate(E['days'])}
    base = Path.home() / '.local/share/correlation2/experiments/published_data'
    ymap = {}
    for line in open('/home/bhoult/programming/correlation/research/published_data/out/tickers_full_stocks_map.tsv'):
        f = line.rstrip('\n').split('\t')
        if f[0] == 'export_ticker' or len(f) < 3:
            continue
        if tag in f[2].split(',') or tag == 'fwd':
            ymap[f[1]] = f[0]
    n_ev = 0
    for fn in sorted((base / 'raw' / 'yahoo_full_stocks').glob('batch_*.csv')):
        with open(fn) as fh:
            next(fh)
            for line in fh:
                f = line.split(',')
                dv = f[8]
                if dv in ('0.0', '0', '') or float(dv) <= 0:
                    continue
                j = row.get(ymap.get(f[0], ''))
                t = col.get(f[1][:10])
                if j is not None and t is not None:
                    m[j, t] = False
                    n_ev += 1
    np.save(cache, m)
    return m


if __name__ == '__main__':
    main(sys.argv[1])
