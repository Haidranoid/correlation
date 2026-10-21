"""The file-name number: the headline book against 1,000 random books with the same structure.

Every day each random book draws a random 10% long and a disjoint random 10% short from the same ranked set as the
headline book, with the same twin hedges, the same netted twin costs and the same 10 bps entry penalty.
Number = 252 x mean daily return of the headline book - the median over random books of the same, in points a year.

    python3 -I randoms.py old     # 2008-01-02..2025-06-27 -> out/randoms_old.txt / .json
    python3 -I randoms.py new     # the unseen months (once, at the end)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import books as B  # noqa: E402
from books import c2eval, np  # noqa: E402

NB = 1000
SEED = 20261021
LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def random_books(R, sub, t_lo, rng):
    keep = ~R['idbad'] & (R['t'] >= t_lo)
    if sub == '500':
        keep &= R['in500']
    t = R['t'][keep]
    h_all = R['rs'][keep] - R['beta'][keep].astype(np.float64) * R['rf'][keep]
    beta_all = R['beta'][keep].astype(np.float64)
    f_all = R['f'][keep]
    u, st, en = B.day_slices(t)
    gross = np.full((NB, len(u)), np.nan)
    tc = np.full((NB, len(u)), np.nan)
    for d, (a, b) in enumerate(zip(st, en)):
        n = b - a
        if n < B.MIN_N:
            continue
        nl = int(round(B.FRAC * n))
        h, beta, f = h_all[a:b], beta_all[a:b], f_all[a:b]
        keys = rng.random((NB, n))
        part = np.partition(keys, [nl - 1, 2 * nl - 1], axis=1)
        thrL, thrS = part[:, nl - 1:nl], part[:, 2 * nl - 1:2 * nl]
        mL = keys <= thrL
        mS = (keys > thrL) & (keys <= thrS)
        assert (mL.sum(1) == nl).all() and (mS.sum(1) == nl).all()
        gross[:, d] = (mL @ h - mS @ h) / nl
        uf, inv = np.unique(f, return_inverse=True)
        M = np.zeros((n, len(uf)))
        M[np.arange(n), inv] = 1.0
        W = (mS.astype(np.float64) - mL.astype(np.float64)) * (beta / nl)
        tc[:, d] = 2 * B.TWIN_SIDE * np.abs(W @ M).sum(1)
    ok = np.isfinite(gross[0])
    return u[ok], gross[:, ok], tc[:, ok]


def main(tag):
    t_start = time.time()
    R = B.load_rows(tag)
    days = R['days']
    first = '2008-01-01' if tag == 'old' else '2025-06-30'
    t_lo = c2eval.day_index(days, first)
    rng = np.random.default_rng(SEED)
    res = {}
    for sub in ('all', '500'):
        Bk = B.books(R, sub)
        m = Bk['t'] >= t_lo
        head = B.net(Bk, 'twin')[m]
        bound = B.net(Bk, 'twin', p=0)[m]
        th = Bk['t'][m]
        u, g, tc = random_books(R, sub, t_lo, rng)
        assert np.array_equal(u, th), 'random books and the headline book cover different days'
        rh = g - 2 * B.P_HEAD - tc   # random books, headline costs
        np.save(B.BIG / f'random_books_{tag}_{sub}.npy', rh.astype(np.float32))
        yr_head = 252 * head.mean() * 100
        yr_rand = 252 * np.nanmean(rh, 1) * 100
        med = float(np.median(yr_rand))
        p = float((yr_rand >= yr_head).mean())
        r = dict(head_pts=yr_head, random_median_pts=med, change=yr_head - med, p=p,
                 random_p5=float(np.percentile(yr_rand, 5)), random_p95=float(np.percentile(yr_rand, 95)),
                 random_gross_median_bps=float(np.median(np.nanmean(g, 1)) * 1e4),
                 random_tc_median_bps=float(np.median(np.nanmean(tc, 1)) * 1e4),
                 head_tc_bps=float(Bk['twin_tc'][m].mean() * 1e4),
                 bound_pts=252 * bound.mean() * 100, days=int(len(th)))
        say(f'\n== {tag}, {sub}: {c2eval.iso(days[th[0]])}..{c2eval.iso(days[th[-1]])}, {len(th)} days ==')
        say(f'headline book: {yr_head:+.1f} % a year (252 x mean daily); random books median {med:+.1f} '
            f'(5-95%: {r["random_p5"]:+.1f}..{r["random_p95"]:+.1f}); change over random {yr_head - med:+.1f} points; '
            f'share of random books at or above the headline: {p:.3f}')
        say(f'random books: median gross {r["random_gross_median_bps"]:+.2f} bps/day, median netted twin cost '
            f'{r["random_tc_median_bps"]:.2f} bps/day (headline book {r["head_tc_bps"]:.2f})')
        # per calendar year
        yrs = np.array([int(c2eval.iso(days[x])[:4]) for x in th])
        per = []
        comp = []
        for y in np.unique(yrs):
            k = yrs == y
            a = 252 * head[k].mean() * 100
            bmed = float(np.median(252 * np.nanmean(rh[:, k], 1) * 100))
            ch = np.prod(1 + head[k]) - 1
            cr = np.median(np.prod(1 + np.nan_to_num(rh[:, k]), 1) - 1)
            per.append(dict(year=int(y), days=int(k.sum()), head=a, random_median=bmed, diff=a - bmed,
                            head_compounded=float(ch * 100), random_compounded_median=float(cr * 100),
                            diff_compounded=float((ch - cr) * 100)))
            say(f'  {y} ({k.sum()} days): head {a:+8.1f}, random median {bmed:+8.1f}, change {a - bmed:+7.1f}; '
                f'compounded: head {ch * 100:+8.1f}%, random median {cr * 100:+7.1f}%, change {(ch - cr) * 100:+7.1f}')
        d = np.array([x['diff'] for x in per])
        dc = np.array([x['diff_compounded'] for x in per])
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float('nan')
        sec = float(dc.std(ddof=1) / np.sqrt(len(dc))) if len(dc) > 1 else float('nan')
        r.update(per_year=per, years=len(d), ahead=int((d > 0).sum()), mean_of_years=float(d.mean()), se_years=se,
                 compounded_mean_of_years=float(dc.mean()), compounded_se=sec)
        say(f'per-year mean change {d.mean():+.1f} points (se {se:.1f}), ahead in {(d > 0).sum()} of {len(d)}; '
            f'judge-style compounded per year: {dc.mean():+.1f} (se {sec:.1f})')
        res[sub] = r
    say(f'\ndone in {time.time() - t_start:.0f}s')
    (B.OUT / f'randoms_{tag}.txt').write_text('\n'.join(LINES) + '\n')
    (B.OUT / f'randoms_{tag}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
