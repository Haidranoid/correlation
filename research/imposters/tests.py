"""Tests 1-5 of experiments/+0.1-imposters.md, from classes/*.npz (classes.py).

    python3 -I tests.py      # prints the tables; writes tests_out.json next to this script
"""
import json
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import HERE, K, ODD, WORK, load, week_ends, np  # noqa: E402

CL = WORK / ('classes_odd' if ODD else 'classes')
MAIN = ('2006-02', '2019-12')
LATE = ('2020-01', '2022-05')
HORIZONS = (1, 2, 4)


def nw_t(x, lags):
    """Mean, Newey-West standard error and t of a time series."""
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, n
    m = x.mean()
    u = x - m
    s = (u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1 - l / (lags + 1)
        s += 2 * w * (u[l:] @ u[:-l]) / n
    se = np.sqrt(max(s, 0) / n)
    return m, se, m / se if se > 0 else np.nan, n


def months_in(rng):
    return sorted(p.stem for p in CL.glob('*.npz') if rng[0] <= p.stem <= rng[1])


# ---------------------------------------------------------------- test 1
def test1(rng):
    """Does the gap close? Classes from odd-day exposures; start = even-day exposure; frozen groups."""
    res = {}
    for arm in ('imp', 'und'):
        for h in (6, '6n', 12):
            lag = 12 if h == 12 else 6
            diffs, share, n_cls = [], [], 0
            for m in months_in(rng):
                c = np.load(CL / f'{m}.npz')
                be, bf, fo = c['beta_even'].astype(float), c[f'beta{h}'].astype(float), c['fitted_odd'].astype(float)
                if arm == 'imp':
                    ii = np.nonzero(c['odd_imp'])[0]
                    gi = c['odd_gstar'][ii]
                    cp = np.nonzero(c['odd_ctrl_imp'])[0]
                    cg = c['odd_gstar'][cp]
                    # population for the remaining-gap share: every stock's pair with the group it moves with most
                    pop_i = np.nonzero(c['odd_gstar'] >= 0)[0]
                    pop_g = c['odd_gstar'][pop_i]
                else:
                    ii, gi = c['odd_und_i'], c['odd_und_g']
                    cp, cg = c['odd_ctrl_und_i'], c['odd_ctrl_und_g']
                    pop_i, pop_g = cp, cg  # filled below with all other pairs
                    n = len(c['rows'])
                    gs = c['odd_gstar']
                    other = (c['label'][:, None] != np.arange(K)[None, :]) & (gs[:, None] != np.arange(K)[None, :]) \
                        & np.isfinite(c['sim'])
                    pop_i, pop_g = np.nonzero(other)
                if len(ii) == 0 or len(cp) == 0:
                    continue
                s0, s1 = be[ii, gi], bf[ii, gi]
                c0, c1 = be[cp, cg], bf[cp, cg]
                okc = np.isfinite(c0) & np.isfinite(c1)
                c0, c1 = c0[okc], c1[okc]
                ok = np.isfinite(s0) & np.isfinite(s1)
                if ok.sum() == 0 or len(c0) == 0:
                    continue
                order = np.argsort(c0)
                pos = np.clip(np.searchsorted(c0[order], s0[ok]), 1, len(c0) - 1)
                left, right = order[pos - 1], order[pos]
                near = np.where(np.abs(c0[left] - s0[ok]) <= np.abs(c0[right] - s0[ok]), left, right)
                d = (s1[ok] - s0[ok]) - (c1[near] - c0[near])
                diffs.append(d.mean())
                n_cls += ok.sum()
                # remaining share of the gap (relative to all pairs of the same kind)
                p0 = be[pop_i, pop_g] - fo[pop_i, pop_g]
                p1 = bf[pop_i, pop_g] - fo[pop_i, pop_g]
                g0 = s0[ok] - fo[ii, gi][ok] - np.nanmean(p0)
                g1 = s1[ok] - fo[ii, gi][ok] - np.nanmean(p1)
                share.append((g1.mean(), g0.mean()))
            mean, se, t, n = nw_t(diffs, lag)
            sh = np.array(share)
            r = sh[:, 0].sum() / sh[:, 1].sum() if len(sh) else np.nan
            # t of (1 - share): per-month remaining gap minus starting gap, scaled by the mean starting gap
            per = (sh[:, 0] - sh[:, 1]) / np.abs(sh[:, 1].mean()) if len(sh) else []
            m2, se2, t2, _ = nw_t(per, lag)
            res[f'{arm}_{h}'] = dict(diff=mean, se=se, t=t, months=n, pairs=int(n_cls), start_gap=float(sh[:, 1].mean()),
                                     end_gap=float(sh[:, 0].mean()), remaining=float(r), remaining_t=float(t2))
    return res


# ---------------------------------------------------------------- daily blocks for tests 2-4
class Blocks:
    def __init__(self):
        e = load()
        self.ret, self.days = e['ret'], e['days']
        self.T = len(self.days)
        self.we = week_ends(self.days)
        self.month_list = sorted(p.stem for p in CL.glob('*.npz'))

    def month(self, m):
        c = np.load(CL / f'{m}.npz')
        d0 = int(c['day'])
        i = self.month_list.index(m)
        if i + 1 < len(self.month_list):
            d1 = int(np.load(CL / f'{self.month_list[i + 1]}.npz')['day'])
        else:  # last month: through the next calendar month end
            dt = (np.datetime64('1970-01-01') + self.days.astype('timedelta64[D]')).astype('datetime64[M]')
            nxt = np.nonzero(dt == dt[d0] + 1)[0]
            d1 = int(nxt[-1]) if len(nxt) else min(d0 + 21, self.T - 1)
        a, b = d0 - 320, min(d1 + 70, self.T - 1)
        rows = c['rows']
        r = self.ret[rows, a:b + 1]
        mkt = np.nanmean(r, axis=0)
        x = r - c['beta_mkt'].astype(float)[:, None] * mkt[None, :]
        lab = c['label']
        ok = np.isfinite(x)
        x0 = np.where(ok, x, 0.0)
        oh = np.zeros((len(rows), K))
        oh[np.arange(len(rows)), lab] = 1
        S, C = oh.T @ x0, oh.T @ ok
        return dict(c=c, d0=d0, d1=d1, a=a, b=b, x=x0, ok=ok, S=S, C=C, lab=lab)

    def weeks(self, blk):
        return [w for w in range(1, len(self.we)) if blk['d0'] < self.we[w] <= blk['d1']]


def group_series(blk, g, i=None):
    """Daily market-relative return of group g over the block (without stock i if it is a member)."""
    S, C = blk['S'][g].copy(), blk['C'][g].copy()
    if i is not None and blk['lab'][i] == g:
        S -= blk['x'][i]
        C -= blk['ok'][i]
    return np.where(C > 0, S / np.maximum(C, 1), 0.0)


def fm(rows_by_week, k):
    """Fama-MacBeth: per week OLS of y on X; returns per-coefficient (mean, se, t, weeks)."""
    coefs = []
    for X, y in rows_by_week:
        if len(y) < 10:
            continue
        X1 = np.column_stack([np.ones(len(y)), X])
        try:
            b = np.linalg.lstsq(X1, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        coefs.append(b[1:])
    coefs = np.array(coefs)
    return [nw_t(coefs[:, j], max(k - 1, 0)) for j in range(coefs.shape[1])] if len(coefs) else []


def tests234(bl, rng, split_age=False):
    t2 = {k: [] for k in HORIZONS}
    t2_age = {(k, a): [] for k in HORIZONS for a in range(3)}
    t2_multi = {(k, mlt): [] for k in HORIZONS for mlt in (0, 1)}
    t3 = {k: [] for k in HORIZONS}
    ev = []  # test 4: (month, h21, h42, h63) per event
    for m in months_in(rng):
        blk = bl.month(m)
        c, a = blk['c'], blk['a']
        imp_i = np.nonzero(c['imp'])[0]
        imp_g = c['gstar'][imp_i]
        und_i, und_g = c['und_i'], c['und_g']
        beta = c['beta'].astype(float)
        age = c['age']
        multi = c['multi']
        cs = np.cumsum(np.concatenate([np.zeros((len(c['rows']), 1)), blk['x']], axis=1), axis=1)  # cs[:, t-a+1]

        def span(i, s, e):  # sum of x_i over days (s, e]
            return cs[i, e - a + 1] - cs[i, s - a + 1]

        for w in bl.weeks(blk):
            t0, tw = bl.we[w - 1], bl.we[w]
            fwd = {k: bl.we[w + k] for k in HORIZONS if w + k < len(bl.we) and bl.we[w + k] <= blk['b']}
            # test 2
            X, Y = [], {k: [] for k in fwd}
            ages, mults = [], []
            for i, g in zip(imp_i, imp_g):
                gs = group_series(blk, g, i)
                R = gs[t0 - a + 1:tw - a + 1].sum()
                own = span(i, t0, tw)
                X.append((beta[i, g] * R, own, R))
                for k, e in fwd.items():
                    Y[k].append(span(i, tw, e))
                ages.append(age[i])
                mults.append(multi[i])
            X = np.array(X)
            ages, mults = np.array(ages), np.array(mults)
            for k in fwd:
                y = np.array(Y[k])
                t2[k].append((X, y))
                if split_age and len(y):
                    for ai, (lo, hi) in enumerate(((0, 4), (4, 8), (8, 99))):
                        mk = (ages >= lo) & (ages < hi)
                        t2_age[(k, ai)].append((X[mk], y[mk]))
                    for mlt in (0, 1):
                        mk = mults == bool(mlt)
                        t2_multi[(k, mlt)].append((X[mk], y[mk]))
            # test 3
            X3, Y3 = [], {k: [] for k in fwd}
            gser = {}
            for i, g in zip(und_i, und_g):
                if g not in gser:
                    gser[g] = group_series(blk, g)
                R = gser[g][t0 - a + 1:tw - a + 1].sum()
                X3.append((R, span(i, t0, tw)))
                for k, e in fwd.items():
                    Y3[k].append(span(i, tw, e))
            for k in fwd:
                t3[k].append((np.array(X3), np.array(Y3[k])))
        # test 4: theme cools
        true_i = np.nonzero(c['true'])[0]
        for g in range(K):
            gi = imp_i[imp_g == g]
            gt = true_i[c['gstar'][true_i] == g]
            if len(gi) == 0 or len(gt) == 0:
                continue
            gs = group_series(blk, g)
            s21 = np.convolve(gs, np.ones(21), 'full')[:len(gs)]  # s21[t] = sum gs[t-20..t]
            for t in range(blk['d0'] + 1, blk['d1'] + 1):
                ti = t - a
                if ti < 252 + 42 or s21[ti] >= 0 or s21[ti - 1] < 0:
                    continue
                hot = False
                for tau in range(ti - 21, ti):
                    p90 = np.percentile(s21[tau - 252:tau], 90)
                    if s21[tau] >= p90:
                        hot = True
                        break
                if not hot:
                    continue
                if t + 63 > blk['b']:
                    break
                # beta-matched true members (nearest exposure to g)
                outs = []
                for h in (21, 42, 63):
                    dif = []
                    for i in gi:
                        j = gt[np.argmin(np.abs(beta[gt, g] - beta[i, g]))]
                        dif.append(span(i, t, t + h) - span(j, t, t + h))
                    outs.append(float(np.mean(dif)))
                ev.append((m, g, *outs))
                break
    out = {'t2': {}, 't3': {}, 't4': {}}
    for k in HORIZONS:
        r = fm(t2[k], k)
        out['t2'][k] = dict(zip(('group_driven', 'own', 'group'), r))
        r = fm(t3[k], k)
        out['t3'][k] = dict(zip(('group', 'own'), r))
        if split_age:
            for ai in range(3):
                r = fm(t2_age[(k, ai)], k)
                out['t2'][f'{k}_age{ai}'] = r[0] if r else None
            for mlt in (0, 1):
                r = fm(t2_multi[(k, mlt)], k)
                out['t2'][f'{k}_multi{mlt}'] = r[0] if r else None
    if ev:
        evs = np.array([e[2:] for e in ev], float)
        mons = [e[0] for e in ev]
        for hi, h in enumerate((21, 42, 63)):
            # clustered by month: average events within a month, then t over months
            per = {}
            for mo, v in zip(mons, evs[:, hi]):
                per.setdefault(mo, []).append(v)
            series = [np.mean(v) for _, v in sorted(per.items())]
            mean, se, t, n = nw_t(series, 0)
            out['t4'][h] = dict(mean=mean, se=se, t=t, months=n, events=len(ev))
    return out


def main():
    res = {}
    for name, rng in (('main', MAIN), ('late', LATE)):
        res[f'test1_{name}'] = test1(rng)
        print(name, 'test 1', json.dumps(res[f'test1_{name}'], indent=1, default=float), flush=True)
    if ODD:  # the odd-day groups map is only a robustness check of test 1
        (HERE / 'tests_odd_out.json').write_text(json.dumps(res, indent=1, default=float))
        return
    bl = Blocks()
    for name, rng in (('main', MAIN), ('late', LATE)):
        res[f'tests234_{name}'] = tests234(bl, rng, split_age=(name == 'main'))
        print(name, 'tests 2-4', json.dumps(res[f'tests234_{name}'], indent=1, default=float), flush=True)
    (HERE / 'tests_out.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
