"""Each month: text similarity of every grouped stock to every group, the expected-exposure fits, gaps, and the
classes (imposters, true members, undiscovered members). Also the frozen-group exposures 6 and 12 months later for
test 1, and the same classes made from odd-day exposures (test 1's guard against selecting on noise).

    python3 -I classes.py     # needs months/*.npz (groups.py), text.npz (text.py), map/cik_to_rows.json (cikmap.py)
                              # writes classes/YYYY-MM.npz in the work folder
"""
import json
import sys as _sys, os as _os
import time

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import K, MONTHS_DIR, ODD, WORK, YEAR, load, month_ends, residuals, np  # noqa: E402
import groups as G  # noqa: E402

OUT = WORK / ('classes_odd' if ODD else 'classes')
FIRST = '2006-02'
MIN_TEXT_MEMBERS = 5
NEAR_ZERO = 0.25      # |gap| <= this x sd counts as "near zero"


def epoch_of(day):
    """Expected latest fiscal year at a date: last year's from June 1, else the year before."""
    dt = np.datetime64('1970-01-01') + np.timedelta64(int(day), 'D')
    y = int(str(dt)[:4])
    return y - 1 if dt >= np.datetime64(f'{y}-06-01') else y - 2


class Text:
    def __init__(self):
        t = np.load(WORK / 'text.npz')
        self.cik, self.fy, self.avail = t['cik'], t['fy'], t['avail']
        self.indptr, self.indices = t['indptr'], t['indices']
        m = json.loads((WORK / 'map' / 'cik_to_rows.json').read_text())
        self.row_cik = {}
        for c, rows in m.items():
            for r in rows:
                self.row_cik.setdefault(r, int(c))
        self.by_cik = {}
        for f in np.argsort(self.fy, kind='stable'):
            self.by_cik.setdefault(int(self.cik[f]), []).append(int(f))
        self.cache_key, self.cache = None, None

    def filing(self, row, day):
        """Index of the row's valid latest filing at a day, or -1."""
        c = self.row_cik.get(int(row))
        if c is None:
            return -1
        exp = epoch_of(day)
        best = -1
        for f in self.by_cik.get(c, []):
            if self.avail[f] <= day and self.fy[f] >= exp - 1:
                best = f
        return best

    def gram(self, files):
        """Dot products of the unit word-set vectors of the given filings (cached per set)."""
        key = tuple(sorted(set(files)))
        if key == self.cache_key:
            return self.cache
        idx = {f: k for k, f in enumerate(key)}
        words = np.unique(np.concatenate([self.indices[self.indptr[f]:self.indptr[f + 1]] for f in key]))
        col = {w: k for k, w in enumerate(words)}
        x = np.zeros((len(key), len(words)), np.float32)
        for f in key:
            ws = self.indices[self.indptr[f]:self.indptr[f + 1]]
            x[idx[f], [col[w] for w in ws]] = 1.0 / np.sqrt(len(ws))
        p = x @ x.T
        self.cache_key, self.cache = key, (idx, p)
        return idx, p


def similarity(p, lab, has, k):
    """sim [n, k]: cosine of each stock's vector with the mean vector of each group's members with text (without
    the stock itself). NaN without text or for groups with too few texted members. p: Gram over the n stocks."""
    n = len(lab)
    oh = np.zeros((n, k))
    oh[np.arange(n), lab] = has
    a = p @ oh                                    # x_i . S_g
    q = np.einsum('ig,ij,jg->g', oh, p, oh)       # |S_g|^2
    cnt = oh.sum(0)
    sim = a / np.sqrt(np.maximum(q, 1e-12))[None, :]
    own = np.arange(n), lab
    num = a[own] - 1.0
    den = q[lab] - 2 * a[own] + 1.0
    sim[own] = np.where(den > 1e-12, num / np.sqrt(np.maximum(den, 1e-12)), np.nan)
    valid = cnt - 0 >= MIN_TEXT_MEMBERS
    sim[:, ~valid] = np.nan
    # own group needs MIN_TEXT_MEMBERS others
    sim[own] = np.where(cnt[lab] - has >= MIN_TEXT_MEMBERS, sim[own], np.nan)
    sim[has == 0] = np.nan
    # relative to the stock's mean similarity over all valid groups: removes the filing-length effect (a long
    # filing is close to every group's centroid)
    mean = np.nanmean(np.where(np.isfinite(sim), sim, np.nan), axis=1, keepdims=True)
    rel = sim / np.where(mean > 0, mean, np.nan)
    return rel, valid, sim


def fit_gap(beta, sim, lab):
    """Two OLS lines, exposure on similarity: own-group pairs and other pairs. Returns gap [n, k], fitted, coefs."""
    n, k = beta.shape
    own = np.zeros((n, k), bool)
    own[np.arange(n), lab] = True
    fitted = np.full((n, k), np.nan)
    coefs = []
    for mask in (own, ~own):
        m = mask & np.isfinite(sim) & np.isfinite(beta)
        x, y = sim[m], beta[m]
        b = np.cov(x, y, bias=True)[0, 1] / max(x.var(), 1e-12)
        a = y.mean() - b * x.mean()
        fitted[mask] = a + b * sim[mask]
        coefs.append((a, b, np.corrcoef(x, y)[0, 1], int(m.sum())))
    return beta - fitted, fitted, coefs


def make_classes(beta, sim, lab, gap, corr):
    """Classes from exposures, gaps and correlations. Returns dict of arrays."""
    n, k = beta.shape
    ok = np.isfinite(sim).any(1)
    b = np.where(np.isfinite(sim), corr, -np.inf)        # the group it moves with most: highest correlation
    gstar = np.where(ok, b.argmax(1), -1)
    gg = np.where(ok, gap[np.arange(n), np.maximum(gstar, 0)], np.nan)
    gg[~np.isfinite(gg)] = np.nan
    fin = np.isfinite(gg)
    imp = fin & (gg >= np.nanpercentile(gg, 90))
    sd = np.nanstd(gg)
    simstar = np.where(fin, sim[np.arange(n), np.maximum(gstar, 0)], np.nan)
    q75 = np.array([np.nanpercentile(sim[:, g], 75) if np.isfinite(sim[:, g]).any() else np.nan for g in range(k)])
    true = fin & (lab == gstar) & (np.abs(gg) <= NEAR_ZERO * sd) & (simstar >= q75[np.maximum(gstar, 0)])
    # undiscovered: not in g on the map, g not the group it moves with most, g its textually closest group
    # outside its map group, and top 5% similarity to g among non-members
    other = (lab[:, None] != np.arange(k)[None, :]) & (gstar[:, None] != np.arange(k)[None, :]) & np.isfinite(sim)
    q95 = np.array([np.nanpercentile(np.where(other[:, g], sim[:, g], np.nan), 95) if other[:, g].any() else np.inf
                    for g in range(k)])
    notown = np.where((lab[:, None] != np.arange(k)[None, :]) & np.isfinite(sim), sim, -np.inf)
    closest = np.zeros((n, k), bool)
    closest[np.arange(n), notown.argmax(1)] = True
    cand = other & closest & (sim >= q95[None, :])
    ci, cg = np.nonzero(cand)
    cgap = gap[ci, cg]
    und = cgap <= np.percentile(cgap, 10) if len(cgap) else np.zeros(0, bool)
    # control pools (test 1): gap near zero, same kind of pair
    ctrl_imp = fin & (np.abs(gg) <= NEAR_ZERO * sd)
    og = np.where(other, gap, np.nan)
    ctrl_und = other & (np.abs(og) <= NEAR_ZERO * np.nanstd(og))
    return dict(gstar=gstar, gstar_gap=gg, imp=imp, true=true, und_i=ci[und], und_g=cg[und],
                ctrl_imp=ctrl_imp, ctrl_und_i=np.nonzero(ctrl_und)[0], ctrl_und_g=np.nonzero(ctrl_und)[1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    e = load()
    ret, dv, days = e['ret'], np.asarray(e['dollar_volume'], dtype=np.float64), e['days']
    all_idx, all_ms = month_ends(days, first='2003-01', last='2025-05')
    pos = {str(m): i for i, m in enumerate(all_ms)}
    txt = Text()
    months = sorted(p.stem for p in MONTHS_DIR.glob('*.npz') if p.stem >= FIRST)
    # each month's filings first, so the Gram matrix is built once per text epoch over the filings it uses
    t0 = time.time()
    univ_cache = {}
    info = {}
    for m in months:
        g = np.load(MONTHS_DIR / f'{m}.npz')
        day = int(days[int(g['day'])])
        info[m] = (epoch_of(day), np.array([txt.filing(r, day) for r in g['stocks']]))
    used = {}
    for m, (ep, files) in info.items():
        used.setdefault(ep, set()).update(int(f) for f in files if f >= 0)
    for m in months:
        out = OUT / f'{m}.npz'
        if out.exists():
            continue
        g = np.load(MONTHS_DIR / f'{m}.npz')
        d_end, rows, lab = int(g['day']), g['stocks'], g['label']
        beta, b_odd, b_even = g['beta'].astype(float), g['beta_odd'].astype(float), g['beta_even'].astype(float)
        day = int(days[d_end])
        ep, files = info[m]
        has = files >= 0
        idx, pfull = txt.gram(sorted(used[ep]))
        sel = np.array([idx.get(int(f), 0) for f in files])
        p = pfull[np.ix_(sel, sel)].astype(float) * (has[:, None] & has[None, :])
        sim, valid, sim_raw = similarity(p, lab, has.astype(float), K)
        r, _, _ = residuals(ret, rows, d_end)
        corr = G.correlations(r, lab, K)
        corr_o = G.correlations(r, lab, K, cols=np.arange(YEAR) % 2 == 1)
        gap, fitted, coefs = fit_gap(beta, sim, lab)
        cl = make_classes(beta, sim, lab, gap, corr)
        gap_o, fitted_o, coefs_o = fit_gap(b_odd, sim, lab)
        clo = make_classes(b_odd, sim, lab, gap_o, corr_o)
        # frozen-group exposures 6 and 12 months later
        fut = {}
        for h in (6, 12):
            j = pos[m] + h
            if j >= len(all_idx):
                fut[h] = np.full_like(beta, np.nan)
                if h == 6:
                    fut['6n'] = np.full_like(beta, np.nan)
                continue
            d2 = all_idx[j]
            if d2 not in univ_cache:
                univ_cache[d2] = G.universe(ret, dv, d2)[0]
            r, _, _ = residuals(ret, rows, d2, univ_cache[d2])
            bf = G.exposures(r, lab, K)
            bf[np.isfinite(r).sum(1) < 0.8 * YEAR] = np.nan
            fut[h] = bf
            if h == 6:  # the 126 days after the classification date only (no overlap with its 252 days)
                r, _, _ = residuals(ret, rows, d2, univ_cache[d2], length=d2 - d_end)
                bf = G.exposures(r, lab, K)
                bf[np.isfinite(r).sum(1) < 0.8 * (d2 - d_end)] = np.nan
                fut['6n'] = bf
        # months since the filing became available; multi-segment flag
        age = np.where(has, (day - txt.avail[np.maximum(files, 0)]) / 30.44, np.nan)
        s2 = np.sort(np.where(np.isfinite(sim), sim, -np.inf), axis=1)
        multi = has & (s2[:, -2] >= 0.9 * s2[:, -1]) & np.isfinite(s2[:, -2])
        np.savez_compressed(out, day=d_end, rows=rows, label=lab, has=has, files=files, sim=sim.astype(np.float32),
                            sim_raw=sim_raw.astype(np.float32), corr=corr.astype(np.float32),
                            valid=valid, beta=beta.astype(np.float32), gap=gap.astype(np.float32),
                            fitted=fitted.astype(np.float32), coefs=np.array(coefs),
                            gap_odd=gap_o.astype(np.float32), fitted_odd=fitted_o.astype(np.float32),
                            coefs_odd=np.array(coefs_o), beta_even=b_even.astype(np.float32),
                            beta_odd=b_odd.astype(np.float32), beta6=fut[6].astype(np.float32),
                            beta12=fut[12].astype(np.float32), beta6n=fut['6n'].astype(np.float32), age=age.astype(np.float32), multi=multi,
                            beta_mkt=g['beta_mkt'], **cl, **{f'odd_{a}': b for a, b in clo.items()})
        print(f'{m}: {has.sum()}/{len(rows)} with text, {valid.sum()} groups valid; own fit b={coefs[0][1]:.2f} '
              f'r={coefs[0][2]:.2f}, other b={coefs[1][1]:.2f} r={coefs[1][2]:.2f}; imposters {cl["imp"].sum()}, '
              f'true {cl["true"].sum()}, undiscovered {len(cl["und_i"])}; {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
