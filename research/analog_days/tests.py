"""Tests 1-3, the baselines, by period, and the story check on the old store/export.

    python3 -I tests.py        # writes out/tests_old.txt and out/tests_old.json
"""
import json
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402

LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def row(label, x, scale=1e4):
    mu, t, n = C.tstat(x)
    return f'{label:<44} {scale * mu:+8.2f} {t:+6.2f} {n:5d}'


def disp(O):
    return float(O.std(0, ddof=1).mean())


def agree(O):
    D = O - O.mean(1, keepdims=True)
    U = D / np.linalg.norm(D, axis=1, keepdims=True)
    s = U.sum(0)
    n = len(O)
    return float((s @ s - n) / (n * (n - 1)))


def random_spaced(rng, cands, k=C.K, gap=C.GAP):
    order = rng.permutation(cands)
    acc = []
    taken = set()
    for a in order:
        if a in taken:
            continue
        acc.append(a)
        taken.update(range(a - gap + 1, a + gap))
        if len(acc) == k:
            break
    return np.array(acc) if len(acc) == k else None


def story(P, forms, n_sets=20, seed=7):
    """Dispersion and agreement of the analogs' next-week outcomes vs random spaced candidates; recency share."""
    rng = np.random.default_rng(seed)
    O, idx, ok = P['O'], P['idx'], P['ok']
    cand = np.nonzero(ok)[0]
    res = dict(d_an=[], d_rn=[], a_an=[], a_rn=[], recent=[], recent_rn=[])
    res_skipped = 0
    for t in forms:
        A = idx[t]
        n = int(np.searchsorted(cand, t - C.H, side='right'))
        pool = cand[:n]
        dr, ar, rr = [], [], []
        tries = 0
        while len(dr) < n_sets and tries < 20 * n_sets:
            tries += 1
            R = random_spaced(rng, pool)
            if R is None:       # a random order packs worse than nearest-first; early pools can't always fit 50
                continue
            dr.append(disp(O[R])); ar.append(agree(O[R])); rr.append(float(np.mean(R >= t - 63)))
        if len(dr) < n_sets:
            res['skipped'] = res.get('skipped', 0) + 1
            for k in ('d_an', 'd_rn', 'a_an', 'a_rn', 'recent', 'recent_rn'):
                res[k].append(np.nan)
            continue
        res['d_an'].append(disp(O[A])); res['d_rn'].append(float(np.mean(dr)))
        res['a_an'].append(agree(O[A])); res['a_rn'].append(float(np.mean(ar)))
        res['recent'].append(float(np.mean(A >= t - 63))); res['recent_rn'].append(float(np.mean(rr)))
    skipped = res.pop('skipped', 0)
    out = {k: np.array(v) for k, v in res.items()}
    out['skipped'] = skipped
    return out


def main():
    P = C.build('old')
    days = P['days']
    T = len(days)
    for n in P['notes']:
        say('data:', n)
    has = P['idx'][:, 0] >= 0
    f0 = C.first_formation(has, C.LAST_OLD)
    forms = np.array([f for f in range(f0, T) if f % C.STEP == 0 and f + C.H <= C.LAST_OLD])
    assert has[forms].all()
    say(f'formations {c2eval.iso(days[forms[0]])} .. {c2eval.iso(days[forms[-1]])} ({len(forms)} weeks); '
        f'first complete state {c2eval.iso(days[np.argmax(P["ok"])])}, first forecast {c2eval.iso(days[np.argmax(has)])}; '
        f'days without a forecast after it: {[c2eval.iso(days[d]) for d in np.nonzero(~has[np.argmax(has):C.LAST_OLD + 1])[0] + np.argmax(has)]}')
    yr = np.array([C.year_of(days, f) for f in forms])
    t1 = yr <= 2019
    t3 = yr >= 2020
    mom, rev = C.signals(P)
    runs = {'analog': C.weekly(P, forms, P['F']), 'momentum': C.weekly(P, forms, mom), 'reversal': C.weekly(P, forms, rev)}
    G, N = C.random_weekly(P, forms)
    res = {'forms': [c2eval.iso(days[f]) for f in forms], 'weekly': {k: {kk: v[kk].tolist() for kk in ('gross', 'net', 'turnover')}
                                                                       for k, v in runs.items()}}

    def block(mask, title):
        say(f'\n--- {title}: weeks formed {c2eval.iso(days[forms[mask][0]])} .. {c2eval.iso(days[forms[mask][-1]])} ---')
        say(f'{"":<44} {"bps/wk":>8} {"t":>6} {"n":>5}')
        out = {}
        for k, r in runs.items():
            for kind in ('gross', 'net'):
                say(row(f'{k} {kind}', r[kind][mask]))
                out[f'{k}|{kind}'] = C.tstat(r[kind][mask])
            say(f'{"":<44} turnover {r["turnover"][mask].mean():.2f}/wk, sd of net {1e4 * r["net"][mask].std(ddof=1):.1f} bps, '
                f'annualized net {52 * 100 * r["net"][mask].mean():+.2f}%, Sharpe {np.sqrt(52) * r["net"][mask].mean() / r["net"][mask].std(ddof=1):+.2f}')
        for base in ('momentum', 'reversal'):
            for kind in ('net', 'gross'):
                d = runs['analog'][kind][mask] - runs[base][kind][mask]
                say(row(f'analog - {base} ({kind}, paired)', d))
                out[f'diff|{base}|{kind}'] = C.tstat(d)
        gm = G[:, mask].mean(1) * 1e4
        nm = N[:, mask].mean(1) * 1e4
        a_g = runs['analog']['gross'][mask].mean() * 1e4
        a_n = runs['analog']['net'][mask].mean() * 1e4
        pg = float((gm >= a_g).mean())
        pn = float((nm >= a_n).mean())
        say(f'random (10,000): gross mean {np.median(gm):+.2f} bps/wk median (5-95%: {np.percentile(gm, 5):+.2f}..{np.percentile(gm, 95):+.2f}); '
            f'net median {np.median(nm):+.2f} (5-95%: {np.percentile(nm, 5):+.2f}..{np.percentile(nm, 95):+.2f})')
        say(f'analog vs random: gross p = {pg:.4f}, net p = {pn:.4f} (share of random strategies at least as good)')
        for b in ('momentum', 'reversal'):
            bn = runs[b]['net'][mask].mean() * 1e4
            say(f'{b} vs random: net p = {float((nm >= bn).mean()):.4f}')
        out['random'] = dict(gross_median=float(np.median(gm)), net_median=float(np.median(nm)), p_gross=pg, p_net=pn)
        return out

    res['test1'] = block(t1, 'Test 1 (first forecast .. 2019)')
    res['test3'] = block(t3, 'Test 3 (2020 .. 2025-06)')
    res['all'] = block(np.ones(len(forms), bool), 'All old-export weeks')
    ex = ~(((yr >= 2008) & (yr <= 2009)) | (yr == 2020))
    res['test1_ex'] = block(t1 & ex, 'Test 1 without 2008-09')
    res['test3_ex'] = block(t3 & ex, 'Test 3 without 2020')
    res['all_ex'] = block(ex, 'All without 2008-09 and 2020')

    # verdict
    m1, tt1, n1 = res['test1']['analog|net']
    say('\n=== Tests ===')
    p1 = tt1 >= 2
    say(f'Test 1: analog net {1e4 * m1:+.2f} bps/wk, t {tt1:+.2f} (n {n1}) -> {"PASS" if p1 else "FAIL"} (needs t >= 2)')
    dm = res['test1']['diff|momentum|net']; dr = res['test1']['diff|reversal|net']
    p2 = dm[1] >= 1.5 and dr[1] >= 1.5
    say(f'Test 2: vs momentum t {dm[1]:+.2f}, vs reversal t {dr[1]:+.2f} -> {"PASS" if p2 else "FAIL"} (needs t >= 1.5 against both)')
    m3 = res['test3']['analog|net']
    p3 = m3[0] > 0
    say(f'Test 3: 2020-2025-06 analog net {1e4 * m3[0]:+.2f} bps/wk, t {m3[1]:+.2f} -> {"PASS" if p3 else "FAIL"} (needs positive)')
    res['verdict'] = dict(test1=bool(p1), test2=bool(p2), test3=bool(p3))

    # by year and by block
    say('\n=== By period (bps/wk; t) ===')
    say(f'{"period":<12} {"weeks":>5} {"analog net":>14} {"momentum net":>14} {"reversal net":>14} {"an-mom":>14} {"an-rev":>14}')
    res['by_period'] = {}
    blocks = [(str(y), yr == y) for y in range(yr.min(), yr.max() + 1)]
    blocks += [('2007-2009', (yr >= 2007) & (yr <= 2009)), ('2010-2014', (yr >= 2010) & (yr <= 2014)),
               ('2015-2019', (yr >= 2015) & (yr <= 2019)), ('2020-2025-06', yr >= 2020)]
    for lab, m in blocks:
        a = C.tstat(runs['analog']['net'][m]); mo = C.tstat(runs['momentum']['net'][m]); rv = C.tstat(runs['reversal']['net'][m])
        dm_ = C.tstat(runs['analog']['net'][m] - runs['momentum']['net'][m])
        dr_ = C.tstat(runs['analog']['net'][m] - runs['reversal']['net'][m])
        f = lambda x: f'{1e4 * x[0]:+7.1f} ({x[1]:+5.2f})'
        say(f'{lab:<12} {int(m.sum()):5d} {f(a):>14} {f(mo):>14} {f(rv):>14} {f(dm_):>14} {f(dr_):>14}')
        res['by_period'][lab] = dict(weeks=int(m.sum()), analog=a, momentum=mo, reversal=rv, diff_mom=dm_, diff_rev=dr_)

    # diagnostics: forecast skill and overlap with the baselines
    ic = np.array([spearman(P['F'][f], P['O'][f]) for f in forms])
    cm = np.array([spearman(P['F'][f], mom[f]) for f in forms])
    cr = np.array([spearman(P['F'][f], rev[f]) for f in forms])
    say('\n=== Diagnostics ===')
    for lab, m in (('test 1', t1), ('test 3', t3)):
        a = C.tstat(ic[m])
        say(f'{lab}: weekly rank IC of forecast vs next-week sector-minus-SPY return {a[0]:+.4f} (t {a[1]:+.2f}); '
            f'rank corr of forecast with momentum {cm[m].mean():+.3f}, with reversal {cr[m].mean():+.3f}')
    res['diag'] = dict(ic_t1=C.tstat(ic[t1]), ic_t3=C.tstat(ic[t3]), corr_mom=float(cm.mean()), corr_rev=float(cr.mean()))
    # same legs as momentum/reversal: how often the analog book's long and short sets equal theirs
    for b in ('momentum', 'reversal'):
        same = (np.sign(runs['analog']['w']) == np.sign(runs[b]['w'])).all(1).mean()
        ov = ((runs['analog']['w'] > 0) & (runs[b]['w'] > 0)).sum(1).mean() + ((runs['analog']['w'] < 0) & (runs[b]['w'] < 0)).sum(1).mean()
        say(f'analog vs {b}: identical book {100 * same:.1f}% of weeks; shared names per week {ov:.2f} of 6')

    # story check
    say('\n=== Story check (weekly formations, old export) ===')
    S = story(P, forms)
    res['story'] = {'skipped_weeks': int(S['skipped'])}
    fin = np.isfinite(S['d_rn'])
    say(f'weeks skipped because 20 random spaced sets of 50 could not be drawn from the pool: {S["skipped"]} '
        f'(formed {[c2eval.iso(days[f]) for f in forms[~fin]][:3]} .. {[c2eval.iso(days[f]) for f in forms[~fin]][-1:]})')
    for lab, m in (('all', fin), ('test 1', t1 & fin), ('test 3', t3 & fin)):
        da, dn, aa, an = S['d_an'][m], S['d_rn'][m], S['a_an'][m], S['a_rn'][m]
        tdd = C.tstat(da - dn); tda = C.tstat(aa - an)
        say(f'{lab}: dispersion analogs {100 * da.mean():.3f}% vs random {100 * dn.mean():.3f}% (ratio {da.mean() / dn.mean():.3f}; '
            f'analogs tighter in {100 * (da < dn).mean():.1f}% of weeks; diff t {tdd[1]:+.2f})')
        say(f'{lab}: agreement (mean pairwise cosine) analogs {aa.mean():+.4f} vs random {an.mean():+.4f} '
            f'(analogs higher in {100 * (aa > an).mean():.1f}% of weeks; diff t {tda[1]:+.2f})')
        say(f'{lab}: share of analogs from the 63 days before t: {100 * S["recent"][m].mean():.1f}% '
            f'(random spaced sets: {100 * S["recent_rn"][m].mean():.1f}%)')
        res['story'][lab] = dict(disp_an=float(da.mean()), disp_rn=float(dn.mean()), tighter=float((da < dn).mean()), disp_t=tdd[1],
                                 agree_an=float(aa.mean()), agree_rn=float(an.mean()), agree_higher=float((aa > an).mean()),
                                 agree_t=tda[1], recent=float(S['recent'][m].mean()), recent_rn=float(S['recent_rn'][m].mean()))
    (C.OUT / 'tests_old.txt').write_text('\n'.join(LINES) + '\n')
    (C.OUT / 'tests_old.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
