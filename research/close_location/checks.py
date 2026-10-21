"""Code checks for close location (no outcome of P is looked at).

    python3 -I checks.py     # -> out/checks.txt

1. P for random (week, stock) rows recomputed by hand from the raw store rows and the twin of each day.
2. The book's legs are exactly the top and bottom round(10% n) of P.
3. An oracle signal (P := the next 5-day return) gives a huge coefficient and book.
4. A planted effect y = 0.01 x rank(P) + noise is recovered by the regression.
5. Turnover: a book unchanged with zero returns trades nothing after the build.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import tests as TS  # noqa: E402
np = C.np
c2eval = C.c2eval

LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def raw_hlc(S, sid, day):
    a, b = S['offs'][sid], S['offs'][sid + 1]
    d = S['date'][a:b]
    k = np.searchsorted(d, day)
    if k < len(d) and d[k] == day:
        return float(S['high'][a + k]), float(S['low'][a + k]), float(S['close'][a + k])
    return None


def hand_clv(h):
    if h is None:
        return None
    H, L, Cc = h
    if not (H > 0 and L > 0 and Cc > 0) or (H - L) / Cc > 0.5 or H < Cc or L > Cc or H == L:
        return None
    return ((Cc - L) - (H - Cc)) / (H - L)


def main():
    R = C.load_rows('old')
    days = R['days']
    E = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
    tw = c2eval.twins(E)
    tw.pop('F')
    S = c2eval.load_store(c2eval.STORE_OLD)
    rng = np.random.default_rng(7)
    pick = rng.choice(len(R['t']), 300, replace=False)
    worst, nchk = 0.0, 0
    for r in pick:
        d, i = int(R['t'][r]), int(R['i'][r])
        diffs = []
        for s in range(d - 4, d + 1):
            k = c2eval.rebal_index(tw, s)
            f = int(tw['fund_row'][k, i])
            cs = hand_clv(raw_hlc(S, int(E['ids'][i]), int(days[s])))
            cf = hand_clv(raw_hlc(S, int(tw['fund'][f]), int(days[s]))) if f >= 0 else None
            if cs is not None and cf is not None:
                diffs.append(np.float32(cs) - np.float32(cf))
        p = np.mean(diffs) if len(diffs) >= 4 else np.nan
        worst = max(worst, abs(p - float(R['P'][r])))
        nchk += 1
    say(f'1. hand P vs code on {nchk} random rows: max abs difference {worst:.2e}')
    # 2. legs
    t = R['t']
    d0 = np.unique(t)[300]
    m = t == d0
    P, ii = R['P'][m], R['i'][m]
    n = int(m.sum())
    q = int(round(0.1 * n))
    order = np.lexsort((ii, P))
    hi, lo = set(ii[order[n - q:]]), set(ii[order[:q]])
    thr_hi, thr_lo = np.sort(P)[n - q], np.sort(P)[q - 1]
    ok = all(P[ii == k][0] >= thr_hi for k in hi) and all(P[ii == k][0] <= thr_lo for k in lo) and len(hi) == len(lo) == q
    say(f'2. week {c2eval.iso(days[d0])}: n {n}, legs {q} each = round(10% n); legs are the top / bottom by P: {ok}')
    # 3. oracle
    Ro = dict(R)
    Ro['P'] = (R['y'] - R['m5']).astype(np.float32)
    sel = (R['t'] >= c2eval.day_index(days, '2010-01-01')) & (R['t'] < c2eval.day_index(days, '2012-01-01'))
    Wo = TS.weekly(Ro, sel)
    m_all = np.ones(len(Wo['d']), bool)
    so = TS.stats(Wo, m_all, +1)
    say(f"3. oracle 2010-2011: P coef {100 * so['b_P']:+.2f}% a week (t {so['t_P']:+.1f}); book gross {100 * so['gross']:+.2f}% "
        f"(t {so['t_gross']:+.1f}), net {100 * so['net']:+.2f}%")
    # 4. planted effect
    Rp = {k: v[sel] for k, v in R.items() if k != 'days'} | {'days': days}
    tt = Rp['t']
    yy = np.empty(len(tt))
    for d in np.unique(tt):
        mm = tt == d
        yy[mm] = 0.01 * C.rank01(Rp['P'][mm].astype(np.float64)) + rng.normal(0, 0.05, mm.sum())
    Rp['y'] = yy
    Rp['m5'] = np.zeros(len(tt))
    Wp = TS.weekly(Rp)
    sp = TS.stats(Wp, np.ones(len(Wp['d']), bool), +1)
    say(f"4. planted 1.000% top-minus-bottom: recovered {100 * sp['b_P']:+.3f}% (t {sp['t_P']:+.1f}); raw {100 * sp['raw_P']:+.3f}%")
    # 5. turnover
    w = np.unique(tt)[:3]
    mm = np.isin(tt, w)
    Rz = {k: v[mm] for k, v in Rp.items() if k != 'days'} | {'days': days}
    first = Rz['t'] == w[0]
    keep_i = Rz['i'][first]
    rows = []
    for d in w:
        rows.append({'t': np.full(len(keep_i), d, np.int32), 'i': keep_i, 'P': Rz['P'][first], 'Pst': Rz['Pst'][first],
                     'r1': Rz['r1'][first], 'r5': Rz['r5'][first], 'y': np.zeros(len(keep_i)), 'm5': np.zeros(len(keep_i)),
                     'in500': Rz['in500'][first], 'f': Rz['f'][first]})
    Rt = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]} | {'days': days}
    Wt = TS.weekly(Rt)
    say(f"5. unchanged book, zero returns: traded per week {[round(x, 6) for x in Wt['traded']]} (2 to build, then 0)")
    C.OUT.mkdir(exist_ok=True)
    (C.OUT / 'checks.txt').write_text('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
