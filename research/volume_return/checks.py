"""Code checks before any result (plan: "Checks before any result").

    checks.py basic  -> out/checks.txt       (hand recomputation, planted effects, oracles)
    checks.py null   -> out/checks_null.txt  (R1 with V shuffled within each stock-year: 200 times, 1,000 draws each)

Planted and oracle checks replace the outcome; they never read the real relation between V*R and the next return.
"""
import sys
import time

sys.path.insert(0, '/home/bhoult/programming/correlation/research/volume_return')
import numpy as np  # noqa: E402

import common as C  # noqa: E402
import m1  # noqa: E402
import r1  # noqa: E402

c2eval = C.c2eval


def basic():
    rep = C.OUT / 'checks.txt'
    rep.write_text('')
    t_start = time.time()
    E = C.export('old')
    days = E['days']
    R, V, cal = C.load_panels('old')
    R = np.array(R)
    V = np.array(V)
    N, T = R.shape
    rng = np.random.default_rng(1)

    # 1. V by hand
    C.log(rep, '== 1. V recomputed by hand (plain loops) at 8 random finite stock-days ==')
    lnsv_all = C.share_volume(E['close'], E['dollar_volume'])
    brk = np.load(C.tagdir('old') / 'broken.npy', mmap_mode='r')
    fin = np.argwhere(np.isfinite(V[:, 3000:3600]))
    worst = 0.0
    for i, t in fin[rng.choice(len(fin), 8, replace=False)]:
        t = t + 3000
        vals = [lnsv_all[i, u] for u in range(t - 200, t) if np.isfinite(lnsv_all[i, u]) and not brk[i, u]]
        x = lnsv_all[i, t] - sum(vals) / len(vals)
        x = max(-5.0, min(5.0, x))
        worst = max(worst, abs(x - float(V[i, t])))
        C.log(rep, f'{E["tickers"][i]} {c2eval.iso(days[t])}: by hand {x:+.6f}, panel {float(V[i, t]):+.6f}')
    C.log(rep, f'largest difference {worst:.2e} (float32 storage)')

    # 2. one stock-year's C2 by lstsq
    C.log(rep, '', '== 2. One stock-year (2012) by numpy lstsq against R1\'s block sums ==')
    a, b = C.day_range(days, '2012-01-01', '2012-12-31')
    blk = r1.period_blocks(R, V, cal, 'UW', a, b, r1.d_paper, C.MIN_PAIRS)
    coef = r1.solve(blk['XX'].sum(1), blk['Xy'].sum(1))
    for j in rng.choice(len(blk['rows']), 3, replace=False):
        i = blk['rows'][j]
        ts = np.arange(a, b + 1)
        r, v, y = R[i, ts].astype(float), V[i, ts].astype(float), R[i, ts + 1].astype(float)
        ok = np.isfinite(r) & np.isfinite(v) & np.isfinite(y)
        X = np.column_stack([np.ones(ok.sum()), r[ok], v[ok] * r[ok]])
        ref = np.linalg.lstsq(X, y[ok], rcond=None)[0]
        C.log(rep, f'{E["tickers"][i]}: lstsq C1 {ref[1]:+.6f} C2 {ref[2]:+.6f}; blocks C1 {coef[j, 1]:+.6f} '
                   f'C2 {coef[j, 2]:+.6f}; pairs {int(ok.sum())}')

    # 3. planted R1: y = c(third) V R + noise, c = +0.03 / 0 / -0.01 by thirds of G on the calendar day
    C.log(rep, '', '== 3. Planted R1: target = c V(t) R(t) + N(0, 0.02^2), c = +0.03 low DV, 0 middle, -0.01 high ==')
    Y = np.full((N, T), np.nan, np.float32)
    cvec = np.array([0.03, 0.0, -0.01])
    for t in range(int(cal['rebal'][0]), T):
        k = C.cal_index(cal['rebal'], t)
        g = C.thirds_of(cal['GW'][k].astype(float))
        ok = g >= 0
        Y[ok, t] = cvec[g[ok]] * V[ok, t] * R[ok, t] + rng.normal(0, 0.02, ok.sum())
    dev = r1.periods_for(days, 'dev')
    res = r1.family(R, V, cal, dev, 'UW', 'GW', draws=1000, Y=Y)
    s = res['stats']['C2_delta']
    C.log(rep, f'planted Delta 0.04; recovered {s["est"]:+.4f}, bootstrap t {s["t"]:+.1f}')
    res = r1.family(R, V, cal, dev, 'UW', 'GW', draws=1000, Y=np.where(np.isfinite(Y), 0.02 * rng.normal(size=Y.shape), np.nan))
    s = res['stats']['C2_delta']
    C.log(rep, f'pure noise target: Delta {s["est"]:+.4f}, bootstrap t {s["t"]:+.2f}')
    del Y

    # 4. planted M1: R'(t) = R(t) + c(third) V(t-2) R(t-2), c = +0.10 / 0 / -0.10, on U_J
    C.log(rep, '', '== 4. Planted M1: R\'(t) = R(t) + c V(t-2) R(t-2), c = +0.10 low, 0, -0.10 high (U_J) ==')
    Rp = R.copy()
    cvec = np.array([0.10, 0.0, -0.10])
    for t in range(int(cal['rebal'][0]) + 2, T):
        k = C.cal_index(cal['rebal'], t - 2)
        g = C.thirds_of(cal['GJ'][k].astype(float))
        ok = (g >= 0) & np.isfinite(V[:, t - 2]) & np.isfinite(R[:, t - 2]) & np.isfinite(R[:, t])
        Rp[ok, t] = R[ok, t] + cvec[g[ok]] * V[ok, t - 2] * R[ok, t - 2]
    s0 = c2eval.day_index(days, C.SCORE_START)
    a, b = C.day_range(days, *C.DEV)
    dec = np.arange(s0, b + 1)
    Y2 = lambda t: np.asarray(Rp[:, t + 2], np.float64)  # noqa: E731
    M = m1.run_model(Rp, V, cal, 'UJ', 'GJ', Y2, 2, 1, 1, dec, s0)
    D = M['bf']['gross'] - M['bb']['gross']
    st = m1.series_stats(D, draws=1000)
    C.log(rep, f'planted: D {st["est"] * 1e4:+.2f} bps a day, bootstrap t {st["t"]:+.1f}; fitted c by third in 2015: '
               + ', '.join(f'{x:+.3f}' for x in M['kf'][C.cal_index(cal['rebal'], C.day_range(days, "2015-01-01", "2015-01-31")[0])][:, 3]))
    del Rp, M

    # 5. oracle books
    C.log(rep, '', '== 5. Oracles ==')
    S = np.full((N, T), np.nan, np.float32)
    for t in dec:
        k = C.cal_index(cal['rebal'], t)
        u = cal['UJ'][k]
        S[u, t] = R[u, t + 2]
    bo = m1.book(S, R, dec, 1, 1)
    so = m1.series_stats(bo['gross'], draws=1000)
    C.log(rep, f'M1 book on the score R(t+2) itself: {so["est"] * 1e4:+.1f} bps a day, bootstrap t {so["t"]:+.1f}')
    C.log(rep, '', f'done in {time.time() - t_start:.0f} s')


def null():
    rep = C.OUT / 'checks_null.txt'
    rep.write_text('')
    t_start = time.time()
    E = C.export('old', arrays=())
    days = E['days']
    R, V, cal = C.load_panels('old')
    R = np.array(R)
    V = np.array(V)
    dev = r1.periods_for(days, 'dev')
    ts_ = []
    shuf = np.random.default_rng(7)
    C.log(rep, '== R1 with V shuffled across days within each stock-year: 200 shuffles, 1,000 draws each ==')
    for i in range(200):
        res = r1.family(R, V, cal, dev, 'UW', 'GW', draws=1000, seed=C.SEED + 100 + i, shuffle=shuf)
        s = res['stats']['C2_delta']
        ts_.append(s['t'])
        if i % 20 == 19:
            C.log(rep, f'{i + 1} shuffles ({time.time() - t_start:.0f} s): t so far: share >= 1.645 '
                       f'{np.mean(np.array(ts_) >= 1.645):.3f}, share >= 2 {np.mean(np.array(ts_) >= 2):.3f}, '
                       f'largest {max(ts_):+.2f}')
    t = np.array(ts_)
    C.log(rep, f'final: mean t {t.mean():+.2f}, sd {t.std(ddof=1):.2f}; share t >= 1.645: {np.mean(t >= 1.645):.3f} '
               f'(5% expected); share t >= 2: {np.mean(t >= 2):.3f}; t >= 3: {int((t >= 3).sum())}; '
               f'|t| >= 1.96: {np.mean(np.abs(t) >= 1.96):.3f}', f'done in {time.time() - t_start:.0f} s')


if __name__ == '__main__':
    {'basic': basic, 'null': null}[sys.argv[1]]()
