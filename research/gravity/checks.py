"""Gravity: code checks before any outcome is computed.

    python -I -B checks.py old

1. Look-ahead, by truncation: for 12 stocks, each cut at a random day, the panel is rebuilt from the raw data through
   the cut (universe, sigma, z, attractors, masses, beta) and every candidate's and benchmark's force (110 variants,
   real and random) recomputed for the refits in force over the 2 refits up to the cut. Forces, sigma, z and the
   attractor tables must be identical to the full run's.
2. Kernels against slow versions: for 40 random universe stock-days, every force recomputed in plain Python from the
   panel and the attractor tables; sigma, z and beta_M recomputed in plain Python; for 6 stock-refits the attractor
   selection (G1's rho and beta, and each G2 kernel's m) recomputed pair by pair with numpy.corrcoef.
3. Hand prints: a few stock-days with the attractors, gaps, closest approach and force printed in full.
Writes out/checks_old.txt / .json.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import data  # noqa: E402
import forces  # noqa: E402
import peers  # noqa: E402
np = K.np

VARS = K.variants()
KEYS = [v['key'] for v in VARS]
GK_C = 2 * math.log(2) - 1


def load_forces(tag, keys=KEYS, mmap=True):
    d = K.BIG / f'forces_{tag}'
    return {k: np.load(d / f'{k}.npy', mmap_mode='r' if mmap else None) for k in keys}


def isnan(x):
    return x != x


# ---------------------------------------------------------------------------------------------------------------
# Slow (plain Python) recomputation

def slow_gap(P, b, a, beta, tau, t):
    lnC, lnH, lnL = P['lnC'], P['lnH'], P['lnL']
    xs = []
    for tp in range(t - 63, t):
        if tp - tau < 0 or tp < 0:
            continue
        cb, ca = float(lnC[b, tp]), float(lnC[a, tp - tau])
        if isnan(cb) or isnan(ca):
            continue
        xs.append(cb - beta * ca)
    if len(xs) < 57:
        return None
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    if not var > 0:
        return None
    sd = math.sqrt(var)
    ca = float(lnC[a, t - tau])
    cb, hb, lb = float(lnC[b, t]), float(lnH[b, t]), float(lnL[b, t])
    if isnan(ca) or isnan(cb) or isnan(hb) or isnan(lb):
        return None
    gc = (cb - beta * ca - mu) / sd
    gh = (hb - beta * ca - mu) / sd
    gl = (lb - beta * ca - mu) / sd
    r = 0.0 if gl <= 0 <= gh else min(abs(gl), abs(gh))
    return gc, gh, gl, r, mu, sd


def sign(x):
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


def attractors(PE, world, k, i, stage, q=None):
    if stage in ('G1', 'G3'):
        if world == 'real':
            return [(int(PE['A1'][k, i, s]), float(PE['M1'][k, i, s]), float(PE['B1'][k, i, s])) for s in range(K.K_ATT)]
        return [(int(PE['AR'][k, i, s]), max(float(PE['MR'][k, i, s]), 0.0) if not isnan(float(PE['MR'][k, i, s])) else float('nan'),
                 float(PE['BR'][k, i, s])) for s in range(K.K_ATT)]
    if world == 'real':
        return [(int(PE['A2'][q, k, i, s]), float(PE['M2'][q, k, i, s]), None) for s in range(K.K_ATT)]
    return [(int(PE['AR'][k, i, s]), float(PE['M2R'][q, k, i, s]), None) for s in range(K.K_ATT)]


def slow_forces(P, PE, k, i, b, t):
    """Every variant's force for stock b (universe position i at refit k) on day t, in plain Python."""
    out = {}
    for world in ('real', 'random'):
        att = attractors(PE, world, k, i, 'G1')
        terms = []
        for a, m, be in att:
            if a < 0 or isnan(m) or isnan(be):
                continue
            g = slow_gap(P, b, a, be, 0, t)
            if g is None:
                continue
            terms.append((m, g[0], g[3]))
        ok = len(terms) >= K.MIN_ATT
        out[f'g1_spring_{world}'] = -sum(m * gc for m, gc, r in terms) if ok else float('nan')
        for j, (p, e) in enumerate(K.G1_PE):
            if p == 0:
                v = -sum(m * sign(gc) for m, gc, r in terms) if ok else float('nan')
                out[f'g1_{j}_{world}'] = v
                out[f'g1_{j}_c_{world}'] = v
            else:
                out[f'g1_{j}_{world}'] = -sum(m * sign(gc) / (r + e) ** p for m, gc, r in terms) if ok else float('nan')
                out[f'g1_{j}_c_{world}'] = -sum(m * sign(gc) / (abs(gc) + e) ** p for m, gc, r in terms) if ok else float('nan')
        # G2
        z = P['z']
        for q, (N, a_) in enumerate(K.G2_ALL):
            w = K.kernel_weights(N, a_)
            att2 = attractors(PE, world, k, i, 'G2', q)
            tot, n = 0.0, 0
            for a, m, _ in att2:
                if a < 0 or isnan(m):
                    continue
                f, pres = 0.0, 0.0
                for tau in range(1, 2 * N):
                    if w[tau - 1] == 0:
                        continue
                    zz = float(z[a, t + 1 - tau])
                    if not isnan(zz):
                        f += w[tau - 1] * zz
                        pres += w[tau - 1]
                if pres < K.W_PRESENT - 1e-12:
                    continue
                tot += m * f
                n += 1
            out[f'g2_{q}_{world}'] = tot if n >= K.MIN_ATT else float('nan')
        # G3
        for j, (N, a_, p) in enumerate(K.G3_NAP):
            w = K.kernel_weights(N, a_)
            totH, totC, n = 0.0, 0.0, 0
            for a, m, be in att:
                if a < 0 or isnan(m) or isnan(be):
                    continue
                sH, sC, pres = 0.0, 0.0, 0.0
                for tau in range(1, 2 * N):
                    g = slow_gap(P, b, a, be, tau, t)
                    if g is None:
                        continue
                    wt = w[tau - 1]
                    pres += wt
                    sH += wt * sign(g[0]) / (g[3] + K.G3_EPS) ** p
                    sC += wt * sign(g[0]) / (abs(g[0]) + K.G3_EPS) ** p
                if pres < K.W_PRESENT - 1e-12:
                    continue
                totH += -m * sH
                totC += -m * sC
                n += 1
            out[f'g3_{j}_{world}'] = totH if n >= K.MIN_ATT else float('nan')
            out[f'g3_{j}_c_{world}'] = totC if n >= K.MIN_ATT else float('nan')
    return out


def slow_sigma_z(P, b, t, mkt):
    """sigma through t - 1, beta_M through t - 1 and z at t, in plain Python."""
    lnC, lnO, lnH, lnL = P['lnC'], P['lnO'], P['lnH'], P['lnL']
    gk, rr = [], []
    for tp in range(t - 63, t):
        h, l, o, c = float(lnH[b, tp]), float(lnL[b, tp]), float(lnO[b, tp]), float(lnC[b, tp])
        if not (isnan(h) or isnan(l) or isnan(o) or isnan(c)):
            gk.append(0.5 * (h - l) ** 2 - GK_C * (c - o) ** 2)
        c0 = float(lnC[b, tp - 1])
        if not (isnan(c) or isnan(c0)):
            rr.append(c - c0)
    if len(gk) >= 57:
        sig = math.sqrt(sum(gk) / len(gk))
    elif len(rr) >= 57:
        m = sum(rr) / len(rr)
        sig = math.sqrt(sum((x - m) ** 2 for x in rr) / (len(rr) - 1))
    else:
        sig = float('nan')
    xs, ys = [], []
    for tp in range(t - 252, t):
        c, c0, x = float(lnC[b, tp]), float(lnC[b, tp - 1]), float(mkt[tp])
        if not (isnan(c) or isnan(c0) or isnan(x)):
            xs.append(x)
            ys.append(c - c0)
    if len(xs) >= 227:
        n = len(xs)
        sx, sy = sum(xs), sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        bm = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    else:
        bm = float('nan')
    r = float(lnC[b, t]) - float(lnC[b, t - 1])
    z = (r - bm * float(mkt[t])) / sig
    z = max(-5.0, min(5.0, z)) if not isnan(z) else z
    return sig, bm, z


def close(a, b, rtol=1e-5, atol=1e-6):
    if isnan(a) and isnan(b):
        return True
    if isnan(a) or isnan(b):
        return False
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


# ---------------------------------------------------------------------------------------------------------------

def truncation(tag, P, PE, F, fh, n_stocks=12):
    rng = np.random.default_rng(K.SEED + 1)
    T, d0 = P['T'], P['d0']
    inU = np.asarray(P['inU'])
    rows = np.asarray(P['rows'])
    res = []
    for s in range(n_stocks):
        while True:
            c = int(rng.integers(d0 + 300, T - 3))
            cand = np.nonzero(inU[:, c] & inU[:, c - 30])[0]
            if len(cand):
                b = int(rng.choice(cand))
                break
        Pt = data.build(tag, cut=c, rows=rows)
        assert Pt['d0'] == d0 and len(Pt['rows']) == len(rows)
        ks = [k for k in range(len(Pt['refit'])) if int(Pt['refit'][k]) <= c]
        ks = ks[-2:]
        PEt = peers.compute(Pt, refits=ks)
        same_peers = True
        for key in ('A1', 'M1', 'B1', 'AR', 'MR', 'BR'):
            for k in ks:
                same_peers &= np.array_equal(PEt[key][k], PE[key][k], equal_nan=True)
        for key in ('A2', 'M2', 'M2R'):
            for k in ks:
                same_peers &= np.array_equal(PEt[key][:, k], PE[key][:, k], equal_nan=True)
        ta = forces._span(Pt, ks[0])[0]
        n_val, n_bad, n_fin = 0, 0, 0
        for stage in ('g1', 'g2', 'g3'):
            if stage == 'g1':
                Ft, _ = forces.g1(Pt, PEt, ks, brows=np.array([b]))
            elif stage == 'g2':
                Ft = forces.g2(Pt, PEt, ks, brows=np.array([b]))
            else:
                Ft = forces.g3(Pt, PEt, ks, brows=np.array([b]))
            for key, v in Ft.items():
                a = np.asarray(F[key][b, ta:c + 1])
                bb = v[b, ta:c + 1]
                n_val += len(a)
                n_fin += int(np.isfinite(a).sum())
                n_bad += int((~((a == bb) | (np.isnan(a) & np.isnan(bb)))).sum())
            del Ft
        same_sz = (np.array_equal(np.asarray(P['sig'])[b, :c + 1], Pt['sig'][b, :c + 1], equal_nan=True)
                   and np.array_equal(np.asarray(P['z'])[b, :c + 1], Pt['z'][b, :c + 1], equal_nan=True))
        res.append({'ticker': str(P['tickers'][b]), 'cut': K.iso(P['days'], c), 'from': K.iso(P['days'], ta), 'values': n_val,
                    'finite': n_fin, 'mismatches': n_bad, 'peers_identical': bool(same_peers), 'sigma_z_identical': bool(same_sz)})
        K.log(f'  truncation {s + 1:2d}: {P["tickers"][b]:>6} cut {K.iso(P["days"], c)}; forces {ta}..{c} '
              f'({n_val} values, {n_fin} finite): mismatches {n_bad}; attractor tables identical {same_peers}; '
              f'sigma and z identical {same_sz}', fh)
        del Pt, PEt
    return res


def slow_check(P, PE, F, fh, n=40):
    rng = np.random.default_rng(K.SEED + 2)
    T, d0 = P['T'], P['d0']
    inU = np.asarray(P['inU'])
    mkt = np.asarray(K.c2eval.load_export(K.SRC['old'][0], arrays=('market',))['market'], np.float64)
    refit = np.asarray(P['refit'])
    worst = {}
    n_bad, n_val, n_fin = 0, 0, 0
    sz_bad = 0
    for s in range(n):
        while True:
            t = int(rng.integers(d0, T - 2))
            cand = np.nonzero(inU[:, t])[0]
            if len(cand):
                b = int(rng.choice(cand))
                break
        k = int(np.searchsorted(refit, t, side='right')) - 1
        i = int(np.nonzero(np.asarray(P['uni'][k]) == b)[0][0])
        sl = slow_forces(P, PE, k, i, b, t)
        for key, v in sl.items():
            fv = float(F[key][b, t])
            n_val += 1
            n_fin += int(not isnan(v))
            if not close(v, fv):
                n_bad += 1
                K.log(f'    MISMATCH {key} {P["tickers"][b]} {K.iso(P["days"], t)}: slow {v} vs kernel {fv}', fh)
            if not (isnan(v) or isnan(fv)):
                worst[key[:2]] = max(worst.get(key[:2], 0.0), abs(v - fv) / max(1e-12, abs(v)))
        sig, bm, z = slow_sigma_z(P, b, t, mkt)
        ok = close(sig, float(P['sig'][b, t - 1]), 1e-9, 1e-12) and close(bm, float(P['beta_m'][b, t]) if 'beta_m' in P else bm, 1e-8, 1e-12) \
            and close(z, float(P['z'][b, t]), 1e-8, 1e-10)
        sz_bad += int(not ok)
    K.log(f'  slow forces: {n} stock-days x {len(KEYS)} variants = {n_val} values ({n_fin} finite): {n_bad} mismatches '
          f'(tolerance 1e-5 relative); worst relative difference by stage {worst}', fh)
    K.log(f'  slow sigma, beta_M and z: {n - sz_bad} of {n} agree', fh)
    return {'values': n_val, 'finite': n_fin, 'mismatches': n_bad, 'worst_rel': worst, 'sigma_z_ok': n - sz_bad, 'n': n}


def slow_peers(P, PE, fh, n=6):
    """Attractor selection recomputed pair by pair with numpy.corrcoef on common valid days."""
    rng = np.random.default_rng(K.SEED + 3)
    lnC = np.asarray(P['lnC'])
    z = np.asarray(P['z'])
    refit = np.asarray(P['refit'])
    ks = [k for k in range(len(refit)) if forces._span(P, k)[1] > forces._span(P, k)[0]]
    bad = 0
    checked = 0
    for s in range(n):
        k = int(rng.choice(ks))
        t0 = int(refit[k])
        u = np.asarray(P['uni'][k])
        u = u[u >= 0]
        i = int(rng.integers(len(u)))
        b = int(u[i])
        r = lnC[:, t0 - 251:t0 + 1] - lnC[:, t0 - 252:t0]
        rho, beta = {}, {}
        for a in u:
            if a == b:
                continue
            m = np.isfinite(r[a]) & np.isfinite(r[b])
            if m.sum() < 227:
                continue
            x, y = r[a][m], r[b][m]
            rho[int(a)] = np.corrcoef(x, y)[0, 1]
            beta[int(a)] = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
        top = sorted([a for a in rho if rho[a] > 0], key=lambda a: -rho[a])[:10]
        got = [int(x) for x in PE['A1'][k, i] if x >= 0]
        ok1 = top == got and all(close(rho[a], float(PE['M1'][k, i, j]), 1e-5, 1e-6) and close(beta[a], float(PE['B1'][k, i, j]), 1e-8, 1e-10)
                                 for j, a in enumerate(got))
        rd = [int(x) for x in PE['AR'][k, i]]
        okr = all(close(rho.get(a, float('nan')), float(PE['MR'][k, i, j]), 1e-5, 1e-6) for j, a in enumerate(rd)) and len(set(rd)) == 10 and b not in rd
        ok2 = True
        for q, (N, a_) in enumerate(K.G2_ALL):
            f = peers.kernel_f(z[u], N, a_)[:, t0 - 252:t0]
            y = z[b, t0 - 251:t0 + 1]
            m2 = {}
            for j, a in enumerate(u):
                if a == b:
                    continue
                mm = np.isfinite(f[j]) & np.isfinite(y)
                if mm.sum() < 227:
                    continue
                m2[int(a)] = np.corrcoef(f[j][mm], y[mm])[0, 1]
            top2 = sorted(m2, key=lambda a: -abs(m2[a]))[:10]
            got2 = [int(x) for x in PE['A2'][q, k, i] if x >= 0]
            ok2 &= top2 == got2 and all(close(m2[a], float(PE['M2'][q, k, i, j]), 1e-5, 1e-6) for j, a in enumerate(got2))
        checked += 1
        bad += int(not (ok1 and okr and ok2))
        K.log(f'  attractors {P["tickers"][b]:>6} at {K.iso(P["days"], t0)}: G1 {ok1}, random {okr}, G2 (18 kernels) {ok2}', fh)
    return {'checked': checked, 'bad': bad}


def hand_prints(P, PE, F, fh):
    cases = [('AAPL', '2016-05-02'), ('XOM', '2015-12-08'), ('JPM', '2011-08-08')]
    tick = list(P['tickers'])
    out = []
    refit = np.asarray(P['refit'])
    for tk, ds in cases:
        if tk not in tick:
            continue
        b = tick.index(tk)
        t = K.day_of(P['days'], ds)
        k = int(np.searchsorted(refit, t, side='right')) - 1
        u = np.asarray(P['uni'][k])
        if b not in u:
            continue
        i = int(np.nonzero(u == b)[0][0])
        K.log(f'  HAND PRINT {tk} on {ds} (refit {K.iso(P["days"], int(refit[k]))}): close {math.exp(P["lnC"][b, t]):.4f}, '
              f'high {math.exp(P["lnH"][b, t]):.4f}, low {math.exp(P["lnL"][b, t]):.4f}', fh)
        rows = []
        for s in range(K.K_ATT):
            a, m, be = int(PE['A1'][k, i, s]), float(PE['M1'][k, i, s]), float(PE['B1'][k, i, s])
            if a < 0:
                continue
            g = slow_gap(P, b, a, be, 0, t)
            if g is None:
                K.log(f'    {tick[a]:>6} rho {m:.3f} beta {be:.3f}: no gap that day', fh)
                continue
            gc, gh, gl, r, mu, sd = g
            K.log(f'    {tick[a]:>6} rho(mass) {m:.3f} beta {be:.3f}: A close {math.exp(P["lnC"][a, t]):.4f}; spread mean '
                  f'{mu:+.5f} sd {sd:.5f}; g_C {gc:+.3f} g_H {gh:+.3f} g_L {gl:+.3f}; closest approach r {r:.3f}; '
                  f'pull p=2 eps=0.3: {-m * sign(gc) / (r + 0.3) ** 2:+.3f}', fh)
            rows.append({'att': tick[a], 'rho': m, 'beta': be, 'gC': gc, 'gH': gh, 'gL': gl, 'r': r, 'mu': mu, 'sd': sd})
        fv = {key: float(F[key][b, t]) for key in KEYS if key.endswith('_real')}
        K.log(f'    forces: spring {fv["g1_spring_real"]:+.3f}; ' + '; '.join(
            f'{K.g1_name(p, e)} {fv[f"g1_{j}_real"]:+.3f}' for j, (p, e) in enumerate(K.G1_PE)), fh)
        K.log('    ' + '; '.join(f'{K.g2_name(N, a)} {fv[f"g2_{q}_real"]:+.3f}' for q, (N, a) in enumerate(K.G2_ALL)), fh)
        K.log('    ' + '; '.join(f'{K.g3_name(N, a, p)} {fv[f"g3_{j}_real"]:+.3f}' for j, (N, a, p) in enumerate(K.G3_NAP)), fh)
        out.append({'ticker': tk, 'day': ds, 'attractors': rows, 'forces': fv})
    return out


def main(tag):
    fh = open(K.OUT / f'checks_{tag}.txt', 'w')
    P = data.load_panel(tag, mmap=False)
    Pb = data.build(tag)           # beta_m and lnO for the slow sigma check
    P['beta_m'] = Pb['beta_m']
    P['lnO'] = Pb['lnO']
    P['ids'] = np.array(P['meta']['ids'])
    assert np.array_equal(Pb['z'], P['z'], equal_nan=True)
    del Pb
    PE = dict(np.load(K.BIG / f'peers_{tag}.npz'))
    F = load_forces(tag)
    res = {}
    K.log('1. Look-ahead by truncation', fh)
    res['truncation'] = truncation(tag, P, PE, F, fh)
    K.log('2. Slow recomputation', fh)
    res['slow'] = slow_check(P, PE, F, fh)
    res['slow_peers'] = slow_peers(P, PE, fh)
    K.log('3. Hand prints', fh)
    res['hand'] = hand_prints(P, PE, F, fh)
    tr = res['truncation']
    K.log(f'SUMMARY: truncation mismatches {sum(x["mismatches"] for x in tr)} of {sum(x["values"] for x in tr)} values '
          f'({sum(x["finite"] for x in tr)} finite), attractor tables identical {sum(x["peers_identical"] for x in tr)}/12, '
          f'sigma and z identical {sum(x["sigma_z_identical"] for x in tr)}/12; slow forces mismatches {res["slow"]["mismatches"]} '
          f'of {res["slow"]["values"]}; slow attractors bad {res["slow_peers"]["bad"]} of {res["slow_peers"]["checked"]}', fh)
    K.jdump(res, K.OUT / f'checks_{tag}.json')
    fh.close()


if __name__ == '__main__':
    main(sys.argv[1])
