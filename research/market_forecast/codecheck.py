"""C9, run before P1's result is read (h = 12, the four markets pooled, development).

(a) 200 sets of nine AR(1) noise series (coefficient 0.95) in place of the inputs, through P1 against the plan's
    one-lag own-history benchmark and against the two-lag benchmark of Decisions 2, with the bootstrap p of every
    set. Also the noise combinations of 1, 2 and 3 series, the sizes of the theses in P2. The percentiles here are
    the reference that P1 and P2 are read against (Decisions 2).
(b) a leaky oracle: the future target plus noise 0.5, 1, 2 and 4 times the target's own standard deviation, as one
    more input; its own forecast and the ten-input combination, against both benchmarks.
Outputs are written before the sanity assertions (Decisions 1): the 0.5x oracle's own forecast must reach R^2 > 0.3
and the 1x oracle's R^2 > 0.2, with bootstrap p < 0.0013, against the one-lag benchmark. Eligibility is asserted
inside every fit.

    python3 -I -B codecheck.py     # -> codecheck.txt, codecheck.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mf_common as M  # noqa: E402
import numpy as np  # noqa: E402

H = 12
HERE = Path(__file__).resolve().parent
SIZES = (1, 2, 3, 9)


def ar1(n, rng, phi=0.95):
    x = np.empty(n)
    x[0] = rng.standard_normal() / np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x


def pct(a):
    a = np.asarray(a, float)
    return {q: float(np.percentile(a, q)) for q in (1, 5, 50, 95, 99)}


def main():
    lines, res = [], {}

    def out(s=''):
        print(s, flush=True)
        lines.append(s)

    X = M.common_inputs()
    P = {b: M.panel(X, M.bea(c), M.bls(M.PRICE[b]), H) for b, c in M.MARKETS.items()}
    months = P['W'][0]
    real_ok = np.all([np.isfinite(p[3]).all(1) for p in P.values()], axis=0)
    dev = M.period_mask(months, H, *M.DEV)
    bench = {'one-lag': {b: p[2] for b, p in P.items()}, 'two-lag': {b: M.two_lag(p[2]) for b, p in P.items()}}
    rng = np.random.default_rng(M.SEED)
    noise = {bn: {sz: {'r2': [], 't': [], 'boot_p': []} for sz in SIZES} for bn in bench}
    for j in range(M.N_NOISE):
        Z = np.column_stack([ar1(len(months), rng) for _ in range(9)])
        Z[~real_ok] = np.nan
        for bn, G in bench.items():
            O = {b: M.oos(P[b][1], G[b], Z, months, H, sink=False) for b in M.MARKETS}
            for sz in SIZES:
                e = M.evaluate([P[b][1] for b in M.MARKETS], [O[b]['fb'] for b in M.MARKETS],
                               [O[b]['fk'][:, :sz].mean(1) for b in M.MARKETS], dev, seed=M.SEED + j)
                for k, kk in (('r2', 'r2os'), ('t', 'cw_t'), ('boot_p', 'cw_boot_p')):
                    noise[bn][sz][k].append(e[kk])
    res['noise'] = {}
    for bn in bench:
        res['noise'][bn] = {}
        for sz in SIZES:
            d = noise[bn][sz]
            r2, t, bp = (np.array(d[k]) for k in ('r2', 't', 'boot_p'))
            full = (t >= 3) & (bp < 0.0013) & (r2 > 0)
            res['noise'][bn][str(sz)] = {**{k: list(map(float, v)) for k, v in d.items()},
                                         'pct_r2': pct(r2), 'pct_t': pct(t), 'pct_boot_p': pct(bp),
                                         'share_t3': float(np.mean(t >= 3)), 'share_t2': float(np.mean(t >= 2)),
                                         'share_boot': float(np.mean(bp < 0.0013)), 'share_full_bar': float(np.mean(full))}
            q = res['noise'][bn][str(sz)]
            out(f'(a) noise, {bn} benchmark, {sz} series: R2 median {q["pct_r2"][50]:+.4f}, 95th {q["pct_r2"][95]:+.4f}, '
                f'99th {q["pct_r2"][99]:+.4f}; t median {q["pct_t"][50]:+.2f}, 95th {q["pct_t"][95]:+.2f}; '
                f't >= 3 in {100 * q["share_t3"]:.1f}%, t >= 2 in {100 * q["share_t2"]:.1f}%, bootstrap p < 0.0013 in '
                f'{100 * q["share_boot"]:.1f}%, the full bar in {100 * q["share_full_bar"]:.1f}%; bootstrap p 5th pct '
                f'{q["pct_boot_p"][5]:.4f}')
    res['oracle'] = {}
    for scale in (0.5, 1.0, 2.0, 4.0):
        orc = {}
        for b, (_, y, g, Xm) in P.items():
            sd = np.nanstd(y[np.isfinite(Xm).all(1)])
            orc[b] = y + scale * sd * rng.standard_normal(len(y))
        res['oracle'][str(scale)] = {}
        for bn, G in bench.items():
            O = {b: M.oos(P[b][1], G[b], np.column_stack([P[b][3], orc[b]]), months, H, sink=False) for b in M.MARKETS}
            ys, fbs = [P[b][1] for b in M.MARKETS], [O[b]['fb'] for b in M.MARKETS]
            eo = M.evaluate(ys, fbs, [O[b]['fk'][:, -1] for b in M.MARKETS], dev)
            ec = M.evaluate(ys, fbs, [O[b]['fk'].mean(1) for b in M.MARKETS], dev)
            res['oracle'][str(scale)][bn] = {'own': eo, 'in_combination': ec}
            out(f'(b) oracle, noise {scale}x, {bn}: its own forecast {M.fmt_ev(eo)}')
            out(f'    as a tenth input in the combination: {M.fmt_ev(ec)}')
    (HERE / 'codecheck.txt').write_text('\n'.join(lines) + '\n')
    M.dump(res, HERE / 'codecheck.json')
    o5, o1 = res['oracle']['0.5']['one-lag']['own'], res['oracle']['1.0']['one-lag']['own']
    assert o5['r2os'] > 0.3 and o5['cw_boot_p'] < 0.0013, o5
    assert o1['r2os'] > 0.2 and o1['cw_boot_p'] < 0.0013, o1
    out('the oracles pass their sanity checks')
    (HERE / 'codecheck.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
