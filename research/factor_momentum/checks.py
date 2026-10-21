"""Checks of the data and the code (no hold returns), on one export:
- coverage per formation: universe, the 500, stocks passing the 90% rule, twin coverage for baseline (b);
- the eigen-decomposition: eigenvalues descending, component 1 one-signed, the in-sample factor returns are the
  principal-component scores (near-diagonal covariance, variance proportional to the eigenvalue);
- the 12-1 copy against stockai's momentum_scores; decile weights; a hold return by hand.

    python3 -I -B checks.py old      # writes out/checks_old.txt, out/coverage_old.tsv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402

LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def main(tag):
    D = C.load(tag)
    say(f'export {tag}: N {D["N"]} T {D["T"]}, {iso0(D, 0)}..{iso0(D, D["T"] - 1)}; gap days filled {D["gap_days"]}')
    k0 = C.first_k(D)
    say(f'first formation with a universe: k {k0} = {C.iso(D, D["rebal"][k0])}; with 500 stocks: k {C.first_k500(D)} = '
        f'{C.iso(D, D["rebal"][C.first_k500(D)])}')
    sc = np.asarray(D['tw']['F']['close'][int(np.nonzero(D['ftick'] == 'SPY')[0][0])], np.float64)
    say(f'SPY store close first day {iso0(D, int(np.nonzero(np.isfinite(sc))[0][0]))}; days with a close before 2006: '
        f'{int(np.isfinite(sc[:c2eval.day_index(D["days"], "2006-01-01")]).sum())}')
    tw = D['tw']
    rows_out = ['k\tdate\tuniverse\tbig\teigen_n\ttwin_frac\tdistinct_twins\tb_valid']
    first_b = None
    for k in range(k0, D['K']):
        t = int(D['rebal'][k])
        big = np.nonzero(D['big'][k])[0]
        fr = tw['fund_row'][k, big]
        has = fr >= 0
        frac = float(has.mean())
        distinct = len(set(fr[has].tolist()))
        c = np.asarray(D['close'][big, t - C.WIN:t + 1], np.float64)
        with np.errstate(all='ignore'):
            r = np.log(c[:, 1:] / c[:, :-1])
        en = int((np.isfinite(r).mean(1) >= C.MIN_FRAC).sum())
        bv = frac >= C.B_MIN_FRAC and distinct >= C.B_MIN_DISTINCT
        if bv and first_b is None:
            first_b = k
        rows_out.append(f'{k}\t{C.iso(D, t)}\t{int(D["uni"][k].sum())}\t{len(big)}\t{en}\t{frac:.3f}\t{distinct}\t{int(bv)}')
    (C.OUT / f'coverage_{tag}.tsv').write_text('\n'.join(rows_out) + '\n')
    say(f'first formation where baseline (b) counts (>= {C.B_MIN_FRAC:.0%} of the 500 with a twin, >= '
        f'{C.B_MIN_DISTINCT} distinct twins): k {first_b} = {C.iso(D, D["rebal"][first_b])}')
    say('coverage, first formations:')
    for line in rows_out[1:50]:
        say('  ' + line)
    arr = np.array([[float(x) for x in l.split('\t')[2:]] for l in rows_out[1:]])
    say(f'over all formations: universe {arr[:, 0].min():.0f}-{arr[:, 0].max():.0f}; big {arr[:, 1].min():.0f}-'
        f'{arr[:, 1].max():.0f}; eigen n {arr[:, 2].min():.0f}-{arr[:, 2].max():.0f} (mean {arr[:, 2].mean():.1f}); '
        f'twin frac {arr[:, 3].min():.2f}-{arr[:, 3].max():.2f}; distinct twins {arr[:, 4].min():.0f}-{arr[:, 4].max():.0f}; '
        f'(b) invalid at {int((arr[:, 5] == 0).sum())} formations')

    say('\n== eigen checks ==')
    for k in [k0, k0 + 50, k0 + 100, k0 + 150, D['K'] - 3]:
        F = C.formation(D, k)
        lam = F['lam']
        assert np.all(np.diff(lam) <= 1e-9), 'eigenvalues not descending'
        assert abs(lam.sum() - F['n']) < 1e-6 * F['n'], 'trace != n'
        v1 = F['V'][:, 0]
        pos = max((v1 > 0).mean(), (v1 < 0).mean())
        cf = np.corrcoef(F['f'])
        off = np.abs(cf - np.eye(len(cf))).max()
        # in-sample factor returns are PC scores: var(f_k) * G_k^2 should be proportional to lam_k
        G = np.abs(F['V'][:, C.FIRST - 1:C.LAST] / F['sd'][:, None]).sum(0)
        ratio = F['f'].var(1, ddof=1) * G ** 2 / lam[C.FIRST - 1:C.LAST]
        # same check exactly on the log-return z-scores (the matrix the eigenvectors come from)
        say(f'k {k} {C.iso(D, F["t"])}: n {F["n"]}; lam1/n {lam[0] / F["n"]:.3f}; top-21 shares '
            f'{np.round(lam[:5] / F["n"], 3).tolist()}...; comp 1 one-signed share {pos:.3f}; '
            f'max |corr| between in-sample factor returns {off:.3f}; var*G^2/lam min/max {ratio.min():.3e}/{ratio.max():.3e} '
            f'(ratio {ratio.max() / ratio.min():.2f}); book gross {np.abs(F["W"]).sum():.3f}, sum of B = W: '
            f'{np.abs(F["B"].sum(1) - F["W"]).max():.1e}; net exposure {F["W"].sum():+.3f}')

    say('\n== sign invariance (flip component 2 and rebuild) ==')
    k = k0 + 60
    F = C.formation(D, k)
    t = F['t']
    rows = F['rows']
    c = np.asarray(D['close'][rows, t - C.WIN:t + 1], np.float64)
    with np.errstate(all='ignore'):
        r = np.log(c[:, 1:] / c[:, :-1])
    rs = np.expm1(r)
    rs[~np.isfinite(rs)] = 0.0
    w2 = F['w'].copy()
    w2[:, 0] *= -1
    f2 = w2.T @ rs
    s2 = np.sign(f2[:, :C.WIN - C.SKIP].sum(1))
    B2 = (w2 * (s2 * F['scale'])[None, :]).sum(1)
    B2 *= C.BOOK_GROSS / np.abs(B2).sum()
    say(f'max |W - W(flipped)| = {np.abs(B2 - F["W"]).max():.2e}; in-sample f matches formation(): '
        f'{np.abs(F["f"] - (F["w"].T @ rs)).max():.2e}')

    say('\n== 12-1 copy vs stockai momentum_scores (gap-filled closes) ==')
    sys.path.insert(0, '/home/bhoult/programming/correlation/stockai')
    from stockai.baselines.momentum import momentum_scores
    ms = momentum_scores(D['close'][:, :], 21, 252)
    mx = 0.0
    nd = 0
    for kk in range(k0, D['K'], 7):
        t = int(D['rebal'][kk])
        rows = np.nonzero(D['big'][kk])[0]
        a = C.mom12_1(D, rows, t)
        b = ms[rows, t]
        nd += int((np.isfinite(a) != np.isfinite(b)).sum())
        both = np.isfinite(a) & np.isfinite(b)
        mx = max(mx, float(np.abs(a[both] - b[both]).max()))
    del ms
    say(f'max abs diff {mx:.2e}; cells finite in only one: {nd}')

    say('\n== decile weights ==')
    rng = np.random.default_rng(0)
    s = rng.normal(size=500)
    s[:7] = np.nan
    w = C.decile_weights(s)
    say(f'random scores: long {w[w > 0].sum():.3f} ({(w > 0).sum()} names), short {w[w < 0].sum():.3f} '
        f'({(w < 0).sum()}), top names are the largest: {bool(np.all(s[w > 0] >= np.nanpercentile(s, 89)))}')
    s = np.repeat(np.arange(25.0), 20)            # 25 tie groups of 20
    w = C.decile_weights(s)
    say(f'tied groups (25 x 20): long {w[w > 0].sum():.3f} over {(w > 0).sum()} names (weights {sorted(set(np.round(w[w > 0], 5)))}), '
        f'short {w[w < 0].sum():.3f} over {(w < 0).sum()}')
    s = np.repeat(np.arange(7.0), [100, 30, 90, 80, 70, 60, 70])
    w = C.decile_weights(s)
    say(f'uneven ties: long {w[w > 0].sum():.3f} over {(w > 0).sum()} names, short {w[w < 0].sum():.3f} over {(w < 0).sum()}')

    say('\n== hold return by hand ==')
    k = k0 + 80
    t = int(D['rebal'][k])
    rows = np.nonzero(D['big'][k])[0][:3]
    for j in rows:
        raw = np.asarray(D['E']['close'][j, t:t + C.HOLD + 1], np.float64)
        last = raw[np.isfinite(raw)][-1]
        first = raw[0] if np.isfinite(raw[0]) else np.nan
        say(f'  {D["E"]["tickers"][j]} {C.iso(D, t)}: by hand {last / first - 1:+.5f}; code {C.hold_growth(D, np.array([j]), t)[0]:+.5f}')
    (C.OUT / f'checks_{tag}.txt').write_text('\n'.join(LINES) + '\n')


def iso0(D, d):
    return c2eval.iso(D['days'][d])


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'old')
