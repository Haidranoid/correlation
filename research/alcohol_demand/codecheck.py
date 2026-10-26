"""C9: code checks, run before P1's result is read. Noise in place of the six inputs (200 sets of independent normal
series, 200 sets of AR(1) series with coefficient 0.95) and a leaky oracle (the future target plus noise) go through
P1's pipeline on the primary target. Development decisions only.

    python3 -I -B codecheck.py      # -> codecheck_old.txt, codecheck_old.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ad_common as K  # noqa: E402
import numpy as np  # noqa: E402


def main():
    z = np.load(K.WORK / 'inputs_old.npz')
    D, dates = z['D'], z['dates']
    y = z['y_ALC-SPY']
    six = np.column_stack([z[f'x_{n}_c0'] for n in K.INPUTS])
    base = np.isfinite(six).all(1) & np.isfinite(y)      # the real inputs' decisions, so the same sample
    dev = K.sel(dates, *K.DEV)
    rng = np.random.default_rng(K.SEED + 9)
    res = {}
    lines = []
    for kind in ('iid', 'ar095'):
        ts, r2s = [], []
        for s in range(K.N_NOISE):
            e = rng.standard_normal((len(D), 6))
            if kind == 'ar095':
                x = np.empty_like(e)
                x[0] = e[0] / np.sqrt(1 - 0.95 ** 2)
                for i in range(1, len(D)):
                    x[i] = 0.95 * x[i - 1] + e[i]
                e = x
            Xn = np.where(base[:, None], e, np.nan)
            o = K.oos(y, Xn, D, sink=False)
            ev = K.evaluate(y, o['f0'], o['fc'], dev, boot=False)
            ts.append(ev['cw_t'])
            r2s.append(ev['r2os'])
        ts, r2s = np.array(ts), np.array(r2s)
        res[kind] = {'n': len(ts), 'share_t_ge_3': float(np.mean(ts >= 3)), 'share_t_ge_2': float(np.mean(ts >= 2)),
                     'share_r2_pos': float(np.mean(r2s > 0)), 't_p95': float(np.percentile(ts, 95)),
                     'r2_median': float(np.median(r2s)), 'r2_p95': float(np.percentile(r2s, 95))}
        lines.append(f'noise {kind}: {len(ts)} sets; Clark-West t >= 3 in {res[kind]["share_t_ge_3"]:.3f}, '
                     f't >= 2 in {res[kind]["share_t_ge_2"]:.3f}, 95th percentile t {res[kind]["t_p95"]:.2f}; '
                     f'R2 > 0 in {res[kind]["share_r2_pos"]:.3f}, median R2 {res[kind]["r2_median"]:+.4f}, '
                     f'95th percentile {res[kind]["r2_p95"]:+.4f}')
    # Leaky oracles: the future y plus noise (six copies, independent noise). Noise of y's own size first; the bar's
    # reach is shown with weaker and stronger noise (C9, reported as a power check).
    sd = np.nanstd(y[base])
    for scale in (1.0, 0.5, 2.0, 4.0):
        Xo = np.where(base[:, None], y[:, None] + scale * sd * rng.standard_normal((len(D), 6)), np.nan)
        o = K.oos(y, Xo, D, sink=False)
        ev = K.evaluate(y, o['f0'], o['fc'], dev)
        res[f'oracle_{scale}'] = ev
        lines.append(f'leaky oracle (future y plus noise {scale} times its size, six copies): n {ev["n"]}, '
                     f'Clark-West t {ev["cw_t"]:.2f}, bootstrap p {ev["cw_boot_p"]:.4f}, R2 {ev["r2os"]:+.3f}')
    ev = res['oracle_1.0']
    assert ev['cw_t'] > 3 and ev['r2os'] > 0.3, 'the oracle must win by far'
    print('\n'.join(lines))
    (K.HERE / 'codecheck_old.txt').write_text('\n'.join(lines) + '\n')
    K.dump(res, K.HERE / 'codecheck_old.json')


if __name__ == '__main__':
    main()
