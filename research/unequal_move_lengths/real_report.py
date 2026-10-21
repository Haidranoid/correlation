"""Real vs chance per combination: strong correlations (|z| >= 4) between real stocks at the 3 predictive offsets,
against the mean of the 8 null copies. Rolling 5-year periods (20) and the full history.

python3 -I real_report.py -> real_vs_chance.txt here
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umlcore as U  # noqa: E402
import engine as E  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TODAY = {1: 1.03, 2: 1.02, 3: 1.02, 5: 0.99, 6: 1.03, 7: 1.03, 9: 0.99, 11: 1.07, 13: 1.09, 15: 1.18, 18: 1.02, 19: 1.30, 20: 1.03}


def stats(x):
    real = float(x[0])
    nul = np.asarray(x[1:], dtype=float)
    mu, sd = nul.mean(), nul.std(ddof=1)
    return (real, mu, (real / mu if mu > 0 else float('nan')), ((real - mu) / sd if sd > 0 else float('nan')),
            (real / np.median(nul) if np.median(nul) > 0 else float('nan')))


def noise_reference(rolling):
    """Calibration reference per combination: continuous noise kinds, noise A -> stocks B, rolling periods."""
    import cal_report as C
    d = C.load(rolling)
    kinds = np.arange(E.N_NOISE_KINDS * E.N_PER_KIND) // E.N_PER_KIND
    cont = np.isin(kinds, [0, 1, 2, 3, 5, 6])
    return {num: C.ratio(C.per_series(d, rolling, ci, 0, 1)[cont])[2] for ci, (num, a, b) in enumerate(U.COMBOS)}


def main():
    rolling = [f'Y{y}' for y in range(2006, 2026)]
    d = {t: np.load(U.OUT / 'real' / f'{t}.npz') for t in rolling + ['full']}
    ref = noise_reference(rolling)
    lines = ['Real vs chance: pairs of real stocks with |z| >= 4 at the 3 predictive offsets (B right after A, +1 week, +1 month),',
             'overlapping observations, against the mean of 8 block sign-flip null copies (residuals flipped, census variable rebuilt).',
             'z excess = (real - mean null) / sd of the null copies. "Excess" = ratio >= 1.10 and z excess >= 3.',
             'median ratio (added after the results): real over the median null copy; the copies are skewed (one copy in 8 often',
             'has 1.5-2.7x the others, when its signs line up with a market-wide episode), which pulls the mean up.',
             'noise ref = the calibration\'s real/chance for continuous noise A -> real stocks B (same null, same periods).',
             'Today\'s census (non-overlapping, old null) for comparison where it has the combination.',
             '',
             f'{"#":>3} {"A":>4} {"B":>4}  {"rolling 5-year periods (20)":>36} {"z exc":>6} {"median":>6}  {"full history":>30} {"z exc":>6}  '
             f'{"|z|>=3 rolling":>15} {"|z|>=5 rolling":>15}  {"noise ref":>9} {"census today":>12}  excess']
    rows = {}
    tot = np.zeros(1 + E.NULLS)
    for ci, (num, a, b) in enumerate(U.COMBOS):
        roll = sum(d[t]['counts'][ci].sum(0) for t in rolling)  # [copy, thr]
        full = d['full']['counts'][ci].sum(0)
        tot += roll[:, 1]
        r4, f4, r3, r5 = stats(roll[:, 1]), stats(full[:, 1]), stats(roll[:, 0]), stats(roll[:, 2])
        exc = r4[2] >= 1.10 and r4[3] >= 3
        rows[num] = {'roll': r4, 'full': f4, 'r3': r3, 'r5': r5, 'excess': exc, 'noise_ref': ref[num]}
        lines.append(f'{num:>3} {a:>4} {b:>4}  {int(r4[0]):>10,} / {r4[1]:>11,.1f} = {r4[2]:>5.2f}  {r4[3]:>6.1f} {r4[4]:>6.2f}  '
                     f'{int(f4[0]):>8,} / {f4[1]:>9,.1f} = {f4[2]:>5.2f} {f4[3]:>6.1f}  {r3[2]:>15.2f} {r5[2]:>15.2f}  '
                     f'{ref[num]:>9.2f} {TODAY.get(num, float("nan")):>12.2f}  {"YES" if exc else "no"}')
    t = stats(tot)
    lines.append(f'all 20, rolling: {int(t[0]):,} / {t[1]:,.1f} = {t[2]:.3f} (z excess {t[3]:.1f})')
    lines.append('')
    lines.append('Stocks per period: ' + ', '.join(f'{t} {len(d[t]["universe"])}' for t in rolling + ['full']))
    out = '\n'.join(lines)
    open(os.path.join(HERE, 'real_vs_chance.txt'), 'w').write(out + '\n')
    np.save(U.OUT / 'real_vs_chance_rows.npy', rows, allow_pickle=True)
    print(out)


if __name__ == '__main__':
    main()
