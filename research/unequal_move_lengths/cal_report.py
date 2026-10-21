"""Noise calibration report: real/chance for 220 pure-noise series under this experiment's null and two older ones.

python3 -I cal_report.py  -> research/unequal_move_lengths/calibration.txt
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umlcore as U  # noqa: E402
import engine as E  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ['this null, noise A -> stocks B', 'this null, stocks A -> noise B',
           'old null (non-overlapping, each observation flipped), noise A -> stocks B',
           'overlapping, blocks of v flipped (not residuals), noise A -> stocks B']


def load(tags):
    return {t: np.load(U.OUT / 'cal' / f'{t}.npz') for t in tags}


def ratio(x):
    """x: [series, copies] summed counts (real copy 0, then the null copies). Returns real, chance mean, ratio, and the
    standard error of the ratio from the spread over the independent noise series (bootstrap-equivalent)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    real_i = x[:, 0]
    null_i = x[:, 1:].mean(1)
    real, mu = real_i.sum(), null_i.sum()
    if mu <= 0:
        return real, mu, float('nan'), float('nan')
    d = real_i - null_i
    se = np.sqrt(len(d) * d.var(ddof=1)) / mu if len(d) > 1 else float('nan')
    return real, mu, real / mu, se


def per_series(d, tags, ci, mi, thr):
    """[series, copies] counts summed over the given periods and the 3 offsets (ci may be a list of combinations)."""
    cis = ci if isinstance(ci, (list, tuple, np.ndarray)) else [ci]
    return sum(d[t]['counts'][c, :, mi, :, :, thr].sum(0) for t in tags for c in cis).astype(float)


def main():
    rolling = [f'Y{y}' for y in range(2006, 2026)]
    tags = rolling + ['full']
    d = load(tags)
    kinds = np.arange(E.N_NOISE_KINDS * E.N_PER_KIND) // E.N_PER_KIND
    cont = np.isin(kinds, [0, 1, 2, 3, 5, 6])  # iid, fat, garch, ar0.3, ar0.9, ar0.97
    lines = []
    w = lines.append
    w('Noise calibration: 220 pure-noise series (11 kinds x 20) through the exact pipeline, 20 move-length combinations,')
    w('3 offsets each, the 20 rolling 5-year correlation periods plus the full history, 8 null copies per matrix.')
    w('real/chance = count of |z| >= 4 with the noise as it is, over the mean count in the 8 null copies. 1.0 = calibrated.')
    w('se = standard error of the ratio from the spread over the 220 independent noise series. Pass: 0.90..1.10, or within 2 se.')
    w('Kinds: ' + ', '.join(E.NOISE_KINDS) + '. "continuous" = iid, fat, garch, ar0.3, ar0.9, ar0.97 (stock-like, plus persistence).')
    w('')
    summary = {}
    fmt = lambda r: f'{r[2]:.2f} (se {r[3]:.2f})' if np.isfinite(r[2]) else '   -'
    for mi, mname in enumerate(METHODS):
        w(f'== {mname} ==')
        w(f'{"#":>3} {"A":>4} {"B":>4}  {"rolling periods + full history":>40}  {"rolling only":>16}  {"full history":>16}  '
          f'{"continuous kinds":>16}  {"|z|>=3":>16}  pass')
        tot = 0
        for ci, (num, a, b) in enumerate(U.COMBOS):
            x_all = per_series(d, tags, ci, mi, 1)
            tot = tot + x_all
            r_all = ratio(x_all)
            r_roll = ratio(per_series(d, rolling, ci, mi, 1))
            r_full = ratio(per_series(d, ['full'], ci, mi, 1))
            r_cont = ratio(x_all[cont])
            r3 = ratio(per_series(d, tags, ci, mi, 0))
            ok = np.isfinite(r_all[2]) and (0.90 <= r_all[2] <= 1.10 or abs(r_all[2] - 1) <= 2 * r_all[3])
            if mi < 2:
                summary.setdefault(num, []).append((ok, r_all[2], r_all[3]))
            w(f'{num:>3} {a:>4} {b:>4}  {int(r_all[0]):>10,} / {r_all[1]:>11,.1f} = {fmt(r_all):>15}  {fmt(r_roll):>16}  '
              f'{fmt(r_full):>16}  {fmt(r_cont):>16}  {fmt(r3):>16}  {"yes" if ok else "NO"}')
        rt = ratio(tot)
        w(f'all 20 combinations: {int(rt[0]):,} / {rt[1]:,.1f} = {rt[2]:.3f} (se {rt[3]:.3f}); continuous kinds {ratio(tot[cont])[2]:.3f} (se {ratio(tot[cont])[3]:.3f})')
        w('')
    w('By noise kind, |z| >= 4, rolling periods + full history, all 20 combinations:')
    for mi in range(4):
        w(f'  {METHODS[mi]}')
        allc = per_series(d, tags, list(range(len(U.COMBOS))), mi, 1)
        for q, kname in enumerate(E.NOISE_KINDS):
            r = ratio(allc[kinds == q])
            w(f'    {kname:>12}: {int(r[0]):>10,} / {r[1]:>12,.1f} = {r[2]:.3f} (se {r[3]:.3f})')
    w('')
    w('Long moves (any move of 42 days or more: #4, 5, 8-10, 12-17, 19, 20) pooled, |z| >= 4:')
    longs = [ci for ci, (num, a, b) in enumerate(U.COMBOS) if max(a, b) >= 42]
    for mi, mname in enumerate(METHODS):
        r = ratio(per_series(d, tags, longs, mi, 1))
        rf = ratio(per_series(d, ['full'], longs, mi, 1))
        w(f'  {mname}: {r[2]:.2f} (se {r[3]:.2f}); full history only {rf[2]:.2f} (se {rf[3]:.2f})')
    w('')
    allpass = all(v[0][0] and v[1][0] for v in summary.values())
    w(f'Calibration with this null, both directions, every combination within the pass rule: {"YES" if allpass else "NO"}')
    for num, v in summary.items():
        if not (v[0][0] and v[1][0]):
            w(f'  outside the rule: #{num} (noise A -> stocks B {v[0][1]:.2f} se {v[0][2]:.2f}; stocks A -> noise B {v[1][1]:.2f} se {v[1][2]:.2f})')
    hi = max(max(v[0][1], v[1][1]) for v in summary.values())
    w(f'Highest ratio with this null in either direction: {hi:.2f}')
    out = '\n'.join(lines)
    open(os.path.join(HERE, 'calibration.txt'), 'w').write(out + '\n')
    print(out)


if __name__ == '__main__':
    main()
