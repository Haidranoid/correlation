"""Replication: do the 2,000 strongest pairs of a 5-year period keep their sign in the next non-overlapping 5-year
period (5 years later) more often than the 2,000 strongest pairs of each null copy?

python3 -I replicate.py  -> replication.npz under the large-file folder, replication.txt here
"""
import os
import sys

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '8')
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umlcore as U  # noqa: E402
import engine as E  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def pair_r(v, p0, p1, a, b, pairs):
    """Exact r in rows [p0, p1) for pairs [m, 5] (z, A row, B row, offset, r), overlapping observations. Returns r
    (NaN where a stock has < 90% of days with a value in the period or too few observations)."""
    out = np.full(len(pairs), np.nan)
    TP = p1 - p0
    rows = np.unique(np.concatenate([pairs[:, 1], pairs[:, 2]]).astype(int))
    V = v[rows, p0:p1]
    cov = np.isfinite(V).sum(1) >= 0.9 * TP
    C = E.cums(V)
    pos = {r: i for i, r in enumerate(rows)}
    for k in np.unique(pairs[:, 3]).astype(int):
        sel = np.where(pairs[:, 3] == k)[0]
        n = TP - k - a + 1
        if n < 30:
            continue
        X = E.norm_rows(E.moves(C, a, a - 1, n))
        Y = E.norm_rows(E.moves(C, b, a - 1 + k, n))
        ia = np.array([pos[int(x)] for x in pairs[sel, 1]])
        ib = np.array([pos[int(x)] for x in pairs[sel, 2]])
        r = np.einsum('ij,ij->i', X[ia].astype(np.float64), Y[ib].astype(np.float64))
        ok = cov[ia] & cov[ib]
        out[sel[ok]] = r[ok]
    return out


def main():
    data = E.Data()
    per = {t: (p0, p1) for t, p0, p1 in data.periods()}
    years = list(range(2006, 2021))
    nc = len(U.COMBOS)
    same = np.zeros((nc, 1 + E.NULLS, len(years)))
    tested = np.zeros((nc, 1 + E.NULLS, len(years)))
    signed = np.zeros((nc, 1 + E.NULLS, len(years)))
    for yi, y in enumerate(years):
        d = np.load(U.OUT / 'real' / f'Y{y}.npz')
        top = d['top']  # [combo, copy, TOP, 5]
        q0, q1 = per[f'Y{y + 5}']
        for ci, (num, a, b) in enumerate(U.COMBOS):
            for c in range(1 + E.NULLS):
                pr = top[ci, c]
                pr = pr[np.isfinite(pr[:, 0])]
                if len(pr) == 0:
                    continue
                rn = pair_r(data.v, q0, q1, a, b, pr)
                ok = np.isfinite(rn)
                same[ci, c, yi] = np.sum(np.sign(pr[ok, 4]) == np.sign(rn[ok]))
                tested[ci, c, yi] = ok.sum()
                signed[ci, c, yi] = np.sum(np.sign(pr[ok, 4]) * rn[ok])
        print(f'Y{y} -> Y{y + 5} done', flush=True)
    np.savez(U.OUT / 'replication.npz', same=same, tested=tested, signed=signed, years=np.array(years))
    lines = ['Replication: the 2,000 strongest pairs of each 5-year period (pooled over the 3 offsets) vs the next',
             'non-overlapping 5-year period (5 years later), test periods 2006..2020 -> 2011..2025. Real pairs vs the 2,000',
             'strongest pairs of each of the 8 null copies (their null r against the real r 5 years later).',
             'same sign % pooled over periods; z = (real - mean of null copies) / sd of the null copies. Pass: z >= 2.',
             '',
             f'{"#":>3} {"A":>4} {"B":>4} {"tested":>9} {"real same %":>12} {"null same %":>12} {"null sd":>8} {"z":>6}'
             f' {"real mean signed r":>19} {"null mean signed r":>19}  pass']
    for ci, (num, a, b) in enumerate(U.COMBOS):
        rate = same[ci].sum(1) / np.maximum(tested[ci].sum(1), 1)
        sr = signed[ci].sum(1) / np.maximum(tested[ci].sum(1), 1)
        mu, sd = rate[1:].mean(), rate[1:].std(ddof=1)
        z = (rate[0] - mu) / sd if sd > 0 else float('nan')
        lines.append(f'{num:>3} {a:>4} {b:>4} {int(tested[ci, 0].sum()):>9,} {100 * rate[0]:>11.1f}% {100 * mu:>11.1f}% '
                     f'{100 * sd:>7.2f} {z:>6.2f} {sr[0]:>19.4f} {sr[1:].mean():>19.4f}  {"yes" if z >= 2 else "no"}')
    out = '\n'.join(lines)
    open(os.path.join(HERE, 'replication.txt'), 'w').write(out + '\n')
    print(out)


if __name__ == '__main__':
    main()
