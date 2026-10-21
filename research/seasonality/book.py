"""The decile books, as monthly return series, for one export and one return variant.

    python3 -I -B book.py old|new raw|fix      -> WORK/book_<tag>_<variant>.npz

variant raw: the export's closes as they are.
variant fix: the shared dividend correction (pubdata.div_fix, summed over each month's days from 2022-07 on, in
    WORK/divfix_<tag>.npy, made by divfix.py) added to the monthly log returns, both to the signal's inputs and to
    the holding returns.

Books for holding month i (formed at the close of e[i-1], held to the close of e[i]):
  same      the universe; long the top tenth by `same`, short the bottom tenth (rank < n/10 short, >= 9n/10 long)
  same500   the same within the 500 largest
  other     the universe sorted on `other` (the other months of the same ten lag-years, lags 1..120)
  other_kln the universe sorted on `other_kln` (Keloharju et al.'s other months: skips lags 1..11)
Equal weight, $1 long and $1 short, buy and hold within the month. Leg return = mean of the stocks' simple returns
exp(g) - 1. Long-short gross = long - short (market-relative legs: leg - the export's equal-weight market). Costs
5 bps x turnover, turnover = sum over stocks of |new weight - drifted old weight|.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import c2eval, np, WORK, COST_BPS, DECILE, log, month_label  # noqa: E402
from build import signals  # noqa: E402

MIN_BOOK = 100   # a book is formed only with >= 100 stocks holding the signal


def rank(x):
    return np.argsort(np.argsort(x, kind='mergesort'), kind='mergesort')


def run(tag, variant):
    z = np.load(WORK / f'panel_{tag}.npz')
    months, e, days = z['months'], z['e'], z['days']
    M = len(months)
    r = z['r'].astype(np.float64)
    g = z['g'].astype(np.float64)
    if variant == 'fix':
        cf = np.load(WORK / f'divfix_{tag}.npy').astype(np.float64)
        assert cf.shape == r.shape
        r = r + cf
        g = g + cf
        S = signals(r)
        sig = {'same': S['same'], 'other': S['other'], 'other_kln': S['other_kln']}
    else:
        sig = {'same': z['same'].astype(np.float64), 'other': z['other'].astype(np.float64),
               'other_kln': z['other_kln'].astype(np.float64)}
    uni_f, big_f = z['uni_f'], z['big_f']
    R = np.expm1(g)
    mkt = z['mkt']
    out = {'months': months, 'e': e, 'days': days, 'mkt': mkt}
    for name, key, mask in (('same', 'same', uni_f), ('same500', 'same', big_f), ('other', 'other', uni_f),
                            ('other_kln', 'other_kln', uni_f)):
        rec = {k: [] for k in ('i', 'L', 'S', 'gross', 'turn', 'cost', 'net', 'nL', 'nS', 'n', 'partial',
                               'sigL', 'sigS', 'otherL', 'otherS')}
        prev = None   # drifted weights of the previous book [N]
        for i in range(1, M):
            s = sig[key][:, i]
            m = np.nonzero(mask[i] & np.isfinite(s))[0]
            if len(m) < MIN_BOOK:
                prev = None
                continue
            n = len(m)
            rk = rank(s[m])
            Sx, Lx = m[rk < n * DECILE], m[rk >= n * (1 - DECILE)]
            rL, rS = R[Lx, i], R[Sx, i]
            assert np.isfinite(rL).all() and np.isfinite(rS).all(), (name, i)
            w = np.zeros(R.shape[0])
            w[Lx] = 1.0 / len(Lx)
            w[Sx] = -1.0 / len(Sx)
            to = np.abs(w - (prev if prev is not None else 0.0)).sum()
            prev = w * (1 + np.nan_to_num(R[:, i]))
            L, Sr = float(rL.mean()), float(rS.mean())
            cost = COST_BPS / 1e4 * to
            for k, v in (('i', i), ('L', L), ('S', Sr), ('gross', L - Sr), ('turn', to), ('cost', cost),
                         ('net', L - Sr - cost), ('nL', len(Lx)), ('nS', len(Sx)), ('n', n), ('partial', i == M - 1),
                         ('sigL', float(np.mean(s[Lx]))), ('sigS', float(np.mean(s[Sx]))),
                         ('otherL', float(np.nanmean(sig['other_kln'][Lx, i]))),
                         ('otherS', float(np.nanmean(sig['other_kln'][Sx, i])))):
                rec[k].append(v)
        for k, v in rec.items():
            out[f'{name}__{k}'] = np.array(v)
        i0 = rec['i'][0]
        log(f'{tag} {variant} {name}: {len(rec["i"])} months from {month_label(months[i0])}, '
            f'mean turnover {np.mean(rec["turn"][1:]):.2f}, mean n {np.mean(rec["n"]):.0f}')
    np.savez(WORK / f'book_{tag}_{variant}.npz', **out)


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
