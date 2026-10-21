"""PRET and ID for every stock and day (rules in common.py's docstring and the .md).

    python3 -I -B signals.py old|new plain|divfix     -> WORK/pret_<tag>_<px>.npy, WORK/id_<tag>_<px>.npy  [N, T] float32

Value at [j, t] uses closes through t - 21 only (NaN where fewer than 207 valid daily returns in t-250 .. t-21).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (c2eval, np, WORK, EXPORTS, FAR, NEAR, MIN_VALID, MAX_LOG, log, prices)  # noqa: E402

CHUNK = 1000
ZERO = 1e-9                  # |PRET| below this counts as 0 (sign 0, so ID 0)


def run(tag, px):
    E = c2eval.load_export(EXPORTS[tag], arrays=('close',))
    N, T = E['close'].shape
    PRET = np.full((N, T), np.nan, np.float32)
    ID = np.full((N, T), np.nan, np.float32)
    dropped = 0
    traded = 0
    for a in range(0, N, CHUNK):
        b = min(N, a + CHUNK)
        P, ok = prices(tag, E, px, slice(a, b))
        n = b - a
        r = np.full((n, T), np.nan)
        with np.errstate(all='ignore'):
            r[:, 1:] = np.log(P[:, 1:] / P[:, :-1])
        r[~ok] = np.nan                                  # gap days: the move lands on the next traded day
        fin = np.isfinite(r)
        big = fin & (np.abs(r) > MAX_LOG)
        dropped += int(big.sum())
        traded += int(fin.sum())
        r[big] = np.nan
        valid = np.isfinite(r)
        rz = np.where(valid, r, 0.0)

        def cum(X):
            cs = np.zeros((n, T + 1))
            np.cumsum(X, 1, out=cs[:, 1:])
            return cs

        # window for t: returns d = t-FAR+1 .. t-NEAR -> cs[:, t-NEAR+1] - cs[:, t-FAR+1]
        t = np.arange(FAR, T)
        hi, lo = t - NEAR + 1, t - FAR + 1

        def win(cs):
            return cs[:, hi] - cs[:, lo]

        cs = cum(rz)
        s = win(cs)
        del cs
        cs = cum(valid.astype(np.float64))
        nv = win(cs)
        cs = cum((valid & (rz > 0)).astype(np.float64))
        npos = win(cs)
        cs = cum((valid & (rz < 0)).astype(np.float64))
        nneg = win(cs)
        del cs, r, rz, valid, P, ok
        good = nv >= MIN_VALID - 0.5
        with np.errstate(all='ignore'):
            s = np.where(np.abs(s) < ZERO, 0.0, s)       # a round trip to the cent is PRET 0, not +-1e-15
            sg = np.sign(s)
            idv = sg * (nneg - npos) / nv
        PRET[a:b, FAR:] = np.where(good, s, np.nan)
        ID[a:b, FAR:] = np.where(good, idv, np.nan)
        log(f'{tag} {px} rows {a}..{b}')
    np.save(WORK / f'pret_{tag}_{px}.npy', PRET)
    np.save(WORK / f'id_{tag}_{px}.npy', ID)
    msg = (f'{tag} {px}: finite PRET stock-days {int(np.isfinite(PRET).sum())}; daily returns with |log r| > {MAX_LOG} '
           f'dropped as bad prints: {dropped} of {traded} ({dropped / traded:.5%})')
    log(msg)
    with open(WORK / f'signals_{tag}_{px}.txt', 'w') as f:
        f.write(msg + '\n')


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
