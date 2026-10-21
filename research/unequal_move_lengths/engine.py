"""Unequal move lengths: real vs chance with overlapping observations and a block sign-flip null.

Modes:
  python3 -I engine.py cal    noise calibration only (no real stock-to-stock correlations)
  python3 -I engine.py real   real stock-to-stock matrices: counts, 2,000 strongest real and null pairs per period
Writes one .npz per correlation period under ~/.local/share/correlation2/experiments/unequal_move_lengths/{cal,real}/.
"""
import os
import sys
import time

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '8')

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umlcore as U  # noqa: E402
from umlcore import c2eval  # noqa: E402

PERIOD = 1260
CAP = 1500
NULLS = 8
TOP = 2000
ZTHR = (3.0, 4.0, 5.0)
SEED = 20261021
N_NOISE_KINDS = 11
N_PER_KIND = 20
NOISE_KINDS = ['iid', 'fat', 'garch', 'ar0.3', 'ar0.7+cycle', 'ar0.9', 'ar0.97', 'weekly', 'monthly', 'sparse', 'counts']


def block_len(a, b):
    return max(63, 2 * max(a, b))


class Data:
    def __init__(self, export=c2eval.EXPORT_OLD):
        t0 = time.time()
        e = c2eval.load_export(export)
        self.export = export
        self.days = e['days']
        self.T = len(self.days)
        self.v = np.asarray(e['v'], dtype=np.float32)
        self.close = e['close']
        self.dv = e['dollar_volume']
        self.market = np.asarray(e['market'], dtype=np.float64)
        self.N = self.v.shape[0]
        fin = np.isfinite(self.v).sum(0)
        self.first_v = int(np.argmax(fin >= 100))
        self._resid = None
        print(f'loaded {export}: N={self.N} T={self.T} first_v={self.first_v} ({c2eval.iso(self.days[self.first_v])}) '
              f'[{time.time() - t0:.0f}s]', flush=True)

    @property
    def resid(self):
        if self._resid is None:
            t0 = time.time()
            out = np.empty((self.N, self.T), dtype=np.float32)
            for ch in np.array_split(np.arange(self.N), 16):
                out[ch] = U.residuals(np.asarray(self.close[ch]), self.market)
            self._resid = out
            print(f'residuals computed [{time.time() - t0:.0f}s]', flush=True)
        return self._resid

    def periods(self):
        """(tag, p0, p1): one 5-year period ending before each test year 2006..2024, one ending at the data end, and
        the full history."""
        out = []
        for y in range(2006, 2025):
            s = c2eval.day_index(self.days, f'{y}-06-28')
            out.append((f'Y{y}', max(s - PERIOD, self.first_v), s))
        out.append(('Y2025', max(self.T - PERIOD, self.first_v), self.T))
        out.append(('full', self.first_v, self.T))
        return out

    def universe(self, p0, p1, cap=CAP):
        vv = self.v[:, p0:p1]
        cov = np.isfinite(vv).sum(1) >= 0.9 * (p1 - p0)
        idx = np.where(cov)[0]
        lo = max(p1 - 252, 0)
        c = np.asarray(self.close[idx, lo:p1], dtype=np.float32)
        d = np.asarray(self.dv[idx, lo:p1], dtype=np.float32)

        def upper_median(m):
            s = np.sort(np.where(np.isfinite(m), m, np.inf), axis=1)
            k = np.isfinite(m).sum(1)
            out = np.zeros(len(m), dtype=np.float32)
            ok = k > 0
            out[ok] = s[ok, k[ok] // 2]
            return out
        mc, md = upper_median(c), upper_median(d)
        ok = (mc >= 5.0) & (md >= 5e6)
        idx, md = idx[ok], md[ok]
        order = np.argsort(-md, kind='stable')[:cap]
        return np.sort(idx[order])


_SIGNS = {}


def global_signs(T, L, num, m, role):
    """One block-sign sequence over all T rows of the export per (combination, null copy, role), sliced by each
    period, so a null copy reuses the same signs on the same days in overlapping periods, as the real data does.
    role 0 = stock residuals, 1 = noise residuals, 3 = v flipped in blocks; role 2 = one iid sign per day (old null)."""
    key = (T, L, num, m, role)
    if key not in _SIGNS:
        rng = np.random.default_rng([SEED, num, m, role])
        _SIGNS[key] = rng.choice([-1.0, 1.0], size=T) if role == 2 else U.block_signs(T, L, rng)
    return _SIGNS[key]


def cums(V):
    """V [N, T] float (NaN -> 0) -> cumulative sums [N, T+1] float64 with a leading 0."""
    V = np.nan_to_num(np.asarray(V, dtype=np.float64), nan=0.0)
    C = np.zeros((V.shape[0], V.shape[1] + 1))
    np.cumsum(V, axis=1, out=C[:, 1:])
    return C


def moves(C, h, start, n, step=1):
    """Moves of h days ending at local rows start, start+step, ... (n of them), in census units (sum / sqrt(h))."""
    e = start + step * np.arange(n)
    return (C[:, e + 1] - C[:, e + 1 - h]) / np.sqrt(h)


def norm_rows(X):
    X = X - X.mean(1, keepdims=True)
    nr = np.linalg.norm(X, axis=1, keepdims=True)
    bad = nr[:, 0] < 1e-9
    nr[bad] = 1.0
    X = X / nr
    X[bad] = 0.0
    return X.astype(np.float32)


def rthr(neff):
    """|r| cut-offs equivalent to |z| >= ZTHR at the given effective number of observations."""
    if neff <= 3.0:
        return None
    return np.tanh(np.array(ZTHR) / np.sqrt(neff - 3.0))


def count(r, thr):
    a = np.abs(r)
    return [int(np.count_nonzero(a >= t)) for t in thr]


def null_v(R, s):
    """Census variable rebuilt from residuals R [N, 252 + T_P] (demeaned) multiplied by signs s; returns the last T_P
    columns (NaN where unavailable)."""
    return U.standardize(R * s[None, :])[:, U.LOOKBACK:]


def make_noise(T, seed=SEED):
    """220 raw noise 'residual' series [220, T]: 11 kinds x 20."""
    rng = np.random.default_rng([seed, 7])
    out = np.zeros((N_NOISE_KINDS * N_PER_KIND, T))
    n = N_PER_KIND
    for kind in range(N_NOISE_KINDS):
        e = rng.standard_normal((n, T))
        name = NOISE_KINDS[kind]
        if name == 'iid':
            x = e
        elif name == 'fat':
            x = e * np.where(rng.random((n, T)) < 0.05, 4.0, 1.0)
        elif name == 'garch':
            x = np.zeros((n, T))
            s2 = np.ones(n)
            prev = np.zeros(n)
            for t in range(T):
                s2 = 0.05 + 0.10 * prev ** 2 + 0.85 * s2
                prev = np.sqrt(s2) * e[:, t]
                x[:, t] = prev
        elif name.startswith('ar'):
            phi = float(name[2:].split('+')[0])
            x = np.zeros((n, T))
            x[:, 0] = e[:, 0] / np.sqrt(1 - phi * phi)
            for t in range(1, T):
                x[:, t] = phi * x[:, t - 1] + e[:, t]
            if 'cycle' in name:
                ph = rng.random((n, 1)) * 2 * np.pi
                x = x + 3.0 * np.sin(2 * np.pi * np.arange(T)[None, :] / 252.0 + ph)
        elif name == 'weekly':
            x = np.where((np.arange(T) % 5 == 4)[None, :], e, 0.0)
        elif name == 'monthly':
            x = np.where((np.arange(T) % 21 == 20)[None, :], e, 0.0)
        elif name == 'sparse':
            x = (rng.random((n, T)) < 0.03).astype(np.float64)
        elif name == 'counts':
            x = rng.poisson(0.7, (n, T)).astype(np.float64)
        out[kind * n:(kind + 1) * n] = x
    return out


def kind_counts(r, thr, axis):
    """Counts of |r| >= thr per noise series (kind = series // 20); the noise is on `axis` (0 rows, 1 columns)."""
    a = np.abs(r)
    if axis == 1:
        a = a.T
    return np.stack([(a >= t).sum(1) for t in thr], axis=-1)  # [series, thr]


def flip_side(a, b):
    """The null flips the side with the longer move (B if b > a, else A); see the experiment file."""
    return 'B' if b > a else 'A'


def _copies(C_real, C_nulls, h, start, n):
    return [norm_rows(moves(C_real, h, start, n))] + [norm_rows(moves(C, h, start, n)) for C in C_nulls]


def _pairs(xs, ys):
    """Yield (copy, X, Y): one side fixed (a list of one), the other real + null copies."""
    if len(xs) == 1:
        for c, Y in enumerate(ys):
            yield c, xs[0], Y
    else:
        for c, X in enumerate(xs):
            yield c, X, ys[0]


def run_period(data, pi, tag, p0, p1, mode, noise_raw=None, noise_v=None, combos=U.COMBOS):
    t0 = time.time()
    TP = p1 - p0
    uni = data.universe(p0, p1)
    NU = len(uni)
    CS = cums(data.v[uni, p0:p1])
    R = data.resid[uni, p0 - U.LOOKBACK:p1].astype(np.float64)
    R = R - np.nanmean(R, axis=1, keepdims=True)
    res = {'tag': tag, 'p0': p0, 'p1': p1, 'universe': uni}
    nc = len(combos)
    if mode == 'real':
        counts = np.zeros((nc, 3, 1 + NULLS, len(ZTHR)), dtype=np.int64)
        tops = {}
    else:
        # methods: 0 = this null, noise A -> stocks B; 1 = this null, stocks A -> noise B;
        # 2 = old method (non-overlapping, per-observation flip of B), noise A -> stocks B;
        # 3 = overlapping, blocks of B's v flipped (not residuals), noise A -> stocks B
        counts = np.zeros((nc, 3, 4, N_NOISE_KINDS * N_PER_KIND, 1 + NULLS, len(ZTHR)), dtype=np.int32)
        CN = cums(noise_v[:, p0:p1])
        RN = noise_raw[:, p0 - U.LOOKBACK:p1]
        RN = RN - RN.mean(1, keepdims=True)
        VB_raw = np.nan_to_num(data.v[uni, p0:p1].astype(np.float64), nan=0.0)
    nobs = np.zeros((nc, 3), dtype=np.int64)
    neffs = np.zeros((nc, 3))
    for ci, (num, a, b) in enumerate(combos):
        L = block_len(a, b)
        side = flip_side(a, b)
        cab = U.c_ab(a, b)
        CSn = []
        for m in range(NULLS):
            s = global_signs(data.T, L, num, m, 0)[p0 - U.LOOKBACK:p1]
            CSn.append(cums(null_v(R, s)))
        if mode == 'cal':
            CNn, CVf = [], []
            for m in range(NULLS):
                s = global_signs(data.T, L, num, m, 1)[p0 - U.LOOKBACK:p1]
                CNn.append(cums(null_v(RN, s)))
                s = global_signs(data.T, L, num, m, 3)[p0:p1]
                CVf.append(cums(VB_raw * s[None, :]))
        if mode == 'real':
            cand = [[] for _ in range(1 + NULLS)]
        for ki, k in enumerate(U.offsets(b)):
            n = TP - k - a + 1
            nobs[ci, ki] = n
            if n < 30:
                continue
            neff = n / cab
            neffs[ci, ki] = neff
            thr = rthr(neff)
            if thr is None:
                continue
            sq = np.sqrt(neff - 3.0)
            a0, b0 = a - 1, a - 1 + k  # first A move ends at local row a-1, its B move at a-1+k
            if side == 'B':
                xs_s = [norm_rows(moves(CS, a, a0, n))]
                ys_s = _copies(CS, CSn, b, b0, n)
            else:
                xs_s = _copies(CS, CSn, a, a0, n)
                ys_s = [norm_rows(moves(CS, b, b0, n))]
            if mode == 'real':
                for c, X, Y in _pairs(xs_s, ys_s):
                    r = X @ Y.T
                    np.fill_diagonal(r, 0.0)
                    counts[ci, ki, c] = count(r, thr)
                    flat = np.abs(r).ravel()
                    top = np.argpartition(-flat, TOP)[:TOP] if flat.size > TOP else np.arange(flat.size)
                    ia, ib = np.unravel_index(top, r.shape)
                    rv = r[ia, ib].astype(np.float64)
                    z = np.arctanh(np.clip(rv, -0.999999, 0.999999)) * sq
                    cand[c].append(np.stack([z, uni[ia], uni[ib], np.full(len(ia), k), rv], axis=1))
            else:
                # method 0: noise A -> stocks B
                if side == 'B':
                    xs, ys = [norm_rows(moves(CN, a, a0, n))], ys_s
                else:
                    xs, ys = _copies(CN, CNn, a, a0, n), [norm_rows(moves(CS, b, b0, n))]
                for c, X, Y in _pairs(xs, ys):
                    counts[ci, ki, 0, :, c] = kind_counts(X @ Y.T, thr, axis=0)
                # method 1: stocks A -> noise B
                if side == 'B':
                    xs, ys = [norm_rows(moves(CS, a, a0, n))], _copies(CN, CNn, b, b0, n)
                else:
                    xs, ys = xs_s, [norm_rows(moves(CN, b, b0, n))]
                for c, X, Y in _pairs(xs, ys):
                    counts[ci, ki, 1, :, c] = kind_counts(X @ Y.T, thr, axis=1)
                # method 3: blocks of B's v flipped (B always), noise A -> stocks B
                XN = norm_rows(moves(CN, a, a0, n))
                for c, Y in enumerate(_copies(CS, CVf, b, b0, n)):
                    counts[ci, ki, 3, :, c] = kind_counts(XN @ Y.T, thr, axis=0)
                # method 2: old method, non-overlapping observations stepping by the longer move, each B observation flipped
                step = max(a, b)
                g0 = step - 1
                n_old = (TP - 1 - k - g0) // step + 1
                if n_old >= 30:
                    thr_old = np.tanh(np.array(ZTHR) / np.sqrt(n_old - 3.0))
                    XNo = norm_rows(moves(CN, a, g0, n_old, step))
                    YBo = moves(CS, b, g0 + k, n_old, step)
                    counts[ci, ki, 2, :, 0] = kind_counts(XNo @ norm_rows(YBo).T, thr_old, axis=0)
                    for m in range(NULLS):
                        so = global_signs(data.T, 1, num, m, 2)[p0 + g0 + k + step * np.arange(n_old)]
                        counts[ci, ki, 2, :, 1 + m] = kind_counts(XNo @ norm_rows(YBo * so[None, :]).T, thr_old, axis=0)
        if mode == 'real':
            for c in range(1 + NULLS):
                if not cand[c]:
                    tops[(num, c)] = np.zeros((0, 5))
                    continue
                allc = np.concatenate(cand[c])
                o = np.argsort(-np.abs(allc[:, 0]), kind='stable')[:TOP]
                tops[(num, c)] = allc[o]
    res['counts'] = counts
    res['nobs'] = nobs
    res['neff'] = neffs
    if mode == 'real':
        res['top'] = np.stack([np.pad(tops[(num, c)], ((0, TOP - len(tops[(num, c)])), (0, 0)), constant_values=np.nan)
                               for num, _, _ in combos for c in range(1 + NULLS)]).reshape(len(combos), 1 + NULLS, TOP, 5)
    print(f'{tag}: rows {p0}..{p1} ({TP} days, {c2eval.iso(data.days[p0])}..{c2eval.iso(data.days[p1 - 1])}), '
          f'{NU} stocks [{time.time() - t0:.0f}s]', flush=True)
    return res


def main():
    mode = sys.argv[1]
    only = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    data = Data()
    _ = data.resid
    outdir = U.OUT / mode
    outdir.mkdir(parents=True, exist_ok=True)
    noise_raw = noise_v = None
    if mode == 'cal':
        noise_raw = make_noise(data.T)
        noise_v = U.standardize(noise_raw)
    for pi, (tag, p0, p1) in enumerate(data.periods()):
        if only and tag not in only:
            continue
        res = run_period(data, pi, tag, p0, p1, mode, noise_raw, noise_v)
        np.savez(outdir / f'{tag}.npz', **{k: v for k, v in res.items()})


if __name__ == '__main__':
    main()
